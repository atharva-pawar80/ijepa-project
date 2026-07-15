"""
Multi-block masking strategy for I-JEPA, following the paper exactly
(Section 3 "Context"/"Targets", and Appendix A.1 "Masking").

Key paper facts this file encodes:
  - Target blocks: M=4 blocks, scale in (0.15, 0.2), aspect ratio in (0.75, 1.5).
    These are (possibly overlapping) blocks on the patch grid.
  - Context block: 1 block, scale in (0.85, 1.0), UNIT aspect ratio (i.e. square,
    not just "wide range" like the targets).
  - The context block has any patches that overlap with ANY target block removed
    -> this is what makes the prediction task non-trivial (paper, Sec 3 "Context").
  - Masking happens on indices into the patch grid, not on pixels. The target
    encoder always sees the FULL image; masking is applied to its OUTPUT
    (see paper Table 11 / Appendix C) -- this file only produces the index
    sets; where you apply them is handled in the model code, not here.
  - "Scale" = fraction of total patch-grid AREA the block covers (paper Table 8/9
    calls this out directly with an Avg. Ratio column).
"""

import math
import random
import torch


def _sample_block_indices(grid_h, grid_w, scale_range, aspect_ratio_range):
    """Sample ONE block's patch indices on a (grid_h x grid_w) patch grid.

    Returns:
        indices: 1D LongTensor of flattened patch indices belonging to the block
        (top, left, h, w): the block's location, for visualization/debugging
    """
    total_patches = grid_h * grid_w

    # Try a handful of times in case a sampled (h, w) doesn't fit the grid
    # (this can happen with extreme aspect ratios) -- standard practice, and
    # what the reference I-JEPA implementation does too.
    for _ in range(10):
        target_area = random.uniform(*scale_range) * total_patches
        aspect_ratio = math.exp(random.uniform(math.log(aspect_ratio_range[0]),
                                                math.log(aspect_ratio_range[1])))
        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))
        h = min(h, grid_h)
        w = min(w, grid_w)
        if h <= 0 or w <= 0:
            continue

        top = random.randint(0, grid_h - h)
        left = random.randint(0, grid_w - w)

        rows = torch.arange(top, top + h)
        cols = torch.arange(left, left + w)
        grid = torch.cartesian_prod(rows, cols)          # (h*w, 2)
        indices = grid[:, 0] * grid_w + grid[:, 1]        # flatten to 1D index
        return indices, (top, left, h, w)

    raise RuntimeError("Could not sample a valid block after 10 attempts")


def sample_target_blocks(grid_h, grid_w, num_blocks=4,
                          scale_range=(0.15, 0.2), aspect_ratio_range=(0.75, 1.5)):
    """Sample M target blocks (paper defaults: M=4).

    Returns:
        list of (indices, (top, left, h, w)) tuples, one per target block.
        Blocks may overlap each other -- the paper allows this.
    """
    return [
        _sample_block_indices(grid_h, grid_w, scale_range, aspect_ratio_range)
        for _ in range(num_blocks)
    ]


def sample_context_block(grid_h, grid_w, target_blocks,
                          scale_range=(0.85, 1.0)):
    """Sample the single context block and remove any patches that overlap
    with any of the target blocks (paper, Sec 3 "Context").

    Note aspect_ratio is fixed to 1.0 (unit) here, per the paper -- this is
    NOT a configurable range like the target blocks.
    """
    context_indices, box = _sample_block_indices(
        grid_h, grid_w, scale_range, aspect_ratio_range=(1.0, 1.0)
    )

    target_index_set = set()
    for idx, _ in target_blocks:
        target_index_set.update(idx.tolist())

    keep_mask = torch.tensor(
        [i.item() not in target_index_set for i in context_indices]
    )
    context_indices = context_indices[keep_mask]

    if context_indices.numel() == 0:
        raise RuntimeError(
            "Context block fully overlapped by target blocks -- resample. "
            "This is rare but possible; caller should retry."
        )

    return context_indices, box


def build_ijepa_masks(grid_h, grid_w, num_target_blocks=4,
                       target_scale_range=(0.15, 0.2),
                       target_aspect_ratio_range=(0.75, 1.5),
                       context_scale_range=(0.85, 1.0),
                       max_resample=10):
    """Convenience wrapper: sample target blocks, then a context block with
    target overlap removed. Retries automatically if the context block ends
    up empty (rare, but possible with unlucky sampling).

    Returns:
        context_indices: 1D LongTensor, patch indices to feed the CONTEXT encoder
        target_blocks: list of (indices, box) for the M target blocks -- used to
            index into the TARGET encoder's (full-image) output, and to know
            where to place mask tokens for the predictor
    """
    for _ in range(max_resample):
        target_blocks = sample_target_blocks(
            grid_h, grid_w, num_target_blocks,
            target_scale_range, target_aspect_ratio_range,
        )
        try:
            context_indices, context_box = sample_context_block(
                grid_h, grid_w, target_blocks, context_scale_range
            )
            return context_indices, context_box, target_blocks
        except RuntimeError:
            continue
    raise RuntimeError("Failed to sample valid masks after max_resample attempts")


if __name__ == "__main__":
    # --- Sanity check: run this file directly ---
    random.seed(0)
    torch.manual_seed(0)

    grid_h = grid_w = 14  # e.g. 224/16 patch grid, like ViT-B/16 at 224x224

    context_indices, context_box, target_blocks = build_ijepa_masks(grid_h, grid_w)

    total_patches = grid_h * grid_w
    print(f"grid: {grid_h}x{grid_w} = {total_patches} patches")
    print(f"context block box (top,left,h,w): {context_box}")
    print(f"context patches kept after removing target overlap: {context_indices.numel()}")
    print(f"context ratio: {context_indices.numel() / total_patches:.3f}  "
          f"(paper's Table 6 reports avg ratio ~0.25 for multi-block)")

    for i, (idx, box) in enumerate(target_blocks):
        print(f"target block {i}: box(top,left,h,w)={box}  "
              f"num_patches={idx.numel()}  scale={idx.numel()/total_patches:.3f}")

    assert context_indices.numel() > 0
    assert len(target_blocks) == 4
    for idx, _ in target_blocks:
        assert idx.numel() > 0
    print("\nAll checks passed.")
