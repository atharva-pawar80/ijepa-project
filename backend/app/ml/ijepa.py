"""
Full I-JEPA model: wires together
  - context encoder (VisionTransformer, sees only visible/context patches)
  - target encoder  (VisionTransformer, EMA copy, sees the FULL image, frozen)
  - predictor       (narrow ViT, context tokens + mask tokens -> predictions)
  - multi-block masking (context block + M target blocks)
  - loss            (avg squared L2 distance in representation space)

This file also contains an "overfit one batch" sanity test in __main__ --
the single most important debugging step before trusting a training pipeline.
If the model can't drive the loss to near-zero on a FIXED batch it's allowed
to see over and over, something upstream is broken (wrong indices, gradient
not flowing, EMA momentum too aggressive, etc).
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(__file__))
from encoders.vision_transformer import VisionTransformer
from predictor import Predictor
from masking import build_ijepa_masks
from training.ema import make_target_encoder, MomentumSchedule, update_target_encoder
from training.loss import ijepa_loss


class IJEPA(nn.Module):
    def __init__(
        self,
        img_size=96,
        patch_size=8,
        in_chans=3,
        embed_dim=192,
        encoder_depth=6,
        encoder_heads=6,
        predictor_embed_dim=384,
        predictor_depth=6,
        predictor_heads=6,
    ):
        super().__init__()
        self.grid_size = img_size // patch_size

        self.context_encoder = VisionTransformer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=encoder_depth, num_heads=encoder_heads,
        )
        # Target encoder: exact weight copy at init, frozen from gradients,
        # only ever updated via EMA (see training/ema.py).
        self.target_encoder = make_target_encoder(self.context_encoder)

        self.predictor = Predictor(
            encoder_embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            grid_size=self.grid_size,
        )

    def forward(self, images, context_indices, target_blocks):
        """
        images: (B, C, H, W)
        context_indices: (B, Nc) long -- same for the whole batch is fine
            (paper restricts mask size to be identical within a batch for
            efficient collation; sampling can still differ per-image if you
            extend this, as long as Nc is constant)
        target_blocks: list of M (indices, box) tuples from masking.py,
            each indices: (Nt_i,) long -- will be expanded across the batch

        Returns: (predictions, targets) -- both lists of M tensors, ready to
            pass straight into training.loss.ijepa_loss
        """
        B = images.shape[0]

        # Context encoder only ever sees the visible/context patches.
        context_tokens = self.context_encoder(images, keep_indices=context_indices)

        # Target encoder ALWAYS sees the full image -- masking is applied to
        # its OUTPUT, never its input (paper Appendix C, Table 11).
        with torch.no_grad():
            full_target_tokens = self.target_encoder(images)  # (B, N, D)

        predictions, targets = [], []
        for indices, _box in target_blocks:
            target_idx_batch = indices.unsqueeze(0).expand(B, -1)  # (B, Nt_i)

            # Predictor gets context tokens + mask tokens at this block's positions.
            pred = self.predictor(context_tokens, context_indices, target_idx_batch)

            # Ground truth: gather the target encoder's own output at those
            # same positions, then detach -- no gradient should ever flow
            # into the target encoder through this path (only EMA touches it).
            D = full_target_tokens.shape[-1]
            gather_idx = target_idx_batch.unsqueeze(-1).expand(-1, -1, D)
            target = torch.gather(full_target_tokens, dim=1, index=gather_idx).detach()

            predictions.append(pred)
            targets.append(target)

        return predictions, targets


def embedding_variance(tokens):
    """Collapse-detection metric: variance of embeddings across the batch.
    If this goes to ~0, the target encoder is producing a constant output
    regardless of input -- representation collapse. Log this every epoch.
    """
    # tokens: (B, N, D) -> flatten batch/patch dims, compute variance per-dim,
    # then average -- a simple, standard collapse signal.
    flat = tokens.reshape(-1, tokens.shape[-1])
    return flat.var(dim=0).mean().item()


if __name__ == "__main__":
    # --- Overfit-one-batch sanity test ---
    torch.manual_seed(0)

    img_size, patch_size = 96, 8
    model = IJEPA(img_size=img_size, patch_size=patch_size, embed_dim=192,
                   encoder_depth=4, predictor_depth=4)  # small/fast for a quick test

    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=1e-3,
    )
    # NOTE: target_encoder parameters are NOT in the optimizer -- it is only
    # ever updated via EMA below, never via gradient descent.

    num_steps = 200
    schedule = MomentumSchedule(base_momentum=0.996, final_momentum=1.0, total_steps=num_steps)

    # Fixed batch + fixed masks, reused every step -- this is the "can it
    # overfit one batch" test. Real training would resample masks per step.
    B = 8
    images = torch.randn(B, 3, img_size, img_size)
    grid_size = img_size // patch_size
    context_indices_single, _, target_blocks = build_ijepa_masks(grid_size, grid_size)
    context_indices = context_indices_single.unsqueeze(0).expand(B, -1)

    print(f"context patches: {context_indices.shape[1]}, "
          f"target blocks: {[idx.numel() for idx, _ in target_blocks]}")

    losses = []
    variances = []
    for step in range(num_steps):
        predictions, targets = model(images, context_indices, target_blocks)
        loss = ijepa_loss(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        momentum = schedule.get_momentum(step)
        update_target_encoder(model.target_encoder, model.context_encoder, momentum)

        losses.append(loss.item())
        if step % 20 == 0 or step == num_steps - 1:
            with torch.no_grad():
                full_target = model.target_encoder(images)
            var = embedding_variance(full_target)
            variances.append(var)
            print(f"step {step:4d}  loss={loss.item():.4f}  "
                  f"momentum={momentum:.4f}  target_embed_variance={var:.4f}")

    print(f"\nloss: first={losses[0]:.4f} -> last={losses[-1]:.4f}")
    assert losses[-1] < losses[0] * 0.1, (
        "Loss did not drop enough on a fixed batch -- something is broken "
        "(check gradient flow, indices, or EMA momentum)."
    )
    assert variances[-1] > 1e-4, (
        "Target embedding variance collapsed near zero -- representation "
        "collapse. Check EMA momentum isn't too low, or LR isn't too high."
    )
    print("\nAll checks passed: loss dropped >90% and target embeddings did not collapse.")
