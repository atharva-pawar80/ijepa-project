"""
Visualize I-JEPA multi-block masking (context + 4 target blocks) so you can
eyeball it against the paper's Figure 4, instead of trusting printed numbers.

Run: python visualize_masking.py
Output: masking_visualization.png
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from masking import build_ijepa_masks


def make_synthetic_image(img_size=224):
    """A simple synthetic image (concentric shapes) -- stands in for a real
    photo so we can see mask placement clearly without needing external files.
    """
    x = np.linspace(-1, 1, img_size)
    y = np.linspace(-1, 1, img_size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    angle = np.arctan2(yy, xx)

    img = np.zeros((img_size, img_size, 3))
    img[..., 0] = 0.5 + 0.5 * np.sin(6 * angle + 3 * r)
    img[..., 1] = 0.5 + 0.5 * np.cos(5 * r * np.pi)
    img[..., 2] = 0.5 + 0.5 * np.sin(4 * angle)
    return np.clip(img, 0, 1)


def draw_patch_grid(ax, boxes_with_colors, grid_h, grid_w, img_size, patch_size, title):
    """boxes_with_colors: list of (indices, color, label) to overlay as patches."""
    ax.imshow(make_synthetic_image(img_size))
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    for indices, color, alpha in boxes_with_colors:
        for idx in indices.tolist():
            row, col = divmod(idx, grid_w)
            rect = patches.Rectangle(
                (col * patch_size, row * patch_size),
                patch_size, patch_size,
                linewidth=0, facecolor=color, alpha=alpha,
            )
            ax.add_patch(rect)


def main():
    random.seed(1)
    torch.manual_seed(1)

    img_size = 224
    patch_size = 16
    grid_h = grid_w = img_size // patch_size  # 14x14 = 196 patches

    context_indices, context_box, target_blocks = build_ijepa_masks(grid_h, grid_w)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: original image, no overlay
    axes[0].imshow(make_synthetic_image(img_size))
    axes[0].set_title("Original image (synthetic)", fontsize=11)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Panel 2: context block (after removing target overlap) in blue
    draw_patch_grid(
        axes[1],
        [(context_indices, "dodgerblue", 0.6)],
        grid_h, grid_w, img_size, patch_size,
        f"Context block\n({context_indices.numel()}/{grid_h*grid_w} patches, "
        f"overlap with targets removed)",
    )

    # Panel 3: the 4 target blocks, each a different color
    colors = ["red", "orange", "yellow", "magenta"]
    overlays = [
        (idx, colors[i % len(colors)], 0.55)
        for i, (idx, _) in enumerate(target_blocks)
    ]
    draw_patch_grid(
        axes[2], overlays, grid_h, grid_w, img_size, patch_size,
        "4 target blocks\n(scale 0.15-0.2 each, independently sampled)",
    )

    plt.tight_layout()
    plt.savefig("masking_visualization.png", dpi=150)
    print("Saved masking_visualization.png")

    # Print scale/ratio numbers to cross-check against paper's Table 6/8/9
    total = grid_h * grid_w
    print(f"\ncontext ratio: {context_indices.numel()/total:.3f}  "
          f"(paper Table 6 multi-block avg ratio: 0.25)")
    for i, (idx, _) in enumerate(target_blocks):
        print(f"target block {i} scale: {idx.numel()/total:.3f}  "
              f"(paper range: 0.15-0.2)")


if __name__ == "__main__":
    main()
