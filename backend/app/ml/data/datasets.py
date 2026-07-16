"""
Dataset loading for I-JEPA pretraining.

Deliberately simple: I-JEPA needs NO hand-crafted augmentations (that's the
whole point of the paper -- contrast with view-invariance methods that need
random-resized-crop, color jitter, etc). We only resize + normalize.

Uses CIFAR-10 by default for small-scale sanity training (paper itself
trains on ImageNet, but that's not practical for a first working pipeline).
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from masking import build_ijepa_masks_batch


def get_cifar10_datasets(data_dir="./data", img_size=96, use_fast_mirror=True):
    """Downloads CIFAR-10 and resizes to img_size x img_size so it's
    compatible with whatever patch_size your ViT uses.
    NOTE: CIFAR-10 images are natively 32x32 -- upsizing to 96 is a practical
    compromise for a fast toy-scale run, not something the paper does.

    use_fast_mirror=True (default): loads via the HuggingFace `datasets`
    library, which serves CIFAR-10 from HuggingFace's own CDN -- a
    genuinely different, fast host than torchvision's default
    (cs.toronto.edu), which is frequently very slow/flaky from Colab.
    Falls back to torchvision automatically if `datasets` isn't installed.
    """
    if use_fast_mirror:
        try:
            return _get_cifar10_via_huggingface(img_size)
        except ImportError:
            print("`datasets` library not available, falling back to "
                  "torchvision download (may be slow)...")

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    return train_set, test_set


class _HFCIFAR10(torch.utils.data.Dataset):
    """Wraps a HuggingFace `datasets` CIFAR-10 split as a torch Dataset,
    applying the same resize + normalize transform torchvision's version uses.
    """

    def __init__(self, hf_dataset, img_size):
        self.hf_dataset = hf_dataset
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        # Field name is "img" on uoft-cs/cifar10, but handle "image" too in
        # case HuggingFace's schema naming changes again in the future.
        raw_img = item.get("img", item.get("image"))
        img = self.transform(raw_img.convert("RGB"))  # PIL Image -> tensor
        label = int(item["label"])
        return img, label


def _get_cifar10_via_huggingface(img_size):
    from datasets import load_dataset  # pip install datasets
    # HuggingFace requires a namespace/name id -- the official CIFAR-10
    # mirror lives at "uoft-cs/cifar10" (plain "cifar10" no longer resolves).
    ds = load_dataset("uoft-cs/cifar10")  # downloads from HuggingFace's CDN
    train_set = _HFCIFAR10(ds["train"], img_size)
    test_set = _HFCIFAR10(ds["test"], img_size)
    return train_set, test_set


def make_collate_fn(grid_h, grid_w, num_target_blocks=4,
                     target_scale_range=(0.15, 0.2),
                     target_aspect_ratio_range=(0.75, 1.5),
                     context_scale_range=(0.85, 1.0)):
    """Builds a collate_fn that: stacks images normally, AND samples one
    shared set of I-JEPA masks for the whole batch (see masking.py's
    build_ijepa_masks_batch -- shared shapes, per-image positions).

    This keeps mask-sampling logic OUT of the Dataset class (masks depend on
    patch-grid size / batch size, not on any individual image) and inside the
    DataLoader's collation step, matching how the paper describes it being
    implemented (Appendix A.1: "a batch-collator function... in the data
    loader processes").
    """
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])  # (B, C, H, W)
        labels = torch.tensor([item[1] for item in batch])  # unused by I-JEPA
        # itself, but harmless to keep around for later probing/eval use.

        B = images.shape[0]
        context_indices, target_blocks = build_ijepa_masks_batch(
            B, grid_h, grid_w, num_target_blocks,
            target_scale_range, target_aspect_ratio_range, context_scale_range,
        )
        return images, labels, context_indices, target_blocks

    return collate_fn


def get_dataloader(dataset, batch_size, grid_h, grid_w, shuffle=True, num_workers=2):
    collate_fn = make_collate_fn(grid_h, grid_w)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=True,
    )


if __name__ == "__main__":
    # --- Sanity check: run this file directly ---
    # NOTE: this downloads CIFAR-10 (~170MB) on first run.
    img_size, patch_size = 96, 8
    grid_size = img_size // patch_size

    train_set, test_set = get_cifar10_datasets(img_size=img_size)
    print(f"train set size: {len(train_set)}, test set size: {len(test_set)}")

    loader = get_dataloader(train_set, batch_size=8, grid_h=grid_size, grid_w=grid_size)
    images, labels, context_indices, target_blocks = next(iter(loader))

    print(f"images shape: {images.shape}")
    print(f"labels shape: {labels.shape}")
    print(f"context_indices shape: {context_indices.shape}")
    for i, t in enumerate(target_blocks):
        print(f"target block {i} shape: {t.shape}")

    assert images.shape == (8, 3, img_size, img_size)
    assert context_indices.shape[0] == 8
    print("\nAll checks passed.")
