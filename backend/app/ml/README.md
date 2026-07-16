# I-JEPA from Scratch

A from-scratch PyTorch implementation of **I-JEPA** (Image-based Joint-Embedding
Predictive Architecture), the self-supervised learning method from
[Assran et al., 2023 (Meta AI)](https://arxiv.org/abs/2301.08243).

Unlike most self-supervised vision methods, I-JEPA learns useful image
representations **without any hand-crafted data augmentations** (no random
crops, color jitter, or multiple views) and **without reconstructing pixels**
(unlike MAE). Instead, it predicts missing regions of an image directly in
*representation space* — a much more abstract, semantic target than raw
pixels.

## How it works

Given an image, split it into patches. Hide most of them, keeping only a
single **context block**. A `context encoder` processes the visible patches.
A separate **target encoder** (an exponential moving average of the context
encoder — never updated by gradient descent) processes the *full* image and
produces target representations at several **target block** locations. A
lightweight `predictor` takes the context encoder's output and, conditioned
on positional mask tokens, tries to predict what the target encoder "sees" at
each target block. The loss is simply the distance between predicted and
actual representations — no pixels, no contrastive negatives.

![masking strategy](masking_visualization.png)

*Multi-block masking: a single context block (left, blue) predicts several
target blocks (right, colored) — sampled independently per image, with target
overlap removed from the context to keep the task non-trivial.*

## Architecture

```
Image
  │
  ├──► Context Encoder (ViT) ──► context tokens ──┐
  │        (sees only visible patches)            │
  │                                                ▼
  │                                          Predictor (narrow ViT)
  │                                          + mask tokens per target
  │                                                │
  ├──► Target Encoder (ViT, EMA of context) ──►    │
  │        (always sees the FULL image)      target embeddings
  │                                                │
  └────────────────────────────────────────────────┴──► L2 loss
```

Built from scratch (no `timm` pretrained weights, no borrowed model code):
- **Vision Transformer** (`encoders/vision_transformer.py`) — patch
  embedding, fixed sin-cos positional embeddings, standard pre-norm
  transformer blocks.
- **Multi-block masking** (`masking.py`) — context block (scale 0.85-1.0,
  unit aspect ratio) + 4 target blocks (scale 0.15-0.2, aspect ratio
  0.75-1.5), overlap removed, matching the paper's Figure 4 exactly.
- **Predictor** (`predictor.py`) — a narrower ViT (width 384) that takes
  context tokens + learnable mask tokens and predicts target representations.
- **EMA target encoder** (`training/ema.py`) — momentum linearly ramped
  0.996 → 1.0, the mechanism that prevents representation collapse without
  needing contrastive negatives.
- **Loss** (`training/loss.py`) — average squared L2 distance in
  representation space, matching the paper's exact formula.

## Results

Trained on CIFAR-10 (resized to 96×96) for 5 epochs on a small ViT
(embed_dim=192, depth=6):

| Metric | Value |
|---|---|
| Training loss | 89.8 → ~34 |
| Target embedding variance | stayed healthy throughout (no collapse) |
| k-NN accuracy on frozen embeddings (test set) | *(fill in from `evaluate.py`)* |
| Random chance (10 classes) | 10% |

This is a small-scale sanity run, not a reproduction of the paper's
ImageNet/ViT-Huge results — the goal was to verify the full pipeline (context
encoder → predictor → EMA target encoder → loss) is implemented correctly
and actually learns.

## Project structure

```
backend/app/ml/
├── ijepa.py                 # wires everything together
├── masking.py                # multi-block masking strategy
├── predictor.py              # narrow ViT predictor
├── evaluate.py                # k-NN eval on frozen embeddings
├── encoders/
│   └── vision_transformer.py # ViT backbone (context + target encoder)
├── data/
│   └── datasets.py            # CIFAR-10 loading + batch mask collation
└── training/
    ├── ema.py                  # EMA target encoder update
    ├── loss.py                 # representation-space L2 loss
    └── trainer.py              # full training loop
```

## Running it

```bash
pip install -r requirements.txt
cd backend/app/ml

# sanity checks for each component
python encoders/vision_transformer.py
python masking.py
python predictor.py
python training/ema.py
python training/loss.py
python ijepa.py          # overfit-one-batch pipeline test

# train on CIFAR-10
python training/trainer.py --epochs 20 --batch_size 128

# evaluate a checkpoint
python evaluate.py --checkpoint checkpoints/ijepa_epoch20.pt
```

## Reference

Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M.,
LeCun, Y., & Ballas, N. (2023). *Self-Supervised Learning from Images with a
Joint-Embedding Predictive Architecture*. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
