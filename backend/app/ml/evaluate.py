"""
Evaluate a trained I-JEPA checkpoint by measuring whether its learned
representations are actually useful -- not just whether the loss went down.

Method: k-NN classification on frozen target-encoder embeddings (a standard,
fast proxy for representation quality in self-supervised learning papers --
much cheaper than a full linear-probe training run, and a meaningful number
you can actually report).

How to read the result: random/untrained features score close to chance
(10% for CIFAR-10's 10 classes). Anything meaningfully above that means the
target encoder is capturing real structure in the images -- exactly what
I-JEPA is supposed to learn without ever seeing a label during pretraining.
"""

import sys
import os
import argparse
import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

sys.path.append(os.path.dirname(__file__))
from ijepa import IJEPA
from data.datasets import get_cifar10_datasets


@torch.no_grad()
def extract_embeddings(encoder, dataset, device, batch_size=256, max_samples=None):
    """Runs the (frozen) target encoder over a dataset and returns
    (embeddings, labels) as numpy arrays. Uses average-pooling over patch
    tokens -- same as the paper's evaluation protocol (Appendix A.2).
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_embeddings, all_labels = [], []
    seen = 0

    for images, labels in loader:
        images = images.to(device)
        tokens = encoder(images)               # (B, N, D)
        pooled = tokens.mean(dim=1)             # (B, D) -- average-pool patches
        all_embeddings.append(pooled.cpu().numpy())
        all_labels.append(labels.numpy())

        seen += images.shape[0]
        if max_samples is not None and seen >= max_samples:
            break

    return np.concatenate(all_embeddings), np.concatenate(all_labels)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    grid_size = args.img_size // args.patch_size
    model = IJEPA(
        img_size=args.img_size, patch_size=args.patch_size, embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth, encoder_heads=args.num_heads,
        predictor_embed_dim=args.predictor_embed_dim,
        predictor_depth=args.predictor_depth, predictor_heads=args.num_heads,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.target_encoder.load_state_dict(ckpt["target_encoder"])
    model.target_encoder.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch'] + 1}")

    train_set, test_set = get_cifar10_datasets(data_dir=args.data_dir, img_size=args.img_size)

    print("Extracting embeddings from training set (used as k-NN reference)...")
    train_emb, train_labels = extract_embeddings(
        model.target_encoder, train_set, device, max_samples=args.max_train_samples
    )
    print(f"train embeddings: {train_emb.shape}")

    print("Extracting embeddings from test set...")
    test_emb, test_labels = extract_embeddings(
        model.target_encoder, test_set, device, max_samples=args.max_test_samples
    )
    print(f"test embeddings: {test_emb.shape}")

    print(f"Fitting {args.k}-NN classifier on frozen embeddings...")
    knn = KNeighborsClassifier(n_neighbors=args.k)
    knn.fit(train_emb, train_labels)
    accuracy = knn.score(test_emb, test_labels)

    print(f"\n{'='*50}")
    print(f"k-NN accuracy on CIFAR-10 test set: {accuracy*100:.2f}%")
    print(f"(random chance for 10 classes: 10.00%)")
    print(f"{'='*50}")

    return accuracy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--img_size", type=int, default=96)
    p.add_argument("--patch_size", type=int, default=8)
    p.add_argument("--embed_dim", type=int, default=192)
    p.add_argument("--encoder_depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--predictor_embed_dim", type=int, default=384)
    p.add_argument("--predictor_depth", type=int, default=6)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--max_train_samples", type=int, default=5000)
    p.add_argument("--max_test_samples", type=int, default=2000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
