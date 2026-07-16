"""
I-JEPA loss (paper Sec 3 "Loss"):

    (1/M) * sum_i  (1/|B_i|) * sum_{j in B_i} || pred_yj - target_yj ||^2

i.e. for each of the M target blocks, average the squared L2 distance
between predicted and target patch representations, then average across
the M blocks. Also averaged across the batch.

Crucial: the target is always DETACHED (no gradient flows into the target
encoder through this loss -- it's only ever updated via EMA, see ema.py).
"""

import torch
import torch.nn.functional as F


def ijepa_loss(predictions, targets):
    """
    predictions: list of M tensors, each (B, Nt_i, D) -- predictor's output
                 for target block i (D = backbone embed_dim)
    targets:     list of M tensors, each (B, Nt_i, D) -- target ENCODER's
                 output at the same positions (must be detached already)

    Returns: scalar loss (mean squared L2 distance, averaged over blocks,
             patches, and batch)
    """
    assert len(predictions) == len(targets), "must have one prediction per target block"

    block_losses = []
    for pred, target in zip(predictions, targets):
        target = target.detach()  # belt-and-suspenders: never backprop into target
        # squared L2 distance per patch, then mean over patches and batch
        per_patch_sq_dist = (pred - target).pow(2).sum(dim=-1)  # (B, Nt_i)
        block_losses.append(per_patch_sq_dist.mean())

    return torch.stack(block_losses).mean()


if __name__ == "__main__":
    # --- Sanity check: run this file directly ---
    torch.manual_seed(0)

    B, D = 4, 192
    Nt = [35, 30, 36, 30]  # matches our masking.py example target block sizes

    # 1) Identical prediction and target -> loss should be exactly 0.
    targets = [torch.randn(B, n, D) for n in Nt]
    predictions_perfect = [t.clone() for t in targets]
    loss_zero = ijepa_loss(predictions_perfect, targets)
    print(f"loss when prediction == target: {loss_zero.item():.6f}")
    assert torch.allclose(loss_zero, torch.tensor(0.0), atol=1e-6)

    # 2) Random (unrelated) prediction -> loss should be clearly positive.
    predictions_random = [torch.randn(B, n, D) for n in Nt]
    loss_random = ijepa_loss(predictions_random, targets)
    print(f"loss with random (unrelated) prediction: {loss_random.item():.4f}")
    assert loss_random.item() > 0.1

    # 3) Verify gradients flow into predictions but the target requires no grad
    #    contribution (simulate target encoder output requiring grad, then
    #    check .grad stays None after the target is detached inside the loss).
    pred_grad_test = [torch.randn(B, n, D, requires_grad=True) for n in Nt]
    target_grad_test = [torch.randn(B, n, D, requires_grad=True) for n in Nt]
    loss = ijepa_loss(pred_grad_test, target_grad_test)
    loss.backward()
    assert all(p.grad is not None for p in pred_grad_test), "predictions should get gradients"
    assert all(t.grad is None for t in target_grad_test), "targets must NOT get gradients"
    print("Check passed: gradients flow into predictions only, never into targets")

    print("\nAll checks passed.")
