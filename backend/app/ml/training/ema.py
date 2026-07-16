"""
Exponential moving average (EMA) update for the target encoder.

Paper facts encoded here (Appendix A.1 "Optimization"):
  - Target encoder weights are IDENTICAL to context encoder weights at init.
  - After every training step: target = m * target + (1-m) * context
  - Momentum m starts at 0.996 and is LINEARLY increased to 1.0 over the
    course of training (not cosine -- the paper is explicit it's linear).
  - The target encoder is NEVER updated by gradient descent -- only by EMA.
    This asymmetry (paper Sec 2, "Joint-Embedding Predictive Architectures")
    is what prevents representation collapse -- it's the whole reason I-JEPA
    doesn't need contrastive negatives or a stop-gradient-only trick.
"""

import copy
import torch


def make_target_encoder(context_encoder):
    """Create the target encoder as an exact copy of the context encoder's
    initial weights (paper: "identical to the context-encoder weights at
    initialization"). The target encoder is frozen -- never touched by the
    optimizer, only by EMA updates.
    """
    target_encoder = copy.deepcopy(context_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False
    return target_encoder


class MomentumSchedule:
    """Linear momentum ramp from base_momentum -> final_momentum over
    total_steps. Paper defaults: 0.996 -> 1.0.
    """

    def __init__(self, base_momentum=0.996, final_momentum=1.0, total_steps=1000):
        self.base_momentum = base_momentum
        self.final_momentum = final_momentum
        self.total_steps = total_steps

    def get_momentum(self, step):
        step = min(step, self.total_steps)  # clamp so it never exceeds 1.0
        progress = step / self.total_steps
        return self.base_momentum + progress * (self.final_momentum - self.base_momentum)


@torch.no_grad()
def update_target_encoder(target_encoder, context_encoder, momentum):
    """target = m * target + (1 - m) * context, applied param-by-param.
    Call this AFTER every optimizer.step() on the context encoder.
    """
    for t_param, c_param in zip(target_encoder.parameters(), context_encoder.parameters()):
        t_param.data.mul_(momentum).add_(c_param.data, alpha=(1.0 - momentum))


if __name__ == "__main__":
    # --- Sanity check: run this file directly ---
    import torch.nn as nn

    torch.manual_seed(0)

    # A tiny dummy "encoder" just to verify the EMA math, independent of the
    # real ViT -- keeps this test fast and focused on the EMA logic itself.
    context_encoder = nn.Linear(8, 8)
    target_encoder = make_target_encoder(context_encoder)

    # 1) Verify target starts IDENTICAL to context.
    for tp, cp in zip(target_encoder.parameters(), context_encoder.parameters()):
        assert torch.allclose(tp, cp), "target encoder should start identical to context"
    print("Check 1 passed: target encoder initialized identical to context encoder")

    # 2) Verify target encoder has no gradients (frozen).
    assert all(not p.requires_grad for p in target_encoder.parameters())
    print("Check 2 passed: target encoder parameters are frozen (requires_grad=False)")

    # 3) Simulate context encoder changing (as if an optimizer step happened),
    #    then verify the EMA update moves target PARTWAY toward it, not all
    #    the way (that would defeat the purpose of a slow-moving average).
    with torch.no_grad():
        for p in context_encoder.parameters():
            p.add_(1.0)  # simulate a big gradient step

    schedule = MomentumSchedule(base_momentum=0.996, final_momentum=1.0, total_steps=1000)
    m = schedule.get_momentum(step=0)
    print(f"momentum at step 0: {m}")

    target_before = [p.clone() for p in target_encoder.parameters()]
    update_target_encoder(target_encoder, context_encoder, momentum=m)

    for t_before, t_after, c_after in zip(
        target_before, target_encoder.parameters(), context_encoder.parameters()
    ):
        # target should have moved TOWARD context, but only by (1-m) of the gap
        moved_correctly = torch.allclose(
            t_after, m * t_before + (1 - m) * c_after, atol=1e-6
        )
        assert moved_correctly
        # and target should NOT equal context (since m is close to 1, barely moved)
        assert not torch.allclose(t_after, c_after)
    print("Check 3 passed: EMA update moved target partway toward context, as expected")

    # 4) Verify momentum schedule ramps correctly over training.
    m_start = schedule.get_momentum(0)
    m_mid = schedule.get_momentum(500)
    m_end = schedule.get_momentum(1000)
    print(f"momentum schedule: step0={m_start}  step500={m_mid}  step1000={m_end}")
    assert m_start == 0.996
    assert abs(m_mid - 0.998) < 1e-6
    assert m_end == 1.0
    print("Check 4 passed: momentum schedule ramps linearly from 0.996 -> 1.0")

    print("\nAll checks passed.")
