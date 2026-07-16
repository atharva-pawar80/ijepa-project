"""
Full I-JEPA training script.

Schedules match paper Appendix A.1 "Optimization":
  - LR: linear warmup 1e-4 -> 1e-3 over first 15 epochs, then cosine decay to 1e-6
  - Weight decay: linear increase 0.04 -> 0.4 over training
  - EMA momentum: linear increase 0.996 -> 1.0 over training
  - Optimizer: AdamW

This is intentionally a SMALL-SCALE script (CIFAR-10, tiny ViT) so you can
run it in minutes and actually see whether the pipeline learns anything --
not a faithful reproduction of the paper's ImageNet/ViT-Huge scale run.
"""

import os
import sys
import math
import time
import argparse
import torch

sys.path.append(os.path.dirname(__file__))
from ijepa import IJEPA, embedding_variance
from training.ema import MomentumSchedule, update_target_encoder
from training.loss import ijepa_loss
from data.datasets import get_cifar10_datasets, get_dataloader


def linear_warmup_cosine_decay(step, total_steps, warmup_steps,
                                base_lr=1e-4, peak_lr=1e-3, final_lr=1e-6):
    if step < warmup_steps:
        return base_lr + (peak_lr - base_lr) * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return final_lr + 0.5 * (peak_lr - final_lr) * (1 + math.cos(math.pi * progress))


def linear_schedule(step, total_steps, start, end):
    progress = min(step / max(1, total_steps), 1.0)
    return start + progress * (end - start)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    grid_size = args.img_size // args.patch_size

    model = IJEPA(
        img_size=args.img_size, patch_size=args.patch_size, embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth, encoder_heads=args.num_heads,
        predictor_embed_dim=args.predictor_embed_dim,
        predictor_depth=args.predictor_depth, predictor_heads=args.num_heads,
    ).to(device)

    train_set, _ = get_cifar10_datasets(data_dir=args.data_dir, img_size=args.img_size)
    loader = get_dataloader(train_set, batch_size=args.batch_size,
                             grid_h=grid_size, grid_w=grid_size, num_workers=args.num_workers)

    total_steps = args.epochs * len(loader)
    warmup_steps = min(15 * len(loader), total_steps)  # paper: 15 epochs of warmup
    momentum_schedule = MomentumSchedule(base_momentum=0.996, final_momentum=1.0,
                                          total_steps=total_steps)

    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=1e-4,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_losses = []
        for images, _labels, context_indices, target_blocks in loader:
            images = images.to(device)
            context_indices = context_indices.to(device)
            target_blocks = [(idx.to(device), None) for idx in target_blocks]

            lr = linear_warmup_cosine_decay(step, total_steps, warmup_steps)
            wd = linear_schedule(step, total_steps, start=0.04, end=0.4)
            for group in optimizer.param_groups:
                group["lr"] = lr
                group["weight_decay"] = wd

            predictions, targets = model(images, context_indices, target_blocks)
            loss = ijepa_loss(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            momentum = momentum_schedule.get_momentum(step)
            update_target_encoder(model.target_encoder, model.context_encoder, momentum)

            epoch_losses.append(loss.item())
            step += 1

            if step % args.log_every == 0:
                with torch.no_grad():
                    var = embedding_variance(model.target_encoder(images))
                elapsed = time.time() - start_time
                print(f"epoch {epoch:3d}  step {step:6d}/{total_steps}  "
                      f"loss={loss.item():.4f}  lr={lr:.2e}  wd={wd:.3f}  "
                      f"momentum={momentum:.4f}  target_var={var:.4f}  "
                      f"elapsed={elapsed:.1f}s")
                if var < 1e-4:
                    print("  WARNING: target embedding variance is very low -- "
                          "possible representation collapse. Check momentum/LR.")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"=== epoch {epoch} done, avg loss: {avg_loss:.4f} ===")

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.checkpoint_dir, f"ijepa_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "context_encoder": model.context_encoder.state_dict(),
                "target_encoder": model.target_encoder.state_dict(),
                "predictor": model.predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, ckpt_path)
            print(f"saved checkpoint: {ckpt_path}")

    print("Training complete.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--img_size", type=int, default=96)
    p.add_argument("--patch_size", type=int, default=8)
    p.add_argument("--embed_dim", type=int, default=192)
    p.add_argument("--encoder_depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--predictor_embed_dim", type=int, default=384)
    p.add_argument("--predictor_depth", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--checkpoint_every", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
