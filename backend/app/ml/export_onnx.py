"""
Export the trained I-JEPA target encoder to ONNX, so the actual trained
weights can run inference directly in a browser (via onnxruntime-web) --
not a JavaScript reimplementation, the real model.

Why the TARGET encoder specifically: it's the one meant for producing final,
usable representations (the paper's own evaluation protocol uses it, not the
context encoder, which only ever sees partial/masked images). It always
takes the full image with no masking -- exactly what a "upload any photo"
demo needs.

Run this AFTER training (i.e. in the same Colab session, or after loading
a saved checkpoint):
    python export_onnx.py --checkpoint checkpoints/ijepa_epoch5.pt
"""

import sys
import os
import argparse
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(__file__))
from ijepa import IJEPA


class EncoderOnlyWrapper(nn.Module):
    """Wraps the target encoder so the exported ONNX graph has a single,
    simple input->output signature: image in, per-patch embeddings out.
    No keep_indices, no masking -- always the full image, matching how a
    browser demo will actually call it.
    """

    def __init__(self, target_encoder):
        super().__init__()
        self.target_encoder = target_encoder

    def forward(self, images):
        return self.target_encoder(images)  # (B, N, D) -- full image, no masking


def export(args):
    device = torch.device("cpu")  # export on CPU -- ONNX/browser inference is CPU-side anyway

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

    wrapper = EncoderOnlyWrapper(model.target_encoder)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, args.img_size, args.img_size)

    # Sanity check BEFORE exporting: run the PyTorch model directly so we
    # have a reference output to compare the ONNX version against afterward.
    with torch.no_grad():
        torch_output = wrapper(dummy_input)
    print(f"PyTorch output shape: {torch_output.shape}")

    torch.onnx.export(
        wrapper,
        dummy_input,
        args.output,
        input_names=["image"],
        output_names=["patch_embeddings"],
        opset_version=18,
        # batch size fixed at 1 -- this demo only ever processes one
        # uploaded image at a time, so we don't need dynamic batching
        # complexity in the browser.
    )
    print(f"Exported ONNX model to {args.output}")

    # Verify the exported model actually matches the PyTorch model's output
    # -- this is the single most important check here. An export that
    # "succeeds" but produces different numbers is worse than no export.
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(args.output)
    onnx_output = session.run(None, {"image": dummy_input.numpy()})[0]

    max_diff = np.abs(onnx_output - torch_output.numpy()).max()
    print(f"Max difference between PyTorch and ONNX outputs: {max_diff:.8f}")
    assert max_diff < 1e-4, (
        "ONNX export diverges from the original PyTorch model -- do not "
        "ship this file, something went wrong in the export."
    )
    print("Verification passed: ONNX model matches PyTorch model output.")

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="ijepa_encoder.onnx")
    p.add_argument("--img_size", type=int, default=96)
    p.add_argument("--patch_size", type=int, default=8)
    p.add_argument("--embed_dim", type=int, default=192)
    p.add_argument("--encoder_depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--predictor_embed_dim", type=int, default=384)
    p.add_argument("--predictor_depth", type=int, default=6)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export(args)
