"""
I-JEPA predictor: a lightweight/narrow ViT (paper fixes width=384, Appendix A.1)
that takes the CONTEXT ENCODER's output tokens and, conditioned on positional
mask tokens, predicts the representation of a target block at a specific
location (paper Sec 3 "Prediction").

Called once per target block (paper applies it M times -- once per target).
"""

import sys
import os
import torch
import torch.nn as nn

# Reuse Block and the sincos positional embedding from the encoder file so the
# predictor's positional embeddings line up exactly with the encoder's grid.
sys.path.append(os.path.dirname(__file__))
from encoders.vision_transformer import Block, get_2d_sincos_pos_embed


class Predictor(nn.Module):
    def __init__(
        self,
        encoder_embed_dim=192,     # must match the context/target encoder's embed_dim
        predictor_embed_dim=384,   # paper: fixed at 384 regardless of backbone size
        depth=6,                   # paper: 6 for ViT-B, 12 for ViT-L/H, 16 for ViT-G
        num_heads=6,               # paper: same as backbone's head count
        grid_size=14,              # e.g. 224/16 patch grid
        mlp_ratio=4.0,
    ):
        super().__init__()

        # Project context tokens from the backbone's width down to the predictor's
        # (narrower) width.
        self.predictor_embed = nn.Linear(encoder_embed_dim, predictor_embed_dim)

        # A single shared learnable mask-token vector. Every masked position gets
        # this same vector, PLUS that position's own positional embedding -- the
        # positional embedding is what tells them apart (paper Sec 3).
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Fixed sin-cos positional embedding over the full patch grid, at the
        # predictor's width -- used both for context tokens and mask tokens.
        pos_embed = get_2d_sincos_pos_embed(predictor_embed_dim, grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))  # (1, N, D)

        self.blocks = nn.ModuleList(
            [
                Block(predictor_embed_dim, num_heads, mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(predictor_embed_dim)

        # Project back up to the backbone's width so predictions are directly
        # comparable (via the loss) to the target encoder's output.
        self.predictor_proj = nn.Linear(predictor_embed_dim, encoder_embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, context_tokens, context_indices, target_indices):
        """
        context_tokens:  (B, Nc, encoder_embed_dim) -- output of the context encoder
        context_indices: (B, Nc) long -- which grid patches these tokens came from
                          (needed so we add the RIGHT positional embedding to each)
        target_indices:  (B, Nt) long -- which grid patches we want to predict
                          (one target block's worth; call this once per block)

        Returns: (B, Nt, encoder_embed_dim) -- predicted representations at the
                 target locations, in the BACKBONE's embedding space.
        """
        B, Nc, _ = context_tokens.shape
        D = self.pos_embed.shape[-1]

        # Project context tokens down to predictor width, add their positions.
        x_context = self.predictor_embed(context_tokens)          # (B, Nc, D)
        ctx_pos = torch.gather(
            self.pos_embed.expand(B, -1, -1), dim=1,
            index=context_indices.unsqueeze(-1).expand(-1, -1, D),
        )
        x_context = x_context + ctx_pos

        # Build mask tokens for the target positions we want to predict.
        Nt = target_indices.shape[1]
        mask_tokens = self.mask_token.expand(B, Nt, D)
        tgt_pos = torch.gather(
            self.pos_embed.expand(B, -1, -1), dim=1,
            index=target_indices.unsqueeze(-1).expand(-1, -1, D),
        )
        mask_tokens = mask_tokens + tgt_pos

        # Concatenate: context tokens first, mask tokens last -- order doesn't
        # matter to a transformer (positions carry the info), but keeping mask
        # tokens last makes them trivial to slice out afterwards.
        x = torch.cat([x_context, mask_tokens], dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Only the mask-token outputs are our predictions.
        pred = x[:, -Nt:]
        pred = self.predictor_proj(pred)
        return pred


if __name__ == "__main__":
    # --- Sanity check: run this file directly ---
    torch.manual_seed(0)

    B = 4
    encoder_embed_dim = 192
    grid_size = 14
    num_patches = grid_size * grid_size

    predictor = Predictor(
        encoder_embed_dim=encoder_embed_dim,
        predictor_embed_dim=384,
        depth=6,
        num_heads=6,
        grid_size=grid_size,
    )

    # Simulate: context encoder saw 80 patches (matches our masking.py example)
    num_context = 80
    num_target = 35  # one target block's worth, matches masking.py example

    context_tokens = torch.randn(B, num_context, encoder_embed_dim)
    context_indices = torch.stack(
        [torch.randperm(num_patches)[:num_context] for _ in range(B)]
    )
    target_indices = torch.stack(
        [torch.randperm(num_patches)[:num_target] for _ in range(B)]
    )

    pred = predictor(context_tokens, context_indices, target_indices)
    print("predicted target representations shape:", pred.shape)
    assert pred.shape == (B, num_target, encoder_embed_dim)

    print("\nAll shape checks passed.")
