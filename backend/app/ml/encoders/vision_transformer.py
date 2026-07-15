"""
Minimal Vision Transformer, built from scratch, used as the backbone for
both the context encoder and the target encoder in I-JEPA.

No CLS token — I-JEPA operates purely on patch tokens.
"""

import math
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Splits an image into non-overlapping patches and linearly projects each
    patch to `embed_dim`. Implemented as a single strided conv, which is
    mathematically identical to "flatten patch -> linear layer" but faster.
    """

    def __init__(self, img_size=96, patch_size=8, in_chans=3, embed_dim=192):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.grid_size = img_size // patch_size          # patches per side
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)                      # (B, embed_dim, grid, grid)
        x = x.flatten(2).transpose(1, 2)      # (B, num_patches, embed_dim)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Fixed (non-learned) 2D sin-cos positional embedding.
    Returns: (grid_size*grid_size, embed_dim)
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sincos"

    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing="ij")  # (2, grid, grid)
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)

    def embed_1d(pos, dim):
        # pos: (grid_size*grid_size,) -> (grid_size*grid_size, dim)
        omega = torch.arange(dim // 2, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / (dim / 2)))
        out = pos.reshape(-1)[:, None] * omega[None, :]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    emb_h = embed_1d(grid[0], embed_dim // 2)
    emb_w = embed_1d(grid[1], embed_dim // 2)
    return torch.cat([emb_h, emb_w], dim=1)  # (grid*grid, embed_dim)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Block(nn.Module):
    """Pre-norm transformer block: x + Attn(LN(x)) then x + MLP(LN(x))."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Plain ViT encoder. Used identically for the context encoder and the
    target encoder (target encoder's weights are an EMA copy — same class,
    different weight-update rule, handled outside this file).
    """

    def __init__(
        self,
        img_size=96,
        patch_size=8,
        in_chans=3,
        embed_dim=192,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size

        # Fixed sin-cos positional embedding (not learned) — matches I-JEPA paper.
        pos_embed = get_2d_sincos_pos_embed(embed_dim, self.grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))  # (1, N, D)

        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_ratio, drop, attn_drop)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, keep_indices=None):
        """
        x: (B, C, H, W) full image
        keep_indices: optional (B, num_keep) long tensor — patch indices to
            keep (used by the CONTEXT encoder to only look at visible patches).
            If None, all patches are used (this is how the TARGET encoder
            is called — it always sees the full image).

        Returns: (B, num_keep_or_all, embed_dim)
        """
        tokens = self.patch_embed(x)               # (B, N, D)
        tokens = tokens + self.pos_embed            # add fixed position info

        if keep_indices is not None:
            B, D = tokens.shape[0], tokens.shape[-1]
            idx = keep_indices.unsqueeze(-1).expand(-1, -1, D)  # (B, K, D)
            tokens = torch.gather(tokens, dim=1, index=idx)

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return tokens


if __name__ == "__main__":
    # --- Sanity check: run this file directly to verify shapes ---
    torch.manual_seed(0)

    B, C, H, W = 4, 3, 96, 96
    patch_size = 8
    embed_dim = 192

    model = VisionTransformer(
        img_size=H, patch_size=patch_size, in_chans=C,
        embed_dim=embed_dim, depth=6, num_heads=6,
    )

    dummy_img = torch.randn(B, C, H, W)
    num_patches = model.patch_embed.num_patches
    print(f"grid_size={model.grid_size}  num_patches={num_patches}")

    # 1) Full forward (this is how the TARGET encoder is always called)
    out_full = model(dummy_img)
    print("full forward output shape:", out_full.shape)
    assert out_full.shape == (B, num_patches, embed_dim)

    # 2) Partial forward with keep_indices (this is how the CONTEXT encoder
    #    is called — only a subset of patches, e.g. 60% of them)
    num_keep = int(num_patches * 0.6)
    keep_indices = torch.stack(
        [torch.randperm(num_patches)[:num_keep] for _ in range(B)]
    )
    out_partial = model(dummy_img, keep_indices=keep_indices)
    print("partial forward output shape:", out_partial.shape)
    assert out_partial.shape == (B, num_keep, embed_dim)

    print("\nAll shape checks passed.")
