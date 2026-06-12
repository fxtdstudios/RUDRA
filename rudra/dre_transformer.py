"""RUDRA-Full Dynamic Range Encoder (DRE).

Maps spatial descriptor maps R(x) in R^5 to spatial tokens Z_R compatible with
LDM-style latent grids: (B, H/8 * W/8, 512) or (B, 512, H/8, W/8).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RUDRADynamicRangeEncoder(nn.Module):
    """Maps spatial descriptor maps R(x) ∈ ℝ⁵ to spatial tokens Z_R.

    Paper §3.4: patchify into 8×8 patches, project to 512-d tokens with learned
    positional encodings, process through 12-layer transformer with pre-LN.
    """

    def __init__(
        self,
        in_channels: int = 5,
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        patch_size: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_grid_size: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Learned positional encoding (paper §3.4).
        # Allocate for a (max_grid_size × max_grid_size) token grid; slice or
        # interpolate in 2-D at runtime (review §4.5).
        self.grid_side = int(max_grid_size)
        max_tokens = self.grid_side * self.grid_side
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def _get_pos_embed(self, h: int, w: int, dtype: torch.dtype) -> torch.Tensor:
        """Return positional embeddings for an (h, w) token grid.

        The learned buffer is a (grid_side × grid_side) 2-D grid. Within bounds we
        slice the top-left (h × w) sub-grid (exact, preserves learned positions);
        beyond bounds we interpolate the whole grid to (h, w) in 2-D with bicubic.
        The previous 1-D interpolation over the flattened sequence mixed spatially
        distant positions (end of one row blended into the next) — review §4.5.
        """
        s = self.grid_side
        grid = self.pos_embed.reshape(1, s, s, self.embed_dim).permute(0, 3, 1, 2)  # (1,C,s,s)
        if h <= s and w <= s:
            pe = grid[:, :, :h, :w]
        else:
            pe = nn.functional.interpolate(grid, size=(h, w), mode="bicubic", align_corners=False)
        return pe.permute(0, 2, 3, 1).reshape(1, h * w, self.embed_dim).to(dtype=dtype)

    def forward(self, r_map: torch.Tensor, return_grid: bool = False) -> torch.Tensor:
        if r_map.ndim != 4 or r_map.shape[1] != self.in_channels:
            raise ValueError(f"Expected (B, {self.in_channels}, H, W), got {tuple(r_map.shape)}")
        z = self.patch_embed(r_map)
        B, C, H8, W8 = z.shape
        tokens = z.flatten(2).transpose(1, 2)
        tokens = tokens + self._get_pos_embed(H8, W8, tokens.dtype)
        tokens = self.norm(self.encoder(tokens))
        if return_grid:
            return tokens.transpose(1, 2).reshape(B, C, H8, W8)
        return tokens
