"""RUDRA-Lite learned projection: global descriptor + format one-hot -> conditioning vector."""

from __future__ import annotations

import torch
import torch.nn as nn


class RUDRAProjection(nn.Module):
    """Projects global DR statistics and format tokens into a compact conditioning vector.

    Fixes from the prototype:
      - descriptor input is normalized with LayerNorm,
      - final projection is near-zero instead of exactly zero, preserving baseline
        behavior while allowing immediate gradient flow into earlier layers.
    """

    def __init__(
        self,
        dr_raw_dim: int = 26,
        format_dim: int = 9,
        proj_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dr_raw_dim = int(dr_raw_dim)
        self.format_dim = int(format_dim)
        self.proj_dim = int(proj_dim)
        in_dim = self.dr_raw_dim + self.format_dim

        self.input_norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, dr_raw: torch.Tensor, format_onehot: torch.Tensor) -> torch.Tensor:
        if dr_raw.ndim != 2:
            raise ValueError(f"dr_raw must be (B, {self.dr_raw_dim}), got {tuple(dr_raw.shape)}")
        if format_onehot.ndim != 2:
            raise ValueError(f"format_onehot must be (B, {self.format_dim}), got {tuple(format_onehot.shape)}")
        if dr_raw.shape[0] != format_onehot.shape[0]:
            raise ValueError("dr_raw and format_onehot batch sizes do not match")
        if dr_raw.shape[-1] != self.dr_raw_dim:
            raise ValueError(f"Expected dr_raw_dim={self.dr_raw_dim}, got {dr_raw.shape[-1]}")
        if format_onehot.shape[-1] != self.format_dim:
            raise ValueError(f"Expected format_dim={self.format_dim}, got {format_onehot.shape[-1]}")
        x = torch.cat([dr_raw, format_onehot.to(dr_raw.dtype)], dim=-1)
        return self.mlp(self.input_norm(x))
