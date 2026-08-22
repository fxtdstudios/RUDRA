"""HDR-safe augmentations for RUDRA training.

These transforms are radiometric-friendly: they avoid destructive clipping of HDR
super-white values and operate in scene-linear space unless explicitly noted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .radiometry import luma_cf


@dataclass
class RUDRAAugmentConfig:
    exposure_ev_min: float = -2.0
    exposure_ev_max: float = 2.0
    white_balance_min: float = 0.92
    white_balance_max: float = 1.08
    highlight_gain_min: float = 0.85
    highlight_gain_max: float = 1.20
    black_lift_max: float = 0.01
    hflip_prob: float = 0.5
    crop_size: Optional[int] = None
    enable: bool = True


class RUDRAHDRAugment:
    """Apply paired HDR-safe augmentation to input/target tensors.

    Args:
        cfg: augmentation config.

    Call signature:
        x_aug, y_aug = aug(x, y)
    """

    def __init__(self, cfg: RUDRAAugmentConfig | None = None):
        self.cfg = cfg or RUDRAAugmentConfig()

    def _rand(self, shape, device, dtype):
        return torch.rand(shape, device=device, dtype=dtype)

    def __call__(self, inp: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.enable:
            return inp, target
        if inp.shape != target.shape:
            raise ValueError(f"Paired augmentation expects same shapes, got {inp.shape} and {target.shape}")
        x, y = inp.clone(), target.clone()
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # Exposure shift, same for input and target so radiometric relation is preserved.
        ev = self.cfg.exposure_ev_min + (self.cfg.exposure_ev_max - self.cfg.exposure_ev_min) * self._rand((B, 1, 1, 1), device, dtype)
        gain = torch.pow(torch.tensor(2.0, device=device, dtype=dtype), ev)
        x = x * gain
        y = y * gain

        # Per-channel white balance, gentle and multiplicative.
        wb = self.cfg.white_balance_min + (self.cfg.white_balance_max - self.cfg.white_balance_min) * self._rand((B, C, 1, 1), device, dtype)
        x = x * wb
        y = y * wb

        # Highlight gain only for high scene-linear values.
        luminance = luma_cf(y)
        mask = torch.sigmoid((torch.log1p(luminance.clamp(min=0.0)) - torch.log(torch.tensor(2.0, device=device, dtype=dtype))) / 0.25)
        hg = self.cfg.highlight_gain_min + (self.cfg.highlight_gain_max - self.cfg.highlight_gain_min) * self._rand((B, 1, 1, 1), device, dtype)
        x = x * (1.0 + mask * (hg - 1.0))
        y = y * (1.0 + mask * (hg - 1.0))

        # Tiny black lift. Do not clamp because ACES workflows may contain small negatives.
        lift = self.cfg.black_lift_max * self._rand((B, 1, 1, 1), device, dtype)
        x = x + lift
        y = y + lift

        # Horizontal flip per batch item.
        if self.cfg.hflip_prob > 0:
            flip = self._rand((B,), device, dtype) < self.cfg.hflip_prob
            if flip.any():
                x[flip] = torch.flip(x[flip], dims=[-1])
                y[flip] = torch.flip(y[flip], dims=[-1])

        # Shared random crop.
        if self.cfg.crop_size is not None and self.cfg.crop_size > 0 and H >= self.cfg.crop_size and W >= self.cfg.crop_size:
            cs = self.cfg.crop_size
            top = torch.randint(0, H - cs + 1, (1,), device=device).item()
            left = torch.randint(0, W - cs + 1, (1,), device=device).item()
            x = x[:, :, top:top + cs, left:left + cs]
            y = y[:, :, top:top + cs, left:left + cs]

        return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4), torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
