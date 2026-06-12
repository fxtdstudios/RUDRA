"""Paper-aligned spatial RUDRA descriptor.

Produces per-pixel R(x) = [L, E, H, x_CIE, y_CIE] in R^5, matching the revised
RUDRA research formulation. This is the input for the transformer DRE.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import FORMAT_TO_ID, LUMA_WEIGHTS
from .normalization import normalize_to_scene_linear
from .descriptor import _RGB_TO_XYZ

_EPS = 1e-8


class RUDRASpatialDescriptor(nn.Module):
    CHANNELS = 5

    def __init__(self, color_space: str = "rec2020", y_max_nits: float = 10000.0, normalize_input: bool = True,
                 highlight_ev_threshold: float = 2.0, highlight_ev_softness: float = 4.0):
        super().__init__()
        if color_space not in LUMA_WEIGHTS:
            raise ValueError(f"Unsupported color_space={color_space}")
        self.color_space = color_space
        self.y_max_nits = float(y_max_nits)
        self.normalize_input = normalize_input
        # Highlight mask calibration in stops above 0.18 middle grey (scene-linear
        # relative), not absolute nits.
        self.hl_ev_thr = float(highlight_ev_threshold)
        self.hl_ev_soft = float(highlight_ev_softness)
        self.register_buffer("luma_weights", torch.tensor(LUMA_WEIGHTS[color_space], dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("rgb_to_xyz", _RGB_TO_XYZ[color_space].float(), persistent=False)

    def forward(self, image: torch.Tensor, format_id: torch.Tensor | int | None = None, input_is_scene_linear: bool = False) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected (B, 3, H, W), got {tuple(image.shape)}")
        if self.normalize_input and not input_is_scene_linear:
            image = normalize_to_scene_linear(image, format_id, y_max=self.y_max_nits)
        image = torch.nan_to_num(image, nan=0.0, posinf=1e4, neginf=0.0).clamp(min=0.0)

        w = self.luma_weights.to(image.device, image.dtype)
        y = (image * w).sum(dim=1, keepdim=True).clamp(min=0.0)
        denom = torch.log10(torch.tensor(1.0 + self.y_max_nits, device=image.device, dtype=image.dtype)).clamp(min=_EPS)
        L = torch.log10(1.0 + y) / denom

        med = y.flatten(2).median(dim=-1).values.view(-1, 1, 1, 1).clamp(min=_EPS)
        E = torch.log2(med / 0.18 + _EPS).expand_as(L)
        # Highlight mask, scene-linear-relative: stops of luminance above 0.18
        # middle grey. The original absolute-nit form H = max(0, L-0.85)/0.15 with
        # a 10000-nit ceiling is UNREACHABLE for relative scene-linear data where
        # diffuse white ≈ 1 (the max attainable L is ~0.19), so the highlight
        # objective was structurally dead (0/500 pairs registered a highlight).
        # EV-over-grey fires correctly on any scene-linear convention. No upper
        # clamp — extreme highlights keep their energy magnitude.
        ev = torch.log2(y.clamp(min=_EPS) / 0.18)
        H = ((ev - self.hl_ev_thr) / self.hl_ev_soft).clamp(min=0.0)

        x_cie, y_cie = self._xy(image)
        return torch.cat([L, E, H, x_cie, y_cie], dim=1)

    def _xy(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = image.shape
        rgb = image.permute(0, 2, 3, 1).reshape(-1, 3)
        m = self.rgb_to_xyz.to(image.device, image.dtype)
        xyz = (rgb @ m.T).reshape(B, H, W, 3)
        s = xyz.sum(dim=-1, keepdim=True).clamp(min=_EPS)
        xy = (xyz[..., :2] / s).permute(0, 3, 1, 2).clamp(0.0, 1.0)
        return xy[:, 0:1], xy[:, 1:2]
