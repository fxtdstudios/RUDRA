"""RUDRA Stage 1: Dynamic Range Analysis.

This file implements the production-friendly RUDRA-Lite global descriptor.
It fixes the earlier prototype by:
  - using configurable Rec.2020 / Rec.709 / ACEScg luminance weights,
  - normalizing supported input formats before analysis,
  - computing highlight descriptors in log-normalized luminance space,
  - adding true CIE xy chromaticity statistics in addition to color-volume proxies.

Output dimension: 26
  8 luminance statistics, 4 exposure statistics, 6 highlight statistics,
  8 color / chromaticity statistics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FORMAT_DIM, FORMAT_NAMES, FORMAT_TO_ID, LUMA_WEIGHTS
from .normalization import normalize_to_scene_linear

_EPS = 1e-8

# RGB-to-XYZ matrices for common linear RGB spaces, D65 except ACEScg adapted approximation.
_RGB_TO_XYZ = {
    "rec709": torch.tensor([
        [0.4123908, 0.3575843, 0.1804808],
        [0.2126390, 0.7151687, 0.0721923],
        [0.0193308, 0.1191948, 0.9505322],
    ]),
    "rec2020": torch.tensor([
        [0.6369580, 0.1446169, 0.1688810],
        [0.2627002, 0.6779981, 0.0593017],
        [0.0000000, 0.0280727, 1.0609851],
    ]),
    "acescg": torch.tensor([
        [0.6624542, 0.1340042, 0.1561877],
        [0.2722287, 0.6740818, 0.0536895],
        [-0.0055746, 0.0040607, 1.0103391],
    ]),
}


class RUDRADescriptor(nn.Module):
    """Extracts a 26-dimensional global dynamic-range descriptor.

    Args:
        color_space: RGB working space used for luminance and CIE xy conversion.
        y_max_nits: Log-luminance normalization ceiling. 10000 matches PQ ceiling.
        normalize_input: If True, convert using format_id. If False, input is assumed scene-linear.
    """

    DR_RAW_DIM = 26

    def __init__(
        self,
        color_space: str = "rec2020",
        y_max_nits: float = 10000.0,
        normalize_input: bool = True,
    ):
        super().__init__()
        if color_space not in LUMA_WEIGHTS:
            raise ValueError(f"Unsupported color_space={color_space}. Expected one of {list(LUMA_WEIGHTS)}")
        self.color_space = color_space
        self.y_max_nits = float(y_max_nits)
        self.normalize_input = normalize_input

        luma_w = torch.tensor(LUMA_WEIGHTS[color_space], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("luma_weights", luma_w, persistent=False)
        self.register_buffer("rgb_to_xyz", _RGB_TO_XYZ[color_space].float(), persistent=False)

    def forward(
        self,
        image: torch.Tensor,
        format_id: torch.Tensor | int | None = None,
        input_is_scene_linear: bool = False,
    ) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected image shape (B, 3, H, W), got {tuple(image.shape)}")

        if self.normalize_input and not input_is_scene_linear:
            image_linear = normalize_to_scene_linear(image, format_id, y_max=self.y_max_nits)
        else:
            image_linear = torch.nan_to_num(image, nan=0.0, posinf=1e4, neginf=0.0)

        image_linear = image_linear.clamp(min=0.0)
        luma = self._luminance(image_linear)
        log_luma = self._log_normalized_luminance(luma)

        lum_desc = self._luminance_descriptor(log_luma)
        exp_desc = self._exposure_descriptor(luma, log_luma)
        hl_desc = self._highlight_descriptor(log_luma)
        color_desc = self._color_volume_descriptor(image_linear)
        return torch.cat([lum_desc, exp_desc, hl_desc, color_desc], dim=-1)

    def _luminance(self, image: torch.Tensor) -> torch.Tensor:
        w = self.luma_weights.to(image.device, image.dtype)
        return (image * w).sum(dim=1, keepdim=True).clamp(min=0.0)

    def _log_normalized_luminance(self, luma: torch.Tensor) -> torch.Tensor:
        denom = torch.log10(torch.tensor(1.0 + self.y_max_nits, device=luma.device, dtype=luma.dtype))
        return torch.log10(1.0 + luma.clamp(min=0.0)) / denom.clamp(min=_EPS)

    @staticmethod
    def _safe_quantile(x: torch.Tensor, q: float) -> torch.Tensor:
        sorted_x, _ = torch.sort(x, dim=-1)
        n = sorted_x.shape[-1]
        idx = min(max(int((n - 1) * q), 0), n - 1)
        return sorted_x[:, 0, idx]

    def _luminance_descriptor(self, log_luma: torch.Tensor) -> torch.Tensor:
        x = log_luma.flatten(2)
        mean = x.mean(dim=-1).squeeze(1)
        std = x.std(dim=-1, unbiased=False).squeeze(1)
        p50 = self._safe_quantile(x, 0.50)
        p90 = self._safe_quantile(x, 0.90)
        p95 = self._safe_quantile(x, 0.95)
        p99 = self._safe_quantile(x, 0.99)
        peak = x.max(dim=-1).values.squeeze(1)
        entropy = 0.5 + torch.log(std.clamp(min=1e-6))
        return torch.stack([mean, std, p50, p90, p95, p99, peak, entropy], dim=-1)

    def _exposure_descriptor(self, luma: torch.Tensor, log_luma: torch.Tensor) -> torch.Tensor:
        x = luma.flatten(2)
        lx = log_luma.flatten(2)
        mean_luma = x.mean(dim=-1).squeeze(1).clamp(min=_EPS)
        median_luma = self._safe_quantile(x, 0.50).clamp(min=_EPS)
        p95_luma = self._safe_quantile(x, 0.95).clamp(min=_EPS)
        peak = x.max(dim=-1).values.squeeze(1)

        ev_median = torch.log2(median_luma / 0.18 + _EPS)
        stops_above_grey = torch.log2(p95_luma / 0.18 + _EPS)
        log_std = lx.std(dim=-1, unbiased=False).squeeze(1)
        peak_to_mean = (peak / (mean_luma + _EPS)).clamp(max=1000.0)
        return torch.stack([ev_median, stops_above_grey, log_std, peak_to_mean], dim=-1)

    @staticmethod
    def _highlight_descriptor(log_luma: torch.Tensor) -> torch.Tensor:
        x = log_luma.flatten(2)
        thresholds = [0.50, 0.75, 0.85, 0.90, 0.95]
        temp = 0.03
        features = []
        for t in thresholds:
            mask = torch.sigmoid((x - t) / temp)
            coverage = mask.mean(dim=-1).squeeze(1)
            energy = ((x - t).clamp(min=0.0) * mask).mean(dim=-1).squeeze(1)
            # Use energy for high thresholds, coverage for lower thresholds by combining smoothly.
            features.append(0.5 * coverage + 0.5 * energy)
        mask_09 = torch.sigmoid((x - 0.90) / temp)
        highlight_mean = (x * mask_09).sum(dim=-1).squeeze(1) / (mask_09.sum(dim=-1).squeeze(1) + _EPS)
        skew = (highlight_mean - 0.90).clamp(min=0.0)
        return torch.stack(features + [skew], dim=-1)

    def _rgb_to_xy(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # image: (B, 3, H, W); matrix multiply in channel dimension.
        B, _, H, W = image.shape
        rgb = image.permute(0, 2, 3, 1).reshape(-1, 3)
        m = self.rgb_to_xyz.to(image.device, image.dtype)
        xyz = rgb @ m.T
        xyz = xyz.reshape(B, H, W, 3)
        s = xyz.sum(dim=-1).clamp(min=_EPS)
        x_cie = (xyz[..., 0] / s).clamp(0.0, 1.0)
        y_cie = (xyz[..., 1] / s).clamp(0.0, 1.0)
        return x_cie, y_cie

    def _color_volume_descriptor(self, image: torch.Tensor) -> torch.Tensor:
        B, _, H, W = image.shape
        rgb = image.permute(0, 2, 3, 1)
        rgb_max = rgb.max(dim=-1).values.clamp(min=_EPS)
        rgb_min = rgb.min(dim=-1).values
        sat = ((rgb_max - rgb_min) / rgb_max).clamp(0.0, 1.0)

        mean_sat = sat.flatten(1).mean(dim=-1)
        max_sat = sat.flatten(1).max(dim=-1).values

        r_norm = (rgb[..., 0] / rgb_max).flatten(1).mean(dim=-1).clamp(min=1e-6)
        g_norm = (rgb[..., 1] / rgb_max).flatten(1).mean(dim=-1).clamp(min=1e-6)
        b_norm = (rgb[..., 2] / rgb_max).flatten(1).mean(dim=-1).clamp(min=1e-6)
        p_total = r_norm + g_norm + b_norm
        p_r, p_g, p_b = r_norm / p_total, g_norm / p_total, b_norm / p_total
        hue_entropy = -(p_r * p_r.log() + p_g * p_g.log() + p_b * p_b.log())

        color_spread = (rgb.std(dim=-1, unbiased=False) + _EPS).flatten(1).mean(dim=-1)
        gamut_area = mean_sat * color_spread

        x_cie, y_cie = self._rgb_to_xy(image)
        x_mean = x_cie.flatten(1).mean(dim=-1)
        y_mean = y_cie.flatten(1).mean(dim=-1)
        x_std = x_cie.flatten(1).std(dim=-1, unbiased=False)
        y_std = y_cie.flatten(1).std(dim=-1, unbiased=False)

        return torch.stack([mean_sat, max_sat, gamut_area, hue_entropy, x_mean, y_mean, x_std, y_std], dim=-1)


def format_onehot(format_id: torch.Tensor | int, device: str | torch.device | None = None) -> torch.Tensor:
    """Convert a scalar or (B,) format id into one-hot format tokens."""
    if isinstance(format_id, int):
        if device is None:
            device = "cpu"
        format_id = torch.tensor([format_id], dtype=torch.long, device=device)
    else:
        if device is None:
            device = format_id.device
        format_id = format_id.to(device=device, dtype=torch.long).view(-1)
    return F.one_hot(format_id, num_classes=FORMAT_DIM).float()
