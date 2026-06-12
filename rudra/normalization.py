"""Format normalization utilities for RUDRA.

The goal is stable scene-linear radiance conditioning, not a color-management
replacement. For production, connect this to OCIO/ACES transforms where possible.
"""

from __future__ import annotations

import torch

from .config import FORMAT_NAMES, FORMAT_TO_ID
from .color_curves import LOG_TO_LINEAR, LINEAR_TO_LOG

_EPS = 1e-8


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(min=0.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))


def _pq_eotf(x: torch.Tensor, y_max: float = 10000.0) -> torch.Tensor:
    """Inverse SMPTE ST.2084 PQ, returning approximate relative scene-linear RGB.

    Note: y_max is accepted for API consistency but is not used in the computation.
    The PQ EOTF formula inherently produces L/L_max ∈ [0, 1] when the input signal
    is in [0, 1], which is already normalized relative scene-linear radiance.
    """
    x = x.clamp(0.0, 1.0)
    m1 = 2610.0 / 16384.0
    m2 = 2523.0 / 32.0
    c1 = 3424.0 / 4096.0
    c2 = 2413.0 / 128.0
    c3 = 2392.0 / 128.0
    xp = x.pow(1.0 / m2)
    num = (xp - c1).clamp(min=0.0)
    den = (c2 - c3 * xp).clamp(min=_EPS)
    return (num / den).pow(1.0 / m1).clamp(min=0.0)


def _hlg_inverse_oetf(x: torch.Tensor) -> torch.Tensor:
    """Approximate inverse ARIB STD-B67 HLG OETF."""
    x = x.clamp(min=0.0)
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073
    return torch.where(x <= 0.5, (x * x) / 3.0, (torch.exp((x - c) / a) + b) / 12.0).clamp(min=0.0)


def _generic_log_decode(x: torch.Tensor, gamma: float = 2.2, stops: float = 14.0) -> torch.Tensor:
    """Safe approximate log-camera decode for conditioning descriptors."""
    x = x.clamp(0.0, 1.0)
    # Map [0, 1] to roughly [-stops/2, stops/2] around middle grey.
    ev = (x - 0.5) * stops
    return (0.18 * torch.pow(torch.tensor(2.0, device=x.device, dtype=x.dtype), ev)).clamp(min=0.0)


def normalize_to_scene_linear(
    image: torch.Tensor,
    format_id: torch.Tensor | int | None = None,
    y_max: float = 10000.0,
) -> torch.Tensor:
    """Normalize mixed SDR/HDR/log/linear inputs into approximate scene-linear RGB.

    Args:
        image: (B, 3, H, W) floating image tensor.
        format_id: scalar int, (B,) tensor, or None. None defaults to linear.
        y_max: PQ peak value, retained for API clarity.
    """
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"Expected image with shape (B, 3, H, W), got {tuple(image.shape)}")

    B = image.shape[0]
    if format_id is None:
        format_id = torch.full((B,), FORMAT_TO_ID["linear"], device=image.device, dtype=torch.long)
    elif isinstance(format_id, int):
        format_id = torch.full((B,), format_id, device=image.device, dtype=torch.long)
    else:
        format_id = format_id.to(device=image.device, dtype=torch.long).view(-1)
        if format_id.numel() == 1 and B > 1:
            format_id = format_id.expand(B)
    if format_id.numel() != B:
        raise ValueError(f"format_id must have {B} values, got {format_id.numel()}")

    out = torch.empty_like(image)
    for idx, name in enumerate(FORMAT_NAMES):
        mask = format_id == idx
        if not mask.any():
            continue
        x = image[mask]
        if name == "sdr":
            y = _srgb_to_linear(x)
        elif name == "pq":
            y = _pq_eotf(x, y_max=y_max)
        elif name == "hlg":
            y = _hlg_inverse_oetf(x)
        elif name == "linear":
            y = x
        elif name in LOG_TO_LINEAR:
            # Exact analytic inverse of the matching vendor curve. This must be
            # the true inverse of the encode used to build training targets
            # (rudra.color_curves) or scene-linear supervision is corrupted
            # (review §3.1). The previous _generic_log_decode approximation is
            # kept only as a fallback for unknown formats below.
            y = LOG_TO_LINEAR[name](x).clamp(min=0.0)
        else:
            y = _generic_log_decode(x)
        out[mask] = y
    return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=0.0)


def encode_scene_linear_to_format(
    image_linear: torch.Tensor,
    format_id: torch.Tensor | int | None = None,
) -> torch.Tensor:
    """Encode scene-linear radiance into a per-sample target format.

    Inverse of :func:`normalize_to_scene_linear` for the log/sdr families.
    Used to re-encode a linear image into a simulated source format so the
    dynamic-range descriptor sees a realistically-coded input (review §3.2
    multi-curve augmentation). Round-trips exactly with the decode above for
    every curve in ``rudra.color_curves``.
    """
    if image_linear.ndim != 4 or image_linear.shape[1] != 3:
        raise ValueError(f"Expected (B, 3, H, W), got {tuple(image_linear.shape)}")

    B = image_linear.shape[0]
    if format_id is None:
        format_id = torch.full((B,), FORMAT_TO_ID["linear"], device=image_linear.device, dtype=torch.long)
    elif isinstance(format_id, int):
        format_id = torch.full((B,), format_id, device=image_linear.device, dtype=torch.long)
    else:
        format_id = format_id.to(device=image_linear.device, dtype=torch.long).view(-1)
        if format_id.numel() == 1 and B > 1:
            format_id = format_id.expand(B)

    out = torch.empty_like(image_linear)
    for idx, name in enumerate(FORMAT_NAMES):
        mask = format_id == idx
        if not mask.any():
            continue
        x = image_linear[mask].clamp(min=0.0)
        if name == "sdr":
            out[mask] = _linear_to_srgb(x)
        elif name in LINEAR_TO_LOG:
            out[mask] = LINEAR_TO_LOG[name](x)
        else:  # linear / pq / hlg fall through unchanged (descriptor handles them)
            out[mask] = x
    return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=0.0)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(min=0.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=_EPS).pow(1.0 / 2.4) - 0.055)
