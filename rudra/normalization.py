"""Format normalization utilities for RUDRA.

The goal is stable scene-linear radiance conditioning, not a color-management
replacement. For production, connect this to OCIO/ACES transforms where possible.
"""

from __future__ import annotations

import torch

from .config import FORMAT_NAMES, FORMAT_TO_ID
from .radiometry import (
    HLG_A, HLG_B, HLG_C, PQ_C1, PQ_C2, PQ_C3, PQ_M1, PQ_M2, PQ_REF_WHITE_NITS,
)
from .color_curves import LOG_TO_LINEAR, LINEAR_TO_LOG

_EPS = 1e-8


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(min=0.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))


# BT.2408 graphics/diffuse ("reference") white for PQ: 203 cd/m².
# All other formats handled here (sdr, camera log curves, linear) decode to
# diffuse-white-relative scene linear (white ≈ 1.0, 18% grey ≈ 0.18), so PQ
# and HLG must land on the same radiometric anchor or the EV/highlight
# descriptors read the same physical scene stops apart depending on the
# container format (AUDIT_2026-08-10 NEW-4: a PQ-tagged scene read ~5.6
# stops darker than S-Log3 and produced identically-zero highlight features).
_PQ_REF_WHITE_NITS = PQ_REF_WHITE_NITS


def _pq_eotf(x: torch.Tensor, y_max: float = 10000.0) -> torch.Tensor:
    """Inverse SMPTE ST.2084 PQ → diffuse-white-relative scene linear.

    The raw PQ EOTF produces L as a fraction of an ABSOLUTE 10,000 cd/m²
    (ST 2084); returning that fraction directly put PQ input on a completely
    different scale than every other format's decode. We convert to the
    project-wide diffuse-white convention by anchoring BT.2408 reference
    white (203 nits) at 1.0: linear = (L * 10000) / 203.
    """
    x = x.clamp(0.0, 1.0)
    m1, m2, c1, c2, c3 = PQ_M1, PQ_M2, PQ_C1, PQ_C2, PQ_C3
    xp = x.pow(1.0 / m2)
    num = (xp - c1).clamp(min=0.0)
    den = (c2 - c3 * xp).clamp(min=_EPS)
    frac = (num / den).pow(1.0 / m1).clamp(min=0.0)   # L / 10000 nits
    return frac * (y_max / _PQ_REF_WHITE_NITS)


def _hlg_inverse_oetf(x: torch.Tensor) -> torch.Tensor:
    """Approximate inverse ARIB STD-B67 HLG OETF → diffuse-white-relative.

    Rescaled so HLG reference/diffuse white (75% signal, BT.2408) maps to
    1.0, matching the anchor used by every other format decode here
    (AUDIT_2026-08-10 NEW-4: unscaled, HLG diffuse white landed ~1.9 stops
    below the other formats).
    """
    x = x.clamp(min=0.0)
    a, b, c = HLG_A, HLG_B, HLG_C
    lin = torch.where(x <= 0.5, (x * x) / 3.0, (torch.exp((x - c) / a) + b) / 12.0).clamp(min=0.0)
    # Inverse OETF of the 0.75 reference-white signal ≈ 0.26496.
    _ref = (torch.exp(torch.tensor((0.75 - c) / a, dtype=x.dtype, device=x.device)) + b) / 12.0
    return lin / _ref


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
        elif name == "pq":
            out[mask] = _pq_inverse_eotf(x)
        elif name == "hlg":
            out[mask] = _hlg_oetf(x)
        elif name in LINEAR_TO_LOG:
            out[mask] = LINEAR_TO_LOG[name](x)
        else:  # linear falls through unchanged
            out[mask] = x
    return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=0.0)


def _pq_inverse_eotf(lin: torch.Tensor, y_max: float = 10000.0) -> torch.Tensor:
    """Diffuse-white-relative scene linear → PQ code (true inverse of _pq_eotf).

    Previously pq/hlg "encode" was a passthrough while decode applied the
    EOTF, so an encode→decode round trip corrupted the data
    (AUDIT_2026-08-10 P2-12).
    """
    frac = (lin.clamp(min=0.0) * (_PQ_REF_WHITE_NITS / y_max)).clamp(0.0, 1.0)
    m1, m2, c1, c2, c3 = PQ_M1, PQ_M2, PQ_C1, PQ_C2, PQ_C3
    yp = frac.pow(m1)
    return ((c1 + c2 * yp) / (1.0 + c3 * yp)).pow(m2)


def _hlg_oetf(lin: torch.Tensor) -> torch.Tensor:
    """Diffuse-white-relative scene linear → HLG signal (inverse of
    _hlg_inverse_oetf, including the 75%-signal reference-white rescale)."""
    a, b, c = HLG_A, HLG_B, HLG_C
    _ref = (torch.exp(torch.tensor((0.75 - c) / a, dtype=lin.dtype, device=lin.device)) + b) / 12.0
    e = (lin.clamp(min=0.0) * _ref).clamp(min=0.0)   # undo reference-white rescale
    return torch.where(
        e <= 1.0 / 12.0,
        torch.sqrt(3.0 * e),
        a * torch.log((12.0 * e - b).clamp(min=_EPS)) + c,
    ).clamp(min=0.0)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(min=0.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=_EPS).pow(1.0 / 2.4) - 0.055)
