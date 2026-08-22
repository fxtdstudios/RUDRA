"""HDR10 mastering helpers for RUDRA scene-linear RGB outputs.

RUDRA's SDR-to-HDR model uses normalized absolute luminance where 1.0 equals
10,000 nits.  This module masters that signal to a selectable display peak and
encodes it with the SMPTE ST 2084 (PQ) transfer function.
"""

from __future__ import annotations

import numpy as np

from .radiometry import (
    LUMA_REC2020, PQ_C1, PQ_C2, PQ_C3, PQ_M1, PQ_M2,
)

REC2020_LUMA = np.asarray(LUMA_REC2020, dtype=np.float32)


def pq_oetf(nits: np.ndarray) -> np.ndarray:
    """Encode absolute luminance in nits to normalized ST 2084 code values."""
    values = np.clip(np.asarray(nits, dtype=np.float32) / 10000.0, 0.0, 1.0)
    m1, m2, c1, c2, c3 = PQ_M1, PQ_M2, PQ_C1, PQ_C2, PQ_C3
    powered = np.power(values, m1)
    return np.power((c1 + c2 * powered) / (1.0 + c3 * powered), m2)


def pq_eotf(code: np.ndarray) -> np.ndarray:
    """Decode normalized ST 2084 values to absolute luminance in nits."""
    values = np.clip(np.asarray(code, dtype=np.float32), 0.0, 1.0)
    m1, m2, c1, c2, c3 = PQ_M1, PQ_M2, PQ_C1, PQ_C2, PQ_C3
    powered = np.power(values, 1.0 / m2)
    ratio = np.maximum(powered - c1, 0.0) / np.maximum(c2 - c3 * powered, 1e-9)
    return 10000.0 * np.power(ratio, 1.0 / m1)


def master_to_peak(rgb_normalized: np.ndarray, peak_nits: float = 1000.0,
                   knee_nits: float | None = None) -> np.ndarray:
    """Apply a hue-preserving highlight shoulder in absolute linear light.

    Values below the knee are unchanged. Above it, luminance approaches the
    mastering peak smoothly. A final per-pixel scale prevents an individual
    RGB component from exceeding the chosen peak without changing hue.
    """
    if not 100.0 <= peak_nits <= 10000.0:
        raise ValueError("peak_nits must be between 100 and 10000")
    knee = 0.75 * peak_nits if knee_nits is None else float(knee_nits)
    if not 0.0 <= knee < peak_nits:
        raise ValueError("knee_nits must be >= 0 and below peak_nits")

    rgb_nits = np.maximum(np.asarray(rgb_normalized, dtype=np.float32), 0.0) * 10000.0
    luminance = np.sum(rgb_nits * REC2020_LUMA, axis=-1, keepdims=True)
    span = peak_nits - knee
    compressed = np.where(
        luminance <= knee,
        luminance,
        knee + span * (1.0 - np.exp(-(luminance - knee) / span)),
    )
    rgb_nits *= compressed / np.maximum(luminance, 1e-6)
    component_max = np.max(rgb_nits, axis=-1, keepdims=True)
    rgb_nits *= np.minimum(1.0, peak_nits / np.maximum(component_max, 1e-6))
    return np.clip(rgb_nits, 0.0, peak_nits)


def master_to_pq(rgb_normalized: np.ndarray, peak_nits: float = 1000.0,
                 knee_nits: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(pq_rgb, mastered_linear_rgb_nits)`` for an RUDRA output."""
    mastered = master_to_peak(rgb_normalized, peak_nits, knee_nits)
    return pq_oetf(mastered), mastered
