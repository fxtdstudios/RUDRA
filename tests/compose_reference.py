"""The composite, in numpy: one reference, three implementations pinned to it.

`SDR2HDRNet.forward` composes its prediction in torch, `ui/compositor.js`
composes it again in GLSL so the Studio's controls can run on the viewer's
GPU, and either could drift from the other without anyone noticing until a
master EXR disagreed with the picture it was made from.

So both are checked against this file instead of against each other:

    tests/test_frame_fields_2026_08_28.py   torch  <-> this
    tests/webgl_parity/parity.py            GLSL   <-> this

Torch-free on purpose — it has to run where torch does not.
"""
from __future__ import annotations

import numpy as np

LOG_SCALE = 16.0
MAX_HDR = 4.0
PEAK_NITS = 10_000.0
DIFFUSE_WHITE_NITS = 203.0
ACES = (2.51, 0.03, 2.43, 0.59, 0.14)
LUMA_REC709 = (0.2126, 0.7152, 0.0722)
LUMA_REC2020 = (0.2627, 0.6780, 0.0593)
REGION_SOFT_STOPS = 1.0
DEFAULT_REGIONS = (
    {"label": "highlights", "low_nits": 400.0, "high_nits": 2000.0, "ev": 0.0},
    {"label": "speculars", "low_nits": 2000.0, "high_nits": 8000.0, "ev": 0.0},
    {"label": "shadows", "low_nits": 0.05, "high_nits": 12.0, "ev": 0.0},
)


def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def inverse_aces(display_linear):
    """The clipped pixel is unknowable, so cap it just under one and solve."""
    y = np.clip(display_linear, 0.0, 0.995)
    a, b, c, d, e = ACES
    qa, qb, qc = y * c - a, y * d - b, y * e
    disc = np.maximum(qb * qb - 4.0 * qa * qc, 0.0)
    den = np.minimum(2.0 * qa, -1e-7)
    root = np.sqrt(disc)
    return np.maximum(np.maximum((-qb - root) / den, (-qb + root) / den), 0.0)


def baseline_of(sdr):
    """sRGB SDR -> the analytic scene-linear baseline, in nits/10 000."""
    return inverse_aces(srgb_to_linear(sdr)) * (2.0 * DIFFUSE_WHITE_NITS / PEAK_NITS)


def compose(sdr, residual, highlight, shadow, strength=1.0, mode="all",
            preserve=True):
    """(H, W, 3) sdr in [0, 1] plus the three head fields -> the prediction.

    A line-for-line port of the tail of SDR2HDRNet.forward: the priors gate
    the residual, the residual is added in the log domain, and preserving
    outside the masks blends back to the baseline.
    """
    base = baseline_of(sdr)
    y = (sdr * np.asarray(LUMA_REC709)).sum(-1, keepdims=True)
    highlight_prior = 1.0 / (1.0 + np.exp(-((y - 0.82) * 24.0)))
    shadow_prior = 1.0 / (1.0 + np.exp(-((0.10 - y) * 24.0)))
    gates = {"all": np.maximum(highlight_prior, shadow_prior),
             "highlights": highlight_prior,
             "shadows": shadow_prior,
             "off": np.zeros_like(highlight_prior)}
    if mode not in gates:
        raise ValueError("mode must be all, highlights, shadows, or off")
    gate = gates[mode] * strength
    ceiling = np.log1p(MAX_HDR * LOG_SCALE)
    pred_log = np.clip(np.log1p(base * LOG_SCALE) + residual * gate, 0.0, ceiling)
    pred = np.expm1(pred_log) / LOG_SCALE
    if preserve:
        pred = base + np.maximum(highlight, shadow) * (pred - base)
    return pred


def display_map(pred, display_nits):
    """The viewer's exposure and clip. No tone curve, deliberately."""
    return linear_to_srgb(np.clip(pred * (PEAK_NITS / max(display_nits, 1e-3)), 0.0, 1.0))


def qualifier_mask(rgb_nits, low_nits, high_nits, softness_stops=REGION_SOFT_STOPS):
    """A copy of rudra.delivery.controls.qualifier_mask, kept deliberately.

    The WebGL harness has to run with numpy and nothing else, so it cannot
    import the product package. test_frame_fields_2026_08_28.py asserts the
    two agree, which is the only thing that makes the duplication safe.
    """
    y = np.maximum((np.asarray(rgb_nits, dtype=np.float64)
                    * np.asarray(LUMA_REC2020)).sum(-1), 1e-6)
    log_y = np.log2(y)
    soft = max(float(softness_stops), 1e-3)
    rise = np.clip((log_y - (np.log2(low_nits) - soft)) / soft, 0.0, 1.0)
    fall = np.clip(((np.log2(high_nits) + soft) - log_y) / soft, 0.0, 1.0)
    mask = np.minimum(rise, fall)
    return mask * mask * (3.0 - 2.0 * mask)


def region_gain(pred, regions, softness_stops=REGION_SOFT_STOPS):
    """Per-pixel linear gain from the Region EV bands, shape (..., 1).

    `pred` is in network units (nits / 10 000); the qualifier works in nits.
    Offsets add in stops so overlapping bands compose predictably.
    """
    nits = np.asarray(pred) * PEAK_NITS
    total = np.zeros(nits.shape[:-1], dtype=np.float64)
    for band in regions or ():
        ev = float(band.get("ev", 0.0))
        if ev == 0.0:
            continue
        total = total + ev * qualifier_mask(nits, float(band["low_nits"]),
                                            float(band["high_nits"]), softness_stops)
    return np.exp2(total)[..., None]


def apply_regions(pred, regions, softness_stops=REGION_SOFT_STOPS):
    """Grade, then hold the network's own ceiling so one limit governs."""
    if not regions or all(float(b.get("ev", 0.0)) == 0.0 for b in regions):
        return pred
    return np.clip(pred * region_gain(pred, regions, softness_stops), 0.0, MAX_HDR)
