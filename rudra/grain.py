"""Settle the grain the expansion puts into flat highlights.

MEASURED, 25 Sep 2026, a sunset plate (flat sky and halo only), luminance
high-frequency noise in nits, binned by the source's max channel:

    max channel        0.80-0.86  0.86-0.94  0.94-0.97  0.97-0.99  0.99-1.0  clipped
    source x 203             1.4        1.3        1.1        1.1       1.1      0.6
    master chain             1.5        3.6        7.4       18.0      33.0     27.3

Below the anchor's knee the master IS the source, grain and all. Above it the
reconstruction takes over, and the reconstruction stands on
``inverse_aces_approx``, whose slope near white is enormous: one 8-bit code is
0.8 nits at mid-grey, 41 at 0.95, 161 at 0.98. The source's own one-code
grain, its dither and its compression noise all come out of that slope as
tens of nits of luminance noise -- colour-neutral since the chroma carry, and
worst exactly where a user's tester found it, in the band between the valid
source and the clip.

WHAT THIS DOES. A guided filter on log luminance, the source as the guide,
applied only where the reconstruction took over (the anchor's knee band,
smoothstepped in). A guided filter fits the output, window by window, as a
linear function of the guide: where the source has structure -- an edge, a
glint on water -- the output keeps following it; where the source is flat to
within its own grain, the fit is a local mean and the grain goes. The epsilon
is the grain level that counts as flat: two 8-bit codes. Only luminance moves;
each pixel's RGB is scaled by one factor, so hue and the chroma carry survive.

WHAT IT DOES NOT DO. Nothing below the knee, where the source is already the
answer. It cannot tell a real one-code texture from grain -- nobody can, from
eight bits -- and treats it as grain. Inside a clipped region the guide is
flat, so the output there is the model's own picture averaged over the window
(9 pixels across by default): gradients stay, pixel-scale fleck goes.
"""
from __future__ import annotations

import cv2
import numpy as np

LUMA_REC2020 = np.array([0.2627, 0.6780, 0.0593])
# The anchor's knee and softness: where the source stops being the answer.
DEFAULT_KNEE = 0.9
DEFAULT_SOFTNESS = 0.04
# The source is FLAT where its own local spread is under one 8-bit code, and
# carries structure from three: in between the weight smoothsteps.
FLAT_CODES = 1.0
STRUCTURE_CODES = 3.0
FLATNESS_RADIUS = 3                               # a 7x7 window
DEFAULT_SIGMA = 2.0                               # pixels
LUMA_FLOOR = 1e-6                                 # nits


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def source_flatness(sdr_srgb: np.ndarray) -> np.ndarray:
    """1 where the source is flat to within its grain, 0 where it has structure.

    Spread is measured on two signals and the larger wins: the max channel,
    which is what the expansion is keyed on, and the luma, so a glint that
    clipped its red but not its green and blue still reads as structure. Not
    on each channel: a dim blue under an orange sky carries its own grain and
    none of the luminance.
    """
    k = 2 * FLATNESS_RADIUS + 1
    box = lambda x: cv2.boxFilter(x, ddepth=-1, ksize=(k, k), normalize=True,
                                  borderType=cv2.BORDER_REFLECT)
    spread = np.zeros(sdr_srgb.shape[:2])
    for x in (sdr_srgb.max(axis=-1), sdr_srgb @ LUMA_REC2020):
        x = x.astype(np.float64)
        mean = box(x)
        spread = np.maximum(spread, np.sqrt(np.maximum(box(x * x) - mean * mean, 0.0)))
    return _smoothstep((STRUCTURE_CODES - spread * 255.0) / (STRUCTURE_CODES - FLAT_CODES))


def settle_highlight_grain(hdr_nits: np.ndarray, sdr_srgb: np.ndarray,
                           knee: float = DEFAULT_KNEE, softness: float = DEFAULT_SOFTNESS,
                           sigma: float = DEFAULT_SIGMA) -> np.ndarray:
    """``hdr_nits`` with the grain of flat highlights settled, luminance only."""
    if hdr_nits.shape != sdr_srgb.shape:
        raise ValueError(f"shape mismatch: {hdr_nits.shape} vs {sdr_srgb.shape}")
    if not 0.0 < knee < 1.0:
        raise ValueError(f"knee must be inside (0, 1), got {knee}")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    code = sdr_srgb.max(axis=-1).astype(np.float64)
    ramp = _smoothstep((code - (knee - softness)) / (2.0 * softness))
    if not np.any(ramp > 0.0):
        return hdr_nits
    flat = source_flatness(sdr_srgb)
    weight = ramp * flat
    if not np.any(weight > 0.0):
        return hdr_nits

    luma = np.maximum(hdr_nits @ LUMA_REC2020, 0.0)
    # Normalised convolution over the FLAT neighbours only: an edge next to a
    # flat sky lends the sky none of its brightness, and is not touched itself.
    # In linear light, not log: a log average sits below the linear one, and
    # measured that way it took 7% off the mean of sparkles on water.
    blur = lambda x: cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
    num = blur(flat * luma)
    den = blur(flat)
    settled = np.where(den > 1e-6, num / np.maximum(den, 1e-6), luma)
    target = luma + weight * (settled - luma)
    gain = np.where(luma > LUMA_FLOOR, target / np.maximum(luma, LUMA_FLOOR), 1.0)
    out = hdr_nits * gain[..., None]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
