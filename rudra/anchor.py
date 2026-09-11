"""Anchor a reconstruction to the SDR it came from.

WHY THIS EXISTS. ``sdr_to_baseline_hdr`` multiplies inverse-ACES by
``2 * 203/10000``. The factor of two is the ``-1 EV`` that
``prepare_training_data.py`` applies BEFORE the ACES curve, so it is correct
for every frame the corpus made -- and wrong for every frame it did not. A
graded plate, a client mp4, a PNG off a workstation: none of them were exposed
down a stop first, so the baseline lifts them. Measured on the shipped
checkpoint, 10 Sep 2026:

    sRGB 0.18 -> +1.43 stops     sRGB 0.50 -> +0.48     sRGB 0.90 -> +1.24

That is a global re-exposure of material that was never clipped, and it is not
what "preserve outside the masks" means to anyone conforming to an SDR grade.

WHAT THIS DOES. Below a knee the output is forced back onto the SDR's own
scene-linear values -- identity, to within the precision of the round trip.
Above the knee the reconstruction is kept, scaled by the gain in force AT the
knee so the two halves meet. So: unclipped picture unchanged, highlights get
the headroom, and the ratio between them -- the part the network actually
reconstructed -- survives.

WHEN IT IS WRONG. Measured 10 Sep 2026 against the bench's own ground truth
(dikhololo_sunset, 3 frames): the reference HDR sits 2.81x above its own SDR on
unclipped pixels, because prepare_training_data.py MADE that SDR by exposing
down a stop before the ACES curve. On corpus data the factor of two is correct
and anchoring is catastrophic -- PU21-PSNR 53.4 dB plain, 22.5 dB anchored, a
loss of 30.9 dB.

So this is a MODE, not a repair. Whether it applies depends on how the source
was exposed, which cannot be read out of the pixels. Anchor a graded plate that
was never exposed down; do not anchor a frame the corpus produced. The bench
path leaves it off, which is why the paper's numbers are unaffected, and it
cannot be used to validate this either way -- it would penalise a fix that is
right for a plate.

WHAT IT DOES NOT DO. It cannot repair the curve's SHAPE, only its level. A
single frame's correction is a function of the source code value, so it is
exact wherever the source is valid and frozen where the source is clipped.
It is applied per PIXEL as one scalar across R, G and B, never per channel:
a per-channel correction would move hue, which is the thing being complained
about elsewhere.
"""
from __future__ import annotations

import numpy as np

DIFFUSE_WHITE_NITS = 203.0
# Rec.2020: the working space the reconstruction is written in.
LUMA_REC2020 = np.array([0.2627, 0.6780, 0.0593])
# 0.9 in sRGB is 160 nits: comfortably inside the SDR's valid range, above the
# grade's working mid-tones, and below where 8-bit codes start to run out.
DEFAULT_KNEE = 0.9


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def anchor_gain(sdr_srgb: np.ndarray, hdr_nits: np.ndarray,
                knee: float = DEFAULT_KNEE, softness: float = 0.04) -> np.ndarray:
    """Per-pixel scalar gain that puts unclipped picture back where it was.

    ``sdr_srgb`` is (H, W, 3) in [0, 1]; ``hdr_nits`` is (H, W, 3) absolute
    nits. Returns (H, W, 1) so it broadcasts across the channels.
    """
    if sdr_srgb.shape != hdr_nits.shape:
        raise ValueError(f"shape mismatch: {sdr_srgb.shape} vs {hdr_nits.shape}")
    if not 0.0 < knee < 1.0:
        raise ValueError(f"knee must be inside (0, 1), got {knee}")

    # ONE luminance definition, used on both sides. This module first used
    # Rec.709 weights while rudra/chroma.py used Rec.2020, so the anchor
    # equalised one quantity and the QC measured another: unclipped pixels
    # inside 1% read 89.5% after anchoring alone and 52.5% once the chroma
    # carry ran, with neither module wrong on its own terms.
    #
    # Rec.2020, because that is the working space the reconstruction is
    # written in and what the EXR is tagged with. Note that the pipeline does
    # NOT convert the sRGB source's primaries on the way in -- sdr_to_baseline_hdr
    # expands sRGB values in place -- so source and output are treated as the
    # same space throughout. That assumption is older than this module and is
    # worth revisiting; what matters here is that everything agrees on it.
    target = srgb_to_linear(sdr_srgb) @ LUMA_REC2020 * DIFFUSE_WHITE_NITS
    actual = hdr_nits @ LUMA_REC2020
    # The knee is driven by the MAX channel, not luma. Clipping happens per
    # channel: a saturated red at 255,40,30 has a luma of 0.29 and is every
    # bit as clipped as white. Keyed on luma it would be treated as ordinary
    # picture and pushed back down, which on the first run cost the clipped
    # region almost all of its headroom -- mean gain 1.14x instead of 2x.
    code = sdr_srgb.max(axis=-1)

    # (target + eps) / (actual + eps), not a guarded divide. Both go to zero
    # together in black, so a guarded divide has to invent a value there -- and
    # substituting 1.0 puts a step in the gain right next to pixels carrying
    # 0.5, which is a discontinuity on any surface that fades to black. With
    # the epsilon the ratio slides smoothly to 1.0 instead, and the pixels it
    # affects are the ones multiplied by nothing anyway.
    eps = 1e-4                                            # nits
    want = (target + eps) / (actual + eps)

    # The gain in force at the knee, measured from the pixels that are there,
    # so the held value follows this frame rather than a constant from a
    # different one. Median, because a handful of specular pixels sitting at
    # the knee should not set the level for the whole highlight range.
    band = (code > knee - softness) & (code < knee + softness) & (actual > 1e-9)
    hold = float(np.median(want[band])) if band.sum() >= 64 else float(np.median(want))

    gain = np.where(code <= knee - softness, want, hold)
    # Blend across the band so the derivative does not step. A step in gain is
    # invisible on a busy plate and a visible edge on a gradient sky.
    ramp = np.clip((code - (knee - softness)) / (2.0 * softness), 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)               # smoothstep
    gain = want * (1.0 - ramp) + hold * ramp
    gain = np.where(code <= knee - softness, want, gain)
    return np.nan_to_num(gain, nan=1.0, posinf=1.0, neginf=1.0)[..., None]


def anchor_to_sdr(hdr_nits: np.ndarray, sdr_srgb: np.ndarray,
                  knee: float = DEFAULT_KNEE, softness: float = 0.04) -> np.ndarray:
    """``hdr_nits`` re-levelled so unclipped picture matches ``sdr_srgb``."""
    return hdr_nits * anchor_gain(sdr_srgb, hdr_nits, knee, softness)
