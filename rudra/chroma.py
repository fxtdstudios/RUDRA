"""Carry the source's chromaticity through the expansion.

MEASURED, 10 Sep 2026, flat sky of hsky.png, chroma noise as % of luma:

                          HF luma %   R-G     B-G
    SDR source                 3.93   3.09   1.47
    analytic baseline         20.66  13.25   2.31
    RUDRA full output         25.82  16.60   3.01
    ... same, luma-only       25.82   3.09   1.47

The last row is the whole finding. ``inverse_aces_approx`` is applied PER
CHANNEL, and the curve is steep near white: adjacent 8-bit codes that differ by
one step in red and none in green come out of it separated far more than they
went in. The result is per-pixel hue flecking in smooth sky -- 5.4x the
source's own -- which the eye reads as dirt long before it notices luma grain.
Reconstructing luminance and carrying the source's chromaticity puts it back
exactly, because the source's chromaticity was never the thing that needed
reconstructing.

Where the source IS clipped, its chromaticity is wrong -- a blown sky is 255,
255, 255 whatever colour it really was -- so above the knee the model's own
chroma is kept. Same knee as ``rudra.anchor``, and the two compose: anchor
fixes the level, this fixes the hue, neither touches the other's business.

Luma noise is NOT addressed here and is not the same problem: 3.93 -> 20.66 is
a steep curve amplifying 8-bit quantisation, and carrying chroma leaves it
untouched. That one needs either more bits at the source or a different curve.
"""
from __future__ import annotations

import cv2
import numpy as np

# Rec.2020 luminance: the working space the reconstruction is written in.
LUMA_REC2020 = np.array([0.2627, 0.6780, 0.0593])
DIFFUSE_WHITE_NITS = 203.0
# Higher than rudra.anchor's knee, and for a different reason. The anchor's
# knee is about LEVEL -- where expansion should start. This one is about
# VALIDITY -- the source's chromaticity is trustworthy right up until a channel
# actually clips, which is close to 1.0, not 0.9. Measured on hsky.png's flat
# sky, R-G chroma noise by knee: 0.90 -> 16.6, 0.97 -> 13.8, 0.99 -> 7.0,
# against 16.6 for doing nothing and 3.1 for the source itself.
DEFAULT_KNEE = 0.99


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def carry_source_chroma(hdr_nits: np.ndarray, sdr_srgb: np.ndarray,
                        knee: float = DEFAULT_KNEE, softness: float = 0.01,
                        mask_sigma: float = 2.0) -> np.ndarray:
    """``hdr_nits`` luminance, the source's chromaticity, below the knee.

    Luminance is preserved exactly BELOW the knee: scaling the source triplet
    by ``hdr_luma / source_luma`` leaves the Rec.2020 luminance equal to
    ``hdr_luma`` by construction, so below the knee this changes hue and
    saturation only. Inside the blend band the two chroma sources are mixed and
    luminance moves by a fraction of a nit; above it nothing is touched.
    """
    if hdr_nits.shape != sdr_srgb.shape:
        raise ValueError(f"shape mismatch: {hdr_nits.shape} vs {sdr_srgb.shape}")
    if not 0.0 < knee < 1.0:
        raise ValueError(f"knee must be inside (0, 1), got {knee}")

    source = srgb_to_linear(sdr_srgb) * DIFFUSE_WHITE_NITS
    eps = 1e-6
    source_luma = source @ LUMA_REC2020
    hdr_luma = hdr_nits @ LUMA_REC2020
    carried = source * ((hdr_luma + eps) / (source_luma + eps))[..., None]

    # Above the knee the source is clipped and its chromaticity is a lie, so
    # the model's own is kept. Smoothstep across the band: a hard switch would
    # draw a coloured edge along the shoulder of a gradient.
    #
    # The mask is BLURRED first, and that is not cosmetic. Driven by the raw
    # per-pixel max channel it flickers between the two chroma sources from one
    # pixel to the next wherever the code straddles the knee -- which
    # manufactures exactly the flecking this function exists to remove. Keyed
    # on the raw code at knee 0.97 the sky measured 22.2 against 16.6 for doing
    # nothing at all; blurred, the mask varies over objects rather than pixels.
    code = sdr_srgb.max(axis=-1)
    ramp = np.clip((code - (knee - softness)) / (2.0 * softness), 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    # Blur the RAMP, not the code. Blurring the code first drags clipped pixels
    # back below the knee whenever their neighbours are not clipped, so the
    # inside of a blown highlight ends up wearing the source's neutral
    # chromaticity -- only 32% of the model's own chroma survived where the
    # source was clipped. Blurring the ramp leaves the interior of a clipped
    # REGION at 1.0 and only softens its edge, which is what was wanted.
    ramp = cv2.GaussianBlur(ramp.astype(np.float32), (0, 0), sigmaX=mask_sigma,
                            borderType=cv2.BORDER_REPLICATE)[..., None]
    out = carried * (1.0 - ramp) + hdr_nits * ramp
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
