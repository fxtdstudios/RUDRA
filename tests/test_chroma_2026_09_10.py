"""Carrying the source's chromaticity through the expansion.

``inverse_aces_approx`` runs per channel and is steep near white, so two 8-bit
codes one step apart in red and level in green come out of it much further
apart than they went in. Measured on hsky.png's flat sky, chroma noise as a
percentage of luma: source 3.09, RUDRA 16.60. The eye reads that as dirt in a
smooth sky long before it notices luma grain.

The regression this file exists to prevent is subtler than the bug itself. The
first implementation drove the blend from the raw per-pixel max channel, which
in that same sky flickers across the knee from one pixel to the next -- so the
mask alternated between the two chroma sources and MANUFACTURED the flecking
it was removing: 22.2 against 16.6 for doing nothing at all. The mask has to
vary over objects, not pixels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.chroma import (DEFAULT_KNEE, DIFFUSE_WHITE_NITS, LUMA_REC2020,  # noqa: E402
                          carry_source_chroma, srgb_to_linear)


def scene(width: int = 128, height: int = 64, seed: int = 20260910):
    """A bright flat field with per-channel 8-bit dither.

    Bright, because that is where inverse-ACES is steep and where the flecking
    actually happens; a mid-grey field sits on the gentle part of the curve and
    shows almost nothing.
    """
    rng = np.random.default_rng(seed)
    codes = np.full((height, width, 3), 235, dtype=np.int16)
    codes += rng.integers(-1, 2, size=codes.shape)
    return np.clip(codes, 0, 255).astype(np.float64) / 255.0


def expanded_per_channel(sdr):
    """The REAL baseline curve, not a stand-in.

    The first version of this helper used ``linear ** (1/2.4)``, which is
    compressive -- it cannot produce the per-channel divergence being tested,
    so two of these tests passed on an effect that was not there. The bug lives
    in inverse-ACES being steep near white; test that.
    """
    import torch
    from rudra.sdr2hdr import sdr_to_baseline_hdr
    x = torch.from_numpy(sdr.astype(np.float32)).permute(2, 0, 1)[None]
    out = sdr_to_baseline_hdr(x)[0].permute(1, 2, 0).numpy().astype(np.float64)
    return out * 10_000.0                                   # network units -> nits


def chroma_noise(img):
    y = np.maximum(img @ LUMA_REC2020, 1e-9)
    return float(np.std((img[..., 0] - img[..., 1]) / y)) * 100


def test_luminance_is_untouched_below_the_knee():
    sdr = scene()
    hdr = expanded_per_channel(sdr)
    out = carry_source_chroma(hdr, sdr)
    before, after = hdr @ LUMA_REC2020, out @ LUMA_REC2020
    assert np.abs(after - before).max() < 1e-6 * max(before.max(), 1.0)


def test_chromaticity_comes_from_the_source_below_the_knee():
    sdr = scene()
    out = carry_source_chroma(expanded_per_channel(sdr), sdr)
    src = srgb_to_linear(sdr)

    def uv(x):
        return x[..., :2] / np.maximum(x.sum(axis=-1, keepdims=True), 1e-12)
    assert np.abs(uv(out) - uv(src)).max() < 1e-6


def test_the_flecking_actually_goes_down():
    sdr = scene()
    hdr = expanded_per_channel(sdr)
    assert chroma_noise(hdr) > 3.0 * chroma_noise(srgb_to_linear(sdr) * DIFFUSE_WHITE_NITS)
    assert chroma_noise(carry_source_chroma(hdr, sdr)) < 0.5 * chroma_noise(hdr)


def test_a_noisy_mask_must_not_manufacture_flecking():
    """The regression: a mask keyed on raw per-pixel code, in a sky that
    straddles the knee, is worse than not carrying chroma at all."""
    rng = np.random.default_rng(7)
    # One channel dithered right across the knee while the others sit well
    # below it, so the MAX channel -- which drives the mask -- genuinely
    # straddles from pixel to pixel. Dithering all three together does not do
    # it: the max of three draws lands near the top almost every time and the
    # mask comes out nearly uniform, which is why the first version of this
    # test compared two identical images and passed for the wrong reason.
    codes = np.zeros((64, 128, 3), dtype=np.int16)
    codes[..., 0] = 251 + rng.integers(-4, 5, size=codes.shape[:2])
    codes[..., 1] = 190
    codes[..., 2] = 170
    sdr = np.clip(codes, 0, 255).astype(np.float64) / 255.0
    hdr = expanded_per_channel(sdr)
    smoothed = carry_source_chroma(hdr, sdr, mask_sigma=2.0)
    pixelwise = carry_source_chroma(hdr, sdr, mask_sigma=1e-4)
    assert chroma_noise(smoothed) < chroma_noise(pixelwise)
    assert chroma_noise(smoothed) < chroma_noise(hdr)


def test_a_clipped_region_keeps_the_model_chroma():
    """A blown highlight is 255,255,255 whatever colour it was, so the source's
    chromaticity there is a lie and must not be carried."""
    sdr = scene(160, 160)
    sdr[40:120, 40:120] = 1.0                               # a big blown patch
    hdr = expanded_per_channel(sdr)
    hdr[40:120, 40:120] = [900.0, 300.0, 120.0]             # what recovery put back
    out = carry_source_chroma(hdr, sdr)
    core = (slice(70, 90), slice(70, 90))                   # well inside the patch
    ratio = out[core] / hdr[core]
    assert np.abs(ratio - 1.0).max() < 0.05, float(np.abs(ratio - 1).max())


def test_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match="shape mismatch"):
        carry_source_chroma(np.zeros((4, 4, 3)), np.zeros((4, 5, 3)))


@pytest.mark.parametrize("knee", [0.0, 1.0, -0.5, 2.0])
def test_a_knee_outside_the_range_is_refused(knee):
    sdr = scene(32, 32)
    with pytest.raises(ValueError, match="knee"):
        carry_source_chroma(expanded_per_channel(sdr), sdr, knee=knee)


def test_black_is_finite():
    sdr = np.zeros((8, 8, 3))
    out = carry_source_chroma(np.zeros((8, 8, 3)), sdr)
    assert np.isfinite(out).all()


def test_the_default_knee_is_about_clipping_not_level():
    # rudra.anchor's knee is 0.9 and answers "where should expansion start".
    # This one answers "where does the source stop being trustworthy", which is
    # where a channel clips. Sharing 0.9 left most of a bright sky above the
    # knee and the fix did nothing at all.
    assert DEFAULT_KNEE > 0.95
