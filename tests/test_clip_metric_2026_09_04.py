"""Scoring a clip as a clip, not as nine unrelated stills.

`hdr_vdp3_jod` loops the batch and calls cvvdp once per frame. ColorVideoVDP
models temporal masking; used that way it cannot see flicker, because every
frame is identical work. Measured here: two clips carrying the SAME 0.0600
relative error on every frame, one steady and one inverting sign each frame,
both score 10.000 per-frame. As clips they score 10.000 and 5.111.

That gap is the entire subject of the v02 temporal track, so the metric has
to be able to see it before any model is trained against it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")

from rudra.hdrvdp import (  # noqa: E402
    colorvideovdp_available, hdr_vdp3_clip_jod, hdr_vdp3_jod,
)

needs_cvvdp = pytest.mark.skipif(
    not colorvideovdp_available(),
    reason="ColorVideoVDP not installed (pip install cvvdp)")

T, H, W = 9, 96, 96
AMPLITUDE = 0.06
NITS = 203.0


def reference() -> "torch.Tensor":
    """A clip with smooth motion: a bright edge sweeping across the frame."""
    x = torch.linspace(0, 1, W)
    ref = torch.zeros(T, 3, H, W)
    for t in range(T):
        ref[t] = (0.15 + 2.5 * torch.sigmoid((x - (0.2 + 0.05 * t)) * 40.0)).expand(3, H, W)
    return ref


def steady_and_flickering(ref):
    """Two errors of identical per-frame magnitude, one of them alternating."""
    steady = ref * (1.0 + AMPLITUDE)
    flicker = torch.stack([ref[t] * (1.0 + AMPLITUDE * (1 if t % 2 == 0 else -1))
                           for t in range(T)])
    return steady, flicker


def test_the_two_errors_really_are_the_same_size_per_frame():
    # If this drifts the comparison below proves nothing, so it is asserted
    # rather than assumed.
    ref = reference()
    steady, flicker = steady_and_flickering(ref)
    rel = lambda c: ((c - ref).abs().mean(dim=(1, 2, 3))      # noqa: E731
                     / ref.mean(dim=(1, 2, 3)))
    assert torch.allclose(rel(steady), rel(flicker), atol=1e-6)
    assert torch.allclose(rel(steady), torch.full((T,), AMPLITUDE), atol=1e-3)


@needs_cvvdp
def test_per_frame_scoring_cannot_see_flicker():
    ref = reference()
    steady, flicker = steady_and_flickering(ref)
    a, _ = hdr_vdp3_jod(steady, ref, diffuse_white_nits=NITS)
    b, _ = hdr_vdp3_jod(flicker, ref, diffuse_white_nits=NITS)
    assert abs(a - b) < 0.05, (
        "per-frame scoring separated these; if it now can, this test's premise "
        "is gone and the clip path needs re-justifying")


@needs_cvvdp
def test_clip_scoring_penalises_flicker():
    ref = reference()
    steady, flicker = steady_and_flickering(ref)
    a, ba = hdr_vdp3_clip_jod(steady, ref, frames_per_second=24.0, diffuse_white_nits=NITS)
    b, bb = hdr_vdp3_clip_jod(flicker, ref, frames_per_second=24.0, diffuse_white_nits=NITS)
    assert ba == bb == "colorvideovdp", "the proxy has no temporal model to test"
    assert a - b > 1.0, f"clip scoring saw no flicker: steady {a:.3f}, flicker {b:.3f}"


@needs_cvvdp
def test_a_bare_clip_and_a_batch_of_one_agree():
    ref = reference()
    _, flicker = steady_and_flickering(ref)
    a, _ = hdr_vdp3_clip_jod(flicker, ref, 24.0, diffuse_white_nits=NITS)
    b, _ = hdr_vdp3_clip_jod(flicker[None], ref[None], 24.0, diffuse_white_nits=NITS)
    assert abs(a - b) < 1e-6


# These need no backend: the guards run before cvvdp is ever reached.

def test_zero_fps_is_refused():
    z = torch.zeros(T, 3, 8, 8)
    with pytest.raises(ValueError, match="frames_per_second"):
        hdr_vdp3_clip_jod(z, z, frames_per_second=0.0)


def test_a_non_clip_shape_is_refused():
    z = torch.zeros(T, 4, 8, 8)
    with pytest.raises(ValueError, match=r"expected \(B,T,3,H,W\)"):
        hdr_vdp3_clip_jod(z, z, frames_per_second=24.0)


def test_mismatched_clips_are_refused():
    with pytest.raises(ValueError, match="clip shapes differ"):
        hdr_vdp3_clip_jod(torch.zeros(9, 3, 8, 8), torch.zeros(8, 3, 8, 8),
                          frames_per_second=24.0)
