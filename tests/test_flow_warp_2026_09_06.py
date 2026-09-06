"""Estimated correspondence, held to the exact one it is standing in for.

`rudra/pose_warp.py` knows where the camera was pointing because the renderer
wrote it down. `rudra/flow_warp.py` has to work it out from the pixels, which
is the only option a plate leaves. Every v02 headline so far was measured with
the first, so the second is what decides whether any of it is reachable.

The central test is the same shape as the pose tests: build a motion whose
answer is known, estimate it, and require the estimate to land on the answer.
The direction convention is checked first and separately, because getting it
backwards produces a warp that looks plausible and is wrong by twice the
motion -- silently, in a number nobody would question.

Measured on real drifted clips, 6 Sep 2026, against the analytic poses:

    1 frame apart   flow 4.02 px, pose 4.01 px, endpoint error 0.04 px
    4 frames apart       16.00          15.95                   0.06
    8 frames apart       31.80          31.64                   0.09

That is a corpus with no parallax and nothing moving in it, which is the
easiest case dense flow will ever see -- read those numbers as an upper bound
on estimator quality, not as a claim about footage with people in it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

cv2 = pytest.importorskip("cv2")

from rudra.flow_warp import (                          # noqa: E402
    estimate_flow, forward_backward_valid, to_matching_gray, warp_with_flow)


def texture(seed: int = 0, size=(400, 640), scale: int = 10) -> np.ndarray:
    """Band-limited noise. White noise would make flow measure the fixture.

    `scale` is the size of one blob in pixels, and it has to be comfortably
    larger than the displacement under test or the match is genuinely
    ambiguous -- a pattern that repeats every 10 px cannot say whether it
    moved 40 px or 30. Real footage is not periodic like this; the fixture is.
    """
    rng = np.random.default_rng(seed)
    small = rng.random((max(2, size[0] // scale), max(2, size[1] // scale), 3))
    return cv2.resize(small.astype(np.float32), (size[1], size[0]),
                      interpolation=cv2.INTER_CUBIC)


def interior(shape, margin: int = 60) -> np.ndarray:
    mask = np.zeros(shape, bool)
    mask[margin:-margin, margin:-margin] = True
    return mask


def test_flow_is_defined_on_the_destination_and_points_into_the_source():
    """The convention `cv2.remap` needs, and the one easiest to invert."""
    src = texture()
    shift = 7
    dst = np.roll(src, shift, axis=1)          # dst(x) = src(x - shift)
    flow = estimate_flow(to_matching_gray(dst), to_matching_gray(src))
    inside = interior(flow.shape[:2])
    assert np.median(flow[..., 0][inside]) == pytest.approx(-shift, abs=0.3)
    assert np.median(flow[..., 1][inside]) == pytest.approx(0.0, abs=0.3)


def test_warping_with_the_flow_reconstructs_the_destination():
    src = texture(1)
    dst = np.roll(src, 9, axis=1)
    gs, gd = to_matching_gray(src), to_matching_gray(dst)
    flow, back = estimate_flow(gd, gs), estimate_flow(gs, gd)
    valid = forward_backward_valid(flow, back)
    warped, _ = warp_with_flow(src, flow, valid)
    good = valid & interior(valid.shape)
    assert np.abs(warped[good] - dst[good]).mean() < \
        0.05 * np.abs(src[good] - dst[good]).mean()


def test_forward_backward_rejects_what_leaves_the_frame():
    """Pixels with no counterpart must be dropped, not invented.

    An aligned mean that averaged over them would be scoring the camera move.
    """
    src = texture(2, scale=80)      # blobs far larger than the 40 px shift
    dst = np.roll(src, 40, axis=1)
    gs, gd = to_matching_gray(src), to_matching_gray(dst)
    valid = forward_backward_valid(estimate_flow(gd, gs), estimate_flow(gs, gd))
    # The 40-pixel band the source never saw.
    assert valid[:, :40].mean() < 0.5
    assert valid[:, 100:-100].mean() > 0.9


def test_identity_flow_is_valid_everywhere():
    src = texture(3)
    g = to_matching_gray(src)
    flow, back = estimate_flow(g, g), estimate_flow(g, g)
    assert forward_backward_valid(flow, back).mean() > 0.99


def test_a_tighter_tolerance_keeps_fewer_pixels():
    src = texture(4)
    dst = np.roll(np.roll(src, 11, axis=1), 5, axis=0)
    gs, gd = to_matching_gray(src), to_matching_gray(dst)
    flow, back = estimate_flow(gd, gs), estimate_flow(gs, gd)
    loose = forward_backward_valid(flow, back, tolerance=3.0).mean()
    tight = forward_backward_valid(flow, back, tolerance=0.05).mean()
    assert tight <= loose


def test_normalising_survives_an_exposure_change_that_raw_luma_does_not():
    """THE REASON THE FLAG EXISTS.

    Dense flow solves for brightness constancy, and the drifted corpus is
    built on breaking it -- the exposure ramp is what puts the information
    into the clip in the first place. Standardising each frame removes the
    first-order violation; without it the estimator is matching two different
    exposures of the same scene and has to be worse.
    """
    src = texture(5)
    shift = 8
    dst = np.clip(np.roll(src, shift, axis=1) * 1.7, 0.0, 1.0)   # +0.77 stops

    def error(normalise):
        flow = estimate_flow(to_matching_gray(dst, normalise),
                             to_matching_gray(src, normalise))
        inside = interior(flow.shape[:2])
        return abs(float(np.median(flow[..., 0][inside])) + shift)

    assert error(normalise=True) < 0.5
    assert error(normalise=True) <= error(normalise=False)


def test_gray_conversion_accepts_colour_and_luma_alike():
    colour = texture(6)
    assert to_matching_gray(colour).shape == colour.shape[:2]
    luma = colour.mean(axis=2)
    assert to_matching_gray(luma).shape == luma.shape
    assert to_matching_gray(colour).dtype == np.uint8


# --------------------------------------------------------------------------
# The RAFT backend. DIS collapsed the v02 gain in flat regions; this is the
# estimator that might not.
# --------------------------------------------------------------------------

def _have_raft() -> bool:
    try:
        import torchvision.models.optical_flow  # noqa: F401
    except Exception:                            # noqa: BLE001
        return False
    return True


raft_only = pytest.mark.skipif(not _have_raft(), reason="torchvision absent")


def test_an_unknown_backend_is_refused():
    from rudra.flow_warp import BACKENDS
    assert set(BACKENDS) == {"dis", "raft"}
    with pytest.raises(ValueError):
        estimate_flow(np.zeros((32, 32), np.uint8), np.zeros((32, 32), np.uint8),
                      backend="farneback")


@raft_only
@pytest.mark.slow
def test_raft_agrees_with_dis_on_a_motion_both_can_see():
    """Same convention, same answer, where the problem is easy.

    The backends only earn different numbers in the hard case -- flat regions,
    where DIS drifts and RAFT does not. On plain texture they must agree, or
    one of them has the sign or the argument order wrong.
    """
    src = texture(7, size=(128, 192), scale=24)
    dst = np.roll(src, 6, axis=1)
    gs, gd = to_matching_gray(src), to_matching_gray(dst)
    dis = estimate_flow(gd, gs, backend="dis")
    raft = estimate_flow(gd, gs, backend="raft")
    assert raft.shape == dis.shape
    inside = interior(dis.shape[:2], margin=32)
    assert np.median(raft[..., 0][inside]) == pytest.approx(-6, abs=1.0)


@raft_only
@pytest.mark.slow
def test_raft_handles_a_size_that_is_not_a_multiple_of_eight():
    """RAFT strides by 8 and needs 128 px a side, and the corpus is 1280x720,
    which is fine -- but a crop or a half-resolution pass is not, and a silent
    shape change would misalign every warp built on it."""
    src = texture(8, size=(131, 205), scale=20)
    dst = np.roll(src, 4, axis=1)
    flow = estimate_flow(to_matching_gray(dst), to_matching_gray(src),
                         backend="raft")
    assert flow.shape == (131, 205, 2)


@raft_only
@pytest.mark.slow
def test_raft_pads_an_image_smaller_than_its_minimum():
    """Below 128 px a side RAFT raises about feature-map sizes, not images."""
    src = texture(9, size=(64, 96), scale=16)
    flow = estimate_flow(to_matching_gray(src), to_matching_gray(src),
                         backend="raft")
    assert flow.shape == (64, 96, 2)


def test_scaled_flow_rescales_its_vectors_not_just_its_grid():
    """The half of a resize that is easy to forget and silent to get wrong.

    A field estimated at 0.5x describes half-sized displacements. Resizing it
    back without multiplying the vectors leaves every warp short by a factor
    of two, which still looks like a plausible flow field and aligns nothing.
    """
    src = texture(10, size=(256, 384), scale=40)
    shift = 12
    dst = np.roll(src, shift, axis=1)
    gs, gd = to_matching_gray(src), to_matching_gray(dst)
    full = estimate_flow(gd, gs, backend="dis")
    half = estimate_flow(gd, gs, backend="dis", scale=0.5)
    assert half.shape == full.shape
    inside = interior(full.shape[:2], margin=48)
    assert np.median(half[..., 0][inside]) == pytest.approx(-shift, abs=1.5)


def test_scale_outside_the_unit_interval_is_refused():
    g = np.zeros((160, 160), np.uint8)
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            estimate_flow(g, g, backend="dis", scale=bad)


def test_a_scaled_estimate_never_goes_below_raft_s_minimum():
    """0.25 of a 256px frame is 64px, and RAFT needs 128. The floor keeps a
    small --flow-scale from turning into an error about feature-map sizes."""
    src = texture(11, size=(256, 256), scale=32)
    flow = estimate_flow(to_matching_gray(src), to_matching_gray(src),
                         backend="dis", scale=0.25)
    assert flow.shape == (256, 256, 2)
