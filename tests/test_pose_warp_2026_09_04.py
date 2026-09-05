"""Frame-to-frame correspondence for rendered camera moves, checked end to end.

The renderer writes each frame's camera pose, and the camera only rotates, so
two frames of a clip are related by an exact homography. That makes temporal
measurements estimator-free -- but only while `rudra.pose_warp`'s camera model
stays identical to the renderer's. Nothing enforces that except this file.

The central test closes the loop: build a panorama, let the RENDERER project
two views of it, then ask pose_warp to put one back on the other. If either
side's convention drifts -- ray grid, rotation order, pixel centre, the +Z
axis -- the residual jumps and this fails.

Measured on real rendered frames, 4 Sep 2026, for scale:
    ground truth warped onto ground truth   0.0149 stops   (resampling only)
    the same warp with the pose 3 deg wrong 0.3706 stops
    no correction at all                    0.3162 stops
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

from pipeline.render_hdri_moves import project, ray_grid          # noqa: E402
from pipeline.render_hdri_moves import rotation as render_rotation  # noqa: E402
from rudra.pose_warp import (                                     # noqa: E402
    homography, rotation, temporal_consistency, warp_frame,
)

W, H = 160, 90
HFOV = 75.0


def panorama(width: int = 1024, height: int = 512) -> np.ndarray:
    """A panorama with structure everywhere, so a misalignment cannot hide.

    Band-limited on purpose. The renderer point-samples, so pixel-level noise
    in the source aliases hard at this output size and a sub-pixel shift then
    changes values by more than any geometric error would -- which measures
    the fixture rather than the warp. The blur keeps plenty of contrast for a
    misalignment to show up in, without that.
    """
    rng = np.random.default_rng(20260904)
    lon = np.linspace(0, 2 * np.pi, width, endpoint=False)[None, :]
    lat = np.linspace(-np.pi / 2, np.pi / 2, height)[:, None]
    base = (np.sin(lon * 7) * np.cos(lat * 5) + 1.2)
    speckle = cv2.GaussianBlur(rng.random((height, width)), (0, 0), 6.0)
    speckle = (speckle - speckle.min()) / max(float(np.ptp(speckle)), 1e-9)
    img = (base + speckle * 0.8).astype(np.float32)
    return np.repeat(img[..., None], 3, axis=-1) * np.array([1.0, 0.9, 0.8], np.float32)


def render(pano, pose):
    return project(pano, ray_grid(W, H, pose["hfov"]),
                   render_rotation(pose["yaw"], pose["pitch"], pose["roll"]))


def stops(a, b, mask):
    floor = 0.005 / 203.0
    d = np.abs(np.log2(np.maximum(a, floor)) - np.log2(np.maximum(b, floor)))
    return float(d[mask].mean())


def test_the_two_rotation_matrices_are_the_same():
    # pose_warp copies the renderer's. If one is edited alone, everything
    # downstream is quietly wrong rather than loudly broken.
    for angles in ((0, 0, 0), (37, -12, 3), (-200, 80, -45)):
        assert np.allclose(rotation(*angles), render_rotation(*angles))


def test_an_unchanged_pose_warps_to_itself():
    pose = {"yaw": 12.0, "pitch": -4.0, "roll": 1.0, "hfov": HFOV}
    frame = render(panorama(), pose)
    warped, valid = warp_frame(frame, pose, pose)
    assert valid.mean() > 0.999
    assert np.allclose(warped, frame, atol=1e-4)
    assert np.allclose(homography(pose, pose), np.eye(3), atol=1e-12)


@pytest.mark.parametrize("d_yaw", [0.25, 0.9, 1.6])
def test_the_warp_undoes_a_real_camera_move(d_yaw):
    """The renderer draws yaw drift from 0.25 to 1.6 degrees per frame."""
    pano = panorama()
    a = {"yaw": 40.0, "pitch": -6.0, "roll": 0.8, "hfov": HFOV}
    b = {"yaw": 40.0 + d_yaw, "pitch": -6.0 + 0.18, "roll": 0.8 + 0.05, "hfov": HFOV}
    fa, fb = render(pano, a), render(pano, b)

    warped, valid = warp_frame(fb, b, a)
    aligned = stops(fa, warped, valid)
    raw = stops(fa, fb, valid)

    assert valid.mean() > 0.9, "a one-frame move should overlap almost entirely"
    assert aligned < 0.05, f"warp left {aligned:.4f} stops; the camera models disagree"
    assert raw > 4 * aligned, (
        f"the warp barely helped ({raw:.4f} -> {aligned:.4f}); a warp that does "
        f"nothing would also pass the line above")


def test_a_wrong_pose_is_much_worse():
    # Guards the test above against passing for a no-op warp.
    pano = panorama()
    a = {"yaw": 40.0, "pitch": 0.0, "roll": 0.0, "hfov": HFOV}
    b = {"yaw": 41.0, "pitch": 0.0, "roll": 0.0, "hfov": HFOV}
    fa, fb = render(pano, a), render(pano, b)

    good, vg = warp_frame(fb, b, a)
    bad, vb = warp_frame(fb, {**b, "yaw": b["yaw"] + 3.0}, a)
    assert stops(fa, bad, vb) > 5 * stops(fa, good, vg)


def test_a_zoom_is_handled_too():
    # hfov changes along a move, so the intrinsics differ between the frames.
    pano = panorama()
    a = {"yaw": 10.0, "pitch": 0.0, "roll": 0.0, "hfov": 75.0}
    b = {"yaw": 10.5, "pitch": 0.0, "roll": 0.0, "hfov": 73.0}
    fa, fb = render(pano, a), render(pano, b)
    warped, valid = warp_frame(fb, b, a)
    assert stops(fa, warped, valid) < 0.05


def test_the_mask_excludes_what_the_other_frame_never_saw():
    pano = panorama()
    a = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "hfov": HFOV}
    coverage = []
    for d in (1.0, 10.0, 30.0):
        b = {**a, "yaw": d}
        coverage.append(warp_frame(render(pano, b), b, a)[1].mean())
    assert coverage[0] > coverage[1] > coverage[2], coverage
    assert coverage[2] < 0.7, "a 30 degree turn should leave a lot unseen"


def test_ground_truth_is_consistent_with_itself():
    pano = panorama()
    poses = [{"yaw": 20.0 + 0.9 * i, "pitch": -3.0 + 0.18 * i,
              "roll": 0.5 + 0.05 * i, "hfov": 75.0 - 0.25 * i} for i in range(9)]
    frames = [render(pano, p) for p in poses]
    out = temporal_consistency(frames, poses)
    assert out["pairs"] == 8
    assert out["mean_overlap"] > 0.9
    assert out["mean_stops"] < 0.05, (
        f"ground truth disagreed with itself by {out['mean_stops']:.4f} stops; "
        f"that is the floor every model result is measured against")


def test_the_guards():
    f = [np.zeros((H, W, 3), np.float32)] * 3
    p = [{"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "hfov": HFOV}] * 3
    with pytest.raises(ValueError, match="frames but"):
        temporal_consistency(f, p[:2])
    with pytest.raises(ValueError, match="at least two frames"):
        temporal_consistency(f[:1], p[:1])
