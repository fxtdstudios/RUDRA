"""Exact frame-to-frame correspondence for rendered HDRI camera moves.

`pipeline/render_hdri_moves.py` flies a virtual camera through an
equirectangular panorama and writes the camera's yaw, pitch, roll and hfov
into every frame's meta JSON. The camera only ROTATES -- a panorama has no
parallax to translate against -- so the mapping between any two frames of a
clip is a plain homography in camera-space rays, and it is known exactly.

That is worth more than it sounds. Temporal work normally has to estimate
optical flow and then carry the estimator's error into every number built on
it. Here the correspondence is analytic, so:

  * a temporal consistency measure has no estimator error in it at all, and
  * a perfectly aligned model is available as an upper bound, which is how
    §7 of the v01 paper stayed honest about what was reachable.

The camera model is copied from the renderer and must stay identical to it;
`warp_is_exact` in the tests holds the two together by warping ground truth
onto ground truth and requiring the residual to vanish.

Torch-free: numpy and cv2 only, so this runs wherever the corpus does.
"""
from __future__ import annotations

import math
from typing import Mapping

import cv2
import numpy as np

__all__ = ["rotation", "camera_rays", "homography", "warp_frame",
           "temporal_consistency"]


def rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """World-from-camera rotation. Identical to the renderer's."""
    y, p, r = (math.radians(v) for v in (yaw_deg, pitch_deg, roll_deg))
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0],
                   [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)],
                   [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0],
                   [0, 0, 1]])
    return ry @ rx @ rz


def camera_rays(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Camera-space ray per pixel, (H,W,3), unnormalised, +Z forward."""
    half = math.tan(math.radians(hfov_deg) * 0.5)
    aspect = height / width
    xs = (np.arange(width, dtype=np.float64) + 0.5 - width * 0.5) / (width * 0.5) * half
    ys = (np.arange(height, dtype=np.float64) + 0.5 - height * 0.5) / (height * 0.5) \
        * half * aspect
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx, -gy, np.ones_like(gx)], axis=-1)


def homography(src_pose: Mapping[str, float], dst_pose: Mapping[str, float]) -> np.ndarray:
    """Camera-space rays of *src* expressed in *dst*'s camera space.

    A pure rotation, because the camera never translates. Compose it with the
    intrinsics of each frame -- which differ whenever the move zooms -- and
    the pixel mapping follows.
    """
    r_src = rotation(src_pose["yaw"], src_pose["pitch"], src_pose["roll"])
    r_dst = rotation(dst_pose["yaw"], dst_pose["pitch"], dst_pose["roll"])
    return r_dst.T @ r_src


def warp_frame(src: np.ndarray,
               src_pose: Mapping[str, float],
               dst_pose: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Resample *src* into *dst*'s view. Returns (warped, valid mask).

    Linear light in, linear light out. Interpolating after a tone curve would
    soften exactly the highlights this corpus exists to preserve, which is the
    same reason the renderer samples the panorama in linear.

    The mask is False wherever *dst* looks somewhere *src* never saw -- behind
    the camera, or outside its frame. Every measurement below is taken over
    the mask only; averaging over pixels that have no counterpart would score
    the camera move rather than the model.
    """
    h, w = src.shape[:2]
    rays = camera_rays(w, h, float(dst_pose["hfov"]))          # dst pixel -> dst ray
    back = rays @ homography(dst_pose, src_pose).T             # -> src camera space

    half = math.tan(math.radians(float(src_pose["hfov"])) * 0.5)
    aspect = h / w
    z = back[..., 2]
    forward = z > 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        gx = np.where(forward, back[..., 0] / z, 0.0)
        gy = np.where(forward, -back[..., 1] / z, 0.0)
    u = gx / half * (w * 0.5) + w * 0.5 - 0.5
    v = gy / (half * aspect) * (h * 0.5) + h * 0.5 - 0.5

    # A hair of tolerance on the bounds. Warping a frame onto its own pose is
    # an exact round trip, but it lands on the first row and column at
    # -1.4e-14 rather than 0, and a strict >= threw the frame's border away.
    eps = 1e-6
    valid = (forward & (u >= -eps) & (u <= w - 1 + eps)
             & (v >= -eps) & (v <= h - 1 + eps))
    warped = cv2.remap(src.astype(np.float32),
                       u.astype(np.float32), v.astype(np.float32),
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    return warped, valid


def temporal_consistency(frames: list[np.ndarray],
                         poses: list[Mapping[str, float]],
                         floor_nits: float = 0.005,
                         nits_per_unit: float = 203.0) -> dict:
    """How much a clip disagrees with itself once the camera move is removed.

    Each frame is warped onto its predecessor and compared in log2 radiance,
    over the pixels both frames actually saw. A reconstruction that is stable
    scores near the resampling floor; one that flickers does not, and unlike a
    flow-based flicker score there is no estimator error in the number.

    Returned in STOPS, which is the unit the rest of the pipeline argues in:
    ``mean_stops`` is the average absolute disagreement, ``p99_stops`` the
    tail that shows up as sparkle rather than drift.
    """
    if len(frames) != len(poses):
        raise ValueError(f"{len(frames)} frames but {len(poses)} poses")
    if len(frames) < 2:
        raise ValueError("temporal consistency needs at least two frames")

    floor = floor_nits / nits_per_unit
    diffs, coverage = [], []
    for i in range(len(frames) - 1):
        warped, valid = warp_frame(frames[i + 1], poses[i + 1], poses[i])
        coverage.append(float(valid.mean()))
        if not valid.any():
            continue
        a = np.log2(np.maximum(frames[i], floor))
        b = np.log2(np.maximum(warped, floor))
        diffs.append(np.abs(a - b)[valid[..., None].repeat(a.shape[-1], axis=-1)]
                     if a.ndim == 3 else np.abs(a - b)[valid])

    if not diffs:
        raise ValueError("no frame pair overlapped; are the poses from one clip?")
    all_diff = np.concatenate(diffs)
    return {
        "mean_stops": float(all_diff.mean()),
        "p99_stops": float(np.percentile(all_diff, 99)),
        "max_stops": float(all_diff.max()),
        "mean_overlap": float(np.mean(coverage)),
        "pairs": len(diffs),
    }
