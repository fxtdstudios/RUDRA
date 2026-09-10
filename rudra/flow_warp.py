"""Frame-to-frame correspondence ESTIMATED from the pixels, not read off the poses.

`rudra/pose_warp.py` warps one rendered frame onto another using the camera
angles the renderer wrote. That is exact, and it is why the v02 gate could
give an honest upper bound. It is also unavailable: a plate arrives as pixels,
and nothing in it says where the camera was pointing.

So the gate's answer -- +0.592 JOD mean achievable on drifted clips, 6 Sep
2026 -- is an answer about PERFECT alignment. This module asks the follow-up
the architecture depends on: how much of it survives when the correspondence
has to be estimated? Whatever is left is what a temporal model can actually
reach; the difference is estimator error, and no architecture recovers it.

Two things make this harder than ordinary optical flow, and both are the point
rather than an inconvenience:

  * EXPOSURE IS WHAT CARRIES THE INFORMATION, and exposure breaks the
    assumption flow is built on. Dense flow solves for brightness constancy --
    a scene point keeps its value between frames -- and the drifted corpus
    exists precisely because that is false there. `normalise=True` removes the
    first-order violation by standardising each frame before matching, which
    is what a practical system would do; it cannot remove the rest, and what
    it cannot remove is a real cost of estimating rather than knowing.
  * FLOW IS LEAST RELIABLE WHERE THE VALUE IS. A blown highlight is a flat
    white region with no texture to match, and that region is exactly the one
    a temporal model is being asked to reconstruct. Forward-backward
    consistency is the guard: a correspondence that does not survive the round
    trip is dropped rather than trusted, so the aligned mean averages fewer
    pixels rather than wrong ones.

Interface matches `pose_warp.warp_frame`: (warped, valid), linear light in and
out, mask False wherever the destination has no trustworthy counterpart.

Torch-free: numpy and cv2 only.
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = ["to_matching_gray", "estimate_flow", "warp_with_flow",
           "forward_backward_valid", "forward_backward_drift",
           "texture_energy", "BACKENDS"]

# DIS is the right estimator here: dense, accurate enough for sub-pixel
# alignment, and fast enough that 72 ordered pairs of a nine-frame 720p clip
# cost seconds rather than minutes. MEDIUM rather than ULTRAFAST because the
# question is what a temporal model COULD reach; handicapping the estimator to
# save time would answer a different one.
_PRESET = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM


def to_matching_gray(frame: np.ndarray, normalise: bool = True) -> np.ndarray:
    """8-bit luma for the matcher, optionally exposure-normalised.

    The input is the DEGRADED SDR the network saw -- not ground truth, and not
    the HDR prediction. A model at inference has the plate and nothing else,
    so estimating correspondence from anything richer would quietly restore
    the oracle this module exists to remove.
    """
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_RGB2GRAY)
    else:
        gray = frame.astype(np.float32)
    if normalise:
        # Standardising each frame cancels a per-frame gain and offset, which
        # is what an exposure ramp mostly is. The residual -- the tone curve's
        # curvature, and clipping at both ends -- does not cancel, and should
        # not: it is part of what estimation costs.
        centre, spread = float(gray.mean()), float(gray.std())
        gray = (gray - centre) / (spread + 1e-6) * 48.0 + 128.0
    else:
        lo, hi = float(gray.min()), float(gray.max())
        gray = (gray - lo) / (hi - lo + 1e-6) * 255.0
    return np.clip(gray, 0, 255).astype(np.uint8)


def texture_energy(gray: np.ndarray, window: int = 15) -> np.ndarray:
    """Local gradient energy: how much there is to match on, per pixel.

    THE MISSING GUARD, measured 6 Sep 2026. Forward-backward consistency is
    the standard check on a flow field and it does not catch the failure that
    matters here: in a FLAT region any displacement round-trips perfectly, so
    the check passes and the match is still guesswork. On an open-sky panorama
    that let 57% of the low-texture pixels through with a median endpoint
    error of 1.35 px against the analytic poses, where textured pixels of the
    same frame sat at 0.15 px -- nine times better.

    That is not a small effect on the result. Averaging neighbours misaligned
    by a pixel across a smooth sky gradient produces an error that is small,
    spatially coherent and different every frame, which PU21-PSNR barely
    registers and a video metric punishes: over 50 drifted clips, estimated
    alignment kept 75% of the PU21 gain and 3% of the JOD gain.

    Typical values from the boxed Sobel magnitude on 8-bit luma, same clips:

        open sky / desert road    median  4.2 - 6.6
        interior / built scene    median 17.8 - 31.0

    which is what makes a floor near 8 separate them.
    """
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    return cv2.boxFilter(np.hypot(gx, gy), -1, (window, window))


BACKENDS = ("dis", "raft")

_RAFT: dict[str, object] = {}


def _raft(device: str):
    """RAFT-large, loaded once. torchvision downloads the weights on first use.

    LARGE, not small, and the difference is the whole reason this backend
    exists. Measured 6 Sep 2026 against the analytic poses, four frames apart,
    median endpoint error in the flattest quarter of the frame:

        scene                       DIS    RAFT-small   RAFT-large
        drackenstein_quarry_puresky 2.61       0.95         0.35
        ostrich_road                0.60       0.34         0.11
        billiard_hall               0.21        --          0.10

    RAFT-small lands between the two and would confound the experiment; the
    question is whether a GOOD estimator recovers what DIS lost in flat
    regions, not whether a cheaper one does.
    """
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    if device not in _RAFT:
        model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        _RAFT[device] = model.eval().to(device)
    return _RAFT[device]


def _raft_flow(dst_gray: np.ndarray, src_gray: np.ndarray,
               device: str) -> np.ndarray:
    import torch

    def prepare(gray):
        three = np.repeat(gray[None].astype(np.float32), 3, axis=0) / 255.0
        return (torch.from_numpy(three) * 2.0 - 1.0)[None].to(device)

    h, w = dst_gray.shape
    # RAFT strides by 8 and its correlation pyramid needs a feature map of at
    # least 16, so nothing under 128 px a side goes in unpadded. 1280x720
    # needs neither, but a crop or a half-resolution pass does, and the error
    # it raises otherwise names feature-map sizes rather than image sizes.
    target_h, target_w = max(h, 128), max(w, 128)
    pad_h = (target_h + (-target_h) % 8) - h
    pad_w = (target_w + (-target_w) % 8) - w
    dst, src = prepare(dst_gray), prepare(src_gray)
    if pad_h or pad_w:
        import torch.nn.functional as F
        dst = F.pad(dst, (0, pad_w, 0, pad_h), mode="replicate")
        src = F.pad(src, (0, pad_w, 0, pad_h), mode="replicate")
    with torch.no_grad():
        flow = _raft(device)(dst, src)[-1]
    return flow[0, :, :h, :w].permute(1, 2, 0).float().cpu().numpy()


def _rescale_flow(flow: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resample a flow field to (height, width) and rescale its VECTORS.

    Both halves are needed and forgetting the second is silent: a field
    estimated at half resolution describes half-sized displacements, so
    resizing it without multiplying leaves every warp short by a factor of
    two -- which looks like a plausible flow and aligns nothing.
    """
    h, w = flow.shape[:2]
    out = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
    out[..., 0] *= width / w
    out[..., 1] *= height / h
    return out


def estimate_flow(dst_gray: np.ndarray, src_gray: np.ndarray,
                  backend: str = "dis", device: str = "cpu",
                  scale: float = 1.0) -> np.ndarray:
    """Displacement on DST's grid pointing into SRC. (H,W,2), x then y.

    Argument order is destination first, which is the opposite of how flow is
    usually written and the only order that composes with `cv2.remap`: remap
    asks, for every output pixel, where to read from. Both backends define
    their flow on the FIRST image's grid, so passing the destination first
    gives exactly the lookup remap wants. Getting this backwards produces a
    warp that looks plausible and is wrong by twice the motion.

    `dis` is the fast classical estimator and the default. `raft` is the
    learned one, and on a CPU it is roughly 200x slower per pair -- 20 s
    against 0.1 s at 720p -- which is why the sweep that uses it belongs on a
    GPU. `scale` estimates at a reduced resolution and rescales the result,
    which is what makes RAFT fit a machine without a large GPU.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown flow backend {backend!r}; expected one of {BACKENDS}")
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"scale must be in (0, 1]; got {scale}")

    full_h, full_w = dst_gray.shape
    if scale < 1.0:
        # RAFT-large builds an all-pairs correlation volume at 1/8 resolution:
        # at 1280x720 that is (160x90) squared per level, several GB, and it
        # OOM-killed a 7 GB box on 6 Sep 2026 with no traceback -- the process
        # simply vanished. Estimating smaller and rescaling the vectors brings
        # it back inside a laptop while keeping most of the accuracy that made
        # RAFT worth using. Measured against the analytic poses, median
        # endpoint error in the flattest quarter of an open-sky clip:
        #
        #     DIS full   2.61 px      RAFT 0.50   0.84 px
        #     RAFT 1.00  0.35 px      RAFT 0.75   0.44 px
        #
        # 0.75 keeps the difference that matters and peaks at 1.3 GB.
        # Multiples of 8 because RAFT strides by 8.
        small_h = max(128, int(round(full_h * scale)) // 8 * 8)
        small_w = max(128, int(round(full_w * scale)) // 8 * 8)
        dst_gray = cv2.resize(dst_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        src_gray = cv2.resize(src_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

    if backend == "dis":
        flow = cv2.DISOpticalFlow_create(_PRESET).calc(dst_gray, src_gray, None)
    else:
        flow = _raft_flow(dst_gray, src_gray, device)
    return flow if scale == 1.0 else _rescale_flow(flow, full_w, full_h)


def _sample(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return cv2.remap(field, x.astype(np.float32), y.astype(np.float32),
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def forward_backward_drift(flow: np.ndarray,
                           back: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(round-trip error in pixels, in-frame mask) for every destination pixel.

    Factored out of `forward_backward_valid` because the drift is worth more
    than the threshold applied to it. A pixel that round-trips to 0.1 px is a
    better correspondence than one that scrapes in at 1.4 px, and a combiner
    that averages them equally throws that away -- which is what the aligned
    mean did, and part of why estimated alignment fell short of the poses.
    """
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    px, py = xs + flow[..., 0], ys + flow[..., 1]
    inside = (px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)
    round_trip = np.stack((px + _sample(back[..., 0], px, py),
                           py + _sample(back[..., 1], px, py)), axis=-1)
    drift = np.hypot(round_trip[..., 0] - xs, round_trip[..., 1] - ys)
    return drift.astype(np.float32), inside


def forward_backward_valid(flow: np.ndarray, back: np.ndarray,
                           tolerance: float = 1.5,
                           energy: np.ndarray | None = None,
                           texture_floor: float = 0.0) -> np.ndarray:
    """Pixels whose correspondence survives the round trip, and stay in frame.

    A pixel of destination lands at p in the source under `flow`; the source's
    own flow back should return it to where it started. Where it does not, the
    match is an occlusion or a boundary, and is dropped.

    `texture_floor` adds the guard the round trip cannot provide -- see
    `texture_energy`. Pass the destination frame's energy, since that is the
    grid the flow is defined on and the region being reconstructed.
    """
    drift, inside = forward_backward_drift(flow, back)
    valid = inside & (drift <= tolerance)
    if texture_floor > 0.0:
        if energy is None:
            raise ValueError("texture_floor needs the destination's energy")
        valid &= energy >= texture_floor
    return valid


def warp_with_flow(src: np.ndarray, flow: np.ndarray,
                   valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resample *src* onto the grid *flow* is defined on. Linear light."""
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    warped = cv2.remap(src.astype(np.float32),
                       (xs + flow[..., 0]).astype(np.float32),
                       (ys + flow[..., 1]).astype(np.float32),
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    return warped, valid
