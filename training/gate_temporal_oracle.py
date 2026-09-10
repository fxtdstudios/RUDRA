#!/usr/bin/env python3
"""What could a perfectly aligned temporal model win? Measure it before training one.

§7 of the v01 paper did not argue that the per-frame model was near its limit,
it measured the limit: an oracle picked the best residual scale per frame, the
gain was +5.84 dB, and a learned head recovered 3% of the variance because the
information was not in the input. That method is why the paper held up. This
applies it to time.

The claim v02 rests on is that neighbouring frames carry information a single
8-bit frame does not -- a clipped region seen at a different exposure and a
different angle. If that is true, an oracle allowed to combine aligned
neighbours must beat the per-frame reconstruction by a wide margin. If it is
not true, no architecture will find what is not there, and the honest move is
to say so now rather than after a month of training.

The oracle here is deliberately unreachable. It gets:

  * EXACT alignment, from the camera poses the renderer wrote -- no optical
    flow, no estimator error (see rudra/pose_warp.py),
  * per-pixel omniscient SELECTION among the aligned candidates, choosing
    whichever is closest to ground truth.

No model can do better than that on these frames. So the number it produces is
a ceiling, and a low ceiling is decisive in a way a low model result never is.

    python training/gate_temporal_oracle.py --clips <dir> --checkpoint <ckpt>

Reports, over the clips given: the per-frame baseline, the oracle, and the gap
between them, in PU21-PSNR and in ColorVideoVDP JOD. Read the JOD column: the
v01 paper's own §5 shows PU21 and CVVDP disagreeing by two orders of magnitude
on small differences, and half a JOD is the threshold this decision was set at.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.delivery.bench import pu_psnr                    # noqa: E402
from rudra.pose_warp import warp_frame                      # noqa: E402

# Two different "1.0"s meet here, and conflating them is a 5.6-stop error that
# still produces plausible-looking numbers. Checked on 4 Sep 2026 against a
# committed benchmark output: with the network scale below, this path
# reproduces bench/clean/shadow_v1 to 0.0034 stops; with the storage scale it
# scored 10.46 dB instead of 53.91.
NETWORK_PEAK_NITS = 10_000.0   # SDR2HDRNet.hdr: 1.0 = 10 000 nits
STORAGE_WHITE_NITS = 203.0     # decode_hdr_u16 / the EXRs: 1.0 = diffuse white
FLOOR_NITS = 0.005

# The image benchmark clamps its reference before scoring
# (export_bench_pairs.REFERENCE_CEILING, 1 000 network units = 10^7 nits) and
# this did not. A rendered panorama carries the sun at 10^8 nits and upwards,
# and an unclamped reference lets those few pixels dominate PU21-PSNR, so the
# two numbers were never on the same footing. Imported rather than restated so
# the two cannot drift apart.


def round_trip(predictions: list[np.ndarray], warp) -> list[list]:
    """Each frame's own prediction, warped out to every neighbour and back.

    THE CONTROL. An oracle picking the per-pixel best of nine candidates wins
    something even when the candidates carry no information at all, purely by
    selecting the luckiest resampling noise. Reporting the raw oracle gain as
    "what time is worth" would bake that in.

    These candidates have been through exactly the same two resamplings as the
    real ones and describe exactly the same instant, so whatever the oracle
    extracts from them is its own noise floor. Real headroom is the difference
    between the two.
    """
    out = []
    for i, pred_i in enumerate(predictions):
        variants = []
        for j in range(len(predictions)):
            if j == i:
                continue
            there, _ = warp(pred_i, i, j)
            variants.append(warp(there, j, i))
        out.append(variants)
    return out


def aligned_mean(predictions: list[np.ndarray], warp,
                 weighting: str = "mean") -> list[np.ndarray]:
    """Combine the aligned neighbours. No ground truth, so a model could do it.

    The oracle picks the best candidate PER PIXEL by consulting ground truth.
    Nothing can do that at inference, so the oracle alone is a very loose
    ceiling -- especially under per-frame degradation, where it amounts to
    choosing the luckiest of nine independent noise draws.

    A mean of the aligned neighbours is the other bracket: it uses only what
    a model has, and it is close to the simplest thing a temporal architecture
    could learn. The real question is not the oracle's number but whether the
    achievable end of the bracket already beats the per-frame model.

    THE MEAN IS A WEAK COMBINER, and that matters once alignment is estimated
    rather than known. Measured 10 Sep 2026 on 40 drifted clips, seed
    20260906: exact poses gave +0.603 JOD achievable against a ceiling of
    +1.090; RAFT gave +0.341 against +0.671. The gap between what the mean
    reaches and what the oracle shows is available is the room a learned
    fusion works in, and under `uniform` a neighbour that round-trips at
    1.4 px counts exactly as much as one at 0.1 px.

    `confidence` weights each neighbour by exp(-(drift/sigma)^2 / 2) on its
    forward-backward residual. It uses nothing a model could not compute at
    inference. Under exact poses there is no residual, every weight is 1
    inside the valid mask, and the two modes agree exactly -- which is the
    property that keeps the pose arm a fixed reference across both, pinned
    by tests/test_gate_resume_2026_09_07.py.
    """
    if weighting not in ("mean", "confidence"):
        raise ValueError(f"weighting must be mean or confidence, got {weighting!r}")
    out = []
    for i in range(len(predictions)):
        total = predictions[i].astype(np.float64).copy()
        count = np.ones_like(total)
        for j, pred_j in enumerate(predictions):
            if j == i:
                continue
            if weighting == "confidence":
                warped, weight = warp(pred_j, j, i, weight=True)
                m = weight[..., None].astype(np.float64)
                total += warped * m
                count += m
                continue
            warped, valid = warp(pred_j, j, i)
            m = valid[..., None]
            total += np.where(m, warped, 0.0)
            count += m
        out.append((total / count).astype(np.float32))
    return out


def oracle_combine(predictions: list[np.ndarray],
                   warp,
                   truth: list[np.ndarray],
                   candidates: list[list] | None = None) -> list[np.ndarray]:
    """Best per-pixel choice among aligned neighbours, for every frame.

    For each reference frame, every other frame's prediction is warped into
    its view and the pixel closest to ground truth wins. Pixels a neighbour
    never saw are not candidates -- an oracle that could pick from outside
    the overlap would be measuring the camera move, not the information.
    """
    out = []
    for i, (base, truth_i) in enumerate(zip(predictions, truth)):
        floor = FLOOR_NITS / STORAGE_WHITE_NITS
        log_truth = np.log2(np.maximum(truth_i, floor))
        best = base.copy()
        best_err = np.abs(np.log2(np.maximum(base, floor)) - log_truth)
        pool = candidates[i] if candidates is not None else [
            warp(p, j, i)
            for j, p in enumerate(predictions) if j != i]
        for warped, valid in pool:
            err = np.abs(np.log2(np.maximum(warped, floor)) - log_truth)
            take = valid[..., None] & (err < best_err)
            best = np.where(take, warped, best)
            best_err = np.where(take, err, best_err)
        out.append(best)
    return out


def apply_hard(frame, index: int, clip_seed: int, mode: str):
    """The hard condition, with a stated assumption about how it moves in time.

    THIS IS THE CALIBRATION. `degrade_like_eval` reseeds from the frame index,
    so the 4 Sep run redrew the exposure, the tone curve, the white balance,
    the saturation, the chroma subsampling, the bit depth and the JPEG quality
    independently on every frame of a nine-frame clip. Nothing shot through a
    real camera does that. It is also, precisely, the condition under which
    averaging nine aligned neighbours wins the most it can ever win: nine
    independent draws of the same corruption average down like sqrt(9), and a
    temporal model gets credit for arithmetic rather than for information.

    The three modes bracket the truth rather than pretending to know it:

      per-frame   every parameter redrawn every frame. The 4 Sep condition,
                  kept so the +9.308 JOD result stays reproducible. Upper
                  bound on what time can be worth; not a claim about footage.
      per-clip    one realisation for the whole clip. A graded, encoded shot
                  with a static grain plate. Lower bound.
      realistic   the grade and the codec are properties of the SHOT and are
                  frozen; the sensor noise is not, and is redrawn per frame.
                  This is the one to read.

    A gain that survives `realistic` is a gain a temporal model can expect to
    see on real footage. A gain that only exists under `per-frame` is a
    measurement of our own degradation model.
    """
    import torch
    from training.export_bench_pairs import degrade_like_eval

    if mode == "per-frame":
        return degrade_like_eval(frame, index)
    graded = degrade_like_eval(frame, clip_seed)
    if mode == "per-clip":
        return graded
    if mode != "realistic":
        raise ValueError(f"unknown degradation mode {mode!r}")

    # degrade_sdr draws sigma once from U(0, 3/255) behind a 0.60 coin. Both
    # are shot constants -- a camera's noise LEVEL does not change frame to
    # frame -- so they are drawn from the clip seed and only the REALISATION
    # is redrawn. Sensor noise is the one term in that function that is
    # genuinely independent across frames, and it is the only one a temporal
    # model is entitled to average away.
    shot = random.Random(clip_seed)
    if shot.random() >= 0.60:
        return graded
    sigma = shot.uniform(0.0, 3.0 / 255.0)
    generator = torch.Generator().manual_seed((clip_seed + index) & 0xFFFFFFFF)
    noise = torch.randn(graded.shape, generator=generator, dtype=graded.dtype)
    return (graded + noise * sigma).clamp(0.0, 1.0)


def codec_round_trip(frames: list[np.ndarray], crf: int, fps: float,
                     codec: str = "libx264") -> list[np.ndarray]:
    """Put the clip through a real video encoder and take it back out.

    The three synthetic modes above argue about how a per-FRAME degradation
    should move down a clip. A video codec settles the argument by not being
    one: it has a GOP, it spends bits unevenly across frames, it predicts each
    frame from its neighbours, and it subsamples chroma in 4:2:0 for real
    rather than by blurring at half resolution. Whatever temporal structure
    real compression has, this has it, and it is the only mode here that was
    not designed by someone who already had a hypothesis.

    That matters for a temporal gate specifically. Motion compensation makes a
    codec's error CORRELATED along the clip wherever the camera move is easy to
    predict -- which is exactly the case a temporal model would otherwise be
    credited with fixing.

    8-bit sRGB in, 8-bit sRGB out, which is what the network reads anyway.
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for index, frame in enumerate(frames):
            eight = np.clip(frame * 255.0 + 0.5, 0, 255).astype(np.uint8)
            if not cv2.imwrite(str(work / f"{index:05d}.png"), eight[..., ::-1]):
                raise SystemExit(f"error: could not stage frame {index} for {codec}")
        encoded = work / "clip.mp4"
        encode = ["ffmpeg", "-y", "-loglevel", "error",
                  "-framerate", f"{fps:g}", "-i", str(work / "%05d.png"),
                  "-c:v", codec, "-crf", str(crf), "-pix_fmt", "yuv420p",
                  str(encoded)]
        decode = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(encoded),
                  str(work / "out_%05d.png")]
        for command in (encode, decode):
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise SystemExit(f"error: {command[0]} failed\n{result.stderr}")
        out = []
        for index in range(len(frames)):
            # ffmpeg numbers its output from 1.
            path = work / f"out_{index + 1:05d}.png"
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise SystemExit(f"error: {codec} returned no frame {index}")
            out.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    if len(out) != len(frames):
        raise SystemExit(f"error: {codec} returned {len(out)} of {len(frames)} frames")
    return out


def degrade_clip(frames: list[np.ndarray], clip_seed: int, mode: str,
                 crf: int, fps: float) -> list[np.ndarray]:
    """The hard condition, applied to a whole clip rather than a frame."""
    import torch

    if mode == "codec":
        graded = [apply_hard(torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1),
                             index, clip_seed, "realistic")
                  for index, f in enumerate(frames)]
        graded = [g.permute(1, 2, 0).numpy() for g in graded]
        return codec_round_trip(graded, crf, fps)
    out = []
    for index, f in enumerate(frames):
        chw = torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1)
        out.append(apply_hard(chw, index, clip_seed, mode).permute(1, 2, 0).numpy())
    return out


def pose_aligner(poses: list[dict]):
    """Exact correspondence from the renderer's camera angles. The upper bound."""
    def warp(image, src_index, dst_index, weight: bool = False):
        warped, valid = warp_frame(image, poses[src_index], poses[dst_index])
        if not weight:
            return warped, valid
        # Exact correspondence: there is no residual to down-weight by, so a
        # confidence combiner must reduce to the mean here. Anything else
        # would move the pose arm and break it as a reference.
        return warped, valid.astype(np.float32)
    return warp


def flow_aligner(shown: list[np.ndarray], normalise: bool, tolerance: float,
                 texture_floor: float = 0.0, backend: str = "dis",
                 device: str = "cpu", scale: float = 1.0,
                 cache_limit: int = 96, sigma: float = 0.75):
    """Correspondence ESTIMATED from the frames a deployed model actually has.

    THE POINT OF THIS ARM. Every headline number the gate has produced used
    poses the renderer wrote down, and a plate does not come with those. The
    difference between the two arms is estimator error: unreachable by any
    architecture, and therefore not part of what training could win.

    Flow is estimated from the DEGRADED SDR -- the pixels the network was fed
    -- and never from ground truth or from the HDR prediction, either of which
    would smuggle the oracle back in through the alignment.

    Flows are cached per ordered pair. A nine-frame clip needs 72 of them and
    every one is asked for twice, once as a correspondence and once as its own
    forward-backward check. The cache is BOUNDED: 72 fields at 720p is half a
    gigabyte before the estimator has allocated anything of its own, and on
    6 Sep 2026 the RAFT run on a 7 GB box was OOM-killed with no traceback --
    the process simply stopped. Peak memory should not be a function of how
    many frames a clip happens to have.
    """
    from rudra.flow_warp import (estimate_flow, forward_backward_drift,
                                 forward_backward_valid, texture_energy,
                                 to_matching_gray, warp_with_flow)

    grays = [to_matching_gray(f, normalise=normalise) for f in shown]
    energies = ([texture_energy(g) for g in grays] if texture_floor > 0
                else [None] * len(grays))
    flows: dict[tuple[int, int], np.ndarray] = {}

    def flow_for(dst_index: int, src_index: int) -> np.ndarray:
        key = (dst_index, src_index)
        if key not in flows:
            if len(flows) >= cache_limit:
                flows.pop(next(iter(flows)))          # oldest out, FIFO
            flows[key] = estimate_flow(grays[dst_index], grays[src_index],
                                       backend=backend, device=device,
                                       scale=scale)
        return flows[key]

    def warp(image, src_index, dst_index, weight: bool = False):
        if src_index == dst_index:
            ones = np.ones(image.shape[:2], np.float32)
            return image.astype(np.float32), (ones if weight else ones.astype(bool))
        forward = flow_for(dst_index, src_index)
        backward = flow_for(src_index, dst_index)
        valid = forward_backward_valid(forward, backward, tolerance,
                                       energies[dst_index], texture_floor)
        warped, _ = warp_with_flow(image, forward, valid)
        if not weight:
            return warped, valid
        drift, _ = forward_backward_drift(forward, backward)
        confidence = np.exp(-0.5 * (drift / max(sigma, 1e-6)) ** 2)
        return warped, (confidence * valid).astype(np.float32)
    return warp


def score(frames: list[np.ndarray], truth: list[np.ndarray],
          fps: float, cvvdp: bool, device: str = "cpu") -> dict:
    """PU21-PSNR and clip JOD for one reconstruction against its reference.

    `device` is not cosmetic. `hdr_vdp3_clip_jod` chooses where to run from
    whether the tensors it is HANDED are on the GPU, so building them on the
    CPU here left ColorVideoVDP on the CPU no matter what --device said -- and
    the metric, not the network, is nearly all of this script's cost. The
    50-clip pair on 6 Sep 2026 took 3.5 hours on two CPU cores for that
    reason, and the forward passes were a few minutes of it.
    """
    db = [pu_psnr(f * STORAGE_WHITE_NITS, t * STORAGE_WHITE_NITS)
          for f, t in zip(frames, truth)]
    row = {"pu21_db": float(np.mean(db))}
    if cvvdp:
        import torch
        from rudra.hdrvdp import hdr_vdp3_clip_jod

        def to_clip(xs, where):
            return torch.from_numpy(np.ascontiguousarray(np.stack(xs))) \
                .permute(0, 3, 1, 2).float().to(where)

        where = device
        try:
            jod, backend = hdr_vdp3_clip_jod(
                to_clip(frames, where), to_clip(truth, where),
                frames_per_second=fps, diffuse_white_nits=STORAGE_WHITE_NITS)
        except RuntimeError as exc:                       # noqa: BLE001
            if "out of memory" not in str(exc).lower() or where == "cpu":
                raise
            # A 9-frame 720p clip is ~100 MB per tensor before cvvdp's own
            # pyramids. Falling back keeps a long run alive rather than losing
            # every clip already scored.
            torch.cuda.empty_cache()
            print(f"   cvvdp: out of memory on {where}, this clip on cpu")
            jod, backend = hdr_vdp3_clip_jod(
                to_clip(frames, "cpu"), to_clip(truth, "cpu"),
                frames_per_second=fps, diffuse_white_nits=STORAGE_WHITE_NITS)
        row["cvvdp_jod"], row["cvvdp_backend"] = jod, backend
    return row


def load_clip(clip_dir: Path, storage, truth_ceiling: float) -> tuple[list, list, list]:
    """(sdr, hdr truth, poses) for one clip, in frame order."""
    from pipeline.hdr_io import decode_hdr_u16

    metas = sorted(clip_dir.glob("meta/*.json"), key=lambda p: p.stem)
    sdr, truth, poses = [], [], []
    for m in metas:
        poses.append(json.loads(m.read_text(encoding="utf-8"))["pose"])
        s = cv2.imread(str(clip_dir / "sdr" / f"{m.stem}.png"), cv2.IMREAD_UNCHANGED)
        h = cv2.imread(str(clip_dir / "hdr" / f"{m.stem}.png"), cv2.IMREAD_UNCHANGED)
        sdr.append(cv2.cvtColor(s, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        truth.append(np.minimum(
            decode_hdr_u16(cv2.cvtColor(h, cv2.COLOR_BGR2RGB), storage),
            truth_ceiling).astype(np.float32))
    return sdr, truth, poses


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True, type=Path,
                    help="directory of clip directories, each with sdr/ hdr/ meta/")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--condition", choices=("clean", "hard"), default="clean",
                    help="hard applies the eval's seeded camera/codec "
                         "degradation, per frame, before inference. Run it. "
                         "On clean rendered panoramas the per-frame model "
                         "already scores 9.97 JOD out of 10, so the oracle "
                         "has nothing to find and the gate cannot say "
                         "anything -- see the note at the bottom of this "
                         "file.")
    ap.add_argument("--degradation",
                    choices=("per-frame", "per-clip", "realistic", "codec"),
                    default="codec",
                    help="How the hard condition varies along the clip. "
                         "per-frame redraws every parameter every frame -- "
                         "the 4 Sep default, and the single most favourable "
                         "case temporal averaging could be handed. per-clip "
                         "freezes the whole realisation. realistic freezes "
                         "the grade and the codec, which are properties of a "
                         "shot, and redraws only the sensor noise, which is "
                         "not. codec does that and then puts the clip through "
                         "a real H.264 encode, which is the only mode here "
                         "with a GOP and motion compensation in it -- read "
                         "this one.")
    ap.add_argument("--crf", type=int, default=28,
                    help="H.264 quality for --degradation codec. Lower is "
                         "better; 28 is a plausible delivery encode.")
    ap.add_argument("--alignment", choices=("pose", "flow"), default="pose",
                    help="pose reads the renderer's camera angles and is "
                         "exact -- an upper bound no plate can offer. flow "
                         "estimates the correspondence with DIS optical flow "
                         "from the degraded SDR, which is all a deployed "
                         "model has. The gap between them is estimator error "
                         "and no architecture recovers it.")
    ap.add_argument("--flow-tolerance", type=float, default=1.5,
                    help="pixels of forward-backward disagreement tolerated "
                         "before a correspondence is dropped as guesswork.")
    ap.add_argument("--flow-backend", choices=("dis", "raft"), default="dis",
                    help="dis is the fast classical estimator. raft is the "
                         "learned one and is far better in the low-texture "
                         "regions where dis lost the v02 gain -- 0.35 px "
                         "against 2.61 px on an open-sky clip -- but roughly "
                         "200x slower per pair on a CPU. Pair it with "
                         "--flow-device cuda.")
    ap.add_argument("--combiner", choices=("mean", "confidence"), default="mean",
                    help="how the aligned neighbours are combined into the "
                         "ACHIEVABLE row. mean is the original: every valid "
                         "neighbour counts the same. confidence weights each "
                         "by exp(-(drift/sigma)^2/2) on its forward-backward "
                         "residual, so a neighbour matched to 0.1 px outvotes "
                         "one that scraped in at 1.4. Uses nothing a model "
                         "could not compute at inference, and reduces to mean "
                         "under exact poses. ONE alternative, declared before "
                         "it was run -- not a search until something passes.")
    ap.add_argument("--fb-sigma", type=float, default=0.75,
                    help="pixels. The confidence weighting's falloff; half "
                         "the FB tolerance by default, so a match at the "
                         "tolerance keeps about 14% of a perfect one's vote.")
    ap.add_argument("--flow-scale", type=float, default=1.0,
                    help="estimate the flow at this fraction of the frame and "
                         "rescale it. 1.0 is full resolution. 0.75 keeps most "
                         "of RAFT's accuracy at about a tenth of the memory, "
                         "which is what lets it run without a large GPU.")
    ap.add_argument("--flow-cache", type=int, default=96,
                    help="how many flow fields to keep. A nine-frame clip "
                         "needs 72 and each is ~7 MB at 720p.")
    ap.add_argument("--flow-device", default="cpu",
                    help="where --flow-backend raft runs. cpu is about 20 s "
                         "per pair at 720p and a nine-frame clip needs 72.")
    ap.add_argument("--flow-texture-floor", type=float, default=0.0,
                    help="drop flow matches where the destination's local "
                         "gradient energy is below this. 0 disables it; 8.0 "
                         "separates open sky from built scenes on this "
                         "corpus. Forward-backward consistency cannot catch a "
                         "bad match in a flat region -- any displacement "
                         "round-trips there -- and that is where estimated "
                         "alignment lost the v02 gain. See "
                         "rudra.flow_warp.texture_energy.")
    ap.add_argument("--no-flow-normalise", action="store_true",
                    help="match on raw luma. Dense flow assumes brightness "
                         "constancy and an exposure ramp violates it, so the "
                         "default standardises each frame first; this turns "
                         "that off to measure what it was worth.")
    ap.add_argument("--tile-size", type=int, default=0,
                    help="0 runs the frame untiled, as the benchmark does "
                         "when it fits. Falls back to 512 on OOM.")
    ap.add_argument("--tile-overlap", type=int, default=64)
    ap.add_argument("--raw-forward", action="store_true",
                    help="Score the bare network instead of RUDRA as "
                         "deployed. This is what the 4 Sep run did by "
                         "accident; kept only so the discrepancy can be "
                         "reproduced.")
    ap.add_argument("--no-cvvdp", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="skip clips already present in --out and append to "
                         "it. Rows are written after every clip either way, "
                         "so an interrupted run is never wasted.")
    args = ap.parse_args()

    import torch
    from pipeline.hdr_io import HDRStorage
    from rudra.sdr2hdr import SDR2HDRNet
    from training.export_bench_pairs import REFERENCE_CEILING
    from training.infer_sdr2hdr import predict_image

    truth_ceiling = REFERENCE_CEILING * (NETWORK_PEAK_NITS / STORAGE_WHITE_NITS)

    print(f"   condition  : {args.condition}")
    print("   inference  : "
          + ("bare network forward (NOT the shipped configuration)"
             if args.raw_forward else
             "predict_image, preserve_outside=True, recovery all @ 1.0"))
    print(f"   reference  : clamped at {truth_ceiling:,.0f} x 203 nits")
    print(f"   alignment  : {args.alignment}"
          + ("" if args.alignment == "pose" else
             f", {args.flow_backend.upper()}"
             + (f" on {args.flow_device}" if args.flow_backend == "raft" else "")
             + (f" @ {args.flow_scale:g}x" if args.flow_scale != 1.0 else "")
             + f", fb tolerance {args.flow_tolerance} px"
             + ("" if args.no_flow_normalise else ", exposure-normalised")
             + ("" if args.flow_texture_floor <= 0 else
                f", texture floor {args.flow_texture_floor:g}")))
    print(f"   combiner   : {args.combiner}"
          + (f", sigma {args.fb_sigma:g} px"
             if args.combiner == "confidence" and args.alignment == "flow" else ""))
    if args.condition == "hard":
        print(f"   degradation: {args.degradation}"
              + (f", H.264 crf {args.crf}" if args.degradation == "codec" else ""))
    clips = sorted(p for p in args.clips.iterdir() if (p / "meta").is_dir())
    if not clips:
        raise SystemExit(f"no clip directories under {args.clips}")

    cfg = json.loads((args.clips / "_ingest_config.json").read_text(encoding="utf-8"))
    storage = HDRStorage(**{k: v for k, v in cfg["storage"].items() if k != "version"})

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SDR2HDRNet.from_config(payload.get("config", {}) or {})
    model.load_state_dict(payload.get("model", payload), strict=True)
    model.eval().to(args.device)

    # Resumable, because these runs are long and the thing that ends them is
    # rarely the code. A 50-clip CPU pass is hours; on 6-7 Sep 2026 one was
    # OOM-killed and another lost to the sandbox being reclaimed, and both
    # times every completed clip went with it. Rows are written after each
    # clip, and --resume skips the ones already in the file.
    rows: list[dict] = []
    done: set[str] = set()
    if args.resume and args.out and args.out.is_file():
        try:
            rows = json.loads(args.out.read_text(encoding="utf-8"))
            done = {row["clip"] for row in rows}
        except (json.JSONDecodeError, KeyError, TypeError):
            # A half-written file from a process that died mid-dump. Starting
            # over is correct; silently scoring a truncated set is not.
            print(f"   resume: {args.out} is unreadable, starting fresh")
            rows, done = [], set()
        else:
            print(f"   resume: {len(done)} clip(s) already scored in {args.out}")

    def flush() -> None:
        if args.out:
            args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    for clip in clips:
        if clip.name in done:
            continue
        sdr, truth, poses = load_clip(clip, storage, truth_ceiling)
        clip_seed = zlib.crc32(clip.name.encode("utf-8"))

        # One frame at a time. A nine-frame batch of 1280x720 is ~100 M
        # activations per layer and gets the process OOM-killed on CPU, which
        # is where this runs when the GPU is busy training.
        # Degradation is a CLIP operation now, not a frame one -- a codec
        # cannot be applied one frame at a time and still be a codec.
        shown = sdr if args.condition == "clean" else degrade_clip(
            sdr, clip_seed, args.degradation, args.crf, args.fps)
        per_frame = []
        with torch.no_grad():
            for frame in shown:
                x = torch.from_numpy(np.ascontiguousarray(frame)) \
                    .permute(2, 0, 1)[None].to(args.device)
                # RUDRA AS DEPLOYED, not the bare network. The benchmark
                # that produced the 7.805 JOD hard figure runs predict_image
                # with preserve_outside=True: outside the learned
                # highlight/shadow masks the output IS the analytic baseline,
                # untouched. A plain model(x) call has preserve_outside=False
                # and so replaces the baseline everywhere, including the
                # well-exposed midtones the baseline already inverts almost
                # exactly. That -- not resolution, not the crop size -- is why
                # the 4 Sep run scored -1.962 JOD where the benchmark scored
                # 7.805 on the same checkpoint and the same 1280x720 frames.
                # An oracle measured on top of a configuration nobody ships
                # answers a question nobody asked.
                if args.raw_forward:
                    out = model(x).hdr
                else:
                    try:
                        out = predict_image(model, x, preserve_outside=True,
                                            tile_size=args.tile_size,
                                            overlap=args.tile_overlap,
                                            recovery_mode="all",
                                            recovery_strength=1.0)
                    except Exception as exc:            # noqa: BLE001
                        if "out of memory" not in str(exc).lower():
                            raise
                        out = predict_image(model, x, preserve_outside=True,
                                            tile_size=512,
                                            overlap=args.tile_overlap,
                                            recovery_mode="all",
                                            recovery_strength=1.0)
                hdr = out[0].permute(1, 2, 0).float().cpu().numpy()
                # Into the storage convention everything else here uses.
                per_frame.append(hdr * (NETWORK_PEAK_NITS / STORAGE_WHITE_NITS))
        warp = (pose_aligner(poses) if args.alignment == "pose"
                else flow_aligner(shown, not args.no_flow_normalise,
                                  args.flow_tolerance,
                                  args.flow_texture_floor,
                                  args.flow_backend, args.flow_device,
                                  args.flow_scale, args.flow_cache,
                                  args.fb_sigma))
        oracle = oracle_combine(per_frame, warp, truth)
        control = oracle_combine(per_frame, warp, truth,
                                 candidates=round_trip(per_frame, warp))

        mean_of = aligned_mean(per_frame, warp, args.combiner)

        base = score(per_frame, truth, args.fps, not args.no_cvvdp, args.device)
        best = score(oracle, truth, args.fps, not args.no_cvvdp, args.device)
        ctrl = score(control, truth, args.fps, not args.no_cvvdp, args.device)
        avg = score(mean_of, truth, args.fps, not args.no_cvvdp, args.device)
        rows.append({"clip": clip.name, "per_frame": base, "oracle": best,
                     "control": ctrl, "aligned_mean": avg})
        print(f"   {clip.name[:34]:34s} {base['pu21_db']:6.2f} | "
              f"ctrl {ctrl['pu21_db']:6.2f} | mean {avg['pu21_db']:6.2f} | "
              f"oracle {best['pu21_db']:6.2f} dB", flush=True)
        flush()

    def mean(kind, key):
        return float(np.mean([r[kind][key] for r in rows if key in r[kind]]))

    print("\n" + "=" * 72)
    print(f"   {len(rows)} clip(s)")
    for key, unit, fmt in (("pu21_db", "dB", "6.2f"), ("cvvdp_jod", "JOD", "6.3f")):
        if not any(key in r["per_frame"] for r in rows):
            continue
        pf, ct, am, orc = (mean(k, key) for k in
                           ("per_frame", "control", "aligned_mean", "oracle"))
        name = "PU21-PSNR" if key == "pu21_db" else "CVVDP"
        print(f"   {name:11s} per-frame {pf:{fmt}}   control {ct:{fmt}}   "
              f"aligned mean {am:{fmt}}   oracle {orc:{fmt}}  {unit}")
        print(f"   {'':11s} selection-on-noise {ct - pf:+.3f}    "
              f"ACHIEVABLE {am - pf:+.3f}    CEILING {orc - ct:+.3f} {unit}")
    if not args.no_cvvdp and any("cvvdp_jod" in r["per_frame"] for r in rows):
        print()
        print("   Read ACHIEVABLE, not CEILING. The oracle consults ground")
        print("   truth per pixel, which nothing can do at inference; the")
        print("   aligned mean uses only what a model has. Below about +0.5")
        print("   JOD achievable, v02 should pivot rather than train.")
        print()
        print("   The gate: below about +0.5 JOD, perfect alignment and perfect")
        print("   selection are not worth an architecture. Above it, they are.")
    print("=" * 72)

    if args.out:
        flush()
        print(f"   wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# MEASURED, 4-5 September 2026, checkpoint sdr2hdr_shadow_v1, PU21 dB / JOD.
#
# 4 Sep, --condition clean, 2 clips
#   per-frame 52.58 / 9.972    oracle 55.12 / 9.988    headroom +1.55 / +0.007
#
# The clean run cannot answer the question it was built for. 9.972 out of 10
# means the reconstruction is already perceptually indistinguishable from
# ground truth, so nothing can be won -- which says nothing about whether time
# carries information. The gate has to be read on --condition hard.
#
# 5 Sep, --condition hard, through predict_image as RUDRA ships
#
#   degradation   clips   per-frame      aligned mean   ACHIEVABLE   ceiling
#   per-frame        2    30.07 / -2.396   35.99 / 7.001   +9.397 JOD  +9.408
#   per-clip         2    31.22 /  7.993   31.20 / 7.982   -0.011 JOD  +0.003
#   realistic        2    31.22 /  7.993   31.20 / 7.982   -0.011 JOD  +0.003
#   codec            2    30.55 /  7.951   30.62 / 7.940   -0.011 JOD  +0.073
#   realistic       40    27.27 /  6.961   27.56 / 6.994   +0.033 JOD  +0.191
#   codec           40    27.15 /  6.864   27.27 / 6.897   +0.034 JOD  +0.165
#
# THE GATE FAILS. Against a +0.5 JOD threshold the achievable gain is +0.033
# and +0.034 on the two coherent conditions at 40 clips, and even the
# unreachable bound -- omniscient per-pixel selection over exactly aligned
# neighbours -- reaches only +0.191.
#
# The +9.397 in the first row is the 4 Sep result and it was an artefact of
# our own degradation. `degrade_like_eval` reseeds from the frame index, so
# that row redraws the exposure, the tone curve, the white balance, the
# saturation, the chroma subsampling, the bit depth and the JPEG quality
# independently on every frame. Averaging nine independent draws of a
# corruption is worth sqrt(9) whether or not the frames carry any information,
# so the number measured the seeding, not the video. Two things had to be
# fixed together: that, and scoring the bare network instead of RUDRA as
# deployed (--raw-forward), which is why the floor sat at -2.396 JOD against
# the image benchmark's 7.805. With both corrected the per-frame floor is
# 6.86-7.99 JOD, on the same footing as the benchmark at last.
#
# WHY IT FAILS IS NOT A FACT ABOUT VIDEO. `training/prepare_training_data.py`
# tone-maps every frame with one fixed curve and one fixed EV offset, and the
# renderer's camera only rotates through a static panorama. A scene point
# therefore carries THE SAME SDR CODE in every frame it appears in: what is
# blown in frame 3 is blown in frame 7. The corpus contains none of the
# information v02's claim is about -- that a neighbour can show what this
# frame clipped -- so a faithful gate had to come back empty. The measurement
# is sound; the corpus cannot test the hypothesis.
#
# `pipeline/render_hdri_moves.py --exposure-drift` puts that axis back (0.12
# stops per frame is about one stop end to end over nine frames, roughly what
# auto-exposure does panning into the sun).
#
# 5 Sep, THE PAIRED PILOT. 12 panoramas rendered twice with PYTHONHASHSEED=0:
# identical camera paths, identical codec degradation, the SDR exposure the
# only difference. Poses checked equal to 1e-9 before scoring.
#
#   exposure          per-frame      aligned mean   ACHIEVABLE   ceiling
#   constant       26.91 / 5.037   27.06 / 5.378   +0.340 JOD   +0.557
#                                                  +0.149 dB    +0.703 dB
#   +/-0.48 stops  26.20 / 4.910   26.87 / 5.448   +0.538 JOD   +1.129
#                                                  +0.665 dB    +5.563 dB
#
# Moving the exposure and changing NOTHING else takes the achievable gain from
# +0.340 to +0.538 JOD -- across the +0.5 threshold -- and the oracle ceiling
# from +0.703 to +5.563 dB, an eight-fold jump in what perfect alignment could
# in principle fetch. That is the v02 hypothesis behaving exactly as stated:
# a neighbour is worth something when it saw the scene at a different
# exposure, and worth nothing when it did not.
#
# 5-6 Sep, THE SAME PAIRING AT 50 CLIPS. 12 pilot scenes plus 38 sampled
# deterministically (_gate50_scenes.txt), same protocol.
#
#   exposure          per-frame      aligned mean   ACHIEVABLE   ceiling
#   constant       26.17 / 6.209   26.29 / 6.319   +0.110 JOD   +0.264
#                                                  +0.120 dB    +0.632 dB
#   +/-0.48 stops  25.83 / 5.786   26.26 / 6.296   +0.511 JOD   +0.851
#                                                  +0.435 dB    +4.554 dB
#
# THE GATE PASSES, and only under drift: +0.511 JOD against the +0.5
# threshold, with the fixed-exposure control at +0.110 on the very same
# scenes and camera paths. The PU21 ceiling is the clearer signal -- +0.632 dB
# without exposure movement, +4.554 dB with it, seven times more for perfect
# alignment to fetch.
#
# The pilot's flat number (+0.340 on 12 clips) was small-sample noise; at 50
# it is +0.110, in line with the 40-clip runs. Read the 12-clip rows above as
# a direction only.
#
# 6 Sep, THREE SEEDS (training/run_drift_gate.ps1, on the 4080).
#
#   seed        flat achievable   drift achievable   drift ceiling   clips
#   20260903        +0.110            +0.511            +0.851         50
#   20260906        +0.035            +0.603            +1.090         40
#   20260907        +0.046            +0.661            +1.142         40
#   ------------------------------------------------------------------
#   mean            +0.064            +0.592            +1.028
#
# THE RESULT HOLDS. Every drift arm clears +0.5; every flat arm is an order
# of magnitude below it, on the same panoramas and the same camera paths.
# The separation -- about +0.53 JOD between arms -- is far larger than the
# spread across seeds, which is what makes it a result rather than a run.
#
# The 40 in the later two rows is a defect worth recording: stage_oracle_clips
# defaulted --count to 40 and the sweep did not pass one, so both arms of
# those seeds took the first 40 of the 50 scenes the list names. The SAME 40
# in both arms, so each pairing is intact and the comparison stands; the
# default is now 0 (all).
#
# What that licenses is narrow: v02's claim holds where the exposure moved
# between neighbours and fails where it did not, so it is a proposition about
# footage whose exposure breathes rather than about video in general.
#
# 6 Sep, --alignment flow. THE ONE THAT ENDS IT. Everything above used the
# camera angles the renderer wrote down. A plate has none, so the same 50
# drifted clips were re-scored with the correspondence ESTIMATED by DIS
# optical flow from the degraded SDR -- what a deployed model actually holds.
#
#   arm            per-frame  control  aligned   oracle | ACHIEVABLE  ceiling
#   pose               5.786    5.835    6.296    6.687 |    +0.511   +0.851
#   flow               5.786    5.844    5.802    5.839 |    +0.016   -0.005
#   flow + texture     5.786    5.828    5.658    5.514 |    -0.128   -0.314
#
# READ THE CEILING COLUMN. Under estimated alignment the ORACLE -- exact
# ground truth consulted per pixel, the best any weighting or architecture
# could ever do and better, since nothing at inference can consult it -- beats
# the per-frame model by +0.053 JOD while the control, which carries no
# information at all, beats it by +0.058. The oracle's entire margin is
# selection on resampling noise. There is no headroom left to build for.
#
# The alignment is not obviously bad, which is what makes this worth stating
# carefully. Against the analytic poses, DIS matched to a median endpoint
# error of 0.04 px at one frame of separation and 0.09 px at eight, with
# coverage within 1% of the poses' and warped frames agreeing to under half an
# 8-bit code value. PU21-PSNR accordingly keeps 75% of the gain (+0.326 of
# +0.435 dB). The JOD keeps 3%. What survives per-frame PSNR and does not
# survive a video metric is a small, spatially coherent error that CHANGES
# EVERY FRAME -- the definition of the artefact a temporal model exists to
# remove. §5 of the v01 paper is the same lesson: read the JOD.
#
# Where it fails is legible. The six worst clips are all open sky
# (drackenstein_quarry_puresky, ostrich_road, fish_hoek_beach); the four best
# are interiors (billiard_hall, burnt_warehouse). On the sky scene, flow error
# in low-texture regions is 1.35 px against 0.15 px in textured regions of the
# same frame -- and forward-backward consistency passed 57% of those bad
# pixels, because in a flat region ANY displacement round-trips perfectly.
# Flow fails exactly where the highlights are.
#
# The obvious repair does not work either. Gating on local texture energy
# (--flow-texture-floor 8.0) made it worse, -0.128 JOD: a hard mask leaves
# nine-frame averaging beside one-frame averaging with a seam between them,
# and a video metric dislikes that more than it disliked the misalignment. A
# softer weighting is not worth trying, because the oracle row already bounds
# every weighting there is.
#
# 10 Sep, THE LEARNED ESTIMATOR, and the close. DIS is a fast classical
# estimator, so the flow arm was re-run with RAFT-large, which is markedly
# better in exactly the low-texture regions where DIS failed -- 0.35 px
# against 2.61 px on an open-sky clip. Seed 20260906, 40 drifted clips, the
# same clips for every row:
#
#   arm            per-frame  control  aligned   oracle | ACHIEVABLE  ceiling
#   pose (exact)       5.925    5.973    6.528    7.063 |    +0.603   +1.090
#   flow (DIS)         5.925    5.993    5.856    6.115 |    -0.069   +0.122
#   flow (RAFT)        5.925    5.988    6.266    6.659 |    +0.341   +0.671
#
# RAFT recovers 57% of what exact poses give and MISSES THE THRESHOLD. +0.341
# against the +0.5 this file was set at before any of it was measured. Not the
# flat zero DIS gave -- the ceiling moved from +0.122 to +0.671, so under RAFT
# there was room a better combiner might have reached.
#
# One combiner was declared and tried, before it was run: weight each
# neighbour by exp(-(drift/sigma)^2/2) on its forward-backward residual rather
# than averaging equally (--combiner confidence). On the same clips:
#
#   RAFT, mean         ACHIEVABLE +0.351      RAFT, confidence  +0.350
#
# Nothing. And the reason is the same wall from a third angle: RAFT's
# forward-backward drift is 0.04-0.09 px on nearly every pixel that passes, so
# the weight is ~1 everywhere and the signal has no dynamic range. The
# failures are not low-confidence matches. They are CONFIDENT WRONG ones, in
# flat regions where any displacement round-trips perfectly -- which is what
# forward-backward consistency cannot see, and therefore what weighting by it
# cannot fix.
#
# LINE D IS CLOSED. Three alignment arms and two combiners, against a
# threshold fixed in advance: exact poses clear it, nothing a plate can supply
# does. The +0.603 belongs to the renderer's camera angles. No temporal model
# was trained because there is nothing measurable for one to learn, and that
# is the result rather than the absence of one.
#
# What would reopen it: a corpus whose neighbouring frames carry information
# these do not -- real parallax, moving subjects, genuine multi-exposure
# capture -- not a better estimator and not a better architecture. The gate
# would have to be re-run from scratch on it; none of the numbers above
# transfer.
