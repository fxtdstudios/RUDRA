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
import sys
from pathlib import Path

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


def round_trip(predictions: list[np.ndarray], poses: list[dict]) -> list[list]:
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
            there, _ = warp_frame(pred_i, poses[i], poses[j])
            variants.append(warp_frame(there, poses[j], poses[i]))
        out.append(variants)
    return out


def aligned_mean(predictions: list[np.ndarray], poses: list[dict]) -> list[np.ndarray]:
    """Average the aligned neighbours. No ground truth, so a model could do it.

    The oracle picks the best candidate PER PIXEL by consulting ground truth.
    Nothing can do that at inference, so the oracle alone is a very loose
    ceiling -- especially under per-frame degradation, where it amounts to
    choosing the luckiest of nine independent noise draws.

    A mean of the aligned neighbours is the other bracket: it uses only what
    a model has, and it is close to the simplest thing a temporal architecture
    could learn. The real question is not the oracle's number but whether the
    achievable end of the bracket already beats the per-frame model.
    """
    out = []
    for i in range(len(predictions)):
        total = predictions[i].astype(np.float64).copy()
        count = np.ones_like(total)
        for j, pred_j in enumerate(predictions):
            if j == i:
                continue
            warped, valid = warp_frame(pred_j, poses[j], poses[i])
            m = valid[..., None]
            total += np.where(m, warped, 0.0)
            count += m
        out.append((total / count).astype(np.float32))
    return out


def oracle_combine(predictions: list[np.ndarray],
                   poses: list[dict],
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
            warp_frame(p, poses[j], poses[i])
            for j, p in enumerate(predictions) if j != i]
        for warped, valid in pool:
            err = np.abs(np.log2(np.maximum(warped, floor)) - log_truth)
            take = valid[..., None] & (err < best_err)
            best = np.where(take, warped, best)
            best_err = np.where(take, err, best_err)
        out.append(best)
    return out


def score(frames: list[np.ndarray], truth: list[np.ndarray],
          fps: float, cvvdp: bool) -> dict:
    db = [pu_psnr(f * STORAGE_WHITE_NITS, t * STORAGE_WHITE_NITS)
          for f, t in zip(frames, truth)]
    row = {"pu21_db": float(np.mean(db))}
    if cvvdp:
        import torch
        from rudra.hdrvdp import hdr_vdp3_clip_jod
        to_clip = lambda xs: torch.from_numpy(          # noqa: E731
            np.ascontiguousarray(np.stack(xs))).permute(0, 3, 1, 2).float()
        jod, backend = hdr_vdp3_clip_jod(to_clip(frames), to_clip(truth),
                                         frames_per_second=fps,
                                         diffuse_white_nits=STORAGE_WHITE_NITS)
        row["cvvdp_jod"], row["cvvdp_backend"] = jod, backend
    return row


def load_clip(clip_dir: Path, storage) -> tuple[list, list, list]:
    """(sdr, hdr truth, poses) for one clip, in frame order."""
    import cv2
    from pipeline.hdr_io import decode_hdr_u16

    metas = sorted(clip_dir.glob("meta/*.json"), key=lambda p: p.stem)
    sdr, truth, poses = [], [], []
    for m in metas:
        poses.append(json.loads(m.read_text(encoding="utf-8"))["pose"])
        s = cv2.imread(str(clip_dir / "sdr" / f"{m.stem}.png"), cv2.IMREAD_UNCHANGED)
        h = cv2.imread(str(clip_dir / "hdr" / f"{m.stem}.png"), cv2.IMREAD_UNCHANGED)
        sdr.append(cv2.cvtColor(s, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        truth.append(decode_hdr_u16(cv2.cvtColor(h, cv2.COLOR_BGR2RGB),
                                    storage).astype(np.float32))
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
    ap.add_argument("--no-cvvdp", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import torch
    from pipeline.hdr_io import HDRStorage
    from rudra.sdr2hdr import SDR2HDRNet
    from training.export_bench_pairs import degrade_like_eval

    print(f"   condition  : {args.condition}")
    clips = sorted(p for p in args.clips.iterdir() if (p / "meta").is_dir())
    if not clips:
        raise SystemExit(f"no clip directories under {args.clips}")

    cfg = json.loads((args.clips / "_ingest_config.json").read_text(encoding="utf-8"))
    storage = HDRStorage(**{k: v for k, v in cfg["storage"].items() if k != "version"})

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SDR2HDRNet.from_config(payload.get("config", {}) or {})
    model.load_state_dict(payload.get("model", payload), strict=True)
    model.eval().to(args.device)

    rows = []
    for clip in clips:
        sdr, truth, poses = load_clip(clip, storage)
        # One frame at a time. A nine-frame batch of 1280x720 is ~100 M
        # activations per layer and gets the process OOM-killed on CPU, which
        # is where this runs when the GPU is busy training.
        per_frame = []
        with torch.no_grad():
            for index, frame in enumerate(sdr):
                x = torch.from_numpy(frame).permute(2, 0, 1)[None].to(args.device)
                if args.condition == "hard":
                    # Seeded per frame, so neighbours carry genuinely
                    # different noise -- which is the whole reason a temporal
                    # model could help here, and why the clean run could not
                    # show it.
                    # degrade_sdr takes CHW, not NCHW -- export_bench_pairs
                    # calls it as degrade_like_eval(sdr[0], i)[None].
                    x = degrade_like_eval(x[0].cpu(), index)[None].to(args.device)
                hdr = model(x).hdr[0].permute(1, 2, 0).cpu().numpy()
                # Into the storage convention everything else here uses.
                per_frame.append(hdr * (NETWORK_PEAK_NITS / STORAGE_WHITE_NITS))
        oracle = oracle_combine(per_frame, poses, truth)
        control = oracle_combine(per_frame, poses, truth,
                                 candidates=round_trip(per_frame, poses))

        mean_of = aligned_mean(per_frame, poses)

        base = score(per_frame, truth, args.fps, not args.no_cvvdp)
        best = score(oracle, truth, args.fps, not args.no_cvvdp)
        ctrl = score(control, truth, args.fps, not args.no_cvvdp)
        avg = score(mean_of, truth, args.fps, not args.no_cvvdp)
        rows.append({"clip": clip.name, "per_frame": base, "oracle": best,
                     "control": ctrl, "aligned_mean": avg})
        print(f"   {clip.name[:34]:34s} {base['pu21_db']:6.2f} | "
              f"ctrl {ctrl['pu21_db']:6.2f} | mean {avg['pu21_db']:6.2f} | "
              f"oracle {best['pu21_db']:6.2f} dB")

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
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"   wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# MEASURED, 4 September 2026, 2 clips of 9 frames, checkpoint sdr2hdr_shadow_v1
#
#   clean    per-frame 52.58 dB / 9.972 JOD
#            control   53.57 dB / 9.980       selection-on-noise +0.99 dB
#            oracle    55.12 dB / 9.988       real headroom +1.55 dB, +0.007 JOD
#
# The clean run cannot answer the question it was built for. A per-frame score
# of 9.972 out of 10 means the reconstruction is already perceptually
# indistinguishable from ground truth on rendered panoramas, so there is no
# room for a temporal model to win any -- which says nothing about whether
# time carries information, only that this condition has nothing to give.
# v01's whole result is that RUDRA earns its keep on DEGRADED input and is
# near-neutral on clean, so the gate has to be read on --condition hard.
