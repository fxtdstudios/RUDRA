"""How much of the corpus actually clips -- and on those pixels, who is right?

The claim "RUDRA reconstructs the range" rests entirely on pixels the SDR
clipped. Every score in RESULTS.md is a whole-frame number, and a whole frame
is mostly pixels that were never clipped, so those numbers cannot answer it.

Two passes, cheapest first:

  1. CLIP CENSUS. What fraction of each SDR frame is clipped. On 19 frames
     sampled 10 Sep 2026 the median was 0.000% -- ten of nineteen had no
     clipped pixels at all -- because prepare_training_data.py exposes the SDR
     down a stop before the ACES curve. A model cannot learn to reverse a
     phenomenon its training data does not contain, and a benchmark cannot
     measure it either.

  2. CLIPPED-PIXEL SCORING (--score). On the frames that DO clip, error in
     stops against the ground-truth HDR, for three contenders: leaving them at
     white, the analytic inverse tone map, and RUDRA. On the one frame with
     enough clipped pixels to judge, the last two were identical to the digit
     -- everything gained was the curve, nothing was the network. That is one
     frame; this settles it across the split.
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

DIFFUSE_WHITE_NITS = 203.0
NETWORK_PEAK_NITS = 10_000.0
LUMA_REC2020 = np.array([0.2627, 0.6780, 0.0593])
CLIPPED_CODE = 254 / 255


def clip_fraction(sdr: np.ndarray) -> tuple[float, float]:
    top = sdr.max(axis=-1)
    return float((top >= CLIPPED_CODE).mean()), float((top >= 250 / 255).mean())


def census(sdr_root: Path, limit: int) -> list[dict]:
    from PIL import Image
    rows = []
    paths = sorted(p for p in sdr_root.rglob("*.png"))
    if limit:
        paths = paths[:limit]
    for path in paths:
        sdr = np.asarray(Image.open(path).convert("RGB"), np.float64) / 255.0
        clipped, near = clip_fraction(sdr)
        rows.append({"clip": str(path.parent.name), "frame": path.stem,
                     "clipped_pct": 100 * clipped, "near_clip_pct": 100 * near,
                     "path": str(path)})
    return rows


def score_clipped(rows: list[dict], ref_root: Path, checkpoint: Path,
                  device: str, min_clip_pct: float, max_frames: int) -> list[dict]:
    """Error in stops ON THE CLIPPED PIXELS ONLY, against the reference."""
    import torch
    from PIL import Image
    sys.path.insert(0, str(REPO / "ui"))
    import server as S
    from rudra.delivery.exr import read_exr
    from rudra.sdr2hdr import sdr_to_baseline_hdr
    from training.infer_sdr2hdr import predict_image

    model, _ = S.load_model(checkpoint, device)
    candidates = [r for r in rows if r["clipped_pct"] >= min_clip_pct][:max_frames or None]
    print(f"\n   {len(candidates)} frame(s) clip at least {min_clip_pct}%")
    scored = []
    for row in candidates:
        ref_path = ref_root / row["clip"] / f"{row['frame']}.exr"
        if not ref_path.is_file():
            print(f"     no reference for {row['frame']}")
            continue
        sdr = np.asarray(Image.open(row["path"]).convert("RGB"), np.float64) / 255.0
        ref = np.asarray(read_exr(ref_path)[0], np.float64)[..., :3] * DIFFUSE_WHITE_NITS
        if ref.shape != sdr.shape:
            print(f"     shape mismatch on {row['frame']}")
            continue
        x = torch.from_numpy(sdr.astype(np.float32)).permute(2, 0, 1)[None]
        with torch.no_grad():
            hdr = predict_image(model, x, preserve_outside=True, tile_size=0, overlap=64,
                                recovery_mode="all", recovery_strength=1.0)
            # The baseline contender must invert the exposure the corpus
            # actually applied, not the legacy -1 EV default: on a 0 EV corpus
            # the default is one stop off and the "RUDRA minus baseline"
            # comparison -- the whole point of this tool -- is corrupted.
            base = sdr_to_baseline_hdr(x, model.corpus_ev)
        preds = {
            "clamp": np.full_like(ref, DIFFUSE_WHITE_NITS),
            "baseline": base[0].numpy().transpose(1, 2, 0).astype(np.float64) * NETWORK_PEAK_NITS,
            "rudra": hdr[0].numpy().transpose(1, 2, 0).astype(np.float64) * NETWORK_PEAK_NITS,
        }
        mask = sdr.max(axis=-1) >= CLIPPED_CODE
        truth = (ref @ LUMA_REC2020)[mask]
        entry = {"frame": row["frame"], "clipped_pct": row["clipped_pct"],
                 "clipped_px": int(mask.sum())}
        for name, pred in preds.items():
            err = np.log2(np.maximum((pred @ LUMA_REC2020)[mask], 1e-6) / np.maximum(truth, 1e-6))
            entry[name] = {"median_stops": float(np.median(err)),
                           "abs_median_stops": float(np.median(np.abs(err))),
                           "within_1_stop": float(np.mean(np.abs(err) < 1.0))}
        scored.append(entry)
        print(f"     {row['frame'][:40]:<40} {row['clipped_pct']:>6.2f}%  "
              f"clamp {entry['clamp']['median_stops']:+6.2f}  "
              f"base {entry['baseline']['median_stops']:+6.2f}  "
              f"RUDRA {entry['rudra']['median_stops']:+6.2f} stops")
    return scored


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", type=Path, required=True,
                    help="the bench root holding clean/ and hard/")
    ap.add_argument("--conditions", nargs="+", default=["clean", "hard"])
    ap.add_argument("--score", action="store_true", help="also score the clipped pixels")
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO / "checkpoints" / "sdr2hdr_shadow_v1.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-clip-pct", type=float, default=0.1)
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    report = {}
    for cond in args.conditions:
        sdr_root = args.bench / cond / "sdr"
        if not sdr_root.is_dir():
            print(f"   no {sdr_root}, skipping")
            continue
        rows = census(sdr_root, args.limit)
        pct = np.array([r["clipped_pct"] for r in rows])
        zero = float((pct == 0.0).mean())
        print(f"\n{'=' * 70}\n  {cond.upper()}  --  {len(rows)} frames\n{'=' * 70}")
        print(f"    clipped %   median {np.median(pct):6.3f}   mean {pct.mean():6.3f}   "
              f"p90 {np.percentile(pct, 90):6.3f}   max {pct.max():6.3f}")
        print(f"    frames with NO clipped pixels at all: {100 * zero:.1f}%")
        for edge in (0.01, 0.1, 1.0, 5.0):
            print(f"    frames clipping more than {edge:>5.2f}% : "
                  f"{100 * float((pct > edge).mean()):5.1f}%")
        entry = {"frames": len(rows), "median_pct": float(np.median(pct)),
                 "mean_pct": float(pct.mean()), "max_pct": float(pct.max()),
                 "zero_clip_fraction": zero}
        if args.score:
            entry["clipped_pixel_scores"] = score_clipped(
                rows, args.bench / cond / "ref", args.checkpoint, args.device,
                args.min_clip_pct, args.max_frames)
            got = entry["clipped_pixel_scores"]
            if got:
                for name in ("clamp", "baseline", "rudra"):
                    med = np.median([g[name]["abs_median_stops"] for g in got])
                    win = np.mean([g[name]["within_1_stop"] for g in got])
                    print(f"    {name:<9} |error| {med:5.2f} stops"
                          f"   within 1 stop {100 * win:5.1f}%")
                d = np.array([g["rudra"]["abs_median_stops"] - g["baseline"]["abs_median_stops"]
                              for g in got])
                print(f"\n    RUDRA minus baseline on clipped pixels: {d.mean():+.3f} stops "
                      f"(better on {100 * float((d < 0).mean()):.0f}% of frames)")
                print("    If that is ~0, the recovery is the analytic curve, not the network.")
        report[cond] = entry

    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n   wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
