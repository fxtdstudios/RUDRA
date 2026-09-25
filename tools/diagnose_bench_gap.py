"""Why does a model that beats the analytic inverse in training lose to it on the bench?

The fp32 check (25 Sep 2026) ruled out precision: v4b is -8.1 dB PU21 against
the inverse on its own ACES render in fp32 as in bf16, while its training evals
read +1.4 dB. What is left differs in two ways, and this separates them on the
bench's own trees, with no inference:

  METRIC. Training scores PSNR of log1p(16 x), x in 10,000-nit units, peak
  log1p(16). Almost all of that range is above ~600 nits: a 10% error at 20 nits
  moves log1p(16 x) by 0.003, at 2,000 nits by 0.07. PU21 weights luminance
  perceptually, so shadows and mid-tones count. The training metric is computed
  here on the bench frames: if it still says the model wins, the model is
  optimising something the bench does not reward.

  WHERE. The PU21 error of the model and of the inverse is split by the
  reference's luminance (per channel, as the bench measures it), with the signed
  level bias (mean log2 of test/ref) in each band. A band where the model's
  error share jumps and its bias moves away from zero is where it goes wrong.

    python tools/diagnose_bench_gap.py bench\\cp_aces --test v4b
    python tools/diagnose_bench_gap.py bench\\cp_aces --test v4b --stride 4 --json reports\\logs\\gap_v4b.json

Reads <root>/ref, <root>/baseline and <root>/<test> (scene-linear EXR, 203 nits
per unit), the layout training/export_bench_pairs.py writes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rudra.delivery.bench import _PU21_MAX, load_frame, pu21_encode  # noqa: E402

BANDS = [(0.0, 1.0), (1.0, 10.0), (10.0, 100.0), (100.0, 1000.0), (1000.0, float("inf"))]
LOG_PEAK = float(np.log1p(16.0))
MAX_HDR = 4.0                                # the network's ceiling, network units


def training_psnr(test_nits: np.ndarray, ref_nits: np.ndarray, clamp: bool) -> float:
    """evaluate_image's psnr_log: PSNR of log1p(16 x), x = nits / 10,000."""
    t, r = test_nits / 10000.0, ref_nits / 10000.0
    if clamp:
        t, r = np.minimum(t, MAX_HDR), np.minimum(r, MAX_HDR)
    mse = max(float(np.mean((np.log1p(16.0 * t) - np.log1p(16.0 * r)) ** 2)), 1e-12)
    return 20.0 * np.log10(LOG_PEAK) - 10.0 * np.log10(mse)


def frame_stats(test: np.ndarray, ref: np.ndarray) -> dict:
    pu_t, pu_r = pu21_encode(test), pu21_encode(ref)
    sq = (pu_t - pu_r) ** 2
    peak = float(pu21_encode(np.array(_PU21_MAX)))
    mse = float(sq.mean())
    out = {
        "pu_psnr": 10.0 * np.log10(peak * peak / mse) if mse > 0 else float("inf"),
        "train_psnr": training_psnr(test, ref, clamp=False),
        "train_psnr_clamped": training_psnr(test, ref, clamp=True),
        "bands": [],
    }
    ratio = np.log2(np.maximum(test, 1e-4) / np.maximum(ref, 1e-4))
    for lo, hi in BANDS:
        m = (ref >= lo) & (ref < hi)
        n = int(m.sum())
        out["bands"].append({
            "share": n / ref.size,
            # this band's contribution to the frame's PU21 MSE
            "mse_part": float(sq[m].sum() / ref.size) if n else 0.0,
            "bias_stops": float(np.median(ratio[m])) if n else float("nan"),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--test", required=True, help="the model's tree, e.g. v4b")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--nits-scale", type=float, default=203.0)
    ap.add_argument("--stride", type=int, default=1, help="every Nth frame, for a quicker read")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    refs = sorted((args.root / "ref").rglob("*.exr"))[:: max(args.stride, 1)]
    if not refs:
        raise SystemExit(f"no EXR under {args.root / 'ref'}")
    rows = []
    for i, ref_path in enumerate(refs):
        rel = ref_path.relative_to(args.root / "ref")
        test_path, base_path = args.root / args.test / rel, args.root / args.baseline / rel
        if not (test_path.exists() and base_path.exists()):
            continue
        ref = load_frame(ref_path, args.nits_scale)
        rows.append({"frame": rel.as_posix(),
                     "test": frame_stats(load_frame(test_path, args.nits_scale), ref),
                     "baseline": frame_stats(load_frame(base_path, args.nits_scale), ref)})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(refs)}", file=sys.stderr)
    if not rows:
        raise SystemExit("no frame has all three trees")

    def delta(key: str) -> np.ndarray:
        return np.array([r["test"][key] - r["baseline"][key] for r in rows])

    print(f"\n{len(rows)} frames, {args.test} against {args.baseline}  (positive = the model is better)\n")
    print(f"{'metric':<36}{'mean':>9}{'median':>9}{'model wins':>12}")
    summary = {}
    for key, label in (("pu_psnr", "PU21-PSNR (the bench)"),
                       ("train_psnr", "log1p(16x) PSNR (training)"),
                       ("train_psnr_clamped", "log1p(16x) PSNR, clamped at max_hdr")):
        d = delta(key)
        summary[key] = {"mean_db": float(d.mean()), "median_db": float(np.median(d)),
                        "wins": int((d > 0).sum())}
        print(f"{label:<36}{d.mean():+9.3f}{np.median(d):+9.3f}{(d > 0).sum():>7}/{len(d)}")

    print(f"\nPU21 error by reference luminance (share of each frame's MSE, mean over frames),"
          f"\nand the median level bias in stops (test / ref):\n")
    print(f"{'nits':<14}{'pixels':>8}{'err ' + args.test:>12}{'err base':>10}{'bias ' + args.test:>12}{'bias base':>11}")
    bands_out = []
    for b, (lo, hi) in enumerate(BANDS):
        share = np.mean([r["test"]["bands"][b]["share"] for r in rows])
        t_part = np.mean([r["test"]["bands"][b]["mse_part"] for r in rows])
        b_part = np.mean([r["baseline"]["bands"][b]["mse_part"] for r in rows])
        t_bias = np.nanmedian([r["test"]["bands"][b]["bias_stops"] for r in rows])
        b_bias = np.nanmedian([r["baseline"]["bands"][b]["bias_stops"] for r in rows])
        name = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
        print(f"{name:<14}{share:8.1%}{t_part:12.4g}{b_part:10.4g}{t_bias:+12.3f}{b_bias:+11.3f}")
        bands_out.append({"nits": name, "pixel_share": float(share), "mse_test": float(t_part),
                          "mse_baseline": float(b_part), "bias_test": float(t_bias),
                          "bias_baseline": float(b_bias)})

    pu, tr = summary["pu_psnr"]["median_db"], summary["train_psnr"]["median_db"]
    print("\nReading it:")
    if tr > 0 > pu:
        print("  training metric says the model wins, PU21 says it loses -> the METRIC. Training"
              "\n  and best.pt selection reward the highlights and ignore the rest; the band"
              "\n  table shows which band the model gives away. Fix the loss/selection metric"
              "\n  (PU21 or a log metric floored near 1 nit) before any retrain.")
    elif tr <= 0 and pu <= 0:
        print("  both metrics say the model loses on these frames -> not the metric. The model"
              "\n  behaves differently here than on its 32 val crops: full frames (conditioning"
              "\n  heads read whole-frame statistics), preserve_outside, or the test split itself.")
    else:
        print("  PU21 says the model wins on these frames; nothing to separate.")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "bands": bands_out, "frames": rows}, indent=1))
        print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
