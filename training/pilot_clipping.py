"""Does the corpus render actually clip now? Measured on your own footage,
before anything is re-rendered.

The full render is hours. This reads a sample of the real sources, tone-maps
each one at both conventions, and reports the two numbers the clipping audit
reported for the shipped corpus:

    median clipped fraction         was 0.000%
    frames with no clipped pixel    was 52.6%

If the numbers do not move on your footage, the fix does not work on your
footage and there is no point re-rendering 27,678 records to find that out.
Exits non-zero when the new convention still fails to clip, so it can gate a
run rather than be read and forgotten.

    python training/pilot_clipping.py --src E:\\source_hdr --limit 200
    python training/pilot_clipping.py --src E:\\source_hdr --limit 200 ^
        --output pilot_clipping.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import prepare_training_data as prep  # noqa: E402

SUFFIXES = (".tif", ".tiff", ".exr")


def load_linear(path: Path):
    """Scene-linear float for one source frame, or None if it cannot be read."""
    try:
        if path.suffix.lower() in (".tif", ".tiff"):
            data, _bits = prep.read_tif(path)
            if data is None:
                return None
            return prep.to_scene_linear(data, prep._detect_encoding(path))
        if path.suffix.lower() == ".exr":
            return prep.read_exr(path)
    except Exception as exc:                       # noqa: BLE001 - report, skip
        print(f"  skipped {path.name}: {exc}")
    return None


def summarise(fractions):
    arr = np.asarray(fractions, dtype=np.float64)
    if arr.size == 0:
        return {"frames": 0}
    return {
        "frames": int(arr.size),
        "median_pct": float(np.median(arr) * 100.0),
        "mean_pct": float(arr.mean() * 100.0),
        "p90_pct": float(np.percentile(arr, 90) * 100.0),
        "zero_clip_pct": float((arr == 0).mean() * 100.0),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="source HDR tree")
    ap.add_argument("--limit", type=int, default=200, help="frames to sample")
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--min-median-pct", type=float, default=0.05,
                    help="fail if the new convention's median clipped fraction "
                         "is below this (default 0.05%%)")
    args = ap.parse_args(argv)

    if not args.src.exists():
        print(f"source not found: {args.src}")
        return 2

    files = [p for p in args.src.rglob("*") if p.suffix.lower() in SUFFIXES]
    if not files:
        print(f"no {'/'.join(SUFFIXES)} under {args.src}")
        return 2
    random.Random(args.seed).shuffle(files)
    print(f"{len(files):,} source frames found; sampling {min(args.limit, len(files))}")

    legacy, clipping, read = [], [], 0
    for path in files:
        if read >= args.limit:
            break
        linear = load_linear(path)
        if linear is None:
            continue
        read += 1
        legacy.append(prep.clipped_fraction(prep.make_sdr(linear, prep.LEGACY_TONEMAP_EV)))
        clipping.append(prep.clipped_fraction(prep.make_sdr(linear, 0.0)))
        if read % 25 == 0:
            print(f"  {read} frames")

    a, b = summarise(legacy), summarise(clipping)
    if a["frames"] == 0:
        print("nothing could be read; check the source tree and the decoders")
        return 2

    print()
    print(f"{'':<30}{'-1 EV (shipped)':>18}{'0 EV (fixed)':>18}")
    for key, label in (("median_pct", "median clipped fraction"),
                       ("mean_pct", "mean clipped fraction"),
                       ("p90_pct", "p90 clipped fraction"),
                       ("zero_clip_pct", "frames with no clipping")):
        print(f"  {label:<28}{a[key]:>17.3f}%{b[key]:>17.3f}%")
    print(f"  {'frames read':<28}{a['frames']:>18}{b['frames']:>18}")

    print()
    if b["median_pct"] < args.min_median_pct:
        print(f"FAIL: at 0 EV the median clipped fraction is still "
              f"{b['median_pct']:.4f}%, under the {args.min_median_pct}% floor.")
        print("The exposure was not what was keeping highlights off the top of")
        print("the curve for this footage. Check the source dynamic range before")
        print("re-rendering: a set whose own ground truth peaks near diffuse")
        print("white has no highlights to clip in the first place.")
        verdict = "fail"
        code = 1
    else:
        print(f"PASS: median clipped fraction {a['median_pct']:.4f}% -> "
              f"{b['median_pct']:.4f}%, and frames with no clipping at all "
              f"{a['zero_clip_pct']:.1f}% -> {b['zero_clip_pct']:.1f}%.")
        print("The corpus now contains the phenomenon the model is trained to")
        print("reconstruct. Re-render, then retrain, then remeasure.")
        verdict = "pass"
        code = 0

    if args.output:
        args.output.write_text(json.dumps(
            {"source": str(args.src), "sampled": read, "verdict": verdict,
             "legacy_minus1ev": a, "fixed_0ev": b}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")
    return code


if __name__ == "__main__":
    sys.exit(main())
