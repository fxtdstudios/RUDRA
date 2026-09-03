#!/usr/bin/env python3
"""Does the model shift colour, and if so, towards the truth or away from it?

§6 characterises the clean-input failure entirely by luminance. Two photographs
tested on 3 Sep 2026 -- a stock sunset and an Iceland landscape, neither in any
split -- both showed something luminance cannot see: in the brightest 1% of
pixels the red share of (R+G+B) fell by about 0.07 while green rose by about
0.05, and in deep shadow green was pulled DOWN harder than red or blue. A hue
rotation at roughly constant luminance is close to invisible to PU21-PSNR and
to CVVDP as we report them, so if it is real it is a defect the paper's own
metrics would miss.

Two photographs are a hypothesis. This measures it on all 429 held-out frames,
and -- unlike the photographs, which have no ground truth -- it reads the
REFERENCE too. That is the question that matters: a shift away from the
reference is an error, a shift towards it is a correction, and the average alone
cannot tell them apart.

    python training/analyze_chroma_shift.py --bench <bench>/clean
    python training/analyze_chroma_shift.py --bench <bench>/clean --method v5 \\
        --json docs/chroma_shift_v5.json

Pixels are banded by the BASELINE's luminance, so every method is scored on the
same partition of the same frames. Chromaticity is the share of each channel in
R+G+B, which is exactly the quantity a luminance metric cannot see.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.delivery.exr import read_exr                           # noqa: E402

DIFFUSE_WHITE_NITS = 203.0
BANDS = [(0.0, 20.0, "deep shadow <20"),
         (20.0, 203.0, "20-203"),
         (203.0, 1000.0, "203-1000"),
         (1000.0, float("inf"), ">1000")]
EPS = 1e-6


# Chromaticity is undefined at black, and averaging it there is worse than
# undefined: dividing a near-zero triple by a floor produces three numbers that
# do not sum to 1, and the mean of a band full of those is meaningless. The
# first run of this script reported a 0.096 green "error" in deep shadow that
# was entirely this. A pixel only counts when all three trees carry enough
# energy for a hue to exist.
MIN_ENERGY_NITS = 0.1


def chroma(a: np.ndarray) -> np.ndarray:
    """Share of each channel in R+G+B, per pixel. Only valid where sum > 0."""
    return a / np.maximum(a.sum(-1, keepdims=True), EPS)


def report(bands: dict, method: str, frames: int) -> None:
    total = sum(b["pixels"] for b in bands.values()) or 1
    print("\n   %-16s %8s | %-22s %-22s %-22s"
          % ("band", "share", "reference r/g/b", "baseline r/g/b", method + " r/g/b"))
    for _, _, name in BANDS:
        if name not in bands:
            continue
        b = bands[name]
        n = b["pixels"]
        r = np.array(b["sum_reference"]) / n
        base = np.array(b["sum_baseline"]) / n
        t = np.array(b["sum_method"]) / n
        dg = np.array(b["frame_green_shifts"])
        same = float(np.mean(np.sign(dg) == np.sign(dg.mean()))) if dg.size else 0.0
        toward = abs(t[1] - r[1]) < abs(base[1] - r[1])
        print("   %-16s %7.2f%% | %6.4f %6.4f %6.4f  %6.4f %6.4f %6.4f  %6.4f %6.4f %6.4f"
              % (name, 100 * n / total, *r, *base, *t))
        print("   %-16s %8s | shift vs baseline r%+.4f g%+.4f b%+.4f   "
              "green: %s the reference (|err| %.4f -> %.4f), same sign on %.0f%% of %d frames"
              % ("", "", *(t - base), "TOWARDS" if toward else "AWAY FROM",
                 abs(base[1] - r[1]), abs(t[1] - r[1]), 100 * same, dg.size))
    print("=" * 78)
    print(f"   {frames} frame(s)")


def merge(paths: list[str], out_json: str | None) -> int:
    bands: dict = {}
    frames, method = 0, None
    for path in paths:
        part = json.loads(Path(path).read_text(encoding="utf-8"))
        method = method or part["method"]
        if part["method"] != method:
            raise SystemExit(f"error: {path} is {part['method']}, not {method}")
        frames += part["frames"]
        for name, b in part["bands"].items():
            acc = bands.setdefault(name, {"pixels": 0, "frame_green_shifts": [],
                                          "sum_reference": [0.0] * 3,
                                          "sum_baseline": [0.0] * 3,
                                          "sum_method": [0.0] * 3})
            acc["pixels"] += b["pixels"]
            acc["frame_green_shifts"] += b["frame_green_shifts"]
            for key in ("sum_reference", "sum_baseline", "sum_method"):
                acc[key] = [x + y for x, y in zip(acc[key], b[key])]
    print("\n" + "=" * 78)
    print(f"   chromaticity shift: {method} vs baseline vs reference "
          f"({len(paths)} partial runs merged)")
    print("=" * 78)
    report(bands, method, frames)
    if out_json:
        Path(out_json).write_text(json.dumps(
            {"method": method, "frames": frames, "bands": bands}, indent=2),
            encoding="utf-8")
        print(f"   -> {out_json}\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", default=None,
                        help="a directory holding ref/, baseline/ and the method. "
                             "Not needed with --merge.")
    parser.add_argument("--method", default="shadow_v1")
    parser.add_argument("--offset", type=int, default=0,
                        help="skip this many frames first. With --limit and "
                             "--merge it splits a long run across several "
                             "invocations, which is how it gets run over a "
                             "shell with a wall-clock cap.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--merge", nargs="+", default=None,
                        help="combine partial JSONs written by earlier runs and "
                             "print the table; no frames are read")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    if args.merge:
        return merge(args.merge, args.json)
    if not args.bench:
        raise SystemExit("error: --bench is required unless --merge is given")

    root = Path(args.bench)
    ref_root, base_root = root / "ref", root / "baseline"
    test_root = root / args.method
    for d in (ref_root, base_root, test_root):
        if not d.is_dir():
            raise SystemExit(f"error: no {d}")

    frames = sorted(p for p in ref_root.rglob("*") if p.suffix.lower() == ".exr")
    total_frames = len(frames)
    frames = frames[args.offset:]
    if args.limit:
        frames = frames[:args.limit]

    # Per band: pixel count, summed chromaticity for each of the three trees,
    # and the per-frame green shift, so consistency can be reported and not
    # just an average that a handful of frames could be driving.
    acc = {name: {"n": 0,
                  "ref": np.zeros(3), "base": np.zeros(3), "test": np.zeros(3),
                  "frame_dg": []} for _, _, name in BANDS}
    started, used, skipped = time.time(), 0, []

    print("\n" + "=" * 78)
    print(f"   chromaticity shift: {args.method} vs baseline vs reference")
    print("=" * 78)

    for position, ref_path in enumerate(frames, 1):
        rel = ref_path.relative_to(ref_root)
        try:
            ref = read_exr(ref_path)[0][..., :3] * DIFFUSE_WHITE_NITS
            base = read_exr(base_root / rel)[0][..., :3] * DIFFUSE_WHITE_NITS
            test = read_exr(test_root / rel)[0][..., :3] * DIFFUSE_WHITE_NITS
        except Exception as exc:                                  # noqa: BLE001
            skipped.append((rel.as_posix(), f"{type(exc).__name__}: {exc}"))
            continue
        if ref.shape != base.shape or ref.shape != test.shape:
            skipped.append((rel.as_posix(), "shape mismatch"))
            continue

        ref, base, test = (np.maximum(x, 0) for x in (ref, base, test))
        lum = base.max(-1)                         # banded by the BASELINE
        lit = ((ref.sum(-1) > MIN_ENERGY_NITS)
               & (base.sum(-1) > MIN_ENERGY_NITS)
               & (test.sum(-1) > MIN_ENERGY_NITS))
        cr, cb, ct = chroma(ref), chroma(base), chroma(test)
        for lo, hi, name in BANDS:
            mask = (lum >= lo) & (lum < hi) & lit
            n = int(mask.sum())
            if n == 0:
                continue
            a = acc[name]
            a["n"] += n
            a["ref"] += cr[mask].sum(0)
            a["base"] += cb[mask].sum(0)
            a["test"] += ct[mask].sum(0)
            a["frame_dg"].append(float(ct[mask][:, 1].mean() - cb[mask][:, 1].mean()))
        used += 1
        if position % 50 == 0 or position == len(frames):
            print("   %4d/%d  %5.1f frames/s"
                  % (position, len(frames), position / max(time.time() - started, 1e-9)))

    if not used:
        raise SystemExit("error: no frames read")

    out = {"method": args.method, "frames": used, "bench": str(root.resolve()),
           "skipped": skipped, "min_energy_nits": MIN_ENERGY_NITS, "bands": {}}
    print("\n   %-16s %8s | %-22s %-22s %-22s"
          % ("band", "share", "reference r/g/b", "baseline r/g/b", args.method + " r/g/b"))
    total = sum(a["n"] for a in acc.values()) or 1
    for _, _, name in BANDS:
        a = acc[name]
        if a["n"] == 0:
            continue
        r, b, t = a["ref"] / a["n"], a["base"] / a["n"], a["test"] / a["n"]
        print("   %-16s %7.2f%% | %6.4f %6.4f %6.4f  %6.4f %6.4f %6.4f  %6.4f %6.4f %6.4f"
              % (name, 100 * a["n"] / total, *r, *b, *t))
        dg = np.array(a["frame_dg"])
        same = float(np.mean(np.sign(dg) == np.sign(dg.mean()))) if dg.size else 0.0
        # Does the model move the green share towards the reference or away?
        toward = abs(t[1] - r[1]) < abs(b[1] - r[1])
        print("   %-16s %8s | shift vs baseline r%+.4f g%+.4f b%+.4f   "
              "green: %s the reference (|err| %.4f -> %.4f), same sign on %.0f%% of frames"
              % ("", "", *(t - b), "TOWARDS" if toward else "AWAY FROM",
                 abs(b[1] - r[1]), abs(t[1] - r[1]), 100 * same))
        out["bands"][name] = {
            "pixels": a["n"], "share": a["n"] / total,
            # Raw sums, so --merge is exact rather than an average of averages.
            "sum_reference": a["ref"].tolist(),
            "sum_baseline": a["base"].tolist(),
            "sum_method": a["test"].tolist(),
            "frame_green_shifts": a["frame_dg"],
            "reference": r.tolist(), "baseline": b.tolist(), "method": t.tolist(),
            "shift_vs_baseline": (t - b).tolist(),
            "green_abs_error_baseline": abs(float(b[1] - r[1])),
            "green_abs_error_method": abs(float(t[1] - r[1])),
            "green_shift_sign_agreement": same,
            "frames_measured": int(dg.size),
        }

    print("=" * 78)
    print(f"   {used} frame(s) read" + (f", {len(skipped)} skipped" if skipped else ""))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"   -> {args.json}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
