#!/usr/bin/env python3
"""Reproduce §6's headroom analysis from the benchmark files and the manifest.

`training/audit_paper_numbers.py` checks every claim derived from the per-pair
benchmark results, and then prints a list of the analyses it CANNOT check
because they need per-frame ground-truth luminance. §6's split of the clean
frames by headroom was on that list: the numbers were real, but no script in the
repo reproduced them, which makes §10's "every number names the command that
produced it" not quite true.

It turns out nothing had to be recomputed from pixels. `peak_nits` is already
in the corpus manifest, written when the pairs were prepared, so the whole
analysis is a join:

    python training/analyze_headroom.py --bench <bench dir> --manifest <manifest>
    python training/analyze_headroom.py --bench <dir> --manifest <m> --check

`--check` compares what it finds against the figures §6 states and exits 1 on
any disagreement, so it can gate a submission alongside the audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# What §6 says, and the tolerance each figure is quoted to.
PAPER = {
    "worst_mean_db": (-10.76, 0.01),
    "worst_median_peak": (238.0, 1.0),
    "best_mean_db": (3.41, 0.01),
    "best_median_peak": (19590.0, 10.0),
    "correlation": (0.46, 0.01),
}
SPLIT_SIZE = 60


def load_results(bench: Path, condition: str, method: str) -> dict:
    path = bench / "results" / f"{condition}_{method}.json"
    if not path.exists():
        raise SystemExit(f"error: no {path}")
    return {r["frame"]: r for r in json.loads(path.read_text())["results"]}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--method", default="v5",
                        help="which tree to analyse (default: v5, what §6 is about)")
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--split-size", type=int, default=SPLIT_SIZE)
    parser.add_argument("--check", action="store_true",
                        help="compare against the figures §6 states; exit 1 on drift")
    parser.add_argument("--csv", default=None, help="write the joined per-frame table")
    args = parser.parse_args()

    bench = Path(args.bench)
    model = load_results(bench, args.condition, args.method)
    base = load_results(bench, args.condition, "baseline")

    # peak_nits is per ASSET, and the benchmark's frame id is the asset id.
    peaks: dict[str, float] = {}
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("peak_nits") is not None:
            peaks[str(record["asset_id"])] = float(record["peak_nits"])

    rows = []
    unmatched = []
    for frame, r in model.items():
        if frame not in base:
            continue
        if frame not in peaks:
            unmatched.append(frame)
            continue
        rows.append((frame, r["pu_psnr_db"] - base[frame]["pu_psnr_db"], peaks[frame]))
    if not rows:
        raise SystemExit("error: nothing joined -- do the manifest and the bench match?")

    gains = np.array([r[1] for r in rows])
    nits = np.array([r[2] for r in rows])
    order = np.argsort(gains)
    n = min(args.split_size, len(rows) // 2)
    worst, best = order[:n], order[-n:]

    got = {
        "worst_mean_db": float(gains[worst].mean()),
        "worst_median_peak": float(np.median(nits[worst])),
        "best_mean_db": float(gains[best].mean()),
        "best_median_peak": float(np.median(nits[best])),
        "correlation": float(np.corrcoef(np.log2(nits), gains)[0, 1]),
    }

    print("\n" + "=" * 70)
    print(f"   headroom analysis: {args.condition} / {args.method}, "
          f"{len(rows)} frames joined")
    print("=" * 70)
    if unmatched:
        print(f"   {len(unmatched)} frame(s) had no peak_nits in the manifest: "
              + ", ".join(unmatched[:3]) + (" ..." if len(unmatched) > 3 else ""))
    print(f"   {'clean frames':<16} {'mean delta vs baseline':>24} "
          f"{'median ground-truth peak':>26}")
    print(f"   {str(n) + ' worst':<16} {got['worst_mean_db']:>21.2f} dB "
          f"{got['worst_median_peak']:>21,.0f} nits")
    print(f"   {str(n) + ' best':<16} {got['best_mean_db']:>21.2f} dB "
          f"{got['best_median_peak']:>21,.0f} nits")
    print(f"\n   correlation between log2(peak_nits) and gain: "
          f"{got['correlation']:+.3f}")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("frame,gain_db,peak_nits\n" + "\n".join(
            f"{f},{g:.4f},{p:.3f}" for f, g, p in sorted(rows, key=lambda r: r[1])
        ) + "\n", encoding="utf-8")
        print(f"   per-frame table -> {out}")

    if not args.check:
        print()
        return 0

    print("\n   against §6:")
    bad = []
    for key, (expected, tol) in PAPER.items():
        ok = abs(got[key] - expected) <= tol + 1e-9
        print("   %-20s paper %12.2f   measured %12.2f   %s"
              % (key, expected, got[key], "ok" if ok else "MISMATCH"))
        if not ok:
            bad.append(key)
    print("=" * 70)
    print(f"   {len(PAPER)} figure(s) checked, {len(bad)} mismatch(es)\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
