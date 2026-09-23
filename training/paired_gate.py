"""Paired per-frame comparison of two `rudra bench` CSVs, with a gate.

    python training/paired_gate.py --a bench/oog/results/oog_v4c.csv \\
        --b bench/oog/results/oog_baseline.csv --require-positive

Prints, per metric (PU21-PSNR, CVVDP): mean(a - b), a 95% bootstrap CI over
frames, wins/losses. With --require-positive, exits 1 unless every metric's
CI lies above zero -- the N3 gate in reports/SDR2HDR_PLAN_2026-09-23.md.
--max-regression X exits 1 if any metric's mean falls below -X instead
(for "not worse than" gates on clean/hard).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

METRICS = ("pu_psnr_db", "cvvdp_jod")


def load(path: Path) -> dict[str, dict[str, float]]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = row.get("frame") or row.get("asset_id")
            vals = {}
            for m in METRICS:
                try:
                    v = float(row.get(m, "nan"))
                except ValueError:
                    v = math.nan
                vals[m] = v
            rows[key] = vals
    return rows


def compare(a: dict, b: dict, draws: int = 5000, seed: int = 20260923) -> dict:
    out = {}
    rng = np.random.default_rng(seed)
    for m in METRICS:
        d = np.array([a[k][m] - b[k][m] for k in a if k in b
                      and math.isfinite(a[k][m]) and math.isfinite(b[k][m])])
        if d.size == 0:
            out[m] = {"frames": 0}
            continue
        boot = d[rng.integers(0, d.size, (draws, d.size))].mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        out[m] = {"frames": int(d.size), "mean": float(d.mean()), "ci95": [float(lo), float(hi)],
                  "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True, help="candidate CSV")
    ap.add_argument("--b", type=Path, required=True, help="reference CSV")
    ap.add_argument("--require-positive", action="store_true")
    ap.add_argument("--max-regression", type=float, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    result = compare(load(args.a), load(args.b))
    ok = True
    for m, r in result.items():
        if not r.get("frames"):
            print(f"{m:<11} no paired frames"); ok = False; continue
        print(f"{m:<11} {r['mean']:+.3f}  CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  "
              f"wins {r['wins']}/{r['frames']}")
        if args.require_positive and r["ci95"][0] <= 0:
            ok = False
        if args.max_regression is not None and r["mean"] < -abs(args.max_regression):
            ok = False
    result["passed"] = ok
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
