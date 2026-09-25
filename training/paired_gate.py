"""Paired per-frame comparison of two `rudra bench` CSVs, with a gate.

    python training/paired_gate.py --a bench/oog/results/oog_v4c.csv \\
        --b bench/oog/results/oog_baseline.csv --require-positive

Prints, per metric (PU21-PSNR, CVVDP): mean(a - b), a 95% bootstrap CI over
frames, wins/losses. With --require-positive, exits 1 unless every metric's
CI lies above zero -- the N3 gate in reports/SDR2HDR_PLAN_2026-09-23.md.
--max-regression X exits 1 if any metric's mean falls below -X instead
(for "not worse than" gates on clean/hard); --max-regression-db and
--max-regression-jod set the two metrics' limits separately (the N3 clean
gate is 0.1 dB / 0.02 JOD).

With no gate flag it is a report: it prints "GATE n/a", writes
"passed": null and exits 0. (24 Sep 2026: CP7 called it with no flags and
every comparison printed PASS, including one lost on 537 of 537 frames.)
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


REGRESSION_KEYS = {"pu_psnr_db": "max_regression_db", "cvvdp_jod": "max_regression_jod"}


def verdict(result: dict, require_positive: bool = False,
            max_regression: dict[str, float] | None = None) -> bool | None:
    """True/False against the gate asked for; None when no gate was asked for.

    ``max_regression`` maps metric -> the largest mean loss allowed.
    A metric with no paired frames fails any gate.
    """
    max_regression = {k: v for k, v in (max_regression or {}).items() if v is not None}
    if not require_positive and not max_regression:
        return None
    ok = True
    for m in METRICS:
        r = result.get(m, {})
        if not r.get("frames"):
            return False
        if require_positive and r["ci95"][0] <= 0:
            ok = False
        if m in max_regression and r["mean"] < -abs(max_regression[m]):
            ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True, help="candidate CSV")
    ap.add_argument("--b", type=Path, required=True, help="reference CSV")
    ap.add_argument("--require-positive", action="store_true")
    ap.add_argument("--max-regression", type=float, default=None,
                    help="same limit for both metrics")
    ap.add_argument("--max-regression-db", type=float, default=None)
    ap.add_argument("--max-regression-jod", type=float, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    result = compare(load(args.a), load(args.b))
    for m in METRICS:
        r = result[m]
        if not r.get("frames"):
            print(f"{m:<11} no paired frames"); continue
        print(f"{m:<11} {r['mean']:+.3f}  CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  "
              f"wins {r['wins']}/{r['frames']}")
    limits = {m: args.max_regression for m in METRICS}
    for m, key in REGRESSION_KEYS.items():
        if getattr(args, key) is not None:
            limits[m] = getattr(args, key)
    ok = verdict(result, args.require_positive, limits)
    result["passed"] = ok
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("GATE", "n/a (report only)" if ok is None else ("PASS" if ok else "FAIL"))
    return 1 if ok is False else 0


if __name__ == "__main__":
    sys.exit(main())
