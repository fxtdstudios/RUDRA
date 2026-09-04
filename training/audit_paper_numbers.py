#!/usr/bin/env python3
"""Recompute every headline number in the paper from the benchmark JSONs.

`PAPER_ERRATA.md` records what happens when a table drifts from the
measurements underneath it: the previous manuscript reported a metric this
codebase never computed. This script exists so that cannot happen quietly
again. It reads <bench>/results/{clean,hard}_<method>.json -- the files
`rudra bench` writes -- recomputes each claim, and diffs against what the
paper says.

    python training/audit_paper_numbers.py --bench E:\\RUDRA_v3_20260822\\bench

Exit status is 1 if any claim disagrees, so it can gate a submission.

It audits the numbers that are DERIVED from the per-pair benchmark results.
The §6 headroom analysis (the 60 worst/best split and the gain-vs-peak
correlation) needs ground-truth peak luminance per frame, which is not in
these files; that analysis has no script in the repo and is listed as a gap
at the end of this run rather than silently passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# What the paper says, and where. Keep the section pointers accurate -- the
# point of the file is to send someone to the line that needs editing.
CLAIMS = [
    # (label, section, kind, args..., expected, tolerance)
    ("v5 clean gain dB",        "abstract, 5",  "gain_db",  "clean", "v5",           -3.00, 0.01),
    ("v5 clean gain JOD",       "abstract, 5",  "gain_jod", "clean", "v5",          -0.046, 0.001),
    ("v5 hard gain dB",         "abstract, 5",  "gain_db",  "hard",  "v5",          +1.43, 0.01),
    ("v5 hard gain JOD",        "abstract, 5",  "gain_jod", "hard",  "v5",          +0.443, 0.001),
    ("v5 hard frames won",      "abstract, 5",  "won",      "hard",  "v5",           348, 0),
    ("v5 clean frames won",     "5",            "won",      "clean", "v5",           115, 0),
    ("v5 hard median dB",       "5",            "median_db", "hard", "v5",          +0.609, 0.001),
    ("v5 clean median dB",      "5",            "median_db", "clean", "v5",         -2.131, 0.001),
    ("v5 hard median JOD",      "5",            "median_jod", "hard", "v5",         +0.134, 0.001),
    ("v5 clean median JOD",     "5",            "median_jod", "clean", "v5",        -0.040, 0.001),

    ("noshadow clean gain dB",  "6.1",          "gain_db",  "clean", "v5_noshadow", +0.51, 0.01),
    ("noshadow clean gain JOD", "6.1",          "gain_jod", "clean", "v5_noshadow", -0.017, 0.001),
    ("noshadow hard gain dB",   "6.1",          "gain_db",  "hard",  "v5_noshadow", +0.33, 0.01),
    ("noshadow hard gain JOD",  "6.1",          "gain_jod", "hard",  "v5_noshadow", +0.134, 0.001),

    ("v6 clean gain dB",        "5.2",          "gain_db",  "clean", "v6",          -2.75, 0.01),
    ("v6 clean gain JOD",       "5.2",          "gain_jod", "clean", "v6",          +0.004, 0.001),
    ("v6 hard gain dB",         "5.2",          "gain_db",  "hard",  "v6",          +0.96, 0.01),
    ("v6 hard gain JOD",        "5.2",          "gain_jod", "hard",  "v6",          +0.344, 0.001),

    ("gate clean gain dB",      "6.2",          "gain_db",  "clean", "shadow_v1",   +0.07, 0.01),
    ("gate clean gain JOD",     "6.2",          "gain_jod", "clean", "shadow_v1",   +0.113, 0.001),
    ("gate hard gain dB",       "6.2",          "gain_db",  "hard",  "shadow_v1",   +1.24, 0.01),
    ("gate hard gain JOD",      "6.2",          "gain_jod", "hard",  "shadow_v1",   +0.389, 0.001),

    # The abstract's headline: the gate against the SHIPPED model, not the baseline.
    ("gate vs v5, clean dB",    "abstract, 11", "delta_db",  "clean", "shadow_v1", "v5", +3.06, 0.01),
    ("gate vs v5, clean JOD",   "abstract, 11", "delta_jod", "clean", "shadow_v1", "v5", +0.159, 0.001),
    ("gate vs v5, hard dB",     "abstract, 11", "delta_db",  "hard",  "shadow_v1", "v5", -0.19, 0.01),
    ("gate vs v5, hard JOD",    "abstract, 11", "delta_jod", "hard",  "shadow_v1", "v5", -0.054, 0.001),

    # The seed sweep (§6.2). Each row is the claim in that table; the mean and
    # sd the abstract quotes are checked separately below.
    ("seed2 clean gain dB",     "6.2",          "gain_db",  "clean", "shadow_s2",  +0.71, 0.01),
    ("seed2 clean gain JOD",    "6.2",          "gain_jod", "clean", "shadow_s2",  +0.068, 0.001),
    ("seed2 hard gain dB",      "6.2",          "gain_db",  "hard",  "shadow_s2",  +0.96, 0.01),
    ("seed2 hard gain JOD",     "6.2",          "gain_jod", "hard",  "shadow_s2",  +0.308, 0.001),
    ("seed2 clean frames won",  "6.2",          "won",      "clean", "shadow_s2",   315, 0),
    ("seed3 clean gain dB",     "6.2",          "gain_db",  "clean", "shadow_s3",  +0.45, 0.01),
    ("seed3 clean gain JOD",    "6.2",          "gain_jod", "clean", "shadow_s3",  +0.089, 0.001),
    ("seed3 hard gain dB",      "6.2",          "gain_db",  "hard",  "shadow_s3",  +1.18, 0.01),
    ("seed3 hard gain JOD",     "6.2",          "gain_jod", "hard",  "shadow_s3",  +0.358, 0.001),
    ("seed3 clean frames won",  "6.2",          "won",      "clean", "shadow_s3",   280, 0),
    ("seed1 clean frames won",  "6.2",          "won",      "clean", "shadow_v1",   251, 0),

    # §5.1: ExpandNet on the same split. The absolute rows are checked as
    # deltas against the baseline so one rule covers every method.
    ("expandnet vs base dB",    "5.1",          "gain_db",  "clean", "expandnet",  -18.49, 0.01),
    ("expandnet vs base JOD",   "5.1",          "gain_jod", "clean", "expandnet",  -1.916, 0.001),
    ("expandnet frames won",    "5.1",          "won",      "clean", "expandnet",     1, 0),
    ("aligned vs base dB",      "5.1",          "gain_db",  "clean", "rudra_aligned", +0.62, 0.01),
    ("aligned vs base JOD",     "5.1",          "gain_jod", "clean", "rudra_aligned", +0.110, 0.001),
    ("expandnet vs RUDRA dB",   "5.1",          "delta_db",  "clean", "expandnet", "shadow_v1", -18.56, 0.01),
    ("expandnet vs RUDRA JOD",  "5.1",          "delta_jod", "clean", "expandnet", "shadow_v1", -2.029, 0.001),
    ("fit worth to RUDRA dB",   "5.1",          "delta_db",  "clean", "rudra_aligned", "shadow_v1", +0.55, 0.01),
    ("fit worth to RUDRA JOD",  "5.1",          "delta_jod", "clean", "rudra_aligned", "shadow_v1", -0.0025, 0.0005),

    # §8: step 72,000 is better on the criterion the selector optimises and
    # worse on CVVDP. Both halves are claims, so both are checked.
    ("72k vs v5, clean dB",     "8",            "delta_db",  "clean", "v5s72k", "v5", +0.55, 0.01),
    ("72k vs v5, clean JOD",    "8",            "delta_jod", "clean", "v5s72k", "v5", -0.052, 0.001),
    ("72k vs v5, hard JOD",     "8",            "delta_jod", "hard",  "v5s72k", "v5", -0.071, 0.001),
    ("72k composite dB",        "8",            "composite", "-",     "v5s72k",      -1.36, 0.01),
    ("v5 composite dB",         "8",            "composite", "-",     "v5",          -1.57, 0.01),
]

# Aggregates the abstract quotes as mean +/- sd over the seed sweep.
SEEDS = ("shadow_v1", "shadow_s2", "shadow_s3")
AGGREGATES = [
    ("gate clean dB, mean",   "abstract, 6.2, 11", "clean", "pu_psnr_db", "baseline", "mean", +0.41, 0.01),
    ("gate clean dB, sd",     "abstract, 6.2, 11", "clean", "pu_psnr_db", "baseline", "sd",    0.33, 0.01),
    ("gate clean JOD, mean",  "abstract, 6.2, 11", "clean", "cvvdp_jod",  "baseline", "mean", +0.090, 0.001),
    ("gate clean JOD, sd",    "abstract, 6.2, 11", "clean", "cvvdp_jod",  "baseline", "sd",    0.023, 0.001),
    ("gate hard dB, mean",    "6.2",               "hard",  "pu_psnr_db", "baseline", "mean", +1.12, 0.01),
    ("gate hard dB, sd",      "6.2",               "hard",  "pu_psnr_db", "baseline", "sd",    0.15, 0.01),
    ("gate hard JOD, mean",   "6.2",               "hard",  "cvvdp_jod",  "baseline", "mean", +0.352, 0.001),
    ("gate hard JOD, sd",     "6.2",               "hard",  "cvvdp_jod",  "baseline", "sd",    0.041, 0.001),
    ("gate vs v5 clean dB, mean",  "abstract, 11", "clean", "pu_psnr_db", "v5", "mean", +3.41, 0.01),
    ("gate vs v5 clean dB, sd",    "abstract, 11", "clean", "pu_psnr_db", "v5", "sd",    0.32, 0.01),
    ("gate vs v5 clean JOD, mean", "abstract, 11", "clean", "cvvdp_jod",  "v5", "mean", +0.135, 0.001),
    ("gate vs v5 clean JOD, sd",   "abstract, 11", "clean", "cvvdp_jod",  "v5", "sd",    0.023, 0.001),
    ("gate vs v5 hard dB, mean",   "abstract, 11", "hard",  "pu_psnr_db", "v5", "mean", -0.30, 0.01),
    ("gate vs v5 hard dB, sd",     "abstract, 11", "hard",  "pu_psnr_db", "v5", "sd",    0.15, 0.01),
    ("gate vs v5 hard JOD, mean",  "abstract, 11", "hard",  "cvvdp_jod",  "v5", "mean", -0.091, 0.001),
    ("gate vs v5 hard JOD, sd",    "abstract, 11", "hard",  "cvvdp_jod",  "v5", "sd",    0.041, 0.001),
]


def load(bench: Path, condition: str, method: str) -> dict:
    path = bench / "results" / f"{condition}_{method}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    return {(r["clip"], r["frame"]): r for r in data["results"]}, data


def paired(a: dict, b: dict, field: str) -> np.ndarray:
    keys = sorted(set(a) & set(b))
    if not keys:
        raise ValueError("no shared frames")
    return np.array([a[k][field] - b[k][field] for k in keys], dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", required=True, help="directory holding results/")
    parser.add_argument("--quiet", action="store_true", help="print only failures")
    args = parser.parse_args()
    bench = Path(args.bench)

    cache: dict[tuple[str, str], tuple[dict, dict]] = {}

    def get(condition: str, method: str):
        if (condition, method) not in cache:
            cache[(condition, method)] = load(bench, condition, method)
        return cache[(condition, method)]

    print("\n" + "=" * 78)
    print("   audit: paper claims against " + str(bench / "results"))
    print("=" * 78)
    print("   %-24s %-12s %10s %10s %8s" % ("claim", "section", "paper", "measured", ""))

    failures, checked = [], 0
    for claim in CLAIMS:
        label, section, kind = claim[0], claim[1], claim[2]
        try:
            if kind in ("delta_db", "delta_jod"):
                condition, method, against, expected, tol = claim[3:]
                a, _ = get(condition, method)
                b, _ = get(condition, against)
                field = "pu_psnr_db" if kind == "delta_db" else "cvvdp_jod"
                got = float(np.mean(paired(a, b, field)))
            elif kind == "composite":
                # hard_gain + min(0, clean_gain) -- the shipped selection rule.
                # Spans both conditions, so it does not take one.
                _, method, expected, tol = claim[3:]
                hard = float(np.mean(paired(get("hard", method)[0],
                                            get("hard", "baseline")[0], "pu_psnr_db")))
                clean = float(np.mean(paired(get("clean", method)[0],
                                             get("clean", "baseline")[0], "pu_psnr_db")))
                got = hard + min(0.0, clean)
            else:
                condition, method, expected, tol = claim[3:]
                a, _ = get(condition, method)
                b, _ = get(condition, "baseline")
                if kind == "won":
                    got = float(np.sum(paired(a, b, "pu_psnr_db") > 0))
                elif kind == "gain_db":
                    got = float(np.mean(paired(a, b, "pu_psnr_db")))
                elif kind == "gain_jod":
                    got = float(np.mean(paired(a, b, "cvvdp_jod")))
                elif kind == "median_db":
                    got = float(np.median(paired(a, b, "pu_psnr_db")))
                elif kind == "median_jod":
                    got = float(np.median(paired(a, b, "cvvdp_jod")))
                else:
                    raise ValueError(f"unknown claim kind {kind!r}")
        except Exception as exc:                                  # noqa: BLE001
            failures.append((label, section, "-", f"{type(exc).__name__}: {exc}"))
            print("   %-24s %-12s %10s  %s" % (label, section, "-", exc))
            continue

        checked += 1
        ok = abs(got - expected) <= tol + 1e-9
        fmt = "%10.0f" if kind == "won" else "%10.3f"
        line = ("   %-24s %-12s " + fmt + " " + fmt + " %8s") % (
            label, section, expected, got, "ok" if ok else "MISMATCH")
        if not ok:
            failures.append((label, section, expected, got))
            print(line)
        elif not args.quiet:
            print(line)

    print("   " + "-" * 74)
    for label, section, condition, field, against, stat, expected, tol in AGGREGATES:
        try:
            ref, _ = get(condition, against)
            values = [float(np.mean(paired(get(condition, seed)[0], ref, field)))
                      for seed in SEEDS]
            got = float(np.mean(values)) if stat == "mean" else float(np.std(values, ddof=1))
        except Exception as exc:                                  # noqa: BLE001
            failures.append((label, section, "-", f"{type(exc).__name__}: {exc}"))
            continue
        checked += 1
        ok = abs(got - expected) <= tol + 1e-9
        line = "   %-24s %-12s %10.3f %10.3f %8s" % (
            label, section, expected, got, "ok" if ok else "MISMATCH")
        if not ok:
            failures.append((label, section, expected, got))
            print(line)
        elif not args.quiet:
            print(line)

    print("=" * 78)
    print(f"   {checked} claim(s) checked, {len(failures)} mismatch(es)")
    for label, section, expected, got in failures:
        print(f"   FIX §{section}: {label} -- paper says {expected}, measured {got}")

    print("\n   AUDITED ELSEWHERE:")
    print("     §6  headroom split and the log2(peak_nits) correlation --")
    print("         training/analyze_headroom.py --check")
    print("\n   NOT AUDITED (no script in the repo reproduces these):")
    print("     §6  trimmed-PSNR table (discard worst 0.1 / 1 / 10%)")
    print("     §7  oracle sweep, ceiling regression, three-lever ablation")
    print("   Both need per-PIXEL statistics over the reference frames, which "
          "the\n   benchmark files do not carry. §10 says so.\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
