"""The critical path's gates, scored from the bench CSVs already on disk.

    python training/cp7_verdicts.py                      # bench/cp_* -> reports/logs/cp_results.json
    python training/cp7_verdicts.py --bench-root bench --out reports/logs/cp_results.json

One place for what each comparison is FOR. Every ``<bench>/<model> vs
baseline`` row asks "does the model beat the analytic inverse it rides on?"
(both CIs above zero). ``vs shadow_v1`` rows are reports only: shadow_v1 was
trained over a -1 EV render and is scored here against 0 EV references, so
a gate against it would measure the exposure, not the model. The named gates
are the ones in reports/SDR2HDR_PLAN_2026-09-23.md section 6.

Needs no GPU and no torch: re-run it whenever a CSV changes. Exit code is 0
unless --strict and a named gate fails (a report, not a stop).
(24 Sep 2026: CP7 had called paired_gate.py with no gate flags, so every row
read PASS, including shadow_v1's 0 of 537 on the ACES bench.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.paired_gate import METRICS, compare, load, verdict  # noqa: E402

REFERENCE_MODEL = "shadow_v1"

# (name, what it asks, bench, candidate, reference, require_positive, max_regression)
NAMED_GATES = (
    ("N1/step4", "v4b beats the analytic inverse on its own ACES render",
     "aces", "v4b", "baseline", True, None),
    ("N1/step5", "the gate head adds something over the model it was trained on",
     "aces", "v4b_gate", "v4b", True, None),
    ("N3", "v4c beats the analytic inverse out of generator (Hable + H.264)",
     "oog", "v4c", "baseline", True, None),
    ("N3/clean", "v4c is not worse than v4b on the ACES render (0.1 dB / 0.02 JOD)",
     "aces", "v4c", "v4b", False, {"pu_psnr_db": 0.1, "cvvdp_jod": 0.02}),
    ("N7/oog", "studio weights within 0.3 dB / 0.03 JOD of v4c, out of generator",
     "oog", "v4c_studio", "v4c", False, {"pu_psnr_db": 0.3, "cvvdp_jod": 0.03}),
    ("N7/mix", "studio weights within 0.3 dB / 0.03 JOD of v4c, mixed curves",
     "mix", "v4c_studio", "v4c", False, {"pu_psnr_db": 0.3, "cvvdp_jod": 0.03}),
)


def _csv(root: Path, bench: str, model: str) -> Path:
    return root / f"cp_{bench}" / "results" / f"{model}.csv"


def _row(root: Path, bench: str, a: str, b: str, require_positive: bool,
         max_regression: dict | None) -> dict | None:
    pa, pb = _csv(root, bench, a), _csv(root, bench, b)
    if not (pa.exists() and pb.exists()):
        return None
    result = compare(load(pa), load(pb))
    result["passed"] = verdict(result, require_positive, max_regression)
    return result


def score(root: Path) -> dict:
    comparisons, gates = {}, {}
    for bench_dir in sorted(root.glob("cp_*")):
        bench = bench_dir.name[3:]
        models = sorted(p.stem for p in (bench_dir / "results").glob("*.csv") if p.stem != "baseline")
        for m in models:
            r = _row(root, bench, m, "baseline", True, None)
            if r is not None:
                comparisons[f"{bench}/{m} vs baseline"] = r
            if m != REFERENCE_MODEL:
                r = _row(root, bench, m, REFERENCE_MODEL, False, None)
                if r is not None:
                    comparisons[f"{bench}/{m} vs {REFERENCE_MODEL}"] = r
    for name, question, bench, a, b, pos, reg in NAMED_GATES:
        r = _row(root, bench, a, b, pos, reg)
        gates[name] = {"question": question, "bench": bench, "candidate": a, "reference": b,
                       "passed": None if r is None else r["passed"],
                       "status": "not run" if r is None else ("PASS" if r["passed"] else "FAIL"),
                       **({} if r is None else {m: r[m] for m in METRICS})}
    return {"gates": gates, "comparisons": comparisons}


def fmt(r: dict) -> str:
    out = []
    for m, unit in (("pu_psnr_db", "dB"), ("cvvdp_jod", "JOD")):
        x = r.get(m) or {}
        if x.get("frames"):
            out.append(f"{x['mean']:+7.3f} {unit} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}] "
                       f"{x['wins']}/{x['frames']}")
    return "   ".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-root", type=Path, default=REPO / "bench")
    ap.add_argument("--out", type=Path, default=REPO / "reports" / "logs" / "cp_results.json")
    ap.add_argument("--strict", action="store_true", help="exit 1 if a named gate that ran failed")
    args = ap.parse_args()
    summary = score(args.bench_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("named gates")
    for name, g in summary["gates"].items():
        print(f"  {g['status']:<8} {name:<10} {g['question']}")
        if g["status"] != "not run":
            print(f"           {fmt(g)}")
    print("\nvs baseline (both CIs above zero = PASS; vs shadow_v1 rows are in the JSON, report only)")
    for key, r in summary["comparisons"].items():
        if key.endswith("vs baseline"):
            print(f"  {'PASS' if r['passed'] else 'FAIL':<5} {key:<28} {fmt(r)}")
    print(f"\n-> {args.out}")
    failed = any(g["passed"] is False for g in summary["gates"].values())
    return 1 if (args.strict and failed) else 0


if __name__ == "__main__":
    sys.exit(main())
