"""Paired comparison of two scored methods on the bench, against seed noise.

run_bench.ps1 and score_checkpoint.ps1 write one row PER FRAME under
bench/results/. Comparing two methods by their means throws that away. Paired
is far stronger: the same 429 frames, the same reference, so the difference is
measured frame by frame and the scene-to-scene variance -- which dwarfs every
effect being looked for -- cancels.

The second half is the part that decides things. An effect is only real if it
is bigger than what retraining with a different SEED produces on its own. On
10 Sep 2026 the shadow gate cleared that bar on three of four measures and the
question of whether to keep it was settled from files already on disk, without
spending a single GPU-hour re-running anything.

    python training/compare_bench_methods.py --results E:\\...\\bench\\results \\
        --treatment shadow_v1 shadow_s2 shadow_s3 --control v5_noshadow
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

METRICS = ((0, "PU21-PSNR", "dB"), (1, "CVVDP", "JOD"))


def load(path: Path) -> dict:
    rows = {}
    # utf-8-sig: these CSVs are written on Windows and several carry a BOM,
    # which turns the first column name into "\ufeffclip" and the first row
    # into a silent KeyError if it is used as a header.
    for line in path.read_text(encoding="utf-8-sig").splitlines()[1:]:
        if not line.strip():
            continue
        clip, frame, pu, jod = line.rsplit(",", 3)
        rows[(clip, frame)] = (float(pu), float(jod))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    return rows


def paired(a: dict, b: dict, index: int):
    keys = sorted(set(a) & set(b))
    if not keys:
        raise SystemExit("the two methods share no frames")
    return np.array([a[k][index] - b[k][index] for k in keys]), len(keys)


def bootstrap_ci(d: np.ndarray, draws: int = 20000, seed: int = 20260910):
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(draws, d.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="bench/results directory")
    ap.add_argument("--treatment", nargs="+", required=True,
                    help="method names; two or more are treated as seeds of one change")
    ap.add_argument("--control", required=True, help="the method to compare against")
    ap.add_argument("--conditions", nargs="+", default=["clean", "hard"])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    report = {}
    for cond in args.conditions:
        def path(name):
            p = args.results / f"{cond}_{name}.csv"
            if not p.is_file():
                raise SystemExit(f"missing {p} -- score it with score_checkpoint.ps1 first")
            return p

        sets = {n: load(path(n)) for n in (*args.treatment, args.control)}
        print(f"\n{'=' * 74}\n  {cond.upper()}\n{'=' * 74}")
        for index, metric, unit in METRICS:
            print(f"\n  {metric}")
            for name, rows in sets.items():
                v = np.array([r[index] for r in rows.values()])
                print(f"    {name:<14} mean {v.mean():8.4f} {unit}   n={v.size}")

            print(f"\n    EFFECT  (each treatment against {args.control}, paired)")
            effects = []
            for name in args.treatment:
                d, n = paired(sets[name], sets[args.control], index)
                lo, hi = bootstrap_ci(d)
                effects.append(float(d.mean()))
                star = "" if lo <= 0.0 <= hi else "  *"
                print(f"      {name:<12} {d.mean():+8.4f} {unit}"
                      f"   95% CI [{lo:+.4f}, {hi:+.4f}]   n={n}{star}")

            spread = [0.0]
            if len(args.treatment) > 1:
                print("\n    SEED NOISE  (treatments against each other, paired)")
                for a, b in itertools.combinations(args.treatment, 2):
                    d, n = paired(sets[a], sets[b], index)
                    lo, hi = bootstrap_ci(d)
                    spread.append(abs(float(d.mean())))
                    print(f"      {a} vs {b}  {d.mean():+8.4f} {unit}"
                          f"   95% CI [{lo:+.4f}, {hi:+.4f}]")

            effect, noise = float(np.mean(effects)), float(np.max(spread))
            real = abs(effect) > noise
            print(f"\n    effect {effect:+.4f} {unit}  vs worst seed-to-seed {noise:.4f} {unit}"
                  f"  ->  {'LARGER than seed noise' if real else 'WITHIN seed noise'}")
            report[f"{cond}/{metric}"] = {
                "effect": effect, "seed_noise": noise, "larger_than_seed_noise": real,
                "per_treatment": dict(zip(args.treatment, effects))}

    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n   wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
