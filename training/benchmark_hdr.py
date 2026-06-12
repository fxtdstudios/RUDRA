#!/usr/bin/env python3
"""
benchmark_hdr.py — apples-to-apples HDR scoring of two methods vs ground truth.

Use this to answer "is RUDRA better than LTX IC-LoRA-HDR?" properly: generate HDR
predictions from each method on the SAME inputs, drop them in two folders, and
this scores both against ground-truth HDR with the real metrics — ColorVideoVDP
(JOD), ΔE2000, EV-error, highlight reconstruction, and tone-mapped PSNR/SSIM.

It is model-agnostic on purpose: it does NOT run inference (each pipeline differs).
You produce the predictions; this scores them identically.

Folder layout (match files by name stem; .exr / .hdr / .png all OK)
  gt/    clip001.exr  clip002.exr ...        (ground-truth scene-linear HDR)
  a/     clip001.exr  ...                    (e.g. RUDRA predictions)
  b/     clip001.exr  ...                    (e.g. IC-LoRA-HDR predictions)

Usage
-----
  pip install -r requirements-metrics.txt    # cvvdp for the real HDR-VDP metric
  python training/benchmark_hdr.py --gt gt --a rudra_preds --b iclora_preds \
      --name-a RUDRA --name-b IC-LoRA-HDR --color-space rec2020 --out bench.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
for p in (_REPO, os.path.join(_REPO, "rudra"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F
from rudra.metrics import validation_metrics
from rudra.hdrvdp import colorvideovdp_available
from dataset_hdr import load_exr_as_linear   # handles .exr/.hdr/.png → scene-linear

# metric name -> True if higher is better
DIRECTION = {
    "hdr_vdp3": True, "psnr_tm": True, "ssim_tm": True, "hra": True,
    "ev_error": False, "delta_e_2000": False, "lpips": False,
}
REPORT = ["hdr_vdp3", "delta_e_2000", "ev_error", "hra", "psnr_tm", "ssim_tm", "lpips"]


def _load(path: Path, device) -> torch.Tensor | None:
    arr = load_exr_as_linear(str(path))
    if arr is None:
        return None
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0).float()
    return torch.nan_to_num(t, nan=0.0, posinf=1e4, neginf=0.0).clamp(min=0.0).to(device)


def _stems(d: Path) -> dict:
    out = {}
    for ext in ("*.exr", "*.hdr", "*.png", "*.tif", "*.tiff"):
        for f in d.glob(ext):
            out.setdefault(f.stem, f)
    return out


def main():
    ap = argparse.ArgumentParser(description="Benchmark two HDR methods vs ground truth.")
    ap.add_argument("--gt", required=True, help="Ground-truth HDR directory.")
    ap.add_argument("--a", required=True, help="Method A predictions directory.")
    ap.add_argument("--b", required=True, help="Method B predictions directory.")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--color-space", default="rec2020", choices=["rec2020", "rec709", "acescg"])
    ap.add_argument("--out", default="benchmark_hdr.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not colorvideovdp_available():
        print("[warn] ColorVideoVDP not installed — hdr_vdp3 will use the labeled proxy.")
        print("       pip install -r requirements-metrics.txt  for the real metric.")

    gt, A, B = _stems(Path(args.gt)), _stems(Path(args.a)), _stems(Path(args.b))
    common = sorted(set(gt) & set(A) & set(B))
    if not common:
        print("[error] no common file stems across gt / a / b."); return 1
    print(f"Scoring {len(common)} clip(s) present in all three folders...\n")

    rows = []
    acc = {args.name_a: {m: [] for m in REPORT}, args.name_b: {m: [] for m in REPORT}}
    for stem in common:
        g = _load(gt[stem], args.device)
        if g is None:
            continue
        for method, src in ((args.name_a, A), (args.name_b, B)):
            p = _load(src[stem], args.device)
            if p is None:
                continue
            if p.shape[2:] != g.shape[2:]:
                p = F.interpolate(p, size=g.shape[2:], mode="bilinear", align_corners=False)
            m = validation_metrics(p.cpu(), g.cpu(), color_space=args.color_space)
            rows.append({"clip": stem, "method": method, **{k: m.get(k) for k in REPORT}})
            for k in REPORT:
                if m.get(k) is not None:
                    acc[method][k].append(float(m[k]))

    # Write per-clip CSV
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "method"] + REPORT)
        w.writeheader()
        w.writerows(rows)

    # Summary table
    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"{'metric':<14}{args.name_a:>14}{args.name_b:>14}   winner")
    print("-" * 60)
    wins = {args.name_a: 0, args.name_b: 0}
    for k in REPORT:
        ma, mb = mean(acc[args.name_a][k]), mean(acc[args.name_b][k])
        higher = DIRECTION[k]
        if ma == ma and mb == mb:  # both finite
            a_better = (ma > mb) if higher else (ma < mb)
            win = args.name_a if a_better else args.name_b
            wins[win] += 1
        else:
            win = "-"
        arrow = "↑" if higher else "↓"
        print(f"{k+' '+arrow:<14}{ma:>14.4f}{mb:>14.4f}   {win}")
    print("-" * 60)
    overall = max(wins, key=wins.get)
    print(f"per-metric wins: {args.name_a}={wins[args.name_a]}  {args.name_b}={wins[args.name_b]}"
          f"  ->  {overall} leads")
    print(f"\nPer-clip results written to {args.out}")
    print("Report hdr_vdp3 as 'ColorVideoVDP (JOD)'. Cite only if cvvdp backend was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
