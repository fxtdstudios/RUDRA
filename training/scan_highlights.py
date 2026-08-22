#!/usr/bin/env python3
"""
scan_highlights.py — does the training data actually contain HDR highlights?

RUDRA's headline contribution is highlight preservation, driven by the mask
H(x) = stops of luminance above 0.18 middle grey. If the training crops contain
no bright-over-white content, H is ~0 everywhere, the highlight loss never fires,
and "HDR superiority" is a claim with no signal behind it (this is exactly why
`hl=0.0000` shows up in the logs).

This script decodes the stored log-coded targets to scene-linear with the
correct vendor curve, computes the *exact* training H mask, and reports how much
of the dataset actually contains highlights.

Usage
-----
  python training/scan_highlights.py hdrdata/hdr_pairs                 # flux/wan = LogC4
  python training/scan_highlights.py hdrdata/ltx_pairs --curve slog3   # ltx = S-Log3
  python training/scan_highlights.py hdrdata/hdr_pairs --samples 1000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
for p in (_REPO, os.path.join(_REPO, "rudra")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from rudra.config import FORMAT_TO_ID
from rudra.spatial_descriptor import RUDRASpatialDescriptor

CURVE_TO_FMT = {
    "logc4": "logc4", "logc3": "logc3", "slog3": "slog3",
    "vlog": "vlog", "log3g10": "log3g10",
}


def main():
    ap = argparse.ArgumentParser(description="Scan training pairs for HDR highlight coverage.")
    ap.add_argument("pair_dir", help="Directory of .npz pairs (with a 'log_coded' target).")
    ap.add_argument("--curve", default="logc4", choices=list(CURVE_TO_FMT),
                    help="Log curve the targets were encoded with (flux/wan=logc4, ltx=slog3).")
    ap.add_argument("--samples", type=int, default=500, help="How many pairs to sample.")
    ap.add_argument("--ev-threshold", type=float, default=2.0,
                    help="Highlight = luminance this many stops above 0.18 grey (matches training).")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    files = sorted(Path(args.pair_dir).rglob("*.npz"))
    if not files:
        print(f"[error] no .npz pairs in {args.pair_dir}"); return 1
    if args.samples > 0 and len(files) > args.samples:
        idx = np.linspace(0, len(files) - 1, args.samples).astype(int)
        files = [files[i] for i in idx]

    fmt_id = FORMAT_TO_ID[CURVE_TO_FMT[args.curve]]
    desc = RUDRASpatialDescriptor(
        normalize_input=True, highlight_ev_threshold=args.ev_threshold,
    ).to(args.device).eval()

    cov = []        # fraction of highlight pixels per image
    energy = []     # mean H energy per image
    n_with = 0      # images with ANY highlight pixel
    n_ok = 0
    for f in files:
        try:
            d = np.load(str(f))
            if "log_coded" not in d:
                continue
            img = d["log_coded"].astype(np.float32)          # (H, W, 3)
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(args.device)
            with torch.no_grad():
                r = desc(t, format_id=fmt_id)                 # (1, 5, H, W)
                H = r[:, 2:3]
            frac = float((H > 0).float().mean())
            cov.append(frac)
            energy.append(float(H.mean()))
            n_with += int(frac > 0)
            n_ok += 1
        except Exception:
            continue

    if n_ok == 0:
        print("[error] could not read any pairs (missing 'log_coded'?)."); return 1

    cov = np.array(cov)
    pct_any = 100.0 * n_with / n_ok
    print("=" * 64)
    print(f"  Highlight scan: {args.pair_dir}")
    print(f"  curve={args.curve}  highlight=+{args.ev_threshold} stops over grey  sampled={n_ok}")
    print("-" * 64)
    print(f"  images with ANY highlight pixel : {pct_any:6.1f}%  ({n_with}/{n_ok})")
    print(f"  images with >0.1% coverage      : {100.0*np.mean(cov > 0.001):6.1f}%")
    print(f"  images with >1%   coverage      : {100.0*np.mean(cov > 0.01):6.1f}%")
    print(f"  highlight coverage  mean/median : {100*cov.mean():.3f}% / {100*np.median(cov):.3f}%")
    print(f"  highlight coverage  p90 / max   : {100*np.percentile(cov,90):.3f}% / {100*cov.max():.3f}%")
    print("=" * 64)

    if pct_any < 10:
        print("VERDICT: FAIL - Almost no highlights; the highlight loss will rarely fire.")
        print("  RUDRA's HDR advantage cannot show up on this data. Options:")
        print("   - curate high-DR sources (skies, sun, neon, specular, fire, windows),")
        print("   - or lower --ev-threshold (default +2 stops over grey).")
    elif pct_any < 40:
        print("VERDICT: WARN - Some highlights, but sparse. Consider enriching the dataset")
        print("  with brighter sources so the highlight objective trains meaningfully.")
    else:
        print("VERDICT: PASS - Plenty of highlight content; the HDR objective will engage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
