#!/usr/bin/env python3
"""
scan_model.py — inspect and evaluate a trained RUDRA decoder checkpoint.

Two things:
  1. Weight health — loads the checkpoint, reports parameter count, flags any
     NaN/inf weights, and prints per-tensor magnitude stats. A clean run should
     never have non-finite weights (the NaN guards prevent it; this verifies).
  2. Held-out evaluation — runs the trained decoder on sample pairs and reports
     aggregate metrics (PSNR, SSIM, HRA, ΔE2000, EV-error, and ColorVideoVDP JOD).

Usage
-----
  # point at a checkpoint OR a run dir (auto-picks the best EMA)
  python training/scan_model.py hdrdata/checkpoints/flux1_turbo --model-type flux
  python training/scan_model.py hdrdata/checkpoints/flux1_turbo/flux_rudra_stage1/rudra_turbo_decoder_ema_best.safetensors \
      --model-type flux --pair_dir hdrdata/hdr_pairs --samples 200
"""

from __future__ import annotations

import argparse
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
from rudra.pipeline import RUDRAPipeline, PipelineMode
from rudra.normalization import normalize_to_scene_linear
from rudra.metrics import (
    validation_metrics, psnr, ssim, highlight_reconstruction_accuracy,
    exposure_ev_error, delta_e_2000,
)
from rudra.config import FORMAT_TO_ID


def _fast_metrics(pred, target, color_space):
    """Cheap metrics only — skips ColorVideoVDP and LPIPS (the slow perceptual nets)."""
    p01, t01 = pred.clamp(0, 1), target.clamp(0, 1)
    return {
        "psnr_tm": float(psnr(p01, t01)),
        "ssim_tm": float(ssim(p01, t01)),
        "hra": float(highlight_reconstruction_accuracy(pred, target)),
        "ev_error": float(exposure_ev_error(pred, target)),
        "delta_e_2000": float(delta_e_2000(pred, target, color_space)),
    }


def _resolve_ckpt(path: Path) -> Path | None:
    """Accept a file or a run dir; prefer the best EMA safetensors, else newest .pth."""
    if path.is_file():
        return path
    cands = list(path.rglob("*ema_best*.safetensors"))
    if cands:
        return cands[0]
    pths = sorted(path.rglob("*decoder*step*.pth"),
                  key=lambda p: p.stat().st_mtime)
    return pths[-1] if pths else None


def _load_into_pipeline(pipeline: RUDRAPipeline, ckpt: Path) -> None:
    if str(ckpt).endswith(".safetensors"):
        pipeline.load_checkpoint(str(ckpt), strict=False)
        return
    obj = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    # Prefer EMA weights when present.
    dec = obj.get("ema_decoder") or obj.get("decoder")
    proj = obj.get("ema_projection") or obj.get("projection")
    if dec:
        pipeline.decoder.load_state_dict(dec, strict=False)
    if proj:
        pipeline.projection.load_state_dict(proj, strict=False)


def weight_health(module: torch.nn.Module, title: str):
    total, nonfinite, big = 0, 0, 0
    worst = []
    for name, p in module.named_parameters():
        n = p.numel()
        total += n
        nf = int((~torch.isfinite(p)).sum())
        nonfinite += nf
        amax = float(p.detach().abs().max()) if n else 0.0
        if amax > 1e3:
            big += 1
        worst.append((amax, name, tuple(p.shape)))
    worst.sort(reverse=True)
    print(f"  [{title}] params={total/1e6:.2f}M  non-finite={nonfinite}  |w|>1e3 tensors={big}")
    for amax, name, shape in worst[:3]:
        print(f"      max|w|={amax:8.3f}  {name} {shape}")
    return nonfinite == 0


def main():
    ap = argparse.ArgumentParser(description="Scan/evaluate a trained RUDRA decoder.")
    ap.add_argument("ckpt", help="Checkpoint file or run directory.")
    ap.add_argument("--model-type", default="flux")
    ap.add_argument("--model-size", default="turbo", choices=["turbo", "full"])
    ap.add_argument("--pair_dir", default="", help="Held-out pairs to evaluate on (optional).")
    ap.add_argument("--curve", default="logc4", help="Target curve (flux/wan=logc4, ltx=slog3).")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--fast", action="store_true",
                    help="Skip ColorVideoVDP + LPIPS (no hdr_vdp3) for an instant read.")
    ap.add_argument("--color-space", default="rec2020", choices=["rec2020", "rec709", "acescg"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = _resolve_ckpt(Path(args.ckpt))
    if ckpt is None:
        print(f"[error] no checkpoint found at {args.ckpt}"); return 1
    print("=" * 64)
    print(f"  Model scan: {ckpt}")
    print(f"  model_type={args.model_type}  size={args.model_size}")
    print("-" * 64)

    pipeline = RUDRAPipeline.from_model_type(
        model_type=args.model_type, mode=PipelineMode.LITE_DECODER,
        decoder_size=args.model_size, color_space=args.color_space,
    ).to(args.device).eval()
    _load_into_pipeline(pipeline, ckpt)

    ok = weight_health(pipeline.decoder, "decoder")
    ok &= weight_health(pipeline.projection, "projection")
    print(f"  weight health: {'✅ all finite' if ok else '❌ NON-FINITE WEIGHTS'}")
    print("=" * 64)

    if not args.pair_dir:
        print("Pass --pair_dir to evaluate on held-out pairs (metrics).")
        return 0 if ok else 2

    files = sorted(Path(args.pair_dir).rglob("*.npz"))
    if args.samples > 0 and len(files) > args.samples:
        idx = np.linspace(0, len(files) - 1, args.samples).astype(int)
        files = [files[i] for i in idx]
    fmt = FORMAT_TO_ID.get(args.curve, FORMAT_TO_ID["logc4"])

    agg: dict[str, list[float]] = {}
    n = 0
    with torch.no_grad():
        for f in files:
            try:
                d = np.load(str(f))
                if "latent" not in d or "log_coded" not in d:
                    continue
                lat = torch.from_numpy(d["latent"].astype(np.float32)).unsqueeze(0).to(args.device)
                lat = torch.nan_to_num(lat, nan=0.0, posinf=1e4, neginf=-1e4)
                tgt = torch.from_numpy(d["log_coded"].astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(args.device)
                fmt_ids = torch.full((1,), fmt, dtype=torch.long, device=args.device)
                tgt_lin = normalize_to_scene_linear(tgt, fmt_ids, y_max=pipeline.config.y_max_nits)
                pred, _ = pipeline(lat, tgt, fmt_ids)
                if pred.shape[2:] != tgt.shape[2:]:
                    pred = torch.nn.functional.interpolate(pred, size=tgt.shape[2:], mode="bilinear", align_corners=False)
                pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=0.0).clamp(min=0.0)
                if args.fast:
                    m = _fast_metrics(pred.cpu(), tgt_lin.cpu(), args.color_space)
                else:
                    m = validation_metrics(pred.cpu(), tgt_lin.cpu(), color_space=args.color_space)
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        agg.setdefault(k, []).append(float(v))
                n += 1
            except Exception:
                continue

    if n == 0:
        print("[error] could not evaluate any pairs."); return 2
    print(f"\nEvaluated {n} held-out pairs:")
    order = ["hdr_vdp3", "psnr_tm", "ssim_tm", "hra", "ev_error", "delta_e_2000", "lpips"]
    for k in order:
        if k in agg:
            xs = np.array(agg[k])
            print(f"  {k:<14} mean={xs.mean():8.3f}   median={np.median(xs):8.3f}")
    backend = "colorvideovdp" if "hdr_vdp3" in agg else "?"
    print(f"\n(hdr_vdp3 = ColorVideoVDP JOD if cvvdp installed, else proxy.)")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
