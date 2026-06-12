#!/usr/bin/env python3
"""
train_full_decoders.py — train the node's FULL HDR decoder for every backbone.

Trains RadianceFullDecoder (fast_vae.py, ~32M params) via train_turbo_decoder.py
--model_size full, then deploys each best-EMA into the ComfyUI models folder with
the exact name the "Radiance HDR VAE Decode" node scans for
(rudra_full_decoder_<type>_ema.safetensors).

This is the higher-capacity sibling of the turbo decoder you already trained —
same pipeline, same pairs, just the deeper architecture.

Usage
-----
  python training/train_full_decoders.py --dry-run            # show what it will do
  python training/train_full_decoders.py                      # train + deploy all
  python training/train_full_decoders.py --only flux          # one backbone
  python training/train_full_decoders.py --steps 30000 --no-deploy
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRAIN = REPO / "training" / "train_turbo_decoder.py"

# ComfyUI / radiance locations (override via env if your install differs).
COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", r"D:\A.I\ComfyUI"))
RADIANCE_PKG = COMFY_ROOT / "custom_nodes" / "radiance"
CUSTOM_NODES = COMFY_ROOT / "custom_nodes"
MODELS_RADIANCE = COMFY_ROOT / "models" / "radiance"

# backbone -> (model_type for train_turbo_decoder, pair dir, node filename type tag)
BACKBONES = {
    "flux": ("flux",      "hdrdata/hdr_pairs"),
    "wan":  ("wan",       "hdrdata/wan_hdr_pairs"),
    "ltx":  ("ltx-video", "hdrdata/ltx_pairs"),
}
# node scans rudra_full_decoder_<type>_ema; <type> is "flux"/"wan"/"ltx-video".
DEPLOY_TYPE = {"flux": "flux", "wan": "wan", "ltx": "ltx-video"}


def build_env() -> dict:
    """Child env with the radiance package importable (fast_vae + radiance.config)."""
    env = dict(os.environ)
    extra = f"{CUSTOM_NODES};{RADIANCE_PKG}"
    env["PYTHONPATH"] = extra + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return env


def main():
    ap = argparse.ArgumentParser(description="Train + deploy full HDR decoders for all backbones.")
    ap.add_argument("--only", default="", help="Comma-separated subset: flux,wan,ltx")
    ap.add_argument("--steps", type=int, default=20000, help="Training steps (full is heavier; 20k is usually plenty).")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--knee", type=float, default=0.6, help="Highlight knee in log-code space (data tops ~0.79).")
    ap.add_argument("--skip-existing", action="store_true", help="Skip a backbone whose deployed file already exists.")
    ap.add_argument("--no-deploy", action="store_true", help="Train only; don't copy into ComfyUI models.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    wanted = [b.strip() for b in args.only.split(",") if b.strip()] or list(BACKBONES)
    bad = [b for b in wanted if b not in BACKBONES]
    if bad:
        print(f"[error] unknown backbone(s): {bad}. Choose from {list(BACKBONES)}"); return 1

    env = build_env()
    print(f"ComfyUI root: {COMFY_ROOT}")
    print(f"Deploy dir:   {MODELS_RADIANCE}")
    print(f"PYTHONPATH +=  {CUSTOM_NODES};{RADIANCE_PKG}\n")

    failures = []
    for i, b in enumerate(wanted, 1):
        model_type, pair_rel = BACKBONES[b]
        pair_dir = REPO / pair_rel
        out_dir = REPO / "hdrdata" / "checkpoints" / f"full_{b}"
        best = out_dir / "full_decoder_ema_best.safetensors"
        deployed = MODELS_RADIANCE / f"rudra_full_decoder_{DEPLOY_TYPE[b]}_ema.safetensors"

        print(f"[{i}/{len(wanted)}] {b}  (model_type={model_type})")
        if args.skip_existing and deployed.exists():
            print(f"  [skip] already deployed: {deployed}\n"); continue
        if not pair_dir.is_dir():
            print(f"  [skip] missing pairs: {pair_dir}\n"); failures.append(b); continue

        cmd = [sys.executable, str(TRAIN),
               "--pair_dir", str(pair_dir),
               "--output_dir", str(out_dir),
               "--model_type", model_type,
               "--model_size", "full",
               "--steps", str(args.steps),
               "--batch_size", str(args.batch_size),
               "--lr", str(args.lr),
               "--knee", str(args.knee)]
        print("  " + " ".join(cmd))
        if args.dry_run:
            print(f"  would deploy -> {deployed}\n"); continue

        rc = subprocess.run(cmd, env=env).returncode
        if rc != 0:
            print(f"  [FAIL] {b} training exited {rc}\n"); failures.append(b); continue

        if not best.exists():
            print(f"  [WARN] no best-EMA at {best}; training may not have improved.\n")
            failures.append(b); continue

        if not args.no_deploy:
            MODELS_RADIANCE.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best, deployed)
            print(f"  deployed -> {deployed}")
        print()

    print("=== done ===")
    if failures:
        print("Failed/skipped:", ", ".join(failures)); return 2
    print("All full decoders trained" + ("" if args.no_deploy else " and deployed."))
    print("In the node: rudra_decoder=Enabled, decoder_size=rudra_full.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
