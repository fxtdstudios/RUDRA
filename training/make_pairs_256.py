#!/usr/bin/env python3
"""
make_pairs_256.py — generate a low-resolution pair set for the backbone stages.

Stage 2/3 (LoRA / DRE) backprop through the full 12B Flux model, so the token
count is the speed/VRAM bottleneck. 256x256 pairs give 4x fewer Flux tokens and
~16x cheaper DRE attention than the 512px decoder pairs, which is the difference
between an overnight crawl and a ~2h run on a 16 GB card.

Usage
-----
  conda activate comfyui
  python training/make_pairs_256.py                      # flux, 256px, 4 crops/img
  python training/make_pairs_256.py --size 384 --crops 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", r"D:\A.I\ComfyUI"))

for p in (str(COMFY_ROOT), str(COMFY_ROOT / "custom_nodes"),
          str(COMFY_ROOT / "custom_nodes" / "radiance"), str(REPO / "training")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser(description="Generate low-res pairs for Stage 2/3 training.")
    ap.add_argument("--size", type=int, default=256, help="Square image size (default 256).")
    ap.add_argument("--crops", type=int, default=4, help="Crops per source image (more data).")
    ap.add_argument("--model-type", default="flux")
    ap.add_argument("--vae", default=str(COMFY_ROOT / "models" / "vae" / "ae.safetensors"),
                    help="VAE checkpoint to encode with (default: Flux ae.safetensors).")
    ap.add_argument("--out", default="", help="Output dir (default hdrdata/<type>_pairs_<size>).")
    args = ap.parse_args()

    out = Path(args.out) if args.out else REPO / "hdrdata" / f"{args.model_type}_pairs_{args.size}"
    src = REPO / "hdrdata" / "Source_HDR"
    exr_dirs = [str(p) for p in src.iterdir() if p.is_dir()] or [str(src)]

    from dataset_hdr import HDRPairDataset, load_vae_standalone
    print(f"VAE: {args.vae}\nout: {out}\nsize: {args.size}  crops/img: {args.crops}")
    vae = load_vae_standalone(args.vae, model_type=args.model_type)
    n = HDRPairDataset.generate_pairs(
        exr_dirs=exr_dirs, output_dir=str(out), vae=vae,
        image_size=(args.size, args.size), crops_per_image=args.crops,
    )
    print(f"generated {n} pairs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
