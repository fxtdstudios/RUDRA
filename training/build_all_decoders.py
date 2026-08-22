#!/usr/bin/env python3
"""
build_all_decoders.py — end-to-end decoder factory for every backbone.

For each model in the registry it does, in order:
  1. DOWNLOAD any missing VAE / text-encoder into the ComfyUI models tree.
  2. GENERATE pairs (latent + log_coded) from Source_HDR via that model's VAE,
     if a pair set doesn't already exist.
  3. TRAIN the turbo decoder, then the full decoder.
  4. DEPLOY both best-EMA files into models/radiance/ with the names the node
     scans for (rudra_turbo_decoder_<type>_ema / rudra_full_decoder_<type>_ema).

Models whose VAE doesn't fit the standard decoder (different channel count or
16x compression — Flux.2 Klein, Qwen-Image-VAE-2.0) are marked unsupported and
skipped with a clear note; they need a modified decoder, not just config.

Usage
-----
  python training/build_all_decoders.py --list                 # show plan + readiness
  python training/build_all_decoders.py --dry-run
  python training/build_all_decoders.py --only flux,wan,ltx     # already-paired ones
  python training/build_all_decoders.py --sizes turbo           # turbo only
  python training/build_all_decoders.py --only sdxl             # download+pairs+train

IMPORTANT: VAE/text-encoder download URLs for NEW backbones must be filled in the
REGISTRY below (marked TODO). I won't invent links; set them to the real HF
resolve URLs for your models, then re-run.
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
SOURCE_HDR = REPO / "hdrdata" / "Source_HDR"   # Alexa / Netflix / PolyHaven / ...

COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", r"D:\A.I\ComfyUI"))
RADIANCE_PKG = COMFY_ROOT / "custom_nodes" / "radiance"
CUSTOM_NODES = COMFY_ROOT / "custom_nodes"
MODELS = COMFY_ROOT / "models"
MODELS_RADIANCE = MODELS / "radiance"

# ── Registry ──────────────────────────────────────────────────────────────────
# type        : model_type passed to train_turbo_decoder / load_vae_standalone
# latent      : VAE latent channels (decoder input width)
# vae         : (filename, subfolder, url)  url="" means "already present / TODO"
# pairs       : pair directory (relative to repo)
# deploy_tag  : <type> used in the node's scan filename
# supported   : False => skipped (needs a non-standard decoder)
REGISTRY = {
    "flux":      dict(type="flux",      latent=16, deploy_tag="flux",
                      vae=("ae.safetensors", "vae",
                           "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors"),
                      pairs="hdrdata/hdr_pairs", supported=True),
    "wan":       dict(type="wan",       latent=16, deploy_tag="wan",
                      vae=("wan_2.1_vae.safetensors", "vae", ""),       # TODO url if missing
                      pairs="hdrdata/wan_hdr_pairs", supported=True),
    # LTX-2 / LTX-2.3 Video-VAE (AutoencoderKLLTX2Video): 32x spatial / 8x temporal /
    # 128ch. Co-locate config.json next to this file so the loader builds the real
    # 2.3 VAE (see load_vae_standalone ltx branch). 2.44 GB — NOT the 46 GB transformer.
    "ltx":       dict(type="ltx-video", latent=128, deploy_tag="ltx-video",
                      vae=("diffusion_pytorch_model.safetensors", "vae/ltx2",
                           "https://huggingface.co/Lightricks/LTX-2/resolve/main/vae/diffusion_pytorch_model.safetensors"),
                      pairs="hdrdata/ltx_pairs", supported=True),
    "sdxl":      dict(type="sdxl",      latent=4, deploy_tag="sdxl",
                      vae=("sdxl_vae.safetensors", "vae",
                           "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors"),
                      pairs="hdrdata/sdxl_pairs", supported=True),
    # Z-Image uses the FLUX.1 VAE → same 16ch latent space → the flux decoder
    # works directly. Reuse the flux pairs; deploying = copy the flux decoder.
    "zimage":    dict(type="zimage",    latent=16, deploy_tag="zimage",
                      vae=("ae.safetensors", "vae",
                           "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors"),
                      pairs="hdrdata/hdr_pairs", supported=True,
                      note="uses Flux.1 VAE — flux decoder also works as-is."),
    "qwen":      dict(type="qwen",      latent=16, deploy_tag="qwen",
                      vae=("qwen_image_vae.safetensors", "vae",
                           "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"),
                      pairs="hdrdata/qwen_pairs", supported=True),
    # Flux.2 Klein: 32ch VAE (NOT the flux.1 VAE). The decoder takes
    # latent_channels=32; the Qwen3 text encoder is irrelevant to the decoder.
    # If pair-gen shows the latent isn't 8x (64x64 for 512 output), the decoder
    # needs an extra upsample stage — the script reports the shape.
    "flux2-klein": dict(type="flux2-klein", latent=128, deploy_tag="flux2-klein",
                        vae=("flux2-vae.safetensors", "vae",
                             "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors"),
                        pairs="hdrdata/flux2_pairs", supported=True),
    # ── Still unsupported: Qwen-Image-VAE-2.0 (16x compression) ────────────────
    "qwen2":     dict(type="qwen2",     latent=64, deploy_tag="qwen2",
                      vae=("", "vae", ""), pairs="hdrdata/qwen2_pairs", supported=False,
                      note="Qwen-Image-VAE-2.0 is 16x / 64-128ch — needs an extra upsample stage."),
}


def child_env() -> dict:
    env = dict(os.environ)
    extra = f"{CUSTOM_NODES};{RADIANCE_PKG}"
    env["PYTHONPATH"] = extra + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return env


def ensure_vae(spec, dry: bool) -> str | None:
    """Make sure the model's VAE is present; download if a url is given. Returns path or None."""
    fname, sub, url = spec["vae"]
    if not fname:
        return None
    dest = MODELS / sub / fname
    if dest.exists():
        return str(dest)
    if not url:
        print(f"    [vae] missing {dest} and no URL set (fill REGISTRY). Skipping this model.")
        return None
    print(f"    [vae] downloading {fname} -> {dest}")
    if dry:
        return str(dest)
    try:
        sys.path.insert(0, str(REPO / "training"))
        from download_models import _download
        dest.parent.mkdir(parents=True, exist_ok=True)
        _download(url, dest)
        # diffusers-format VAE folders also need config.json beside the weights so
        # AutoencoderKL*.from_pretrained(local_dir) can build the model (LTX-2.3).
        if fname == "diffusion_pytorch_model.safetensors":
            cfg_url = url.rsplit("/", 1)[0] + "/config.json"
            cfg_dest = dest.parent / "config.json"
            if not cfg_dest.exists():
                print(f"    [vae] downloading companion config.json -> {cfg_dest}")
                try:
                    _download(cfg_url, cfg_dest)
                except Exception as ce:
                    print(f"    [vae] config.json fetch failed (comfy path still works): {ce}")
        return str(dest) if dest.exists() else None
    except Exception as e:
        print(f"    [vae] download failed: {e}")
        return None


def ensure_pairs(spec, vae_path: str, dry: bool) -> bool:
    """Generate pairs from Source_HDR if the pair dir is empty/missing."""
    pair_dir = REPO / spec["pairs"]
    if pair_dir.is_dir() and any(pair_dir.glob("*.npz")):
        return True
    if not SOURCE_HDR.is_dir():
        print(f"    [pairs] no Source_HDR at {SOURCE_HDR}; can't generate. Skipping.")
        return False
    if not vae_path:
        print(f"    [pairs] no VAE available; can't generate pairs. Skipping.")
        return False
    print(f"    [pairs] generating into {pair_dir} from {SOURCE_HDR}")
    if dry:
        return True
    try:
        # VAE encoding needs ComfyUI's `comfy` package — put the ComfyUI root and
        # the radiance package on the path so load_vae_standalone can import them.
        for p in (str(COMFY_ROOT), str(CUSTOM_NODES), str(RADIANCE_PKG), str(REPO / "training")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from dataset_hdr import HDRPairDataset, load_vae_standalone
        exr_dirs = [str(p) for p in SOURCE_HDR.iterdir() if p.is_dir()] or [str(SOURCE_HDR)]
        vae = load_vae_standalone(vae_path, model_type=spec["type"])
        n = HDRPairDataset.generate_pairs(exr_dirs=exr_dirs, output_dir=str(pair_dir), vae=vae)
        print(f"    [pairs] generated {n} pairs")
        # Report the actual latent shape vs output — confirms channels and the
        # spatial compression (8x = standard decoder; 16x needs an extra upsample).
        try:
            import numpy as np, glob as _g
            d0 = np.load(sorted(_g.glob(str(pair_dir / "*.npz")))[0])
            lat, out = d0["latent"].shape, d0["log_coded"].shape
            comp = out[0] / lat[-1]
            warn = "" if abs(comp - 8) < 0.5 else f"  ⚠️ {comp:.0f}x compression — decoder needs {int(comp).bit_length()-1} upsample stages (standard=3)"
            print(f"    [pairs] latent={lat}  output={out}  -> {comp:.0f}x{warn}")
        except Exception:
            pass
        return n > 0
    except Exception as e:
        print(f"    [pairs] generation failed: {e}")
        return False


def train_and_deploy(spec, size: str, args) -> bool:
    pair_dir = REPO / spec["pairs"]
    out_dir = REPO / "hdrdata" / "checkpoints" / f"{size}_{spec['deploy_tag'].replace('-', '')}"
    best = out_dir / f"{size}_decoder_ema_best.safetensors"
    deployed = MODELS_RADIANCE / f"rudra_{size}_decoder_{spec['deploy_tag']}_ema.safetensors"

    if args.skip_existing and deployed.exists():
        print(f"    [{size}] already deployed: {deployed.name}"); return True
    # The full decoder (~16x params) is unstable at the turbo LR — use a gentler
    # LR and more steps so it converges instead of oscillating.
    steps = args.steps_full if size == "full" else args.steps_turbo
    lr = args.lr_full if size == "full" else args.lr_turbo
    cmd = [sys.executable, str(TRAIN), "--pair_dir", str(pair_dir), "--output_dir", str(out_dir),
           "--model_type", spec["type"], "--model_size", size, "--steps", str(steps),
           "--lr", str(lr), "--knee", str(args.knee), "--batch_size", str(args.batch_size),
           # AUDIT_2026-07-15 P0-5: without these the trainer "validates" on a
           # training batch and deploys a noise-selected "best" checkpoint.
           "--val_split", str(args.val_split), "--patience", str(args.patience)]
    print(f"    [{size}] " + " ".join(cmd))
    if args.dry_run:
        print(f"    [{size}] would deploy -> {deployed}"); return True
    if subprocess.run(cmd, env=child_env()).returncode != 0:
        print(f"    [{size}] training FAILED"); return False
    if not best.exists():
        print(f"    [{size}] no best-EMA produced"); return False
    MODELS_RADIANCE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, deployed)
    print(f"    [{size}] deployed -> {deployed}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Download + pair + train turbo & full decoders for all backbones.")
    ap.add_argument("--only", default="", help=f"Subset of: {','.join(REGISTRY)}")
    ap.add_argument("--sizes", default="turbo,full", help="turbo, full, or both (default).")
    ap.add_argument("--steps-turbo", type=int, default=20000)
    ap.add_argument("--steps-full", type=int, default=40000)   # full needs longer to converge
    ap.add_argument("--lr-turbo", type=float, default=3e-4)
    ap.add_argument("--lr-full", type=float, default=1e-4)      # gentler LR — full oscillates at 3e-4
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--knee", type=float, default=0.6)
    ap.add_argument("--val-split", dest="val_split", type=float, default=0.05,
                    help="Held-out fraction for best-checkpoint selection / early stopping.")
    ap.add_argument("--patience", type=int, default=8,
                    help="Evals without PSNR improvement before early stop (0 = off).")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--list", action="store_true", help="Show readiness and exit.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(REGISTRY)
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]

    if args.list:
        print(f"{'model':<14}{'latent':<8}{'supported':<11}{'pairs?':<9}note")
        for n in names:
            s = REGISTRY[n]
            has = (REPO / s["pairs"]).is_dir() and any((REPO / s["pairs"]).glob("*.npz"))
            print(f"{n:<14}{s['latent']:<8}{str(s['supported']):<11}{('yes' if has else 'no'):<9}{s.get('note','')}")
        return 0

    failures = []
    for i, n in enumerate(names, 1):
        spec = REGISTRY[n]
        print(f"\n[{i}/{len(names)}] {n}  (type={spec['type']}, latent={spec['latent']})")
        if not spec["supported"]:
            print(f"    [skip] unsupported by standard decoder — {spec.get('note','')}")
            continue
        vae_path = ensure_vae(spec, args.dry_run)
        if not ensure_pairs(spec, vae_path, args.dry_run):
            failures.append(n); continue
        for size in sizes:
            if not train_and_deploy(spec, size, args):
                failures.append(f"{n}:{size}")

    print("\n=== done ===")
    if failures:
        print("Failed/skipped:", ", ".join(failures)); return 2
    print("All requested decoders built and deployed.")
    print("In the node: rudra_decoder=Enabled, decoder_size=rudra_turbo or rudra_full.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
