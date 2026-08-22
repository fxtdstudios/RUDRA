"""
train_all_decoders.py — Train Radiance decoders for ALL supported models in one run.

Usage:
    python train_all_decoders.py --hdr_dir G:\data\hdr --output_dir G:\data\checkpoints --device cuda

    # Train only specific models:
    python train_all_decoders.py --hdr_dir G:\data\hdr --output_dir G:\data\checkpoints --models flux wan sdxl

    # Resume skipped models (skip completed ones):
    python train_all_decoders.py --hdr_dir G:\data\hdr --output_dir G:\data\checkpoints --skip_existing

    # Train FullDecoder (production quality) instead of Turbo:
    python train_all_decoders.py --hdr_dir G:\data\hdr --output_dir G:\data\checkpoints --model_size full

    # Quick smoke test (500 steps per model):
    python train_all_decoders.py --hdr_dir G:\data\hdr --output_dir G:\data\checkpoints --steps 500 --pairs_per_model 200
"""

import gc
import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

import torch

logger = logging.getLogger("radiance.train_all")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

_RADIANCE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _RADIANCE not in sys.path:
    sys.path.insert(0, _RADIANCE)
_SCRIPTS = os.path.abspath(os.path.dirname(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

try:
    from radiance.config.model_map import MODEL_VAE_CONFIG, resolve_model_vae_config
except ImportError:
    from config.model_map import MODEL_VAE_CONFIG, resolve_model_vae_config


MODEL_REGISTRY = {
    "flux": {
        "vae_file": "ae.safetensors",
        "vae_aliases": ["flux_ae.safetensors", "flux_vae.safetensors"],
        "steps_turbo": 50_000,
        "steps_full": 200_000,
        "batch_turbo": 8,
        "batch_full": 4,
        "pairs_target": 5000,
        "crops_per_image": 6,
    },
    "wan": {
        "vae_file": "wan_2.1_vae.safetensors",
        "vae_aliases": ["wan_vae.safetensors", "Wan2.1_VAE.safetensors"],
        "steps_turbo": 50_000,
        "steps_full": 200_000,
        "batch_turbo": 4,
        "batch_full": 2,
        "pairs_target": 5000,
        "crops_per_image": 6,
    },
    "hunyuanvideo": {
        "vae_file": "hunyuan_vae.safetensors",
        "vae_aliases": ["hunyuan_video_vae.safetensors", "HunyuanVideo_VAE.safetensors"],
        "steps_turbo": 50_000,
        "steps_full": 200_000,
        "batch_turbo": 4,
        "batch_full": 2,
        "pairs_target": 5000,
        "crops_per_image": 6,
    },
    "ltx-video": {
        "vae_file": "ltx_vae.safetensors",
        "vae_aliases": ["ltx-video_vae.safetensors", "LTX_VAE.safetensors"],
        "steps_turbo": 50_000,
        "steps_full": 200_000,
        "batch_turbo": 2,
        "batch_full": 1,
        "pairs_target": 3000,
        "crops_per_image": 4,
    },
    "sd3": {
        "vae_file": "sd3_vae.safetensors",
        "vae_aliases": ["SD3_VAE.safetensors", "stable_diffusion_3_vae.safetensors"],
        "steps_turbo": 50_000,
        "steps_full": 200_000,
        "batch_turbo": 8,
        "batch_full": 4,
        "pairs_target": 5000,
        "crops_per_image": 6,
    },
    "sdxl": {
        "vae_file": "vae-ft-mse-840000-ema-pruned.safetensors",
        "vae_aliases": ["sdxl_vae.safetensors", "SDXL_VAE.safetensors"],
        "steps_turbo": 50_000,
        "steps_full": 200_000,
        "batch_turbo": 16,
        "batch_full": 8,
        "pairs_target": 5000,
        "crops_per_image": 6,
    },
    "sd15": {
        "vae_file": "vae-ft-mse-840000-ema-pruned.safetensors",
        "vae_aliases": ["sd_vae.safetensors", "SD1.5_VAE.safetensors"],
        "steps_turbo": 50_000,
        "steps_full": 200_000,
        "batch_turbo": 16,
        "batch_full": 8,
        "pairs_target": 5000,
        "crops_per_image": 6,
    },
    "cogvideox": {
        "vae_file": "cogvideox_vae.safetensors",
        "vae_aliases": ["CogVideoX_VAE.safetensors"],
        "steps_turbo": 50_000,
        "steps_full": 200_000,
        "batch_turbo": 4,
        "batch_full": 2,
        "pairs_target": 5000,
        "crops_per_image": 6,
    },
    "lumina2": {
        "vae_file": "qwen_image_vae.safetensors",
        "vae_aliases": ["lumina2_vae.safetensors", "Lumina2_VAE.safetensors"],
        "steps_turbo": 50_000,
        "steps_full": 200_000,
        "batch_turbo": 8,
        "batch_full": 4,
        "pairs_target": 5000,
        "crops_per_image": 6,
    },
    "pixart": {
        "vae_file": "vae-ft-mse-840000-ema-pruned.safetensors",
        "vae_aliases": ["pixart_vae.safetensors"],
        "steps_turbo": 50_000,
        "steps_full": 200_000,
        "batch_turbo": 16,
        "batch_full": 8,
        "pairs_target": 5000,
        "crops_per_image": 6,
    },
}


def find_vae(vae_dir: str, vae_file: str, aliases: list = None) -> str:
    """Search for VAE file in the models directory, trying aliases."""
    search_dirs = [
        vae_dir,
        os.path.join(vae_dir, "vae"),
        os.path.join(vae_dir, "VAE"),
    ]
    filenames = [vae_file] + (aliases or [])
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for f in filenames:
            candidate = os.path.join(d, f)
            if os.path.isfile(candidate):
                return candidate
    for root, dirs, files in os.walk(vae_dir):
        for f in filenames:
            if f in files:
                return os.path.join(root, f)
    return ""


def free_gpu():
    """Force-release GPU memory between training runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def train_all(args):
    start_time = time.time()
    results = {}

    models_to_train = args.models if args.models else list(MODEL_REGISTRY.keys())

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        from dataset_hdr import HDRPairDataset, load_vae_standalone
    except ImportError:
        logger.error("Cannot import dataset_hdr.py. Run from scripts/training/ directory.")
        sys.exit(1)

    try:
        from train_turbo_decoder import train as train_decoder
    except ImportError:
        logger.error("Cannot import train_turbo_decoder.py. Run from scripts/training/ directory.")
        sys.exit(1)

    # Resolve ComfyUI models directory
    vae_dir = args.vae_dir
    if not vae_dir:
        for candidate in [
            os.path.join(os.path.dirname(_RADIANCE), "models"),
            os.path.join(os.path.dirname(_RADIANCE), "..", "..", "models"),
        ]:
            if os.path.isdir(os.path.join(candidate, "vae")):
                vae_dir = candidate
                break
    if not vae_dir:
        vae_dir = args.output_dir

    logger.info("=" * 70)
    logger.info("  RADIANCE ALL-MODEL DECODER TRAINING")
    logger.info("=" * 70)
    logger.info(f"  HDR source:    {args.hdr_dir}")
    logger.info(f"  Output:        {args.output_dir}")
    logger.info(f"  VAE dir:       {vae_dir}")
    logger.info(f"  Device:        {args.device}")
    logger.info(f"  Model size:    {args.model_size}")
    logger.info(f"  Models:        {models_to_train}")
    logger.info(f"  Skip existing: {args.skip_existing}")
    logger.info("=" * 70)

    # ── Pre-flight: check which models can proceed ──────────────────────────────
    logger.info("")
    logger.info("Pre-flight VAE check:")
    vae_available = {}
    for model_type in models_to_train:
        reg = MODEL_REGISTRY.get(model_type)
        if not reg:
            logger.info(f"  {model_type:15s} SKIP — no registry entry")
            continue
        vae_path = find_vae(vae_dir, reg["vae_file"], reg.get("vae_aliases"))
        if vae_path:
            vae_available[model_type] = vae_path
            logger.info(f"  {model_type:15s} OK   — {vae_path}")
        else:
            logger.info(f"  {model_type:15s} MISS — {reg['vae_file']} not found in {vae_dir}")
    logger.info("")

    # ── Train each model sequentially ────────────────────────────────────────────
    for model_type in models_to_train:
        cfg = resolve_model_vae_config(model_type)
        if cfg is None:
            logger.error(f"[SKIP] Unknown model type: {model_type}")
            results[model_type] = "SKIPPED — unknown model type"
            continue

        reg = MODEL_REGISTRY.get(model_type)
        if reg is None:
            logger.error(f"[SKIP] No registry entry for: {model_type}")
            results[model_type] = "SKIPPED — no registry entry"
            continue

        # ── Find VAE ────────────────────────────────────────────────────────
        vae_path = vae_available.get(model_type)
        if not vae_path:
            logger.warning(f"[SKIP] VAE not found for {model_type}: {reg['vae_file']}")
            results[model_type] = f"SKIPPED — VAE missing: {reg['vae_file']}"
            continue

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"  TRAINING: {model_type.upper()} DECODER")
        logger.info("=" * 70)
        logger.info(f"  VAE:            {vae_path}")
        logger.info(f"  Latent channels: {cfg['latent_channels']}")
        logger.info(f"  Scale factor:    {cfg['scale_factor']}")
        logger.info(f"  Log curve:       {cfg['log_curve']}")
        logger.info(f"  Model size:      {args.model_size}")
        steps = args.steps or reg.get(f"steps_{args.model_size}", 50_000)
        batch_size = args.batch_size or reg.get(f"batch_{args.model_size}", 8)
        logger.info(f"  Steps:           {steps}")
        logger.info(f"  Batch size:      {batch_size}")

        pair_dir = os.path.join(args.output_dir, f"{model_type}_pairs")
        checkpoint_dir = os.path.join(args.output_dir, f"{model_type}_{args.model_size}_decoder")

        # ── Skip if checkpoint already exists ───────────────────────────────
        best_ema = os.path.join(checkpoint_dir, f"{args.model_size}_decoder_ema_best.safetensors")
        if args.skip_existing and os.path.isfile(best_ema):
            logger.info(f"[SKIP] {model_type}: best EMA checkpoint already exists")
            results[model_type] = "SKIPPED — checkpoint exists"
            continue

        # ── Step 1: Generate pairs ──────────────────────────────────────────
        npz_files = list(Path(pair_dir).glob("*.npz")) if os.path.isdir(pair_dir) else []
        pairs_target = reg["pairs_target"] if args.pairs_per_model <= 0 else args.pairs_per_model

        if len(npz_files) >= pairs_target * 0.9:
            logger.info(f"  [PAIRS] Found {len(npz_files)} existing pairs (target: {pairs_target}) — skipping generation")
        else:
            logger.info(f"  [PAIRS] Generating training pairs (existing: {len(npz_files)}, target: {pairs_target})...")
            try:
                vae = load_vae_standalone(vae_path, model_type, args.device)
                n_pairs = HDRPairDataset.generate_pairs(
                    exr_dirs=[args.hdr_dir],
                    output_dir=pair_dir,
                    vae=vae,
                    image_size=(args.size, args.size),
                    log_curve=cfg["log_curve"],
                    device=args.device,
                    crops_per_image=reg["crops_per_image"],
                    target_count=pairs_target,
                )
                logger.info(f"  [PAIRS] Generated {n_pairs} pairs for {model_type}")
            except Exception as e:
                logger.error(f"  [PAIRS] Failed for {model_type}: {e}")
                results[model_type] = f"FAILED — pair generation: {e}"
                free_gpu()
                continue
            finally:
                # Free VAE from GPU before training
                del vae
                free_gpu()
                logger.info(f"  [MEM] VAE freed, GPU cache cleared")

        # ── Step 2: Train decoder ────────────────────────────────────────────
        logger.info(f"  [TRAIN] Starting {model_type} {args.model_size} decoder ({steps} steps, batch {batch_size})...")
        t_model = time.time()

        try:
            final_path = train_decoder(
                pair_dir=pair_dir,
                output_dir=checkpoint_dir,
                model_type=model_type,
                model_size=args.model_size,
                steps=steps,
                batch_size=batch_size,
                lr=args.lr,
                highlight_weight=2.0,
                # 0.6, not 0.96 — log codes top out ~0.79; at 0.96 the
                # highlight loss never fires (AUDIT_2026-07-15 P1).
                knee=0.6,
                ema_decay=0.999,
                val_split=args.val_split,
                patience=args.patience,
                log_every=100,
                eval_every=2000,
                save_every=5000,
                device_str=args.device,
            )

            elapsed = time.time() - t_model
            hours = int(elapsed // 3600)
            mins = int((elapsed % 3600) // 60)

            # Deploy: copy best checkpoint to ComfyUI models/radiance/
            radiance_dir = os.path.join(
                os.path.dirname(_RADIANCE), "models", "radiance"
            )
            os.makedirs(radiance_dir, exist_ok=True)
            # Node filename contract: rudra_{size}_decoder_{tag}_ema.safetensors
            # (AUDIT_2026-07-15 P0-3 — without the rudra_ prefix the ComfyUI
            # radiance node never sees the deployed file).
            deploy_name = f"rudra_{args.model_size}_decoder_{model_type}_ema.safetensors"
            deploy_path = os.path.join(radiance_dir, deploy_name)

            if final_path and os.path.isfile(final_path):
                import shutil
                shutil.copy2(final_path, deploy_path)
                logger.info(f"  [DEPLOY] {final_path} -> {deploy_path}")

            results[model_type] = f"OK — {hours}h{mins:02d}m -> {final_path}"

        except Exception as e:
            logger.error(f"  [TRAIN] Failed for {model_type}: {e}")
            import traceback
            traceback.print_exc()
            results[model_type] = f"FAILED — training: {e}"

        # Free decoder memory before next model
        free_gpu()
        logger.info(f"  [MEM] Decoder freed, GPU cache cleared")

    # ── Summary ──────────────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    total_h = int(total_time // 3600)
    total_m = int((total_time % 3600) // 60)

    logger.info("")
    logger.info("=" * 70)
    logger.info("  TRAINING COMPLETE — SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Total wall time: {total_h}h {total_m:02d}m")
    logger.info("")
    for model_type, result in results.items():
        status = result.split(" — ")[0]
        icon = "[OK]" if status == "OK" else ("[SKIP]" if "SKIP" in result else "[FAIL]")
        logger.info(f"  {icon} {model_type:15s} {result}")
    logger.info("")

    results_path = os.path.join(args.output_dir, "train_all_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "total_time_h": total_h,
            "total_time_m": total_m,
            "model_size": args.model_size,
            "results": results,
        }, f, indent=2)
    logger.info(f"  Results saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Radiance decoders for ALL supported models in one run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models (Turbo, default):
  python train_all_decoders.py --hdr_dir G:\\data\\hdr --output_dir G:\\data\\checkpoints --vae_dir D:\\A.I\\ComfyUI\\models

  # Train only specific models:
  python train_all_decoders.py --hdr_dir G:\\data\\hdr --output_dir G:\\data\\checkpoints --models flux wan sdxl

  # Train FullDecoder (production quality):
  python train_all_decoders.py --hdr_dir G:\\data\\hdr --output_dir G:\\data\\checkpoints --model_size full

  # Quick smoke test (500 steps):
  python train_all_decoders.py --hdr_dir G:\\data\\hdr --output_dir G:\\data\\checkpoints --steps 500 --pairs_per_model 200

  # Skip models that already have checkpoints:
  python train_all_decoders.py --hdr_dir G:\\data\\hdr --output_dir G:\\data\\checkpoints --skip_existing
        """,
    )
    parser.add_argument("--hdr_dir", required=True,
                        help="Directory with HDR source images (EXR/HDR/PNG)")
    parser.add_argument("--output_dir", required=True,
                        help="Base directory for checkpoints and pairs")
    parser.add_argument("--vae_dir", default="",
                        help="ComfyUI models directory (auto-detected if empty)")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Models to train (default: all)")
    parser.add_argument("--model_size", default="turbo", choices=["turbo", "full"],
                        help="Decoder size: turbo (~2M params) or full (~18M params)")
    parser.add_argument("--steps", type=int, default=0,
                        help="Override training steps (0 = use model-specific default)")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Override batch size (0 = use model-specific default)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="AdamW learning rate")
    parser.add_argument("--pairs_per_model", type=int, default=0,
                        help="Override target pair count per model (0 = use default)")
    parser.add_argument("--size", type=int, default=512,
                        help="Training image size (square)")
    parser.add_argument("--val_split", type=float, default=0.05,
                        help="Validation split fraction (0.0 = no val)")
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip models that already have a best EMA checkpoint")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])

    args = parser.parse_args()
    train_all(args)