"""
train_rudra.py — Train RUDRA (Dynamic Range Conditioned Adapters)
================================================================================

End-to-end training of the RUDRA pipeline:

  Stage 1 (Decoder):  RUDRADescriptor (fixed) → RUDRAProjection (learned)
                      → RUDRADecoder with FiLM (learned)
                      Learns to decode latents into log-coded images,
                      conditioned on the dynamic range of the source content.

  Stage 2 (LoRA):    RUDRADescriptor → RUDRAProjection → RUDRALoRA
                      DR-gated LoRA injected into the diffusion backbone.
                      Learns HDR-aware denoising, conditioned on DR info.

  Stage 3 (DRE):     RUDRASpatialDescriptor → DRE Transformer → Cross-Attention Projection
                      Concatenated to text embeddings in diffusion backbone.
                      Learns spatial HDR awareness with DRE tokens.

Usage:
    # Stage 1: Train RUDRA decoder
    python train_rudra.py --pair_dir G:\\data\\flux_pairs --output_dir G:\\data\\checkpoints \\
        --stage decoder --steps 50000

    # Stage 2: Train RUDRA LoRA for Flux (Backbone)
    python train_rudra.py --pair_dir G:\\data\\flux_pairs --output_dir G:\\data\\checkpoints \\
        --stage lora --model_path flux1-dev-fp8.safetensors --steps 10000

    # Stage 3: Train RUDRA DRE
    python train_rudra.py --pair_dir G:\\data\\flux_pairs --output_dir G:\\data\\checkpoints \\
        --stage dre --model_path flux1-dev-fp8.safetensors --steps 10000
"""

import gc
import os
import re
import sys
import json
import math
import time
import random
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import safetensors.torch

logger = logging.getLogger("radiance.train_rudra")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# ── Path setup ─────────────────────────────────────────────────────────────
_SCRIPTS = os.path.abspath(os.path.dirname(__file__))
_RADIANCE = os.path.abspath(os.path.join(_SCRIPTS, ".."))
_RUDRA_PKG = os.path.join(_RADIANCE, "rudra")
if _RADIANCE not in sys.path:
    sys.path.insert(0, _RADIANCE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
if _RUDRA_PKG not in sys.path:
    sys.path.insert(0, _RUDRA_PKG)

import importlib
_model_map = importlib.util.spec_from_file_location(
    "model_map", os.path.join(_RADIANCE, "config", "model_map.py")
)
_mm = importlib.util.module_from_spec(_model_map)
_model_map.loader.exec_module(_mm)
MODEL_VAE_CONFIG = _mm.MODEL_VAE_CONFIG
resolve_model_vae_config = _mm.resolve_model_vae_config

from rudra.config import RUDRAConfig, RUDRA_MODEL_CONFIGS, FORMAT_TO_ID, FORMAT_DIM, make_rudra_config
from rudra.normalization import normalize_to_scene_linear, encode_scene_linear_to_format
from rudra.descriptor import RUDRADescriptor, format_onehot
from rudra.spatial_descriptor import RUDRASpatialDescriptor
from rudra.encoder import RUDRAProjection
from rudra.decoder import RUDRADecoder, RUDRAFullDecoder
from rudra.adapter import RUDRALoRALinear, inject_rudra_lora, rudra_gate_regularization_loss
from rudra.cross_attention import RUDRACrossAttentionProjection, RUDRACrossAttentionInjector, inject_rudra_cross_attention
from rudra.pipeline import RUDRAPipeline, PipelineMode
from rudra.losses import rudra_reconstruction_loss
from rudra.train_modes import LossWeights, scheduled_loss_weights
from rudra.metrics import validation_metrics

from dataset_hdr import HDRPairDataset, load_vae_standalone
from train_turbo_decoder import EMA


# ─── Log curve registry for random format simulation ────────────────────────

LOG_CURVES = {
    "ARRI LogC4":           "logc4",
    "ARRI LogC3":           "logc3",
    "Sony S-Log3":          "slog3",
    "Panasonic V-Log":      "vlog",
    "RED Log3G10":          "log3g10",
    "DaVinci Intermediate": "davinci",
    "PQ (ST.2084)":         "pq",
    "HLG (BT.2100)":        "hlg",
    "sRGB Gamma":           "srgb",
}

FORMAT_TO_CURVE = {
    "logc4":  "logc4",
    "logc3":  "logc3",
    "slog3":  "slog3",
    "vlog":   "vlog",
    "log3g10": "log3g10",
    "davinci": "davinci",
    "pq":     "pq",
    "hlg":    "hlg",
    "srgb":   "srgb",
}

CURVE_TO_FORMAT_ID = {
    "logc4":  FORMAT_TO_ID["logc4"],
    "logc3":  FORMAT_TO_ID["logc3"],
    "slog3":  FORMAT_TO_ID["slog3"],
    "vlog":   FORMAT_TO_ID["vlog"],
    "log3g10": FORMAT_TO_ID["log3g10"],
    "davinci": FORMAT_TO_ID["logc3"],   # closest match
    "pq":     FORMAT_TO_ID["pq"],
    "hlg":    FORMAT_TO_ID["hlg"],
    "srgb":   FORMAT_TO_ID["sdr"],
}

# ─── Helper for Dynamic Backbone Training Imports ───────────────────────────

def load_yaml_overrides(path: str) -> dict:
    """Map a configs/*.yaml file onto train_rudra CLI argument names.

    Wires the previously-dead YAML configs into the trainer (review §5). CLI
    flags still win because these are applied via ``parser.set_defaults``.
    """
    import yaml  # local import; only needed when --config is used

    with open(path, "r") as f:
        y = yaml.safe_load(f) or {}

    _MODE = {
        "decoder_only": "decoder", "rudra_lite": "decoder",
        "lora_only": "lora", "rudra_full_dre": "dre", "rudra_full_cross_attn": "dre",
    }
    _COLOR = {"acescg": "acescg", "rec2020": "rec2020", "rec709": "rec709"}
    _ODOMAIN = {
        "scene_linear": "scene_linear_positive",
        "scene_linear_positive": "scene_linear_positive",
        "scene_linear_signed": "scene_linear_signed",
        "log": "log", "raw": "raw",
    }

    out: dict = {}
    if "model_type" in y:       out["model_type"] = y["model_type"]
    if "batch_size" in y:       out["batch_size"] = int(y["batch_size"])
    if "learning_rate" in y:    out["lr"] = float(y["learning_rate"])
    if "max_steps" in y:        out["steps"] = int(y["max_steps"])
    if "validate_every" in y:   out["eval_every"] = int(y["validate_every"])
    if "mode" in y and y["mode"] in _MODE:
        out["stage"] = _MODE[y["mode"]]
    outputs = y.get("outputs", {}) or {}
    if "color_space" in outputs:
        out["color_space"] = _COLOR.get(str(outputs["color_space"]).lower(), "rec2020")
    if "output_domain" in outputs:
        out["output_domain"] = _ODOMAIN.get(str(outputs["output_domain"]).lower(), "scene_linear_positive")
    return out


def _compute_dtype(model):
    """Activation/compute dtype for a model. fp8 weights have no autograd matmul,
    so inputs must flow in bf16 even though the frozen weights are fp8."""
    try:
        pdt = next(model.parameters()).dtype
    except StopIteration:
        return torch.float32
    if pdt in (torch.float8_e4m3fn, torch.float8_e5m2):
        return torch.bfloat16
    return pdt


def _select_text_embed(batch, model_type, batch_size, device, get_null_embed, text_dropout):
    """Return a text-conditioning embedding for backbone training.

    Uses a real caption embedding when the dataset provides one (key
    ``text_embed``), applying classifier-free-guidance style dropout to the null
    embedding with probability ``text_dropout``. Falls back to null-only when no
    captions are available (review §4.3).
    """
    null = get_null_embed(model_type, batch_size, device)
    real = batch.get("text_embed") if isinstance(batch, dict) else None
    if real is None:
        return null
    if random.random() < text_dropout:
        return null
    return real.to(device=device)


def import_backbone_helpers():
    try:
        from train_hdr_lora import load_diffusion_model, LoRAEMA, export_lora_safetensors
        from dataset_hdr_lora import add_training_noise, make_schedule, get_null_embed
        return load_diffusion_model, LoRAEMA, export_lora_safetensors, add_training_noise, make_schedule, get_null_embed
    except ImportError:
        logger.error("Failed to import backbone helpers from train_hdr_lora.py/dataset_hdr_lora.py. Stage 2/3 will fail.")
        raise


# ─── Stage 1: RUDRA Decoder Training ────────────────────────────────────────

def train_decoder_stage(
    pair_dir: str,
    output_dir: str,
    model_type: str = "flux",
    model_size: str = "turbo",
    steps: int = 50_000,
    batch_size: int = 8,
    lr: float = 3e-4,
    dr_dim: int = 64,
    ema_decay: float = 0.999,
    grad_clip: float = 1.0,
    log_every: int = 100,
    eval_every: int = 2000,
    save_every: int = 5000,
    num_workers: int = 4,
    device_str: str = "cuda",
    val_split: float = 0.05,
    patience: int = 8,
    multi_curve: bool = True,
    color_space: str = "rec2020",
    output_domain: str = "scene_linear_positive",
    resume: Optional[str] = None,
) -> str:
    """Train RUDRA decoder (Stage 1) with dynamic format scaling and updated losses."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # ── Pipeline setup (single working color space, review §5) ────────────────
    pipeline = RUDRAPipeline.from_model_type(
        model_type=model_type,
        mode=PipelineMode.LITE_DECODER,
        decoder_size=model_size,
        dr_proj_dim=dr_dim,
        color_space=color_space,
        output_domain=output_domain,
    ).to(device)
    logger.info(f"Working color space: {pipeline.config.color_space} | output_domain: {pipeline.config.output_domain}")

    trainable_params_count = pipeline.freeze_for_training(PipelineMode.LITE_DECODER)
    logger.info(f"RUDRAPipeline Stage 1 Decoder initialized: Mode={pipeline.mode}")
    logger.info(f"Trainable parameters: {trainable_params_count / 1e6:.2f}M")

    # ── Dataset ─────────────────────────────────────────────────────────────
    full_dataset = HDRPairDataset(pair_dir=pair_dir, augment=True)
    n_total = len(full_dataset)

    if val_split > 0.0:
        from torch.utils.data import DataLoader
        train_pairs, val_pairs = full_dataset.split_by_scene(val_split=val_split, seed=42)
        train_ds = HDRPairDataset(pair_dir=pair_dir, augment=True, pairs=train_pairs)
        val_ds = HDRPairDataset(pair_dir=pair_dir, augment=False, pairs=val_pairs)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, drop_last=False)
        logger.info(f"Train dataset: {len(train_ds)} samples. Val dataset: {len(val_ds)} samples.")
    else:
        from torch.utils.data import DataLoader
        train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, drop_last=True)
        val_loader = None
        logger.info(f"Dataset: {n_total} pairs, {len(train_loader)} batches/epoch")

    # ── Optimiser ───────────────────────────────────────────────────────────
    trainable_params = [p for p in pipeline.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.05)

    # ── EMA ────────────────────────────────────────────────────────────────
    ema = EMA(pipeline.decoder, decay=ema_decay)
    ema_proj = EMA(pipeline.projection, decay=ema_decay)

    # ── Resume ─────────────────────────────────────────────────────────────
    start_step = 0
    if resume and os.path.exists(resume):
        if str(resume).lower().endswith(".safetensors"):
            sd = safetensors.torch.load_file(resume, device=str(device))
            decoder_sd = {k: v for k, v in sd.items() if not k.startswith("projection.")}
            projection_sd = {k.replace("projection.", ""): v for k, v in sd.items() if k.startswith("projection.")}
            if decoder_sd:
                pipeline.decoder.load_state_dict(decoder_sd, strict=False)
                ema.shadow = {k: v.detach().clone().to(device) for k, v in pipeline.decoder.state_dict().items()}
            if projection_sd:
                pipeline.projection.load_state_dict(projection_sd, strict=False)
                ema_proj.shadow = {k: v.detach().clone().to(device) for k, v in pipeline.projection.state_dict().items()}
            # Recover the step count from the filename (…_step000000.safetensors)
            # so a resumed cosine schedule / early-stopping continues correctly
            # instead of restarting from 0 (review §3.5).
            _m = re.search(r"step0*?(\d+)", os.path.basename(resume))
            start_step = int(_m.group(1)) if _m else 0
            logger.info(f"Loaded Stage 1 weights from safetensors: {resume} (resuming at step {start_step})")
        else:
            ckpt = torch.load(resume, map_location=device, weights_only=False)
            pipeline.decoder.load_state_dict(ckpt["decoder"])
            pipeline.projection.load_state_dict(ckpt["projection"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            if "ema_decoder" in ckpt:
                ema.shadow = {k: v.to(device) for k, v in ckpt["ema_decoder"].items()}
            if "ema_projection" in ckpt:
                ema_proj.shadow = {k: v.to(device) for k, v in ckpt["ema_projection"].items()}
            start_step = ckpt.get("step", 0)
            logger.info(f"Resumed Stage 1 from step {start_step}: {resume}")

    CURVE_KEYS = list(CURVE_TO_FORMAT_ID.keys())

    # The dataset stores targets log-coded with the model's canonical curve.
    # Decoding must use *that* curve, not a random one, or the scene-linear
    # ground truth is wrong (review §3.1/§3.2).
    _vae_cfg = resolve_model_vae_config(model_type) or {}
    _true_curve_name = _vae_cfg.get("log_curve", "ARRI LogC4")
    _true_curve_key = LOG_CURVES.get(_true_curve_name, "logc4")
    true_format_id = CURVE_TO_FORMAT_ID.get(_true_curve_key, FORMAT_TO_ID["logc4"])
    logger.info(f"Target log curve: {_true_curve_name} -> format_id={true_format_id}")

    pipeline.decoder.train()
    pipeline.projection.train()
    pipeline.descriptor.eval()

    data_iter = iter(train_loader)
    step = start_step
    t0 = time.time()
    
    # Running loss components
    running = {"total": 0.0, "l1": 0.0, "highlight": 0.0, "chromaticity": 0.0, "exposure": 0.0}
    log_path = os.path.join(output_dir, "rudra_decoder_log.jsonl")

    best_psnr = -1.0
    best_ema_path = ""
    best_step = 0
    stall_count = 0

    pbar = tqdm(total=steps, initial=start_step, desc="RUDRA Decoder") if _HAS_TQDM else None

    # Write training metadata config
    config_dict = {
        "model_type": model_type, "model_size": model_size, "dr_dim": dr_dim,
        "steps": steps, "batch_size": batch_size, "lr": lr,
        "multi_curve": multi_curve, "stage": "decoder",
        "val_split": val_split, "patience": patience, "ema_decay": ema_decay,
    }
    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    while step < steps:
        try:
            latent_batch, target_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            latent_batch, target_batch = next(data_iter)

        # Pre-encoded VAE latents can contain occasional NaN/inf; scrub before use
        # so a single bad pair can't poison the decoder.
        latent_batch = torch.nan_to_num(latent_batch.to(device), nan=0.0, posinf=1e4, neginf=-1e4)
        target_batch = target_batch.to(device)  # (B, H, W, 3) log-coded target

        optimizer.zero_grad()

        # Target image representation in B, C, H, W — log-coded in the model's
        # canonical curve.
        target_bchw = target_batch.permute(0, 3, 1, 2)

        # ── Scene-linear ground truth (decode once, with the TRUE curve) ──────
        true_format_ids = torch.full((latent_batch.shape[0],), true_format_id,
                                     dtype=torch.long, device=device)
        target_linear = normalize_to_scene_linear(
            target_bchw, true_format_ids, y_max=pipeline.config.y_max_nits
        )

        # ── Descriptor source view + matching format id ───────────────────────
        # multi_curve simulates that the conditioning source could arrive in any
        # log format: re-encode the linear target into a random curve and tell
        # the descriptor which curve it is, so it linearizes consistently. The
        # reconstruction target stays scene-linear regardless (review §3.2).
        if multi_curve:
            curve_key = CURVE_KEYS[random.randint(0, len(CURVE_KEYS) - 1)]
            sim_format_id = CURVE_TO_FORMAT_ID[curve_key]
            desc_format_ids = torch.full((latent_batch.shape[0],), sim_format_id,
                                         dtype=torch.long, device=device)
            src_image = encode_scene_linear_to_format(target_linear, desc_format_ids)
        else:
            curve_key = _true_curve_key
            desc_format_ids = true_format_ids
            src_image = target_bchw

        # ── Forward pass ──────────────────────────────────────────────────────
        # The decoder's output_domain is already scene-linear (softplus), so the
        # prediction is compared to target_linear directly. Do NOT re-run it
        # through normalize_to_scene_linear — that previously treated a linear
        # tensor as a [0,1] log code and clamped away every highlight, which is
        # why `highlight` logged as 0.0 and `exposure` as ~15 EV (review §3.2).
        pred_bchw, cond = pipeline(latent_batch, src_image, desc_format_ids)

        if pred_bchw.shape[2:] != target_bchw.shape[2:]:
            pred_bchw = F.interpolate(pred_bchw, size=target_bchw.shape[2:],
                                       mode="bilinear", align_corners=False)

        # Sanitize the prediction before the loss. The old path did this implicitly
        # via normalize_to_scene_linear's nan_to_num; with the domain fix we must do
        # it explicitly, or NaN/inf in the pre-encoded latents poisons the loss
        # (chromaticity stayed finite only because the descriptor sanitizes its input).
        pred_linear = torch.nan_to_num(pred_bchw, nan=0.0, posinf=1e4, neginf=0.0).clamp(min=0.0)

        # ── Staged HDR schedule and Losses ────────────────────────────────────
        weights = scheduled_loss_weights(step)
        loss_dict = rudra_reconstruction_loss(
            pred=pred_linear,
            target=target_linear,
            alpha_highlight=weights.highlight,
            beta_color=weights.chromaticity,
            gamma_perceptual=weights.perceptual,
            delta_exposure=weights.exposure,
            eta_align=weights.align,
            color_space=pipeline.config.color_space,
        )

        # Guard: never let a non-finite loss reach the weights (one bad batch
        # would otherwise poison the decoder permanently via NaN gradients).
        if not torch.isfinite(loss_dict["total"]):
            logger.warning(f"step {step}: non-finite loss, skipping update.")
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if pbar:
                pbar.update(1)
            continue

        loss_dict["total"].backward()

        # Scrub non-finite gradients before clipping: a single inf grad makes
        # clip_grad_norm_ compute inf*0 = NaN and poison the weights permanently.
        for p in trainable_params:
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

        optimizer.step()
        scheduler.step()
        ema.update()
        ema_proj.update()

        for k in running:
            if k in loss_dict:
                running[k] += loss_dict[k].item()
            elif k == "total" and "total" in loss_dict:
                running["total"] += loss_dict["total"].item()

        step += 1
        if pbar:
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss_dict['total'].item():.4f}")

        # ── Logging ─────────────────────────────────────────────────────────
        if step % log_every == 0:
            avg = {k: v / log_every for k, v in running.items()}
            elapsed = time.time() - t0
            sps = log_every / elapsed
            lr_now = scheduler.get_last_lr()[0]
            
            log_entry = {
                "step": step, "loss": round(avg["total"], 5),
                "l1": round(avg["l1"], 5), "highlight": round(avg["highlight"], 5),
                "chromaticity": round(avg["chromaticity"], 5), "exposure": round(avg["exposure"], 5),
                "lr": round(lr_now, 7), "sps": round(sps, 2),
                "format": curve_key if multi_curve else "default",
            }
            logger.info(
                f"step {step:>6d}/{steps} | loss={avg['total']:.4f} "
                f"l1={avg['l1']:.4f} hl={avg['highlight']:.4f} color={avg['chromaticity']:.4f} | "
                f"lr={lr_now:.2e} | {sps:.1f} steps/s"
            )
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            running = {k: 0.0 for k in running}
            t0 = time.time()

        # ── Evaluation ─────────────────────────────────────────────────────
        if eval_every and eval_every > 0 and step % eval_every == 0:
            pipeline.decoder.eval()
            pipeline.projection.eval()
            orig_params = {n: p.data.clone() for n, p in pipeline.decoder.named_parameters()}
            orig_proj = {n: p.data.clone() for n, p in pipeline.projection.named_parameters()}
            ema.apply_shadow(pipeline.decoder)
            ema_proj.apply_shadow(pipeline.projection)

            with torch.no_grad():
                try:
                    if val_loader is not None:
                        # Bound eval memory: cap at a couple of batches instead of
                        # concatenating the entire val set (review §5 OOM risk).
                        max_eval_batches = 2
                        eval_l, eval_t = [], []
                        for bi, (vl, vt) in enumerate(val_loader):
                            eval_l.append(vl)
                            eval_t.append(vt)
                            if bi + 1 >= max_eval_batches:
                                break
                        eval_latent = torch.cat(eval_l, dim=0).to(device)
                        eval_target = torch.cat(eval_t, dim=0).to(device)
                    else:
                        eval_latent, eval_target = next(iter(train_loader))
                        eval_latent = eval_latent.to(device)
                        eval_target = eval_target.to(device)

                    ef_id = torch.full((eval_latent.shape[0],), true_format_id,
                                       dtype=torch.long, device=device)

                    eval_target_bchw = eval_target.permute(0, 3, 1, 2)
                    # Decode target with the true curve; prediction is already scene-linear.
                    eval_target_linear = normalize_to_scene_linear(
                        eval_target_bchw, ef_id, y_max=pipeline.config.y_max_nits
                    )
                    eval_pred_bchw, _ = pipeline(eval_latent, eval_target_bchw, ef_id)

                    if eval_pred_bchw.shape[2:] != eval_target_bchw.shape[2:]:
                        eval_pred_bchw = F.interpolate(eval_pred_bchw, size=eval_target_bchw.shape[2:],
                                                       mode="bilinear", align_corners=False)

                    eval_pred_linear = torch.nan_to_num(eval_pred_bchw, nan=0.0, posinf=1e4, neginf=0.0).clamp(min=0.0)

                    # Compute metrics on CPU so the heavy perceptual nets (LPIPS-VGG,
                    # ColorVideoVDP) do not permanently occupy GPU memory and OOM the
                    # next training step. Decoder eval is small, so CPU is fine.
                    metrics = validation_metrics(
                        eval_pred_linear.detach().cpu().float(),
                        eval_target_linear.detach().cpu().float(),
                        color_space=pipeline.config.color_space,
                    )
                    val_psnr = metrics["psnr_tm"]
                    val_ssim = metrics["ssim_tm"]
                    val_de = metrics["delta_e_2000"]
                    val_hra = metrics["hra"]

                    logger.info(
                        f"  [EVAL] step {step} — PSNR_tm={val_psnr:.2f}dB  SSIM_tm={val_ssim:.4f}  "
                        f"HRA={val_hra:.4f}  DeltaE={val_de:.2f}"
                    )
                    
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "step": step, "eval": True,
                            **metrics
                        }) + "\n")

                    if val_psnr > best_psnr:
                        best_psnr = val_psnr
                        best_step = step
                        best_ema_path = os.path.join(
                            output_dir, f"rudra_{model_size}_decoder_ema_best.safetensors"
                        )
                        ema_weights = {k: v.cpu().contiguous() for k, v in ema.shadow.items()}
                        proj_weights = {f"projection.{k}": v.cpu().contiguous()
                                       for k, v in ema_proj.shadow.items()}
                        safetensors.torch.save_file({**ema_weights, **proj_weights}, best_ema_path)
                        logger.info(f"  ★ New best PSNR: {best_psnr:.2f} dB (step {step})")
                        stall_count = 0
                    else:
                        stall_count += 1

                except Exception as e:
                    # Record eval failures in the run log too — a silently-failing
                    # eval is why the previous run produced zero eval lines.
                    logger.warning(f"Eval failed: {e}", exc_info=True)
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"step": step, "eval": True, "error": str(e)}) + "\n")

            ema.restore(pipeline.decoder, orig_params)
            ema_proj.restore(pipeline.projection, orig_proj)
            pipeline.decoder.train()
            pipeline.projection.train()

            # Release eval tensors / param clones and reclaim GPU memory so the
            # eval spike can't OOM the next training step.
            orig_params = orig_proj = None
            try:
                del eval_latent, eval_target, eval_target_bchw
                del eval_target_linear, eval_pred_bchw, eval_pred_linear
            except NameError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if patience > 0 and stall_count >= patience:
                logger.info(f"Early stopping at step {step} — no improvement for {patience} evals.")
                break

        # ── Checkpoint saving ───────────────────────────────────────────────
        if step % save_every == 0 or step == steps:
            ckpt_path = os.path.join(output_dir, f"rudra_{model_size}_decoder_step{step:06d}.pth")
            torch.save({
                "step": step,
                "best_step": best_step,
                "decoder": pipeline.decoder.state_dict(),
                "projection": pipeline.projection.state_dict(),
                "ema_decoder":    {k: v.detach().cpu() for k, v in ema.shadow.items()},
                "ema_projection": {k: v.detach().cpu() for k, v in ema_proj.shadow.items()},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, ckpt_path)
            logger.info(f"  Checkpoint saved: {ckpt_path}")

    if pbar:
        pbar.close()

    logger.info(f"RUDRA decoder training complete — {steps} steps.")
    final_ema_path = best_ema_path if best_ema_path else os.path.join(
        output_dir, f"rudra_{model_size}_decoder_ema_step{steps:06d}.safetensors"
    )
    if not os.path.exists(final_ema_path):
        ema_weights = {k: v.cpu().contiguous() for k, v in ema.shadow.items()}
        proj_weights = {f"projection.{k}": v.cpu().contiguous()
                        for k, v in ema_proj.shadow.items()}
        safetensors.torch.save_file({**ema_weights, **proj_weights}, final_ema_path)
        logger.info(f"Final EMA safetensors saved: {final_ema_path}")
    return final_ema_path


# ─── Stage 2: RUDRA LoRA Backbone Training ──────────────────────────────────

def train_lora_stage(
    pair_dir: str,
    output_dir: str,
    model_path: str,
    cache_dir: str = "",
    model_type: str = "flux",
    rank: int = 16,
    alpha: float = 1.0,
    steps: int = 10_000,
    batch_size: int = 2,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    ema_decay: float = 0.999,
    highlight_weight: float = 0.5,
    log_every: int = 50,
    save_every: int = 100,
    eval_every: int = 500,
    num_workers: int = 4,
    device_str: str = "cuda",
    text_dropout: float = 0.1,
    lora_include_mlp: bool = False,
    resume: Optional[str] = None,
) -> str:
    """Train dynamic-range gated LoRA adapters (Stage 2) with frozen backbone."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # 1. Load backbone helpers
    load_diffusion_model, LoRAEMA, export_lora_safetensors, add_training_noise, make_schedule, get_null_embed = import_backbone_helpers()

    # 2. Load diffusion model backbone
    diffusion_model = load_diffusion_model(model_path, model_type, device_str)

    # 3. Setup RUDRAPipeline in LITE_LORA mode
    # Safer default for backbone text/context embedding dimension.
    # Do not infer this from the first parameter shape: for Wan/Flux it may be a Conv weight.
    text_embed_dim = 768
    pipeline = RUDRAPipeline.from_model_type(
        model_type=model_type,
        mode=PipelineMode.LITE_LORA,
        text_embed_dim=text_embed_dim,
        dr_proj_dim=64,
    ).to(device)

    pipeline.freeze_for_training(PipelineMode.LITE_LORA)

    # 4. Inject RUDRALoRA layers into the backbone
    lora_layers, trainable_lora_params = inject_rudra_lora(
        diffusion_model,
        rank=rank,
        alpha=alpha,
        dr_proj_dim=pipeline.config.dr_proj_dim,
        gate_init=pipeline.config.lora_gate_init,
        include_mlp=lora_include_mlp,
    )
    logger.info(f"RUDRA-Lite LoRA: {len(lora_layers)} layers injected. Trainable params: {trainable_lora_params/1e6:.2f}M")

    # 5. Define trainable parameters (LoRA layers + Projection layer)
    trainable_params = [p for p in diffusion_model.parameters() if p.requires_grad] + list(pipeline.projection.parameters())
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.05)

    # EMA trackers
    ema = LoRAEMA({k: v for k, v in lora_layers}, decay=ema_decay)
    ema_proj = EMA(pipeline.projection, decay=ema_decay)

    # 6. Load Datasets
    if cache_dir and os.path.isdir(cache_dir):
        from dataset_hdr_lora import HDRLoRADataset
        dataset = HDRLoRADataset(exr_dirs=[], vae=None, model_name=model_type, cache_dir=cache_dir, augment=True)
    else:
        dataset = HDRPairDataset(pair_dir=pair_dir, augment=True)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    schedule = make_schedule(model_type)

    start_step = 0
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device)
        pipeline.projection.load_state_dict(ckpt["projection"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        ema_proj.shadow = {k: v.to(device) for k, v in ckpt["ema_projection"].items()}
        
        lora_safetensors = resume.replace(".pth", ".safetensors")
        if os.path.exists(lora_safetensors):
            from train_hdr_lora import load_lora_safetensors
            load_lora_safetensors(diffusion_model, lora_safetensors)
        start_step = ckpt.get("step", 0)
        logger.info(f"Resumed Stage 2 from step {start_step}")

    diffusion_model.train()
    pipeline.descriptor.eval()
    pipeline.projection.train()

    data_iter = iter(loader)
    step = start_step
    running = {"loss": 0.0, "base_mse": 0.0, "gate_reg": 0.0}
    t0 = time.time()
    log_path = os.path.join(output_dir, "rudra_stage2_log.jsonl")

    pbar = tqdm(total=steps, initial=start_step, desc="RUDRA Stage 2") if _HAS_TQDM else None

    while step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        if isinstance(batch, list) and len(batch) == 2:
            latent, target = batch
            clean_latent = latent.to(device)
            # Compute bright_ratio on CPU to avoid holding the full-res image on GPU
            # during the expensive model forward pass.
            target_cpu = target if target.device.type == "cpu" else target.cpu()
            # target shape: (B, H, W, C) — permute to (B, C, H, W) on CPU
            target_bchw_cpu = target_cpu.permute(0, 3, 1, 2)
            bright_ratio = (
                (target_bchw_cpu.mean(dim=1, keepdim=True) > pipeline.config.highlight_knee)
                .float()
                .mean(dim=[2, 3])
                .to(device)
            )
            del target_cpu, target_bchw_cpu  # keep off GPU
            has_bright_ratio = True
        elif "dr_raw" in batch:
            # Cache carries a precomputed descriptor — use it directly.
            clean_latent = batch["clean_latent"].to(device)
            bright_ratio = None
            has_bright_ratio = False
        else:
            # The cached-latent path stores only clean_latent, with no image or
            # descriptor. The old code substituted an all-zero image here, which
            # makes the dynamic-range descriptor (and gate target) degenerate —
            # silently removing the conditioning that is the whole point of RUDRA
            # (review §3.4). Fail loudly instead of training on garbage.
            raise RuntimeError(
                "Stage 2 cache_dir provides clean_latent but no paired image or "
                "cached DR descriptor, so dynamic-range conditioning would be "
                "all-zero. Use --pair_dir (HDRPairDataset) for DR-gated LoRA "
                "training, or extend the latent cache to store 'dr_raw'."
            )

        # 1. Compute dynamic range projection
        format_ids = torch.full((clean_latent.shape[0],), FORMAT_TO_ID["logc4"], dtype=torch.long, device=device)
        if has_bright_ratio:
            # Re-derive dr_proj from the clean latent (image already freed from GPU)
            # For gate conditioning, use a zero descriptor when no image is on GPU;
            # bright_ratio is kept for gate regularization separately.
            fmt_oh = pipeline._format_onehot(format_ids, clean_latent.shape[0], device)
            # Use zero dr_raw as a proxy when image is not available on GPU.
            # The gate_reg loss is driven by bright_ratio computed on CPU above.
            dr_raw_proxy = torch.zeros(clean_latent.shape[0], getattr(pipeline.config, "dr_raw_dim", 26), device=device)
            dr_proj = pipeline.projection(dr_raw_proxy, fmt_oh)
        else:
            fmt_oh = pipeline._format_onehot(format_ids, clean_latent.shape[0], device)
            dr_proj = pipeline.projection(batch["dr_raw"].to(device), fmt_oh)

        # 2. Inject dr_proj into LoRA layers dynamically
        for name, lora in lora_layers:
            lora.current_dr_proj = dr_proj

        # 3. Add noise target.
        # Use real captions when the dataset provides them, with CFG-style
        # dropout to null; fall back to null-only otherwise (review §4.3). The
        # current pair/cache datasets carry no captions, so this stays null-only
        # until a caption-bearing dataset is wired in — but the adapter no longer
        # hard-codes the null-only assumption.
        text_embed = _select_text_embed(batch, model_type, clean_latent.shape[0], device,
                                        get_null_embed, text_dropout)
        noisy_batch = add_training_noise(
            {"clean_latent": clean_latent, "text_embed": text_embed},
            schedule, device
        )

        _cdt = _compute_dtype(diffusion_model)
        noisy = noisy_batch["noisy_latent"].to(_cdt)
        t_val = noisy_batch["timestep"].to(_cdt)
        t_emb = noisy_batch["text_embed"].to(_cdt)
        # noise_target = velocity v_t (flow) or epsilon (DDPM) — used for direct model calls.
        # clean_latent = x0 — used when apply_model returns a denoised prediction.
        target_velocity = noisy_batch["noise_target"].to(_cdt)
        target_x0 = noisy_batch["clean_latent"].to(_cdt)

        # Free clean_latent from GPU now that we have what we need
        del clean_latent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        optimizer.zero_grad()

        # Detect model architecture from the ComfyUI BaseModel wrapper class name.
        # Raw transformer class names: Flux -> "Flux", Wan -> "WanModel", LTX -> "LTXVModel".
        # ComfyUI BaseModel class names: Flux -> "Flux"/"Flux2", Wan -> "WAN21", LTX -> "LTXV"/"LTXAV".
        # We prefer the comfy_model class name (more reliable) when available.
        _comfy_cls = type(getattr(diffusion_model, "comfy_model", None)).__name__ if hasattr(diffusion_model, "comfy_model") else ""
        _dm_cls = type(diffusion_model).__name__
        # Models that use (x, timestep, context) positional calling convention
        _CONTEXT_MODELS = {
            # raw transformer class names
            "Flux", "WanModel", "LTXVModel", "LTXAVModel",
            # ComfyUI BaseModel subclass names
            "Flux2", "LongCatImage", "WAN21", "LTXV", "LTXAV",
            "HunyuanVideo", "HunyuanVideoI2V", "HunyuanVideoSkyreelsI2V",
        }
        _uses_context = (_dm_cls in _CONTEXT_MODELS) or (_comfy_cls in _CONTEXT_MODELS)

        # IMPORTANT: ComfyUI's apply_model() runs calculate_denoised() internally,
        # so its output is a *denoised x0 prediction*, not a velocity/noise.
        # The raw diffusion_model forward (bypassing apply_model) returns raw velocity v_t.
        # We track which path was taken to use the correct loss target.
        used_apply_model = False
        if hasattr(diffusion_model, "comfy_model"):
            try:
                pred = diffusion_model.comfy_model.apply_model(noisy, t_val, c_crossattn=t_emb)
                used_apply_model = True  # pred is x0_pred — compare against clean x0
            except Exception as e:
                logger.warning(f"[Train] Stage 2 comfy_model.apply_model failed: {e}. Falling back to direct call.")
                # diffusion_model IS the raw transformer (e.g. comfy.ldm.flux.model.Flux).
                # Direct call returns raw velocity/noise — compare against noise_target.
                try:
                    pred = diffusion_model(noisy, t_val, t_emb)
                    if hasattr(pred, "sample"):
                        pred = pred.sample
                except Exception as e2:
                    logger.warning(f"[Train] Stage 2 direct call failed: {e2}. Last resort.")
                    pred = diffusion_model(noisy, t_val, t_emb)
                    if hasattr(pred, "sample"):
                        pred = pred.sample
        else:
            try:
                if _uses_context:
                    pred = diffusion_model(noisy, t_val, t_emb)
                else:
                    pred = diffusion_model(noisy, timestep=t_val, encoder_hidden_states=t_emb)
                if hasattr(pred, "sample"):
                    pred = pred.sample
            except Exception:
                pred = diffusion_model(noisy, t_val, t_emb)
                if hasattr(pred, "sample"):
                    pred = pred.sample

        # Choose the correct target based on what the model returned:
        # - apply_model -> denoised x0 -> compare against target_x0 (clean latent)
        # - raw forward -> velocity v_t -> compare against target_velocity
        if used_apply_model:
            training_target = target_x0
        else:
            training_target = target_velocity

        base_mse = F.mse_loss(pred.to(torch.float32), training_target.to(torch.float32))

        # 4. Gate regularization loss to align gating with image dynamic range
        if has_bright_ratio and bright_ratio is not None:
            gate_reg = rudra_gate_regularization_loss([l for name, l in lora_layers], bright_ratio, weight=0.1)
        else:
            gate_reg = torch.zeros((), device=device)

        loss = base_mse + gate_reg
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

        optimizer.step()
        scheduler.step()

        ema.update({k: v for k, v in lora_layers})
        ema_proj.update()

        # Cleanup current_dr_proj context
        for name, lora in lora_layers:
            lora.current_dr_proj = None

        running["loss"] += loss.item()
        running["base_mse"] += base_mse.item()
        running["gate_reg"] += gate_reg.item()

        step += 1
        if pbar:
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.5f}")


        if step % log_every == 0:
            avg = {k: v / log_every for k, v in running.items()}
            logger.info(f"step {step}/{steps} | loss={avg['loss']:.5f} base={avg['base_mse']:.5f} gate_reg={avg['gate_reg']:.5f}")
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, **avg}) + "\n")
            running = {k: 0.0 for k in running}

        if step % save_every == 0 or step == steps:
            ckpt_path = os.path.join(output_dir, f"rudra_stage2_step{step:06d}.pth")
            torch.save({
                "step": step,
                "projection": pipeline.projection.state_dict(),
                "ema_projection": {k: v.cpu() for k, v in ema_proj.shadow.items()},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, ckpt_path)
            
            lora_path = os.path.join(output_dir, f"rudra_stage2_lora_step{step:06d}.safetensors")
            export_lora_safetensors(
                {k: v for k, v in lora_layers}, lora_path,
                metadata={"radiance_stage": "lora", "step": str(step), "compression_ratio": str(getattr(pipeline.config, "compression_ratio", 1.0))}
            )

    if pbar:
        pbar.close()
    return os.path.join(output_dir, f"rudra_stage2_step{steps:06d}.pth")


# ─── Stage 3: RUDRA DRE + Cross Attention Training ──────────────────────────

def train_dre_stage(
    pair_dir: str,
    output_dir: str,
    model_path: str,
    cache_dir: str = "",
    model_type: str = "flux",
    steps: int = 10_000,
    batch_size: int = 2,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    ema_decay: float = 0.999,
    log_every: int = 50,
    save_every: int = 100,
    eval_every: int = 500,
    num_workers: int = 4,
    device_str: str = "cuda",
    text_dropout: float = 0.1,
    lambda_init: float = 0.8,
    learnable_lambda: bool = True,
    desc_channels: str = "L,E,H,x,y",
    resume: Optional[str] = None,
) -> str:
    """Train dynamic range encoder (DRE) spatial transformer & cross-attention tokens (Stage 3).

    lambda_init / learnable_lambda: paper §7.3 conditioning-strength sweep — pass a
    fixed λ (learnable_lambda=False) per sweep point.
    desc_channels: paper §7.2 descriptor channel ablation — comma subset of
    L,E,H,x,y; omitted channels are zeroed before the DRE.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # 1. Load backbone helpers
    load_diffusion_model, LoRAEMA, export_lora_safetensors, add_training_noise, make_schedule, get_null_embed = import_backbone_helpers()

    # 2. Load diffusion model backbone
    diffusion_model = load_diffusion_model(model_path, model_type, device_str)

    # 3. Setup RUDRAPipeline in FULL_DRE mode
    # Dynamic text embedding dimension based on architecture family
    if model_type.strip().lower() in ["flux", "wan", "hunyuanvideo", "ltx-video", "sd3"]:
        text_embed_dim = 4096
    elif model_type.strip().lower() == "sdxl":
        text_embed_dim = 2048
    else:
        text_embed_dim = 768
        
    pipeline = RUDRAPipeline.from_model_type(
        model_type=model_type,
        mode=PipelineMode.FULL_DRE,
        text_embed_dim=text_embed_dim,
    ).to(device)

    pipeline.freeze_for_training(PipelineMode.FULL_DRE)

    # 4. Inject cross-attention layers into the backbone (λ per paper §3.5/§7.3;
    # the old code misused highlight_knee as λ-init).
    injectors, trainable_attn_params = inject_rudra_cross_attention(
        diffusion_model,
        text_embed_dim=text_embed_dim,
        lambda_init=lambda_init,
        learnable_lambda=learnable_lambda,
    )

    # Descriptor channel ablation mask (§7.2): [L, E, H, x, y]
    _CH_ORDER = ["L", "E", "H", "x", "y"]
    _sel = {c.strip() for c in desc_channels.split(",") if c.strip()}
    channel_mask = [1.0 if c in _sel else 0.0 for c in _CH_ORDER]
    if all(channel_mask):
        channel_mask = None
    else:
        logger.info(f"Descriptor ablation active: channels={sorted(_sel)} mask={channel_mask}")
    logger.info(f"RUDRA-Full Cross-Attention: {len(injectors)} blocks injected. Trainable params: {trainable_attn_params/1e6:.2f}M")

    # 5. Define trainable parameters (DRE transformer + Cross-attn projection + learnable lambdas)
    trainable_params = (
        list(pipeline.dre.parameters()) + 
        list(pipeline.cross_attn_proj.parameters()) + 
        [p for name, inj in injectors for p in inj.parameters() if p.requires_grad]
    )
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.05)

    # 6. Load Datasets
    if cache_dir and os.path.isdir(cache_dir):
        from dataset_hdr_lora import HDRLoRADataset
        dataset = HDRLoRADataset(exr_dirs=[], vae=None, model_name=model_type, cache_dir=cache_dir, augment=True)
    else:
        dataset = HDRPairDataset(pair_dir=pair_dir, augment=True)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    schedule = make_schedule(model_type)

    start_step = 0
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device)
        pipeline.dre.load_state_dict(ckpt["dre"])
        pipeline.cross_attn_proj.load_state_dict(ckpt["cross_attn_proj"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt.get("step", 0)
        logger.info(f"Resumed Stage 3 from step {start_step}")

    diffusion_model.train()
    pipeline.spatial_descriptor.eval()
    pipeline.dre.train()
    pipeline.cross_attn_proj.train()

    data_iter = iter(loader)
    step = start_step
    running = {"loss": 0.0}
    t0 = time.time()
    log_path = os.path.join(output_dir, "rudra_stage3_log.jsonl")

    pbar = tqdm(total=steps, initial=start_step, desc="RUDRA Stage 3") if _HAS_TQDM else None

    while step < steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        if isinstance(batch, list) and len(batch) == 2:
            latent, target = batch
            clean_latent = latent.to(device)
            target_bchw = target.permute(0, 3, 1, 2).to(device)
        else:
            # Stage 3 needs the spatial descriptor R(x), which requires the
            # source image. The cached-latent path cannot supply it, and the old
            # all-zero substitution made the DRE tokens content-free (review
            # §3.4). Fail loudly rather than train on a black frame.
            raise RuntimeError(
                "Stage 3 DRE training requires paired images for the spatial "
                "descriptor R(x); the cached-latent path provides only "
                "clean_latent. Use --pair_dir (HDRPairDataset)."
            )

        # 1. Compute spatial dynamic range tokens C_R
        format_ids = torch.full((clean_latent.shape[0],), FORMAT_TO_ID["logc4"], dtype=torch.long, device=device)
        cond = pipeline.extract_spatial_conditioning(target_bchw, format_ids, channel_mask=channel_mask)
        c_r = cond.c_r

        # 2. Inject C_R tokens into U-Net Cross-Attention inputs dynamically
        text_embed = _select_text_embed(batch, model_type, clean_latent.shape[0], device,
                                        get_null_embed, text_dropout)
        if len(injectors) > 0:
            for name, inj in injectors:
                inj.current_dr_tokens = c_r
        else:
            # For joint-attention architectures (like Flux), append c_r directly to the text embedding
            text_embed = torch.cat([text_embed, c_r.to(dtype=text_embed.dtype)], dim=1)

        # 3. Add noise target
        noisy_batch = add_training_noise(
            {"clean_latent": clean_latent, "text_embed": text_embed},
            schedule, device
        )

        _cdt = _compute_dtype(diffusion_model)
        noisy = noisy_batch["noisy_latent"].to(_cdt)
        t_val = noisy_batch["timestep"].to(_cdt)
        t_emb = noisy_batch["text_embed"].to(_cdt)
        target_noise = noisy_batch["noise_target"].to(_cdt)

        optimizer.zero_grad()

        # Same model-type detection as Stage 2 — derive from comfy_model class name first.
        _comfy_cls_s3 = type(getattr(diffusion_model, "comfy_model", None)).__name__ if hasattr(diffusion_model, "comfy_model") else ""
        _dm_cls_s3 = type(diffusion_model).__name__
        _CONTEXT_MODELS_S3 = {
            "Flux", "WanModel", "LTXVModel", "LTXAVModel",
            "Flux2", "LongCatImage", "WAN21", "LTXV", "LTXAV",
            "HunyuanVideo", "HunyuanVideoI2V", "HunyuanVideoSkyreelsI2V",
        }
        _uses_context_s3 = (_dm_cls_s3 in _CONTEXT_MODELS_S3) or (_comfy_cls_s3 in _CONTEXT_MODELS_S3)

        if hasattr(diffusion_model, "comfy_model"):
            try:
                pred = diffusion_model.comfy_model.apply_model(noisy, t_val, c_crossattn=t_emb)
            except Exception as e:
                logger.warning(f"[Train] Stage 3 comfy_model.apply_model failed: {e}. Falling back to direct call.")
                try:
                    pred = diffusion_model(noisy, t_val, t_emb)
                    if hasattr(pred, "sample"):
                        pred = pred.sample
                except Exception as e2:
                    logger.warning(f"[Train] Stage 3 direct call failed: {e2}. Last resort.")
                    pred = diffusion_model(noisy, t_val, t_emb)
                    if hasattr(pred, "sample"):
                        pred = pred.sample
        else:
            try:
                if _uses_context_s3:
                    pred = diffusion_model(noisy, t_val, t_emb)
                else:
                    pred = diffusion_model(noisy, timestep=t_val, encoder_hidden_states=t_emb)
                if hasattr(pred, "sample"):
                    pred = pred.sample
            except Exception:
                pred = diffusion_model(noisy, t_val, t_emb)
                if hasattr(pred, "sample"):
                    pred = pred.sample

        loss = F.mse_loss(pred.to(torch.float32), target_noise.to(torch.float32))
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

        optimizer.step()
        scheduler.step()

        # Cleanup current_dr_tokens context
        for name, inj in injectors:
            inj.current_dr_tokens = None

        running["loss"] += loss.item()

        step += 1
        if pbar:
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        if step % log_every == 0:
            avg_loss = running["loss"] / log_every
            logger.info(f"step {step}/{steps} | loss={avg_loss:.5f}")
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "loss": round(avg_loss, 5)}) + "\n")
            running["loss"] = 0.0

        if step % save_every == 0 or step == steps:
            ckpt_path = os.path.join(output_dir, f"rudra_stage3_step{step:06d}.pth")
            lambda_weights = {f"injector.{name}.lambda": inj.lambda_param.data.cpu() for name, inj in injectors}
            torch.save({
                "step": step,
                "dre": pipeline.dre.state_dict(),
                "cross_attn_proj": pipeline.cross_attn_proj.state_dict(),
                "injector_lambdas": lambda_weights,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, ckpt_path)
            logger.info(f"  Stage 3 Checkpoint saved: {ckpt_path}")

    if pbar:
        pbar.close()
    return os.path.join(output_dir, f"rudra_stage3_step{steps:06d}.pth")


# ─── CLI Entrypoint ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RUDRA (Dynamic Range Conditioned Adapters)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pair_dir", default="",
                        help="Directory with .npz paired datasets (latents + targets)")
    parser.add_argument("--output_dir", required=True,
                        help="Output checkpoints directory")
    parser.add_argument("--stage", default="decoder", choices=["decoder", "lora", "dre"],
                        help="Training stage: Stage 1 (decoder), Stage 2 (lora), or Stage 3 (dre)")
    parser.add_argument("--model_path", default="",
                        help="Path to backbone model weights (Required for lora / dre stages)")
    parser.add_argument("--cache_dir", default="",
                        help="Optional cached pre-encoded lora NPZ directory")
    parser.add_argument("--model_type", default="flux",
                        choices=list(MODEL_VAE_CONFIG.keys()),
                        help="Backbone model architecture type")
    parser.add_argument("--model_size", default="turbo", choices=["turbo", "full"],
                        help="Decoder capacity size")
    parser.add_argument("--dr_dim", default=64, type=int,
                        help="Global dynamic-range projection dimension")
    parser.add_argument("--steps", default=50_000, type=int,
                        help="Number of training steps")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Training batch size")
    parser.add_argument("--lr", default=3e-4, type=float,
                        help="Learning rate")
    parser.add_argument("--multi_curve", action="store_true",
                        help="Random log-format augmentation during Stage 1 decoder training")
    parser.add_argument("--ema_decay", default=0.999, type=float)
    parser.add_argument("--highlight_weight", default=0.5, type=float)
    parser.add_argument("--val_split", default=0.05, type=float)
    parser.add_argument("--patience", default=8, type=int)
    parser.add_argument("--text_dropout", default=0.1, type=float,
                        help="CFG-style dropout to null text embedding (Stage 2/3)")
    parser.add_argument("--lora_mlp", action="store_true",
                        help="Also inject DR-gated LoRA into MLP layers (Stage 2 ablation)")
    parser.add_argument("--lambda_init", default=0.8, type=float,
                        help="Stage 3 radiometric conditioning strength λ (paper §3.5)")
    parser.add_argument("--fixed_lambda", action="store_true",
                        help="Stage 3: keep λ fixed (for the §7.3 conditioning-strength sweep)")
    parser.add_argument("--desc_channels", default="L,E,H,x,y",
                        help="Stage 3 descriptor channels to keep (§7.2 ablation), e.g. 'L' or 'L,E'")
    parser.add_argument("--config", default="",
                        help="Optional YAML config (configs/*.yaml) to override defaults")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--color_space", default="rec2020", choices=["rec2020", "rec709", "acescg"],
                        help="Single working color space for descriptor, losses, and metrics")
    parser.add_argument("--output_domain", default="scene_linear_positive",
                        choices=["scene_linear_positive", "scene_linear_signed", "log", "raw"],
                        help="Decoder HDR output parameterization")
    parser.add_argument("--resume", default=None,
                        help="Checkpoint path to resume training from")
    parser.add_argument("--save_every", default=100, type=int,
                        help="Save checkpoint every N steps (default: 100)")

    # Apply YAML overrides as defaults so explicit CLI flags still win (review §5).
    _pre, _ = parser.parse_known_args()
    if getattr(_pre, "config", ""):
        _ov = load_yaml_overrides(_pre.config)
        parser.set_defaults(**_ov)
        logger.info(f"Loaded YAML config {_pre.config}: {_ov}")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    if args.stage == "decoder":
        if not args.pair_dir:
            args.pair_dir = os.path.join(args.output_dir, f"{args.model_type}_pairs")
            if not os.path.isdir(args.pair_dir):
                logger.error(f"No pair directory found at {args.pair_dir}. "
                             f"Please generate pairs or provide --pair_dir explicitly.")
                sys.exit(1)

        logger.info(f"Starting RUDRA Decoder Stage 1 training...")
        final_path = train_decoder_stage(
            pair_dir=args.pair_dir,
            output_dir=os.path.join(args.output_dir, f"{args.model_type}_rudra_stage1"),
            model_type=args.model_type,
            model_size=args.model_size,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            dr_dim=args.dr_dim,
            ema_decay=args.ema_decay,
            device_str=args.device,
            val_split=args.val_split,
            patience=args.patience,
            multi_curve=args.multi_curve,
            color_space=args.color_space,
            output_domain=args.output_domain,
            resume=args.resume,
            save_every=args.save_every,
        )
        logger.info(f"Stage 1 Decoder training complete. Saved to: {final_path}")

    elif args.stage == "lora":
        if not args.model_path:
            logger.error("Stage 2 LoRA training requires a base diffusion --model_path!")
            sys.exit(1)

        logger.info(f"Starting Stage 2 Gated LoRA training...")
        final_path = train_lora_stage(
            pair_dir=args.pair_dir,
            output_dir=os.path.join(args.output_dir, f"{args.model_type}_rudra_stage2"),
            model_path=args.model_path,
            cache_dir=args.cache_dir,
            model_type=args.model_type,
            rank=args.dr_dim // 4 if args.dr_dim >= 16 else 16, # Rank adaptation
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            highlight_weight=args.highlight_weight,
            device_str=args.device,
            text_dropout=args.text_dropout,
            lora_include_mlp=args.lora_mlp,
            resume=args.resume,
            save_every=args.save_every,
        )
        logger.info(f"Stage 2 LoRA training complete. Checkpoint: {final_path}")

    elif args.stage == "dre":
        if not args.model_path:
            logger.error("Stage 3 DRE training requires a base diffusion --model_path!")
            sys.exit(1)

        logger.info(f"Starting Stage 3 Cross-Attention DRE training...")
        final_path = train_dre_stage(
            pair_dir=args.pair_dir,
            output_dir=os.path.join(args.output_dir, f"{args.model_type}_rudra_stage3"),
            model_path=args.model_path,
            cache_dir=args.cache_dir,
            model_type=args.model_type,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device_str=args.device,
            text_dropout=args.text_dropout,
            lambda_init=args.lambda_init,
            learnable_lambda=not args.fixed_lambda,
            desc_channels=args.desc_channels,
            resume=args.resume,
            save_every=args.save_every,
        )
        logger.info(f"Stage 3 DRE training complete. Checkpoint: {final_path}")