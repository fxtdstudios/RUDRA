"""
train_turbo_decoder.py — Radiance TurboDecoder Training
════════════════════════════════════════════════════════════════════════════════

Trains the RadianceTurboDecoder (fast_vae.py) to predict log-coded images
directly from VAE latents, replacing the combination of:
  _denoise_log_highlights() + _soft_log_shoulder()
with a learned network that implicitly handles highlight noise and the
soft shoulder — without explicit engineering patches.

SIGNAL PATH (trained):
    latent z  →  RadianceTurboDecoder  →  log_coded_pred
    log_coded_pred  →  inverse_log (existing pipeline)  →  scene-linear

LOSS FUNCTION:
    Log-space L1 + perceptual weight on highlights.
    Standard pixel MSE in linear space is dominated by the absolute
    magnitude of highlights (a clipped sky at linear 200 vs 198 = 4 units
    of loss). In log space those same pixels differ by a perceptually
    uniform amount. Log-space L1 gives equal weight across the tonal range.

    Loss = mean(|log_coded_pred - log_coded_target|)
         + highlight_weight × mean(|pred - target| where target > knee)

TRAINING RECIPE:
    - Architecture : RadianceTurboDecoder (~2M params for Flux 16ch)
    - Optimizer    : AdamW (lr=3e-4, weight_decay=1e-4)
    - Scheduler    : Cosine annealing with warm restarts
    - EMA          : Exponential moving average of weights (decay=0.999)
    - Gradient clip: 1.0
    - Batch size   : 8 (A100 40GB), 4 (3090/4090)
    - Steps        : ~50k for a useful decoder, 200k for production quality

USAGE:
    python train_turbo_decoder.py \
        --pair_dir   /data/hdr_pairs \
        --output_dir /checkpoints/turbo_decoder \
        --model_type flux \
        --steps      50000 \
        --batch_size 8
"""

import os
import sys
import math
import time
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Optional
from copy import deepcopy

import sys
import os
_RADIANCE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _RADIANCE not in sys.path:
    sys.path.insert(0, _RADIANCE)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import safetensors.torch

logger = logging.getLogger("radiance.train_turbo")
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


# ─────────────────────────────────────────────────────────────────────────────
#  LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

class HDRLogLoss(nn.Module):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    HDR-aware loss in log-coded space.

    Components:
      1. L1 in log space (main signal — perceptually uniform)
      2. MSE in log space (sharpness — penalizes large outliers/noise)
      3. Highlight penalty (extra weight above `knee`)
      4. Structural loss (gradient-based)
    """

    def __init__(
        self,
        highlight_weight: float = 2.0,
        mse_weight: float = 0.5,
        knee: float = 0.96,
        structural_weight: float = 0.1,
    ):
        super().__init__()
        self.highlight_weight = highlight_weight
        self.mse_weight = mse_weight
        self.knee = knee
        self.structural_weight = structural_weight

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # 1. Base L1 (Perceptual Brightness)
        l1 = F.l1_loss(pred, target)

        # 2. Base MSE (Stability & Sharpness)
        mse = F.mse_loss(pred, target)

        # 3. Highlight penalty
        highlight_mask = (target > self.knee).float()
        if highlight_mask.sum() > 0:
            highlight_loss = (
                F.mse_loss(pred * highlight_mask, target * highlight_mask)
                * self.highlight_weight
            )
        else:
            highlight_loss = torch.tensor(0.0, device=pred.device)

        # 4. Structural gradient loss
        if self.structural_weight > 0:
            pred_bchw   = pred.permute(0, 3, 1, 2)
            target_bchw = target.permute(0, 3, 1, 2)
            pred_gx   = pred_bchw[:, :, :, 1:] - pred_bchw[:, :, :, :-1]
            pred_gy   = pred_bchw[:, :, 1:, :] - pred_bchw[:, :, :-1, :]
            tgt_gx    = target_bchw[:, :, :, 1:] - target_bchw[:, :, :, :-1]
            tgt_gy    = target_bchw[:, :, 1:, :] - target_bchw[:, :, :-1, :]
            struct_loss = (
                F.l1_loss(pred_gx, tgt_gx) + F.l1_loss(pred_gy, tgt_gy)
            ) * self.structural_weight
        else:
            struct_loss = torch.tensor(0.0, device=pred.device)

        total = l1 + (self.mse_weight * mse) + highlight_loss + struct_loss

        return {
            "loss":           total,
            "l1":             l1.detach(),
            "mse":            mse.detach(),
            "highlight":      highlight_loss.detach(),
            "structural":     struct_loss.detach(),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  EMA (Exponential Moving Average of weights)
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    Maintains an exponential moving average of a model's parameters.
    Use EMA weights for inference — they are smoother and generalise better.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            self.shadow[name] = (
                self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
            )

    def apply_shadow(self, model: nn.Module):
        """Copy EMA weights into model for evaluation."""
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module, original_params: Dict):
        """Restore original weights after evaluation."""
        for name, param in model.named_parameters():
            param.data.copy_(original_params[name])


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────────────────────

def psnr_log_space(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    PSNR computed in log space.
    Physically meaningful: equal error at all luminance levels.
    """
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(
    pair_dir: str,
    output_dir: str,
    model_type: str = "flux",
    steps: int = 50_000,
    batch_size: int = 8,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    ema_decay: float = 0.9995,
    grad_clip: float = 1.0,
    log_every: int = 100,
    eval_every: int = 2_000,
    save_every: int = 5_000,
    highlight_weight: float = 2.0,
    mse_weight: float = 0.5,
    warmup_frac: float = 0.03,
    knee: float = 0.96,
    num_workers: int = 4,
    resume: Optional[str] = None,
    device_str: str = "cuda",
    model_size: str = "turbo",
    val_split: float = 0.0,
    patience: int = 0,
):
    """
    Main training entry point.

    Args:
        pair_dir:         Directory of pre-generated .npz pairs (dataset_hdr.py)
        output_dir:       Directory to write checkpoints and logs
        model_type:       "flux" (16ch latents) or "sdxl" (4ch latents)
        steps:            Total training steps
        batch_size:       Batch size (per GPU)
        lr:               AdamW learning rate
        weight_decay:     AdamW weight decay
        ema_decay:        EMA decay factor (0.999 = slow, 0.99 = faster)
        grad_clip:        Gradient clipping norm
        log_every:        Log loss every N steps
        eval_every:       Evaluate on val split every N steps
        save_every:       Save checkpoint every N steps
        highlight_weight: Weight of HDR highlight penalty in loss
        knee:             Log-code threshold for highlight penalty
        num_workers:      DataLoader workers
        resume:           Path to checkpoint .pth to resume from
        device_str:       "cuda", "cpu", or "mps"
        val_split:        Fraction of pairs held out for validation (0.0 = no val)
        patience:         Early stop after N evals without PSNR improvement (0 = disabled)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save training config alongside checkpoints
    config = {k: v for k, v in locals().items() if k not in ("resume",)}
    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    try:
        from radiance.config.model_map import resolve_model_vae_config
    except ImportError:
        try:
            from config.model_map import resolve_model_vae_config
        except ImportError:
            resolve_model_vae_config = None

    try:
        from .fast_vae import RadianceTurboDecoder, RadianceFullDecoder
    except (ImportError, ValueError):
        try:
            from fast_vae import RadianceTurboDecoder, RadianceFullDecoder
        except ImportError:
            from hdr.fast_vae import RadianceTurboDecoder, RadianceFullDecoder

    cfg = resolve_model_vae_config(model_type) if resolve_model_vae_config else None
    if cfg:
        latent_channels = cfg.get("latent_channels", 16)
    else:
        latent_channels = 16 if model_type in ("flux", "wan", "sd3", "sd3.5", "hunyuanvideo", "ltx-video", "lumina2", "cogvideox") else 4
    # Upsample stages = log2(VAE spatial factor): 8x -> 3, 16x (Flux.2 Klein) -> 4.
    _spatial = cfg.get("vae_spatial_factor", 8) if cfg else 8
    n_upsample = max(1, int(round(math.log2(_spatial))))
    if model_size == "full":
        model = RadianceFullDecoder(latent_channels=latent_channels, output_channels=3,
                                    n_upsample=n_upsample).to(device)
    else:
        model = RadianceTurboDecoder(latent_channels=latent_channels, output_channels=3,
                                     n_upsample=n_upsample).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: Radiance{model_size.capitalize()}Decoder — {n_params/1e6:.1f}M parameters")

    # ── Dataset ───────────────────────────────────────────────────────────────
    try:
        from .dataset_hdr import make_hdr_dataloader, HDRPairDataset
    except (ImportError, ValueError):
        try:
            from dataset_hdr import make_hdr_dataloader, HDRPairDataset
        except ImportError:
            # Fallback for complex package structures
            sys.path.append(os.path.dirname(__file__))
            from dataset_hdr import make_hdr_dataloader, HDRPairDataset

    full_dataset = HDRPairDataset(pair_dir=pair_dir, augment=True)
    n_total = len(full_dataset)

    val_loader = None
    if val_split > 0.0:
        n_val = max(1, int(n_total * val_split))
        n_train = n_total - n_val
        from torch.utils.data import random_split, DataLoader as _DL
        train_ds, val_ds = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        train_loader = _DL(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True,
        )
        val_loader = _DL(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=False,
        )
        logger.info(f"Train: {len(train_ds)} pairs ({len(train_loader)} batches)  "
                    f"Val: {len(val_ds)} pairs ({len(val_loader)} batches)")
    else:
        train_loader = make_hdr_dataloader(
            pair_dir=pair_dir, batch_size=batch_size,
            num_workers=num_workers, augment=True,
        )
        logger.info(f"Dataset: {n_total} pairs, {len(train_loader)} batches/epoch")

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Linear warmup → cosine decay (to 5% of base lr). The plain cosine without
    # warmup let the LR hit full value immediately, which spiked/oscillated the
    # eval early; warmup settles it into a higher, stabler peak.
    warmup_steps = max(1, int(steps * warmup_frac))

    def _lr_lambda(s: int) -> float:
        if s < warmup_steps:
            return (s + 1) / warmup_steps
        progress = (s - warmup_steps) / max(1, steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = HDRLogLoss(
        highlight_weight=highlight_weight,
        mse_weight=mse_weight,
        knee=knee,
    ).to(device)

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema = EMA(model, decay=ema_decay)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema_shadow"].items()}
        start_step = ckpt.get("step", 0)
        logger.info(f"Resumed from step {start_step}: {resume}")

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    data_iter = iter(train_loader)
    step = start_step
    t0 = time.time()

    # Running averages for logging
    running = {"loss": 0.0, "l1": 0.0, "mse": 0.0, "highlight": 0.0, "structural": 0.0}
    log_path = os.path.join(output_dir, "train_log.jsonl")

    best_psnr = -1.0
    best_ema_path = ""
    best_step = 0
    stall_count = 0

    pbar = tqdm(total=steps, initial=start_step, desc="Training") if _HAS_TQDM else None

    while step < steps:
        # Refresh iterator at epoch boundary
        try:
            latent_batch, target_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            latent_batch, target_batch = next(data_iter)

        latent_batch = latent_batch.to(device)  # (B, C, H//8, W//8)
        target_batch = target_batch.to(device)  # (B, H, W, 3)

        # Forward: decoder outputs (B, 3, H, W) → permute to (B, H, W, 3)
        optimizer.zero_grad()
        pred_bchw = model(latent_batch)
        pred_bhwc = pred_bchw.permute(0, 2, 3, 1)

        # Resize pred to match target if spatial dimensions differ
        if pred_bhwc.shape[1:3] != target_batch.shape[1:3]:
            pred_bchw = F.interpolate(
                pred_bchw,
                size=(target_batch.shape[1], target_batch.shape[2]),
                mode="bilinear",
                align_corners=False,
            )
            pred_bhwc = pred_bchw.permute(0, 2, 3, 1)

        losses = criterion(pred_bhwc, target_batch)
        losses["loss"].backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()
        ema.update()

        # Accumulate for logging
        for k in running:
            running[k] += losses[k].item()

        step += 1
        if pbar:
            pbar.update(1)
            pbar.set_postfix(loss=f"{losses['loss'].item():.4f}")

        # ── Logging ───────────────────────────────────────────────────────────
        if step % log_every == 0:
            avg = {k: v / log_every for k, v in running.items()}
            elapsed = time.time() - t0
            steps_per_sec = log_every / elapsed
            lr_now = scheduler.get_last_lr()[0]

            log_entry = {
                "step": step,
                "loss": round(avg["loss"], 5),
                "l1": round(avg["l1"], 5),
                "mse": round(avg["mse"], 5),
                "highlight": round(avg["highlight"], 5),
                "structural": round(avg["structural"], 5),
                "lr": round(lr_now, 7),
                "steps_per_sec": round(steps_per_sec, 2),
            }
            logger.info(
                f"step {step:>6d}/{steps} | "
                f"loss={avg['loss']:.4f} l1={avg['l1']:.4f} mse={avg['mse']:.4f} "
                f"hl={avg['highlight']:.4f} struct={avg['structural']:.4f} | "
                f"lr={lr_now:.2e} | {steps_per_sec:.1f} steps/s"
            )
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            running = {k: 0.0 for k in running}
            t0 = time.time()

        # ── Evaluation ────────────────────────────────────────────────────────
        if step % eval_every == 0:
            model.eval()
            orig_params = {
                name: param.data.clone()
                for name, param in model.named_parameters()
            }
            ema.apply_shadow(model)

            with torch.no_grad():
                try:
                    if val_loader is not None:
                        eval_latents, eval_targets = [], []
                        for vl, vt in val_loader:
                            eval_latents.append(vl)
                            eval_targets.append(vt)
                        eval_latent = torch.cat(eval_latents, dim=0)
                        eval_target = torch.cat(eval_targets, dim=0)
                    else:
                        eval_latent, eval_target = next(iter(train_loader))

                    eval_latent = eval_latent.to(device)
                    eval_target = eval_target.to(device)
                    eval_pred   = model(eval_latent).permute(0, 2, 3, 1)
                    if eval_pred.shape[1:3] != eval_target.shape[1:3]:
                        eval_pred = F.interpolate(
                            eval_pred.permute(0, 3, 1, 2),
                            size=eval_target.shape[1:3],
                            mode="bilinear",
                            align_corners=False,
                        ).permute(0, 2, 3, 1)
                    val_psnr = psnr_log_space(eval_pred, eval_target)
                    val_loss = criterion(eval_pred, eval_target)["loss"].item()
                    logger.info(
                        f"  [EVAL EMA] step {step} — "
                        f"val_loss={val_loss:.4f}  PSNR_log={val_psnr:.2f}dB"
                    )
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "step": step, "eval": True,
                            "val_loss": round(val_loss, 5),
                            "psnr_log": round(val_psnr, 2),
                        }) + "\n")

                    # Track best PSNR for early stopping
                    if val_psnr > best_psnr:
                        best_psnr = val_psnr
                        best_step = step
                        best_ema_path = os.path.join(
                            output_dir, f"{model_size}_decoder_ema_best.safetensors"
                        )
                        ema_weights = {k: v.cpu().contiguous() for k, v in ema.shadow.items()}
                        safetensors.torch.save_file(ema_weights, best_ema_path)
                        logger.info(f"  ★ New best PSNR: {best_psnr:.2f} dB (step {step})")
                        stall_count = 0
                    else:
                        stall_count += 1
                        logger.info(
                            f"  PSNR stall {stall_count}/{patience if patience else '∞'} — "
                            f"best: {best_psnr:.2f} dB @ step {best_step}"
                        )

                except Exception as e:
                    logger.warning(f"Eval failed: {e}")

            ema.restore(model, orig_params)
            model.train()

            # ── Early stopping ────────────────────────────────────────────────
            if patience > 0 and stall_count >= patience:
                logger.info(
                    f"Early stopping triggered at step {step} — "
                    f"no PSNR improvement for {patience} evals."
                )
                step = steps  # break out of main loop

        # ── Checkpoint ────────────────────────────────────────────────────────
        if step % save_every == 0 or step == steps:
            ckpt_path = os.path.join(output_dir, f"{model_size}_decoder_step{step:06d}.pth")
            torch.save(
                {
                    "step":       step,
                    "model":      model.state_dict(),
                    "ema_shadow": {k: v.cpu() for k, v in ema.shadow.items()},
                    "optimizer":  optimizer.state_dict(),
                    "scheduler":  scheduler.state_dict(),
                    "config":     config,
                },
                ckpt_path,
            )

            # Also save EMA-only weights for inference (smaller file, faster load)
            ema_path = os.path.join(output_dir, f"{model_size}_decoder_ema_step{step:06d}.safetensors")
            ema_weights = {k: v.cpu().contiguous() for k, v in ema.shadow.items()}
            safetensors.torch.save_file(ema_weights, ema_path)

            logger.info(f"  Checkpoint saved: {ckpt_path}")
            logger.info(f"  EMA weights saved: {ema_path}")

    if pbar:
        pbar.close()

    logger.info(f"Training complete — {steps} steps.")
    # Return best EMA path if available, otherwise last step
    final_path = best_ema_path if best_ema_path else (
        os.path.join(output_dir, f"{model_size}_decoder_ema_step{steps:06d}.safetensors")
    )
    return final_path


# ─────────────────────────────────────────────────────────────────────────────
#  FAST_VAE weight loader update — load from checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_trained_turbo_decoder(
    checkpoint_path: str,
    model_type: str = "flux",
    device: str = "cuda",
    use_ema: bool = True,
) -> "RadianceTurboDecoder":
    """
    Load a trained RadianceTurboDecoder from a checkpoint.

    Args:
        checkpoint_path: Path to .pth file (EMA weights or full checkpoint).
        model_type:      "flux" or "sdxl".
        device:          Target device.
        use_ema:         If True and full checkpoint, loads EMA weights.

    Returns the loaded model in eval mode.

    Example:
        decoder = load_trained_turbo_decoder(
            "checkpoints/turbo_decoder_ema_step050000.pth",
            model_type="flux",
        )
        # Register with fast_vae.py:
        import fast_vae
        fast_vae._TRAINED_DECODER = decoder
    """
    try:
        from .hdr.fast_vae import RadianceTurboDecoder
    except ImportError:
        from fast_vae import RadianceTurboDecoder

    try:
        from radiance.config.model_map import resolve_model_vae_config
    except ImportError:
        resolve_model_vae_config = None
    if resolve_model_vae_config:
        cfg = resolve_model_vae_config(model_type)
        latent_channels = cfg.get("latent_channels", 16) if cfg else 16
    else:
        latent_channels = 16 if model_type in ("flux", "wan", "sd3", "sd3.5", "hunyuanvideo", "ltx-video", "lumina2", "cogvideox") else 4
    if "full" in checkpoint_path.lower():
        model = RadianceFullDecoder(latent_channels=latent_channels)
    else:
        model = RadianceTurboDecoder(latent_channels=latent_channels)

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(ckpt, dict) and "ema_shadow" in ckpt and use_ema:
        # Full checkpoint — load EMA weights
        state_dict = ckpt["ema_shadow"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        # Full checkpoint — load model weights
        state_dict = ckpt["model"]
    else:
        # EMA-only checkpoint
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(torch.device(device))

    logger.info(
        f"Loaded Radiance Decoder from {checkpoint_path} "
        f"(ema={use_ema}) on {device}"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Radiance TurboDecoder on HDR pairs"
    )
    parser.add_argument("--pair_dir",    required=True,       help="Directory of .npz pairs")
    parser.add_argument("--output_dir",  required=True,       help="Output directory for checkpoints")
    # Unified model type choices from MODEL_VAE_CONFIG
    try:
        from config.model_map import MODEL_VAE_CONFIG as _CFG
        _model_choices = list(_CFG.keys())
    except ImportError:
        _model_choices = ["flux", "sdxl", "wan", "sd3", "hunyuanvideo", "ltx-video",
                          "sd15", "cogvideox", "lumina2", "pixart", "kolors", "aura_flow"]
    parser.add_argument("--model_type",  default="flux",      choices=_model_choices,
                        help="Model family — determines latent channels and scale factor from MODEL_VAE_CONFIG")
    parser.add_argument("--model_size",  default="turbo",     choices=["turbo", "full"])
    parser.add_argument("--steps",       default=50_000,      type=int)
    parser.add_argument("--batch_size",  default=8,           type=int)
    parser.add_argument("--lr",          default=3e-4,        type=float)
    parser.add_argument("--ema_decay",   default=0.9995,      type=float)
    parser.add_argument("--highlight_weight", default=2.0,    type=float)
    parser.add_argument("--mse_weight",  default=0.5,         type=float,
                        help="Weight of the MSE term — raise to push PSNR (trades vs perceptual sharpness)")
    parser.add_argument("--warmup_frac", default=0.03,        type=float,
                        help="Fraction of steps for linear LR warmup before cosine decay")
    parser.add_argument("--knee",        default=0.96,        type=float,
                        help="Log-code threshold for highlight penalty (match encode profile)")
    parser.add_argument("--num_workers", default=4,           type=int)
    parser.add_argument("--log_every",   default=100,         type=int)
    parser.add_argument("--eval_every",  default=2000,        type=int)
    parser.add_argument("--save_every",  default=5000,        type=int)
    parser.add_argument("--resume",      default=None,        help="Checkpoint to resume from")
    parser.add_argument("--device",      default="cuda",      choices=["cuda", "cpu", "mps"])
    parser.add_argument("--val_split",   default=0.0,         type=float,
                        help="Fraction of pairs held out for validation (0.0 = no val)")
    parser.add_argument("--patience",    default=0,           type=int,
                        help="Early stop after N evals without PSNR improvement (0 = disabled)")

    args = parser.parse_args()

    train(
        pair_dir=args.pair_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        ema_decay=args.ema_decay,
        highlight_weight=args.highlight_weight,
        mse_weight=args.mse_weight,
        warmup_frac=args.warmup_frac,
        knee=args.knee,
        num_workers=args.num_workers,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        resume=args.resume,
        device_str=args.device,
        model_size=args.model_size,
        val_split=args.val_split,
        patience=args.patience,
    )
