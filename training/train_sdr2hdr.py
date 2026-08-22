"""Train the direct pixel-space SDR-to-HDR image model or temporal refiner.

Examples::

  python training/train_sdr2hdr.py --mode image --manifest hdrdata/sdr_hdr_manifest.jsonl
  python training/train_sdr2hdr.py --mode temporal --manifest hdrdata/g_data_video_clips_9.jsonl \
      --image-checkpoint hdrdata/checkpoints/sdr2hdr_image/best.pt
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.sdr2hdr import (  # noqa: E402
    SDR2HDRNet, TemporalHDRRefiner, sdr2hdr_loss, temporal_consistency_loss,
)
from training.sdr2hdr_dataset import SDRHDRDataset, SDRHDRVideoDataset  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def manifest_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image_checkpoint(path: str | Path, device: torch.device) -> tuple[SDR2HDRNet, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model = SDR2HDRNet(base_channels=int(config.get("base_channels", 32)))
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    return model.to(device), checkpoint


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def _tone_map(x: torch.Tensor) -> torch.Tensor:
    return x.clamp_min(0.0) / (1.0 + x.clamp_min(0.0))


@torch.no_grad()
def evaluate_image(model: SDR2HDRNet, loader: DataLoader, device: torch.device,
                   max_batches: int = 8, shadow_chroma_weight: float = 0.15,
                   shadow_smoothness_weight: float = 0.02) -> dict[str, float]:
    model.eval()
    sums = {
        "loss": 0.0, "log_l1": 0.0, "highlight": 0.0, "shadow": 0.0,
        "chroma": 0.0, "shadow_chroma": 0.0, "shadow_smoothness": 0.0,
        "psnr_tm": 0.0, "psnr_log": 0.0,
    }
    count = 0
    for batch in loader:
        sdr, target = batch["sdr"].to(device), batch["hdr"].to(device)
        output = model(sdr)
        losses = sdr2hdr_loss(
            output, sdr, target, shadow_chroma_weight, shadow_smoothness_weight,
        )
        mse_tm = F.mse_loss(_tone_map(output.hdr), _tone_map(target)).clamp_min(1e-12)
        pred_log, target_log = torch.log1p(output.hdr * 16.0), torch.log1p(target * 16.0)
        peak = torch.log1p(torch.tensor(16.0, device=device))
        mse_log = F.mse_loss(pred_log, target_log).clamp_min(1e-12)
        sums["loss"] += float(losses["total"])
        sums["log_l1"] += float(F.l1_loss(pred_log, target_log))
        for key in ("highlight", "shadow", "chroma", "shadow_chroma", "shadow_smoothness"):
            sums[key] += float(losses[key])
        sums["psnr_tm"] += float(-10.0 * torch.log10(mse_tm))
        sums["psnr_log"] += float(20.0 * torch.log10(peak) - 10.0 * torch.log10(mse_log))
        count += 1
        if count >= max_batches:
            break
    model.train()
    return {key: value / max(count, 1) for key, value in sums.items()}


@torch.no_grad()
def evaluate_temporal(image_model: SDR2HDRNet, temporal: TemporalHDRRefiner,
                      loader: DataLoader, device: torch.device, max_batches: int = 4) -> dict[str, float]:
    temporal.eval()
    sums = {"log_l1": 0.0, "temporal": 0.0, "initial_log_l1": 0.0, "initial_temporal": 0.0}
    count = 0
    for batch in loader:
        sdr, target = batch["sdr"].to(device), batch["hdr"].to(device)
        b, t, c, h, w = sdr.shape
        initial = image_model(sdr.reshape(b * t, c, h, w)).hdr.reshape(b, t, c, h, w)
        pred = temporal(sdr, initial)
        sums["log_l1"] += float(F.l1_loss(torch.log1p(pred * 16.0), torch.log1p(target * 16.0)))
        sums["temporal"] += float(temporal_consistency_loss(pred, target))
        sums["initial_log_l1"] += float(F.l1_loss(torch.log1p(initial * 16.0), torch.log1p(target * 16.0)))
        sums["initial_temporal"] += float(temporal_consistency_loss(initial, target))
        count += 1
        if count >= max_batches:
            break
    temporal.train()
    return {key: value / max(count, 1) for key, value in sums.items()}


def build_loaders(args: argparse.Namespace):
    common = dict(crop_size=args.crop_size, max_items=args.max_items)
    if args.mode == "image":
        train = SDRHDRDataset(args.manifest, split="train", augment=True,
                              augmentation_strength=args.augmentation_strength,
                              degradation_probability=args.degradation_probability,
                              **common)
        val = SDRHDRDataset(args.manifest, split="val", augment=False,
                            augmentation_strength=0.0, max_items=args.max_val_items,
                            crop_size=args.crop_size)
    else:
        train = SDRHDRVideoDataset(args.manifest, augment=True, split="train",
                                   val_fraction=args.val_fraction, **common)
        val = SDRHDRVideoDataset(args.manifest, augment=False, split="val",
                                 val_fraction=args.val_fraction, crop_size=args.crop_size,
                                 max_items=args.max_val_items)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers,
                       pin_memory=torch.cuda.is_available(), persistent_workers=args.workers > 0)
    sampler = None
    if args.mode == "image" and args.scene_balanced_sampling:
        fraction = float(args.video_sample_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("--video-sample-fraction must be between 0 and 1")
        records = train.records
        scene_sizes = Counter(str(record["scene_id"]) for record in records)
        scenes_by_category = {
            True: {str(record["scene_id"]) for record in records if bool(record.get("is_video", False))},
            False: {str(record["scene_id"]) for record in records if not bool(record.get("is_video", False))},
        }
        category_mass = {True: fraction, False: 1.0 - fraction}
        weights = []
        for record in records:
            is_video = bool(record.get("is_video", False))
            scene = str(record["scene_id"])
            scene_count = max(len(scenes_by_category[is_video]), 1)
            weights.append(category_mass[is_video] / (scene_count * scene_sizes[scene]))
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(records), replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    return (
        DataLoader(train, shuffle=sampler is None, sampler=sampler,
                   drop_last=len(train) >= args.batch_size, **loader_args),
        DataLoader(val, shuffle=False, drop_last=False, **loader_args),
    )


def train(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    train_loader, val_loader = build_loaders(args)
    image_model: SDR2HDRNet | None = None
    if args.mode == "image":
        model: torch.nn.Module = SDR2HDRNet(base_channels=args.base_channels).to(device)
    else:
        if not args.image_checkpoint:
            raise ValueError("--image-checkpoint is required for temporal training")
        image_model, _ = load_image_checkpoint(args.image_checkpoint, device)
        image_model.eval().requires_grad_(False)
        model = TemporalHDRRefiner(channels=args.temporal_channels).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_steps = max(math.ceil(args.steps / args.grad_accum), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimizer_steps, eta_min=args.lr * 0.05)
    start_step, best = 0, math.inf
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"])
        previous_metric = checkpoint.get("config", {}).get("best_metric", "log_l1")
        if args.reset_best or previous_metric != args.best_metric:
            best = math.inf
        else:
            best = float(checkpoint.get("best", best))

    config = vars(args).copy()
    config.update({
        "manifest_sha256": manifest_hash(args.manifest),
        "architecture": type(model).__name__,
        "input_contract": "8-bit normalized sRGB RGB",
        "output_contract": "scene-linear RGB normalized to 10000 nits",
    })
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    log_path = output_dir / "train.jsonl"
    data_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    started = time.time()

    if start_step == 0:
        if args.mode == "image":
            initial_metrics = evaluate_image(
                model, val_loader, device, args.eval_batches,
                args.shadow_chroma_weight, args.shadow_smoothness_weight,
            )
            best = initial_metrics[args.best_metric]
        else:
            assert image_model is not None
            initial_metrics = evaluate_temporal(image_model, model, val_loader, device, args.eval_batches)
            best = initial_metrics["log_l1"] + args.temporal_weight * initial_metrics["temporal"]
        _atomic_save({"model": model.state_dict(), "step": 0, "best": best, "config": config},
                     output_dir / "best.pt")
        print(f"[baseline] {json.dumps(initial_metrics)}")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": 0, "eval": initial_metrics, "baseline": True}) + "\n")

    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        sdr, target = batch["sdr"].to(device, non_blocking=True), batch["hdr"].to(device, non_blocking=True)
        amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
        with amp:
            if args.mode == "image":
                output = model(sdr)
                losses = sdr2hdr_loss(
                    output, sdr, target, args.shadow_chroma_weight,
                    args.shadow_smoothness_weight,
                )
                loss = losses["total"]
            else:
                assert image_model is not None
                b, t, c, h, w = sdr.shape
                with torch.no_grad():
                    initial = image_model(sdr.reshape(b * t, c, h, w)).hdr.reshape(b, t, c, h, w)
                pred = model(sdr, initial)
                spatial = F.l1_loss(torch.log1p(pred * 16.0), torch.log1p(target * 16.0))
                temporal = temporal_consistency_loss(pred, target)
                loss = spatial + args.temporal_weight * temporal
                losses = {"total": loss, "log_l1": spatial, "temporal": temporal}
            scaled_loss = loss / args.grad_accum
        scaled_loss.backward()
        did_optimizer_step = step % args.grad_accum == 0 or step == args.steps
        if did_optimizer_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        record = {"step": step, "lr": scheduler.get_last_lr()[0],
                  **{key: float(value.detach()) for key, value in losses.items()}}
        if step == 1 or step % args.log_every == 0:
            rate = step * args.batch_size / max(time.time() - started, 1e-6)
            print(f"[{args.mode}] step {step}/{args.steps} loss={record['total']:.5f} samples/s={rate:.2f}")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        if step % args.eval_every == 0 or step == args.steps:
            if args.mode == "image":
                metrics = evaluate_image(
                    model, val_loader, device, args.eval_batches,
                    args.shadow_chroma_weight, args.shadow_smoothness_weight,
                )
                score = metrics[args.best_metric]
            else:
                metrics = evaluate_temporal(image_model, model, val_loader, device, args.eval_batches)
                score = metrics["log_l1"] + args.temporal_weight * metrics["temporal"]
            print(f"[eval] step {step}: {json.dumps(metrics)}")
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, "eval": metrics}) + "\n")
            if score < best:
                best = score
                _atomic_save({"model": model.state_dict(), "step": step, "best": best, "config": config},
                             output_dir / "best.pt")

        if step % args.save_every == 0 or step == args.steps:
            payload = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "step": step, "best": best,
                "config": config,
            }
            _atomic_save(payload, output_dir / f"step_{step:07d}.pt")
    return output_dir / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("image", "temporal"), default="image")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="hdrdata/checkpoints/sdr2hdr_image")
    parser.add_argument("--image-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--init-checkpoint",
                        help="Load model weights only and start a fresh optimizer/schedule")
    parser.add_argument("--reset-best", action="store_true",
                        help="Reset best score when exactly resuming a run")
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--temporal-channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--augmentation-strength", type=float, default=1.0)
    parser.add_argument("--degradation-probability", type=float, default=0.65)
    parser.add_argument("--video-sample-fraction", type=float, default=0.25,
                        help="Target share of video frames in scene-balanced image batches")
    parser.add_argument("--no-scene-balanced-sampling", dest="scene_balanced_sampling",
                        action="store_false",
                        help="Restore record-uniform image sampling")
    parser.set_defaults(scene_balanced_sampling=True)
    parser.add_argument("--temporal-weight", type=float, default=0.5)
    parser.add_argument("--shadow-chroma-weight", type=float, default=0.15)
    parser.add_argument("--shadow-smoothness-weight", type=float, default=0.02)
    parser.add_argument("--best-metric", choices=("loss", "log_l1"), default="loss")
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=2_000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--max-val-items", type=int)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    checkpoint = train(args)
    print(f"Best checkpoint: {checkpoint}")
