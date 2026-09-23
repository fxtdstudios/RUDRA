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
from statistics import median
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.sdr2hdr import (  # noqa: E402
    SDR2HDRNet, TemporalHDRRefiner, sdr2hdr_loss, temporal_consistency_loss,
    temporal_spatial_loss,
)
from training.sdr2hdr_dataset import (  # noqa: E402
    SDRHDRDataset, SDRHDRVideoDataset, corpus_ev_of)


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
    # Temporal training refines an IMAGE model's per-frame output; it cannot
    # start without one. A bare FileNotFoundError from torch.load buries that
    # under six frames of serialization internals, so say it here instead.
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        siblings = sorted(checkpoint_path.parent.parent.glob("*/best.pt")) if \
            checkpoint_path.parent.parent.is_dir() else []
        hint = ("\n       image checkpoints that DO exist:\n         "
                + "\n         ".join(str(s) for s in siblings)) if siblings else ""
        raise SystemExit(
            f"error: --image-checkpoint {checkpoint_path} does not exist.\n"
            f"       Temporal training starts from a trained image model -- run "
            f"--mode image first, or point at an existing checkpoint.{hint}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model = SDR2HDRNet.from_config(config)
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
        # The analytic inverse-ACES baseline, scored on the same batches. The
        # network is a residual on top of it, so "is it better than nothing?"
        # is the only question that matters, and it should not have to be
        # reconstructed from the step-0 row of a log file.
        "baseline_log_l1": 0.0, "baseline_psnr_log": 0.0,
        # The same prediction under preserve_outside=True, which blends back to
        # the baseline wherever the learned masks are cold. Free to compute from
        # the fields we already have, and it is the lever for the 26 Aug split:
        # +1.63 dB on degraded SDR but -4.63 dB on clean SDR, with nothing at
        # inference to say which kind of SDR arrived.
        "preserved_psnr_log": 0.0,
    }
    count = 0
    for batch in loader:
        sdr, target = batch["sdr"].to(device), batch["hdr"].to(device)
        ceiling = batch["ceiling"].to(device) if "ceiling" in batch else None
        output = model(sdr)
        losses = sdr2hdr_loss(
            output, sdr, target, shadow_chroma_weight, shadow_smoothness_weight,
            target_ceiling=ceiling,
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
        # "Gain" is always against the ANALYTIC baseline -- the one thing that
        # is genuinely "nothing learned". A CurveHead's corrected baseline is
        # part of the model and is scored separately as curve_baseline_psnr_log.
        analytic = output.analytic_baseline if output.analytic_baseline is not None \
            else output.baseline
        if output.curve_params is not None:
            cb_mse = F.mse_loss(torch.log1p(output.baseline.clamp_min(0.0) * 16.0),
                                target_log).clamp_min(1e-12)
            sums["curve_baseline_psnr_log"] = sums.get("curve_baseline_psnr_log", 0.0) + float(
                20.0 * torch.log10(peak) - 10.0 * torch.log10(cb_mse))
        base_log = torch.log1p(analytic * 16.0)
        base_mse = F.mse_loss(base_log, target_log).clamp_min(1e-12)
        sums["baseline_log_l1"] += float(F.l1_loss(base_log, target_log))
        sums["baseline_psnr_log"] += float(20.0 * torch.log10(peak) - 10.0 * torch.log10(base_mse))
        recovery = torch.maximum(output.highlight_mask, output.shadow_mask)
        preserved = output.baseline + recovery * (output.hdr - output.baseline)
        pres_mse = F.mse_loss(torch.log1p(preserved.clamp_min(0.0) * 16.0),
                              target_log).clamp_min(1e-12)
        sums["preserved_psnr_log"] += float(20.0 * torch.log10(peak) - 10.0 * torch.log10(pres_mse))
        count += 1
        if count >= max_batches:
            break
    model.train()
    metrics = {key: value / max(count, 1) for key, value in sums.items()}
    # Positive = the network improved on the analytic baseline. Negative means
    # it is actively making the baseline worse, which is the failure the August
    # 2026 run and the 26 Aug clean-SDR eval both showed.
    metrics["gain_db"] = metrics["psnr_log"] - metrics["baseline_psnr_log"]
    metrics["preserved_gain_db"] = metrics["preserved_psnr_log"] - metrics["baseline_psnr_log"]
    return metrics


def selection_score(history: list[float], window: int) -> float:
    """The score best.pt is actually selected on: a trailing median.

    The v5 run selected best.pt on a single evaluation, and that was a mistake
    worth naming. The clean/hard eval set is already deterministic -- centre
    crops, `shuffle=False`, the same 256 records every time -- so its spread is
    not sampling noise. It is the model genuinely oscillating from step to
    step: across the 102 evals of the v5 run `clean_gain_db` had mean -1.43 dB
    and standard deviation 1.29 dB, with only 10 of them positive. Taking a raw
    maximum over that series does not find a better model, it finds the
    luckiest step, and `composite_gain = hard_gain + min(0, clean_gain)` climbs
    exactly the term that oscillates. The shipped step-81 000 checkpoint's
    `clean_gain_db` of +0.02 ranked 9th of 102, and the independent benchmark
    later measured +1.43 dB on held-out frames where selection had promised
    +1.80.

    A trailing median over `window` evaluations cannot be won by one lucky
    step: a checkpoint is only best if the neighbourhood it sits in is best.
    best.pt then holds the most recent weights from that neighbourhood, which
    is a member of a good region rather than the peak of a noisy one.

    `window = 1` restores the old single-eval behaviour.
    """
    if window <= 1:
        return history[-1]
    return float(median(history[-window:]))


def temporal_score(metrics: dict, temporal_weight: float) -> float:
    """What selects the refiner's best.pt -- and it must match what it trains on.

    The training objective is censored: a pixel on a grading ceiling means
    ">= ceiling", so reconstructing above it is free. Selecting on the plain
    log_l1 would have graded that behaviour as WORSE, because plain L1 measures
    against a target that was capped -- so best.pt would have preferred exactly
    the flattening the censored loss exists to prevent. Selection follows the
    loss; `log_l1` stays in the metrics for comparison with image mode and with
    the runs before it.
    """
    spatial = metrics.get("censored_log_l1", metrics["log_l1"])
    return spatial + temporal_weight * metrics["temporal"]


@torch.no_grad()
def evaluate_temporal(image_model: SDR2HDRNet, temporal: TemporalHDRRefiner,
                      loader: DataLoader, device: torch.device, max_batches: int = 4) -> dict[str, float]:
    temporal.eval()
    sums = {"log_l1": 0.0, "temporal": 0.0, "initial_log_l1": 0.0, "initial_temporal": 0.0,
            "censored_log_l1": 0.0, "censored_fraction": 0.0}
    count = 0
    for batch in loader:
        sdr, target = batch["sdr"].to(device), batch["hdr"].to(device)
        ceiling = batch["ceiling"].to(device) if "ceiling" in batch else None
        b, t, c, h, w = sdr.shape
        initial = image_model(sdr.reshape(b * t, c, h, w)).hdr.reshape(b, t, c, h, w)
        pred = temporal(sdr, initial)
        # log_l1 stays plain so it is comparable with image mode and with the
        # runs before this; censored_log_l1 is what the refiner is optimising.
        censored, fraction = temporal_spatial_loss(pred, target, ceiling)
        sums["censored_log_l1"] += float(censored)
        sums["censored_fraction"] += float(fraction)
        sums["log_l1"] += float(F.l1_loss(torch.log1p(pred * 16.0), torch.log1p(target * 16.0)))
        sums["temporal"] += float(temporal_consistency_loss(pred, target))
        sums["initial_log_l1"] += float(F.l1_loss(torch.log1p(initial * 16.0), torch.log1p(target * 16.0)))
        sums["initial_temporal"] += float(temporal_consistency_loss(initial, target))
        count += 1
        if count >= max_batches:
            break
    temporal.train()
    return {key: value / max(count, 1) for key, value in sums.items()}


def deterministic_eval_order(size: int, seed: int = 20260829) -> list[int]:
    """A fixed, representative order for the validation set.

    `DataLoader(val, shuffle=False)` plus `evaluate_image(max_batches=N)` does
    not sample the validation split -- it reads the FRONT of it. On 29 Aug 2026
    the gate run's eval (8 batches of 4) was therefore 32 records covering 11
    scenes, every one of them alphabetically between "abandoned_factory" and
    "blau_river", with 403 records and 87 scenes never evaluated at all. The v5
    run's 256 records reached 86 of 97 scenes but still skewed high: median
    reference peak 836.6 nits against the split's own 547.6, and 54.7% of
    frames below 1 000 nits against 69.0%.

    That matters because RUDRA's error correlates with headroom (+0.46 against
    log2 peak nits): an eval slice biased towards bright scenes reports a clean
    gain near zero while the held-out benchmark measures -3.0 dB. It is why the
    conditioning head trained on 29 Aug learned nothing -- at its eval slice
    there was no defect to fix, so staying at scale 1.0 was correct.

    A seeded permutation keeps every property the old order had that mattered
    -- identical records in an identical order at every step, so scores are
    comparable across a run -- and drops the one that did not: alphabetical
    position deciding what gets measured.
    """
    import random
    order = list(range(size))
    random.Random(seed).shuffle(order)
    return order


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
        # Same records, same crops, deterministically degraded: the condition
        # the model is for. See SDRHDRDataset.deterministic_degradation.
        val_hard = SDRHDRDataset(args.manifest, split="val", augment=False,
                                 augmentation_strength=0.0, max_items=args.max_val_items,
                                 crop_size=args.crop_size, deterministic_degradation=True)
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
    # The clean and hard loaders MUST get the identical order: the two are the
    # same records under two conditions, and clean_gain minus hard_gain is only
    # meaningful if they describe the same frames.
    order = deterministic_eval_order(len(val))
    val_eval = Subset(val, order)
    hard_eval = Subset(val_hard, order[:len(val_hard)]) if args.mode == "image" else None
    return (
        DataLoader(train, shuffle=sampler is None, sampler=sampler,
                   drop_last=len(train) >= args.batch_size, **loader_args),
        DataLoader(val_eval, shuffle=False, drop_last=False, **loader_args),
        (DataLoader(hard_eval, shuffle=False, drop_last=False, **loader_args)
         if hard_eval is not None else None),
    )


def train(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    train_loader, val_loader, val_hard_loader = build_loaders(args)
    image_model: SDR2HDRNet | None = None
    # The baseline the network learns a residual over is the inverse of the
    # corpus's own render, exposure included. Read from the manifest, never
    # assumed: until 16 Sep 2026 this constructor took the class default, the
    # legacy -1 EV, whatever the corpus said.
    corpus_ev = corpus_ev_of(args.manifest)
    if args.mode == "image":
        model: torch.nn.Module = SDR2HDRNet(
            base_channels=args.base_channels,
            gate_conditioning=args.gate_conditioning,
            curve_head=args.curve_head,
            corpus_ev=corpus_ev).to(device)
    else:
        if not args.image_checkpoint:
            raise ValueError("--image-checkpoint is required for temporal training")
        image_model, _ = load_image_checkpoint(args.image_checkpoint, device)
        image_model.eval().requires_grad_(False)
        # The refiner learns a residual over the image model's output, and the
        # image model's output is a residual over its baseline at ITS corpus
        # exposure. If that exposure is not the video corpus's, the refiner
        # spends itself undoing a constant stop and every eval reads as a gain.
        # sdr2hdr_temporal_v4 (22 Sep 2026) was started exactly this way:
        # image model at -1 EV, clips at 0 EV.
        if abs(float(image_model.corpus_ev) - float(corpus_ev)) > 1e-6:
            raise SystemExit(
                f"error: --image-checkpoint {args.image_checkpoint} was trained against a "
                f"{image_model.corpus_ev:+.2f} EV baseline but {args.manifest} was rendered "
                f"at {corpus_ev:+.2f} EV. One model has one baseline: train the image model "
                f"on this corpus first, or point at one that was.")
        model = TemporalHDRRefiner(channels=args.temporal_channels).to(device)

    if args.freeze_except_gate:
        if getattr(model, "gate", None) is None:
            raise ValueError("--freeze-except-gate needs --gate-conditioning")
        trainable = 0
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("gate."))
            if parameter.requires_grad:
                trainable += parameter.numel()
        print(f"[freeze] training the conditioning head only: {trainable:,} parameters")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    optimizer_steps = max(math.ceil(args.steps / args.grad_accum), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimizer_steps, eta_min=args.lr * 0.05)
    start_step, best = 0, math.inf
    # Trailing window of raw eval scores; selection_score() reads it. A resumed
    # run starts it empty and the median simply covers fewer evals until it fills.
    score_history: list[float] = []
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(
            checkpoint.get("model", checkpoint), strict=False)
        # Adding the conditioning head to a checkpoint that predates it is the
        # whole point of --gate-conditioning + --init-checkpoint, so gate.*
        # keys are allowed to be missing. Nothing else is: a silently
        # half-loaded backbone would train from noise and look like a bad idea
        # rather than a bad load.
        stray = [k for k in missing if not k.startswith(("gate.", "curve."))]
        if stray or unexpected:
            raise RuntimeError(
                f"--init-checkpoint {args.init_checkpoint} does not match this "
                f"architecture: missing {stray or '[]'}, unexpected {list(unexpected) or '[]'}")
        if missing:
            print(f"[init] {len(missing)} fresh gate parameter tensors; the rest loaded "
                  f"from {args.init_checkpoint}")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"])
        previous_config = checkpoint.get("config", {})
        previous_metric = previous_config.get("best_metric", "log_l1")
        # A smoothed best and a raw best are different quantities; carrying one
        # across as if it were the other would let a stale single-eval maximum
        # block every honest checkpoint for the rest of the run.
        previous_window = int(previous_config.get("best_smoothing", 1))
        if (args.reset_best or previous_metric != args.best_metric
                or previous_window != args.best_smoothing):
            best = math.inf
        else:
            best = float(checkpoint.get("best", best))

    config = vars(args).copy()
    config.update({
        "manifest_sha256": manifest_hash(args.manifest),
        "corpus_ev": corpus_ev,
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

    def image_eval() -> tuple[dict, float]:
        """Both held-out conditions, and the score best.pt is selected on.

        clean_*  : val SDR exactly as prepare_pairs wrote it -- the ACES output
                   the target was tone-mapped from. sdr_to_baseline_hdr is very
                   nearly its analytic inverse here, so a zero-residual network
                   is already near-optimal and gain_db near zero is CORRECT, not
                   a failure. It is a do-no-harm check.
        hard_*   : the same records under a seeded camera/codec degradation --
                   an unknown tone curve, 4:2:0 chroma, banding, JPEG. This is
                   the deployment condition, so this is what selects best.pt.
        """
        clean = evaluate_image(model, val_loader, device, args.eval_batches,
                               args.shadow_chroma_weight, args.shadow_smoothness_weight)
        merged = {f"clean_{k}": v for k, v in clean.items()}
        chosen = clean
        if val_hard_loader is not None:
            hard = evaluate_image(model, val_hard_loader, device, args.eval_batches,
                                  args.shadow_chroma_weight, args.shadow_smoothness_weight)
            merged.update({f"hard_{k}": v for k, v in hard.items()})
            if args.best_eval == "hard":
                chosen = hard
        if args.best_metric in ("composite_gain", "preserved_composite_gain") and val_hard_loader is not None:
            # Reward the gain on degraded SDR, subtract any HARM done to clean
            # SDR, and count a clean gain as worth nothing -- clean input is a
            # constraint, not an objective. Negated because lower wins.
            #
            # This is not academic. The 26 Aug 2026 run selected on hard_loss
            # alone, which falls monotonically, so best.pt landed on step 48,000
            # (hard +1.81, clean -4.22) while step 44,000 sat right there at
            # hard +1.71 for only -0.90 clean -- nearly all of the upside for a
            # fifth of the damage.
            gain_key = "preserved_gain_db" if args.best_metric == "preserved_composite_gain" else "gain_db"
            merged["composite_gain"] = (merged[f"hard_{gain_key}"]
                                        + min(0.0, merged[f"clean_{gain_key}"]))
            return merged, -merged["composite_gain"]
        return merged, chosen[args.best_metric]

    if start_step == 0:
        if args.mode == "image":
            initial_metrics, best = image_eval()
        else:
            assert image_model is not None
            initial_metrics = evaluate_temporal(image_model, model, val_loader, device, args.eval_batches)
            best = temporal_score(initial_metrics, args.temporal_weight)
        score_history.append(best)
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
                    target_ceiling=(batch["ceiling"].to(device, non_blocking=True)
                                    if "ceiling" in batch else None),
                    baseline_weight=args.baseline_weight if args.curve_head else 0.0,
                )
                loss = losses["total"]
            else:
                assert image_model is not None
                b, t, c, h, w = sdr.shape
                with torch.no_grad():
                    initial = image_model(sdr.reshape(b * t, c, h, w)).hdr.reshape(b, t, c, h, w)
                pred = model(sdr, initial)
                # Censored, like the image loss. Trained on plain L1 the refiner
                # learns to pull the image model's reconstructed highlights back
                # down to the grading cap -- and 84% of the video clips carry one.
                spatial, censored_fraction = temporal_spatial_loss(
                    pred, target,
                    target_ceiling=(batch["ceiling"].to(device, non_blocking=True)
                                    if "ceiling" in batch else None),
                )
                temporal = temporal_consistency_loss(pred, target)
                loss = spatial + args.temporal_weight * temporal
                losses = {"total": loss, "log_l1": spatial, "temporal": temporal,
                          "censored_fraction": censored_fraction}
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
                metrics, score = image_eval()
            else:
                metrics = evaluate_temporal(image_model, model, val_loader, device, args.eval_batches)
                score = temporal_score(metrics, args.temporal_weight)
            print(f"[eval] step {step}: {json.dumps(metrics)}")
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, "eval": metrics}) + "\n")
            score_history.append(score)
            smoothed = selection_score(score_history, args.best_smoothing)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, "score_raw": score,
                                         "score_smoothed": smoothed,
                                         "best_smoothing": args.best_smoothing}) + "\n")
            if smoothed < best:
                best = smoothed
                _atomic_save({"model": model.state_dict(), "step": step, "best": best,
                              "config": config}, output_dir / "best.pt")

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
    parser.add_argument("--best-eval", choices=("hard", "clean"), default="hard",
                        help="Which held-out condition selects best.pt. 'hard' is the "
                             "seeded camera/codec degradation -- what the model is for. "
                             "'clean' scores the untouched ACES output, where the analytic "
                             "baseline is already near-optimal and every model looks the "
                             "same (26 Aug 2026: 12,000 steps, not one eval beat step 0).")
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
    parser.add_argument("--curve-head", action="store_true",
                        help="Add rudra.sdr2hdr.CurveHead: estimate each frame's tone curve "
                             "and exposure and correct the analytic inverse before the "
                             "residual. Needed for SDR that did not come through the "
                             "corpus's own ACES render (bench/oog, 23 Sep 2026). Train on a "
                             "--sdr-render mix corpus, or the head has nothing to learn.")
    parser.add_argument("--baseline-weight", type=float, default=0.25,
                        help="With --curve-head: weight of the direct loss on the corrected "
                             "baseline (default %(default)s)")
    parser.add_argument("--gate-conditioning", action="store_true",
                        help="Add the per-frame residual-scale head (rudra.sdr2hdr."
                             "ConditionGate). The per-pixel priors decide WHERE to "
                             "reconstruct and nothing decided HOW MUCH: on clean input "
                             "v5 loses 3.0 dB to its own analytic baseline, all of it in "
                             "low-dynamic-range frames. An oracle per-frame scale is "
                             "worth +5.84 dB clean and +0.29 dB hard at once, and no "
                             "constant can do it. A fresh head emits exactly 1.0, so "
                             "enabling this changes nothing until it trains.")
    parser.add_argument("--freeze-except-gate", action="store_true",
                        help="Train only the conditioning head, everything else frozen. "
                             "This is how to add the head to a checkpoint that already "
                             "works: the residual is not the problem (the oracle wants "
                             "0.97 of it on high-headroom clean frames), the missing "
                             "piece is knowing when to apply it. Few parameters, fast, "
                             "and it cannot damage what already works. Use with "
                             "--init-checkpoint and --gate-conditioning.")
    parser.add_argument("--best-smoothing", type=int, default=5,
                        help="best.pt is selected on the trailing median of this many "
                             "evaluations instead of a single one. The eval set is "
                             "deterministic, so its spread is the model oscillating, not "
                             "sampling noise -- and a raw maximum over 100+ evals finds "
                             "the luckiest step rather than the best model. The v5 run "
                             "shipped a checkpoint whose clean_gain_db ranked 9th of 102 "
                             "and promised +1.80 dB where the benchmark measured +1.43. "
                             "Pass 1 to restore the old single-eval behaviour.")
    parser.add_argument("--best-metric",
                        choices=("composite_gain", "preserved_composite_gain", "loss", "log_l1"), default="composite_gain",
                        help="composite_gain = hard_gain_db + min(0, clean_gain_db): the "
                             "improvement on degraded SDR, less any damage done to clean "
                             "SDR. 'loss' selects on the raw objective, which falls "
                             "monotonically and therefore always picks the last checkpoint "
                             "-- on 26 Aug 2026 that meant -4.22 dB on clean input for "
                             "0.1 dB more on degraded. Image mode only; temporal ignores it.")
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
