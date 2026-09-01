"""Train the per-frame residual scale by direct supervision on the oracle.

    python training/train_gate.py --manifest <manifest> \
        --init-checkpoint <dir>/sdr2hdr_image_v5/shipped_v5_step81000.pt \
        --output-dir <dir>/sdr2hdr_gate_v9 --steps 4000

Why this exists rather than another `--freeze-except-gate` run of
train_sdr2hdr.py: that path asks the head to infer the scale from the
reconstruction loss, and 16 000 steps across gate_v7 and gate_v8 showed it
cannot -- the loss does not contain the defect (training/gate_oracle.py has the
numbers). Here the target is computed, not inferred, and the head sees whole
frames as it will at inference rather than the 384-pixel crops the ordinary
training step feeds it.

The backbone is frozen throughout. This trains a few thousand parameters, so it
is minutes of GPU, and it cannot damage what already works.

Batch size defaults to 1 with gradient accumulation: the gate view keeps each
frame's aspect ratio, so frames of different shapes cannot be stacked, and
padding a batch would corrupt the pooled statistics the head reads. One 512-
pixel forward is a few tens of milliseconds, so this costs little.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.sdr2hdr import SDR2HDRNet                                    # noqa: E402
from training.gate_oracle import (NETWORK_PEAK_NITS, composite_at,      # noqa: E402
                                  effective_residual, gate_view,
                                  oracle_alpha, pu_psnr)
from training.sdr2hdr_dataset import degrade_sdr, load_rgb, read_jsonl  # noqa: E402
from training.train_sdr2hdr import (deterministic_eval_order,           # noqa: E402
                                    selection_score, _atomic_save)


class WholeFrameDataset(Dataset):
    """Whole frames at the gate's own working resolution, clean or degraded.

    `degradation_probability` matters more here than anywhere else: telling a
    clean frame from a degraded one is the head's entire job, and the oracle
    wants opposite answers for the two (0.125 versus full strength on the same
    low-headroom scene). A mix that leans either way teaches it the wrong prior.
    """

    def __init__(self, manifest: str | Path, split: str, max_side: int = 512,
                 degradation_probability: float = 0.5, deterministic: bool = False,
                 seed: int = 20260829):
        # Frames are handed over at NATIVE resolution. The gate downscales for
        # its pooled features but reads its statistics from these pixels, and
        # the oracle target is computed here rather than on a downscaled copy,
        # so it is the same quantity `rudra bench` reports.
        self.records = [r for r in read_jsonl(manifest) if r.get("split") == split]
        if not self.records:
            raise SystemExit(f"error: no {split!r} records in {manifest}")
        self.max_side = int(max_side)
        self.degradation_probability = float(degradation_probability)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        sdr = torch.from_numpy(load_rgb(record["sdr_path"], hdr=False)).permute(2, 0, 1)
        hdr = torch.from_numpy(load_rgb(record["hdr_path"], hdr=True)).permute(2, 0, 1)
        if sdr.shape[-2:] != hdr.shape[-2:]:
            raise ValueError(f"geometry mismatch for {record['asset_id']}")

        if self.deterministic:
            # Every other eval must see the identical frame, or the score moves
            # for reasons that have nothing to do with the head.
            degrade = (index % 2 == 1)
            if degrade:
                py, tor = random.getstate(), torch.random.get_rng_state()
                try:
                    random.seed(self.seed + index)
                    torch.manual_seed(self.seed + index)
                    sdr = degrade_sdr(sdr, 1.0)
                finally:
                    random.setstate(py)
                    torch.random.set_rng_state(tor)
        else:
            degrade = random.random() < self.degradation_probability
            if degrade:
                sdr = degrade_sdr(sdr, 1.0)
        return {"sdr": sdr, "hdr": hdr.clamp_min(0.0), "degraded": bool(degrade),
                "asset_id": str(record["asset_id"])}


def fields(model: SDR2HDRNet, sdr: torch.Tensor):
    """One frozen forward: baseline, the term alpha scales, and the preserve mask."""
    out = model(sdr, preserve_outside=False, recovery_mode="all",
                residual_strength=1.0, residual_scale=1.0)
    recovery = torch.maximum(out.highlight_mask, out.shadow_mask)
    return out.baseline, effective_residual(out.hdr, out.baseline), recovery


def evaluate(model: SDR2HDRNet, loader: DataLoader, device: torch.device,
             max_batches: int, max_side: int = 512) -> dict[str, float]:
    """How much of the oracle's benefit does the head actually recover?

    Reports the achieved PU21 gain at the PREDICTED scale against the two
    reference points that bracket it: what ships (alpha 1) and the ceiling
    (the oracle). A head that helps sits between them; one that has learned
    nothing sits on top of alpha 1.
    """
    model.eval()
    sums, count = {}, 0
    with torch.no_grad():
        for batch in loader:
            sdr = batch["sdr"].to(device)
            target = batch["hdr"].to(device)
            baseline, effective, recovery = fields(model, sdr)
            alpha_star, gain_star = oracle_alpha(baseline, effective, recovery, target)
            alpha_hat = model.gate(*model_gate_inputs(model, sdr, max_side)).flatten()
            ones = torch.ones_like(alpha_hat)

            def gain(alpha):
                pred = composite_at(baseline, effective, recovery, alpha)
                base = composite_at(baseline, effective, recovery, torch.zeros_like(alpha))
                return (pu_psnr(pred * NETWORK_PEAK_NITS, target * NETWORK_PEAK_NITS)
                        - pu_psnr(base * NETWORK_PEAK_NITS, target * NETWORK_PEAK_NITS))

            key = "hard" if bool(batch["degraded"][0]) else "clean"
            for name, value in (
                (f"{key}_gain_predicted", gain(alpha_hat).mean()),
                (f"{key}_gain_ships", gain(ones).mean()),
                (f"{key}_gain_oracle", gain_star.mean()),
                (f"{key}_alpha_predicted", alpha_hat.mean()),
                (f"{key}_alpha_oracle", alpha_star.mean()),
                ("alpha_abs_error", (alpha_hat - alpha_star).abs().mean()),
            ):
                sums[name] = sums.get(name, 0.0) + float(value)
                sums[name + "__n"] = sums.get(name + "__n", 0) + 1
            count += 1
            if count >= max_batches:
                break
    model.train()
    return {k: sums[k] / max(sums.get(k + "__n", 1), 1)
            for k in sums if not k.endswith("__n")}


def model_gate_inputs(model: SDR2HDRNet, sdr: torch.Tensor, max_side: int = 512):
    """The gate's three arguments, with the backbone run under no_grad.

    The backbone is frozen, so its features are constants; only the head's own
    MLP needs a graph. Running the encoder without one keeps this cheap.
    """
    from rudra.sdr2hdr import luminance, sdr_to_baseline_hdr
    sdr = sdr.float().clamp(0.0, 1.0)
    view = gate_view(sdr, max_side)
    with torch.no_grad():
        _, _, mid = model.encode(view, sdr_to_baseline_hdr(view))
    # Features from the view, statistics from the native frame -- see
    # SDR2HDRNet.predict_residual_scale for why the two resolutions differ.
    return mid, sdr, luminance(sdr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Higher than the backbone's 2e-4 on purpose. The head's "
                             "last layer starts zeroed with a bias solving "
                             "sigmoid(b)*alpha_max == 1, so moving the OUTPUT from 1.0 "
                             "to the 0.125 the oracle wants on low-headroom frames means "
                             "moving the pre-sigmoid by about 3.6. Adam moves a parameter "
                             "by roughly lr per step, so at 1e-3 the bias alone needs "
                             "~3 600 steps and there is nothing left over for the "
                             "feature-dependent part.")
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=512,
                        help="must match predict_residual_scale's cap, or the head "
                             "is trained on a view it never sees at inference")
    parser.add_argument("--degradation-probability", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=96)
    parser.add_argument("--best-smoothing", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    base_config = payload.get("config", {}) or {}
    model = SDR2HDRNet.from_config(base_config, gate_conditioning=True)
    missing, unexpected = model.load_state_dict(payload.get("model", payload), strict=False)
    stray = [k for k in missing if not k.startswith("gate.")]
    if stray or unexpected:
        raise RuntimeError(f"checkpoint does not match: missing {stray}, unexpected {list(unexpected)}")
    model.to(device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("gate."))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_set = WholeFrameDataset(args.manifest, "train", args.max_side,
                                  args.degradation_probability, seed=args.seed)
    val_set = WholeFrameDataset(args.manifest, "val", args.max_side,
                                deterministic=True, seed=args.seed)
    from torch.utils.data import Subset
    val_set = Subset(val_set, deterministic_eval_order(len(val_set)))
    loader_args = dict(batch_size=1, num_workers=args.workers,
                       persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=0.0)

    config = vars(args).copy()
    config.update({"base_channels": base_config.get("base_channels", 32),
                   "gate_conditioning": True, "objective": "oracle_alpha_regression"})
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log_path = output_dir / "train.jsonl"

    print("\n" + "=" * 66)
    print("   gate head -- direct oracle supervision")
    print("=" * 66)
    print(f"   backbone   : {args.init_checkpoint}  (frozen)")
    print(f"   trainable  : {trainable:,} parameters")
    print(f"   gate view  : whole frame, max side {args.max_side}")
    print(f"   device     : {device}")
    print("=" * 66 + "\n")

    probe = evaluate(model, val_loader, device, min(args.eval_batches, 24), args.max_side)
    print(f"[start] oracle alpha  clean {probe.get('clean_alpha_oracle', float('nan')):.4f}"
          f"   hard {probe.get('hard_alpha_oracle', float('nan')):.4f}")
    print(f"[start] gain at alpha=1  clean {probe.get('clean_gain_ships', float('nan')):+.3f} dB"
          f"   hard {probe.get('hard_gain_ships', float('nan')):+.3f} dB")
    print(f"[start] gain at oracle   clean {probe.get('clean_gain_oracle', float('nan')):+.3f} dB"
          f"   hard {probe.get('hard_gain_oracle', float('nan')):+.3f} dB")
    print("        (if clean oracle alpha is not well below 1, the targets are wrong "
          "and nothing below will help)\n")

    best, history, step, started = math.inf, [], 0, time.time()
    model.train()
    iterator = iter(train_loader)
    while step < args.steps:
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for _ in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            sdr, target = batch["sdr"].to(device), batch["hdr"].to(device)
            with torch.no_grad():
                baseline, effective, recovery = fields(model, sdr)
                alpha_star, _ = oracle_alpha(baseline, effective, recovery, target)
            alpha_hat = model.gate(*model_gate_inputs(model, sdr, args.max_side)).flatten()
            loss = F.smooth_l1_loss(alpha_hat, alpha_star.to(alpha_hat.dtype), beta=0.05)
            (loss / args.grad_accum).backward()
            accumulated += float(loss.detach()) / args.grad_accum
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        step += 1

        if step % 20 == 0 or step == 1:
            rate = step * args.grad_accum / max(time.time() - started, 1e-6)
            print(f"[gate] step {step}/{args.steps} alpha_l1={accumulated:.5f} frames/s={rate:.2f}")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step, "alpha_loss": accumulated}) + "\n")

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, val_loader, device, args.eval_batches, args.max_side)
            print(f"[eval] step {step}: {json.dumps(metrics)}")
            # Selection follows the thing this run exists to improve: the gain
            # actually achieved at the predicted scale, both conditions.
            score = -(metrics.get("clean_gain_predicted", 0.0)
                      + metrics.get("hard_gain_predicted", 0.0))
            history.append(score)
            smoothed = selection_score(history, args.best_smoothing)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, "eval": metrics,
                                         "score_raw": score,
                                         "score_smoothed": smoothed}) + "\n")
            if smoothed < best:
                best = smoothed
                _atomic_save({"model": model.state_dict(), "step": step, "best": best,
                              "config": config}, output_dir / "best.pt")
                print(f"       new best (smoothed {smoothed:+.4f}) -> best.pt")

    print(f"\n   done in {(time.time() - started) / 60:.1f} min -> {output_dir / 'best.pt'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
