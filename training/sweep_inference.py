"""Score a finished checkpoint across input conditions and inference modes.

The 26 Aug 2026 image run came back split:

    hard_gain_db   +1.63   the model beats inverse-ACES on degraded SDR
    clean_gain_db  -4.63   the model is WORSE than inverse-ACES on clean SDR

Both matter, because at inference nothing tells RUDRA which kind of SDR it was
handed. A well-graded master must not come out worse than doing nothing.

SDR2HDRNet already has a lever for exactly this: ``preserve_outside=True``
blends the prediction back toward the analytic baseline everywhere the learned
highlight/shadow masks are cold, so the network can only act where the SDR
mapping was genuinely non-invertible. Training never used it and neither did
the eval, so nobody has ever measured it. This does -- on a checkpoint that
already exists, without retraining.

    python training\\sweep_inference.py ^
        --checkpoint E:\\RUDRA_v3_20260822\\checkpoints\\sdr2hdr_image_v3b\\best.pt ^
        --manifest  E:\\RUDRA_v3_20260822\\sdr_hdr_manifest.jsonl

Read the table by column: pick the inference mode whose CLEAN gain is >= 0
while its HARD gain stays as high as possible. If no mode manages both, the
model is over-correcting valid input and the fix is in training, not here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.sdr2hdr import SDR2HDRNet  # noqa: E402
from training.sdr2hdr_dataset import SDRHDRDataset  # noqa: E402

SCALE = 16.0
PEAK = math.log1p(SCALE)


def _psnr_log(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(torch.log1p(pred.clamp_min(0.0) * SCALE),
                     torch.log1p(target.clamp_min(0.0) * SCALE)).clamp_min(1e-12)
    return float(20.0 * math.log10(PEAK) - 10.0 * torch.log10(mse))


def _clipped_mask(sdr: torch.Tensor) -> torch.Tensor:
    """Pixels the SDR could not represent: where its brightest channel is at clip.

    Defined on the SDR, not the target. An earlier version thresholded the
    target at 0.85 in network units -- 8,500 nits -- which almost no pixel
    reaches, so the mask was empty and every number came back NaN.
    """
    return (sdr.max(1, keepdim=True).values >= 0.98).float()


def _masked_log_l1(pred: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor) -> tuple[float, float]:
    """Return (weighted error sum, weight) so batches aggregate correctly.

    Averaging per-batch means would let a batch with three clipped pixels count
    as much as one that is half sky.
    """
    delta = (torch.log1p(pred.clamp_min(0.0) * SCALE)
             - torch.log1p(target.clamp_min(0.0) * SCALE)).abs()
    return float((delta * mask).sum()), float(mask.sum() * 3.0)


@torch.no_grad()
def score_condition(model: SDR2HDRNet, loader: DataLoader, device: torch.device,
                    max_batches: int) -> dict:
    modes = {"plain": False, "preserve_outside": True}
    names = (*modes, "baseline")
    psnr = {name: 0.0 for name in names}
    hi_num = {name: 0.0 for name in names}
    hi_den = {name: 0.0 for name in names}
    count = 0
    for batch in loader:
        sdr, target = batch["sdr"].to(device), batch["hdr"].to(device)
        mask = _clipped_mask(sdr)
        for name, preserve in modes.items():
            out = model(sdr, preserve_outside=preserve)
            psnr[name] += _psnr_log(out.hdr, target)
            num, den = _masked_log_l1(out.hdr, target, mask)
            hi_num[name] += num
            hi_den[name] += den
            if name == "plain":
                psnr["baseline"] += _psnr_log(out.baseline, target)
                num, den = _masked_log_l1(out.baseline, target, mask)
                hi_num["baseline"] += num
                hi_den["baseline"] += den
        count += 1
        if count >= max_batches:
            break
    return {name: {"psnr_log": psnr[name] / max(count, 1),
                   "highlight": (hi_num[name] / hi_den[name]) if hi_den[name] else float("nan"),
                   "clipped_pixels": hi_den[name] / 3.0}
            for name in names}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batches", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model = SDR2HDRNet(base_channels=int(config.get("base_channels", 32)))
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.to(device).eval()
    print(f"checkpoint : {args.checkpoint}  (step {checkpoint.get('step', '?')})")

    results = {}
    for condition, degrade in (("clean", False), ("hard", True)):
        dataset = SDRHDRDataset(args.manifest, split=args.split, crop_size=args.crop_size,
                                augment=False, augmentation_strength=0.0,
                                deterministic_degradation=degrade)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)
        results[condition] = score_condition(model, loader, device, args.batches)

    print(f"\n{args.split} split, {args.batches} batches x {args.batch_size}\n")
    print(f"{'':<18} {'clean psnr_log':>15} {'gain':>8}   {'hard psnr_log':>14} {'gain':>8}")
    base_clean = results["clean"]["baseline"]["psnr_log"]
    base_hard = results["hard"]["baseline"]["psnr_log"]
    print(f"{'inverse-ACES':<18} {base_clean:>15.2f} {'--':>8}   {base_hard:>14.2f} {'--':>8}")
    for name in ("plain", "preserve_outside"):
        c = results["clean"][name]["psnr_log"]
        h = results["hard"][name]["psnr_log"]
        print(f"{name:<18} {c:>15.2f} {c - base_clean:>+8.2f}   "
              f"{h:>14.2f} {h - base_hard:>+8.2f}")

    clipped = results["clean"]["baseline"]["clipped_pixels"]
    print(f"\nhighlight log-L1 over the {clipped:,.0f} clipped SDR pixels "
          f"(where the model has to invent; lower is better)")
    print(f"{'':<18} {'clean':>10} {'hard':>10}")
    for name in ("baseline", "plain", "preserve_outside"):
        print(f"{name:<18} {results['clean'][name]['highlight']:>10.4f} "
              f"{results['hard'][name]['highlight']:>10.4f}")

    winner = max(("plain", "preserve_outside"),
                 key=lambda n: results["hard"][n]["psnr_log"] - base_hard
                 if results["clean"][n]["psnr_log"] >= base_clean - 0.5 else -math.inf)
    safe = results["clean"][winner]["psnr_log"] >= base_clean - 0.5
    print()
    if safe:
        print(f"  SHIP WITH: {winner} -- keeps {results['hard'][winner]['psnr_log'] - base_hard:+.2f} dB "
              f"on degraded input without damaging clean input.")
    else:
        print("  NEITHER MODE IS SAFE: every inference mode regresses clean SDR by more "
              "than 0.5 dB.\n  The model is over-correcting already-valid input -- fix it in "
              "training (raise the\n  clean-input fraction, or the residual_outside weight), "
              "not at inference.")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
