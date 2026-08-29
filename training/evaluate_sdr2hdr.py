"""Evaluate direct SDR-to-HDR recovery against its inverse-tone-map baseline."""

from __future__ import annotations

import argparse
import csv
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

from rudra.hdrvdp import hdr_vdp3_jod  # noqa: E402
from rudra.sdr2hdr import SDR2HDRNet  # noqa: E402
from training.sdr2hdr_dataset import SDRHDRDataset  # noqa: E402


def psnr(mse: torch.Tensor, peak: float = 1.0) -> float:
    return float(20.0 * math.log10(peak) - 10.0 * torch.log10(mse.clamp_min(1e-12)))


def score(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_log, target_log = torch.log1p(pred * 16.0), torch.log1p(target * 16.0)
    pred_tm, target_tm = pred / (1.0 + pred), target / (1.0 + target)
    mask = (target.max(1, keepdim=True).values > 0.85).float()
    highlight_error = ((pred_log - target_log).abs() * mask).sum() / (mask.sum() * 3.0 + 1e-6)
    return {
        "psnr_log": psnr(F.mse_loss(pred_log, target_log), math.log1p(16.0)),
        "psnr_tm": psnr(F.mse_loss(pred_tm, target_tm)),
        "log_l1": float(F.l1_loss(pred_log, target_log)),
        "highlight_log_l1": float(highlight_error),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--output", type=Path, default=Path("hdrdata/eval/sdr2hdr_test.csv"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cvvdp", action="store_true")
    # Match training/infer_sdr2hdr.py: the shipped inference mode is the one
    # that must be measured. See training/sweep_inference.py for why it is on.
    parser.add_argument("--preserve-outside", action=argparse.BooleanOptionalAction,
                        default=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model = SDR2HDRNet.from_config(config)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.to(device).eval()
    dataset = SDRHDRDataset(args.manifest, split=args.split, crop_size=args.crop_size,
                            augment=False, max_items=args.max_items)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    rows = []
    with torch.inference_mode():
        for index, batch in enumerate(loader, 1):
            sdr, target = batch["sdr"].to(device), batch["hdr"].to(device)
            output = model(sdr, preserve_outside=args.preserve_outside)
            row: dict[str, object] = {"asset_id": batch["asset_id"][0]}
            for prefix, prediction in (("model", output.hdr), ("baseline", output.baseline)):
                for name, value in score(prediction, target).items():
                    row[f"{prefix}_{name}"] = value
                if args.cvvdp:
                    # Dataset normalization defines 1.0 as 10000 nits. CVVDP's
                    # wrapper defines 1.0 as 200 nits, so scale radiance by 50.
                    jod, backend = hdr_vdp3_jod(prediction * 50.0, target * 50.0)
                    row[f"{prefix}_cvvdp_jod"] = jod
                    row[f"{prefix}_cvvdp_backend"] = backend
            rows.append(row)
            print(f"[{index}/{len(dataset)}] {row['asset_id']} model PSNR_log={row['model_psnr_log']:.2f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in rows[0]
        if key != "asset_id" and not key.endswith("backend")
    }
    summary["items"] = len(rows)
    summary["checkpoint"] = str(Path(args.checkpoint).resolve())
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
