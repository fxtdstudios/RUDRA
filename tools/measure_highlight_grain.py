"""Luminance grain of the master chain, binned by the source, before and after
rudra.grain.settle_highlight_grain.

The method a tester used on 25 Sep 2026: high-frequency luminance noise
(output minus its sigma-2 Gaussian) in nits, in areas the SOURCE is flat
(7x7 spread under 1.5 codes), binned by source luma.

    python tools/measure_highlight_grain.py plate.png
    python tools/measure_highlight_grain.py plate.png --checkpoint checkpoints/sdr2hdr_shadow_v1.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rudra.anchor import anchor_to_sdr, srgb_to_linear  # noqa: E402
from rudra.chroma import carry_source_chroma  # noqa: E402
from rudra.decode import decode_sdr  # noqa: E402
from rudra.grain import LUMA_REC2020, settle_highlight_grain  # noqa: E402
from training.infer_sdr2hdr import load_models, predict_image  # noqa: E402

BINS = [(0.0, 0.5), (0.5, 0.9), (0.9, 0.95), (0.95, 0.98), (0.98, 0.999), (0.999, 1.01)]


def grain_by_band(nits: np.ndarray, sdr: np.ndarray) -> list[tuple[float, float, int]]:
    luma = nits @ LUMA_REC2020
    hf = luma - cv2.GaussianBlur(luma, (0, 0), 2.0)
    y = (sdr @ np.array([0.2126, 0.7152, 0.0722])).astype(np.float64)
    spread = np.sqrt(np.maximum(cv2.blur(y * y, (7, 7)) - cv2.blur(y, (7, 7)) ** 2, 0.0))
    flat = spread < 1.5 / 255.0
    rows = []
    for lo, hi in BINS:
        m = (y >= lo) & (y < hi) & flat
        rows.append((float(hf[m].std()), float(luma[m].mean()), int(m.sum())) if m.sum() > 100
                    else (float("nan"), float("nan"), int(m.sum())))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--checkpoint", default=str(REPO / "checkpoints" / "sdr2hdr_shadow_v1.pt"))
    args = ap.parse_args()

    sdr = decode_sdr(Path(args.image).read_bytes()).rgb.astype(np.float64)
    model, _ = load_models(args.checkpoint, None, torch.device("cpu"))
    tensor = torch.from_numpy(sdr.astype(np.float32)).permute(2, 0, 1)[None]
    hdr = predict_image(model, tensor, preserve_outside=True, tile_size=0, overlap=64)
    nits = np.transpose(hdr[0].numpy(), (1, 2, 0)).astype(np.float64) * 10000.0
    nits = carry_source_chroma(anchor_to_sdr(nits, sdr, knee=0.9), sdr, knee=0.99)
    settled = settle_highlight_grain(nits, sdr)

    print(f"{'source luma':>12}  {'source':>8}  {'before':>8}  {'after':>8}  {'mean nits':>10}  pixels")
    for (lo, hi), s, b, a in zip(BINS, grain_by_band(srgb_to_linear(sdr) * 203.0, sdr),
                                 grain_by_band(nits, sdr), grain_by_band(settled, sdr)):
        print(f"{lo:5.3f}-{min(hi, 1.0):5.3f}  {s[0]:8.1f}  {b[0]:8.1f}  {a[0]:8.1f}  {a[1]:10.0f}  {a[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
