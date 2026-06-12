"""Validation preview helpers for RUDRA training."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .config import LUMA_WEIGHTS

# Default luma weights (Rec.2020). Override via function parameters.
_DEFAULT_LUMA = LUMA_WEIGHTS["rec2020"]


def reinhard_tonemap(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(min=0.0)
    return x / (1.0 + x)


def exposure_false_color(x: torch.Tensor, luma_weights: tuple[float, ...] = _DEFAULT_LUMA) -> torch.Tensor:
    """Exposure false-color visualization."""
    wr, wg, wb = luma_weights
    y = wr * x[:, 0:1] + wg * x[:, 1:2] + wb * x[:, 2:3]
    ev = torch.log2(y.clamp(min=1e-8) / 0.18 + 1e-8).clamp(-6, 6)
    norm = (ev + 6) / 12
    return torch.cat([norm, 1.0 - (norm - 0.5).abs() * 2.0, 1.0 - norm], dim=1).clamp(0, 1)


def highlight_error_map(pred: torch.Tensor, target: torch.Tensor, luma_weights: tuple[float, ...] = _DEFAULT_LUMA) -> torch.Tensor:
    """Highlight error heatmap: red = high error, green = low error."""
    wr, wg, wb = luma_weights
    y_t = wr * target[:, 0:1] + wg * target[:, 1:2] + wb * target[:, 2:3]
    y_p = wr * pred[:, 0:1] + wg * pred[:, 1:2] + wb * pred[:, 2:3]
    err = ((y_p - y_t).abs() / y_t.abs().clamp(min=1e-6)).clamp(0, 1)
    return torch.cat([err, 1.0 - err, torch.zeros_like(err)], dim=1)


def save_tensor_png(path: str | Path, image: torch.Tensor) -> None:
    """Save first RGB tensor as PNG using PIL. Tensor is expected in 0-1."""
    from PIL import Image
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = image[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray((img * 255.0 + 0.5).astype("uint8")).save(path)


def write_validation_previews(out_dir: str | Path, inp: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, step: int) -> dict[str, str]:
    out = Path(out_dir) / f"step_{step:07d}"
    files = {
        "input_tm": out / "input_tonemap.png",
        "pred_tm": out / "pred_tonemap.png",
        "target_tm": out / "target_tonemap.png",
        "highlight_error": out / "highlight_error.png",
        "exposure_false_color": out / "exposure_false_color.png",
    }
    save_tensor_png(files["input_tm"], reinhard_tonemap(inp))
    save_tensor_png(files["pred_tm"], reinhard_tonemap(pred))
    save_tensor_png(files["target_tm"], reinhard_tonemap(target))
    save_tensor_png(files["highlight_error"], highlight_error_map(pred, target))
    save_tensor_png(files["exposure_false_color"], exposure_false_color(pred))
    return {k: str(v) for k, v in files.items()}
