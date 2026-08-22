"""EXR I/O policy for RUDRA.

OpenEXR is optional in this package. When it is unavailable, use tensor caches
(.pt/.safetensors) and export EXR from the host application.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_tensor_cache(path: str | Path, tensor: torch.Tensor, metadata: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tensor": tensor.detach().cpu(), "metadata": metadata or {}}, path)


def load_tensor_cache(path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
    # Try safe load first (PyTorch 2.6+), fall back for legacy pickle caches.
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "tensor" in data:
        return data["tensor"], data.get("metadata", {})
    if isinstance(data, torch.Tensor):
        return data, {}
    raise ValueError(f"Unsupported cache format: {path}")


def save_exr(path: str | Path, tensor: torch.Tensor, metadata: dict[str, Any] | None = None,
             half: bool = True) -> Path:
    """Write a (3,H,W) or (H,W,3) linear tensor as an uncompressed EXR.

    Uses the dependency-free writer in ``rudra.delivery.exr`` — no OpenEXR
    wheel required. Metadata values are stored as string attributes.
    """
    from .delivery.exr import write_exr

    array = tensor.detach().cpu().float().numpy()
    if array.ndim == 3 and array.shape[0] in (3, 4):
        array = array.transpose(1, 2, 0)
    attributes = {f"rudra:{k}": str(v) for k, v in (metadata or {}).items()}
    return write_exr(path, array, half=half, attributes=attributes)


# Backwards-compatible alias: the placeholder used to refuse; now it writes.
save_exr_placeholder = save_exr
