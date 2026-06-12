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


def save_exr_placeholder(path: str | Path, tensor: torch.Tensor, metadata: dict[str, Any] | None = None) -> None:
    """Placeholder with a clear error unless OpenEXR is installed.

    This avoids silently writing broken HDR data. Integrate OpenImageIO, OpenEXR,
    or ComfyUI's image pipeline for production EXR export.
    """
    try:
        import OpenEXR  # noqa: F401
        import Imath  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "OpenEXR Python bindings are not installed. Use save_tensor_cache() or install OpenEXR/OpenImageIO."
        ) from exc
    raise NotImplementedError("EXR writing is intentionally host-specific. Wire this to your studio EXR backend.")
