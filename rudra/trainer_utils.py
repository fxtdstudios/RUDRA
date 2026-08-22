"""Run-integrity helpers shared by the trainers (audit 2026-08-22).

Three things every long training run needs and three of the four trainers
lacked (train_sdr2hdr already had its own versions):

  - ``seed_everything``: one call seeds python/numpy/torch/cuda so a run can
    be reproduced. Deliberately does NOT force deterministic kernels —
    ``torch.use_deterministic_algorithms`` costs real speed and some HDR ops
    have no deterministic implementation; seeding is about run-to-run
    comparability, not bitwise replay.
  - ``atomic_torch_save`` / ``atomic_safetensors_save``: write to a temp file
    in the same directory, then ``os.replace`` — so a crash or power cut
    mid-save can never leave a truncated ``best.pt``/``*_ema_best`` behind on
    a 50k-step run. ``os.replace`` is atomic on the same filesystem on both
    NTFS and POSIX.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

__all__ = ["seed_everything", "atomic_torch_save", "atomic_safetensors_save"]


def seed_everything(seed: int) -> int:
    """Seed python, numpy, torch (CPU + all CUDA devices). Returns the seed."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _replace(tmp: Path, path: Path) -> None:
    os.replace(tmp, path)


def atomic_torch_save(payload, path: str | Path) -> Path:
    """``torch.save`` with write-temp-then-rename semantics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    _replace(tmp, path)
    return path


def atomic_safetensors_save(state_dict, path: str | Path, metadata: dict | None = None) -> Path:
    """``safetensors.torch.save_file`` with write-temp-then-rename semantics."""
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if metadata is not None:
        save_file(state_dict, str(tmp), metadata=metadata)
    else:
        save_file(state_dict, str(tmp))
    _replace(tmp, path)
    return path
