"""Descriptor normalization statistics for stable RUDRA training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class DescriptorStats:
    mean: list[float]
    std: list[float]
    min: list[float]
    max: list[float]
    p01: list[float]
    p99: list[float]
    count: int

    @classmethod
    def from_tensor(cls, x: torch.Tensor) -> "DescriptorStats":
        if x.ndim != 2:
            raise ValueError(f"Expected descriptor tensor (N, D), got {tuple(x.shape)}")
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=1e6, neginf=-1e6)
        return cls(
            mean=x.mean(0).tolist(),
            std=x.std(0).clamp(min=1e-6).tolist(),
            min=x.min(0).values.tolist(),
            max=x.max(0).values.tolist(),
            p01=torch.quantile(x, 0.01, dim=0).tolist(),
            p99=torch.quantile(x, 0.99, dim=0).tolist(),
            count=int(x.shape[0]),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DescriptorStats":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


class DescriptorNormalizer(torch.nn.Module):
    """Normalize raw descriptors using precomputed dataset statistics."""

    def __init__(self, mean: torch.Tensor | list[float], std: torch.Tensor | list[float], eps: float = 1e-6):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, -1))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).view(1, -1).clamp(min=eps))
        self.eps = eps

    @classmethod
    def from_json(cls, path: str | Path) -> "DescriptorNormalizer":
        s = DescriptorStats.load(path)
        return cls(s.mean, s.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        # std is already clamped to min=eps in __init__, no need for + eps here.
        return torch.nan_to_num((x - mean) / std, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
