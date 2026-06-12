"""Dataset manifest, validation, and tensor-cache helpers for RUDRA."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.utils.data import Dataset

from .config import FORMAT_TO_ID
from .descriptor import RUDRADescriptor
from .exr_io import load_tensor_cache, save_tensor_cache
from .normalization import normalize_to_scene_linear
from .sampler import SampleHDRStats


@dataclass
class RUDRASample:
    input_path: str
    target_path: str
    format_name: str = "linear"
    shot_id: str = "unknown"
    frame_id: str = "unknown"


@dataclass
class DatasetIssue:
    index: int
    input_path: str
    target_path: str
    severity: str
    message: str


def read_manifest(path: str | Path) -> list[RUDRASample]:
    path = Path(path)
    rows: list[RUDRASample] = []
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data:
            rows.append(RUDRASample(**item))
    else:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(RUDRASample(**{k: row[k] for k in RUDRASample.__annotations__.keys() if k in row and row[k] != ""}))
    return rows


def _load_tensor(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if path.suffix.lower() in {".pt", ".pth"}:
        t, _ = load_tensor_cache(path)
        return t
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        from PIL import Image
        import numpy as np
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img).astype("float32") / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    raise ValueError(f"Unsupported file type for lightweight loader: {path}")


def _sample_stats(index: int, target: torch.Tensor) -> SampleHDRStats:
    y = 0.2627 * target[:, 0:1] + 0.6780 * target[:, 1:2] + 0.0593 * target[:, 2:3]
    flat = y.flatten()
    return SampleHDRStats(
        index=index,
        peak=float(flat.max().cpu()),
        p95=float(torch.quantile(flat.float(), 0.95).cpu()),
        mean=float(flat.mean().cpu()),
        highlight_fraction=float((flat > 1.0).float().mean().cpu()),
    )


def validate_manifest(samples: list[RUDRASample], max_items: Optional[int] = None) -> tuple[list[DatasetIssue], list[SampleHDRStats]]:
    issues: list[DatasetIssue] = []
    stats: list[SampleHDRStats] = []
    for idx, s in enumerate(samples[: max_items or len(samples)]):
        ip, tp = Path(s.input_path), Path(s.target_path)
        if not ip.exists():
            issues.append(DatasetIssue(idx, s.input_path, s.target_path, "error", "input file missing")); continue
        if not tp.exists():
            issues.append(DatasetIssue(idx, s.input_path, s.target_path, "error", "target file missing")); continue
        try:
            inp = _load_tensor(ip).float()
            target = _load_tensor(tp).float()
            if inp.ndim != 4 or target.ndim != 4 or inp.shape[1] != 3 or target.shape[1] != 3:
                issues.append(DatasetIssue(idx, s.input_path, s.target_path, "error", f"invalid tensor shapes {tuple(inp.shape)} {tuple(target.shape)}")); continue
            if inp.shape[-2:] != target.shape[-2:]:
                issues.append(DatasetIssue(idx, s.input_path, s.target_path, "warning", f"resolution mismatch {tuple(inp.shape[-2:])} vs {tuple(target.shape[-2:])}"))
            if not torch.isfinite(inp).all() or not torch.isfinite(target).all():
                issues.append(DatasetIssue(idx, s.input_path, s.target_path, "error", "NaN or Inf values detected")); continue
            if target.abs().max() < 1e-6:
                issues.append(DatasetIssue(idx, s.input_path, s.target_path, "error", "target is fully black")); continue
            stats.append(_sample_stats(idx, target))
        except Exception as exc:
            issues.append(DatasetIssue(idx, s.input_path, s.target_path, "error", str(exc)))
    return issues, stats


class RUDRACacheDataset(Dataset):
    """Dataset for pre-normalized tensor caches.

    Each cache file is expected to hold a tensor in (3,H,W) or (1,3,H,W).
    """

    def __init__(self, manifest_path: str | Path, descriptor_stats_path: str | Path | None = None):
        self.samples = read_manifest(manifest_path)
        self.descriptor = RUDRADescriptor()
        self.normalizer: DescriptorNormalizer | None = None
        if descriptor_stats_path is not None:
            from .stats import DescriptorNormalizer
            self.normalizer = DescriptorNormalizer.from_json(descriptor_stats_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        s = self.samples[index]
        fmt = FORMAT_TO_ID.get(s.format_name, FORMAT_TO_ID["linear"])
        inp = _load_tensor(s.input_path).float()
        target = _load_tensor(s.target_path).float()
        if inp.ndim == 3: inp = inp.unsqueeze(0)
        if target.ndim == 3: target = target.unsqueeze(0)
        inp = normalize_to_scene_linear(inp, fmt)[0]
        target = normalize_to_scene_linear(target, FORMAT_TO_ID["linear"])[0]
        dr = self.descriptor(target.unsqueeze(0))[0]
        if self.normalizer is not None:
            dr = self.normalizer(dr.unsqueeze(0))[0]
        return {"input": inp, "target": target, "dr_raw": dr, "format_id": fmt, "shot_id": s.shot_id, "frame_id": s.frame_id}


def build_cache(samples: list[RUDRASample], out_dir: str | Path, overwrite: bool = False) -> list[RUDRASample]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cached: list[RUDRASample] = []
    for idx, s in enumerate(samples):
        fmt = FORMAT_TO_ID.get(s.format_name, FORMAT_TO_ID["linear"])
        inp_out = out_dir / f"{idx:07d}_input.pt"
        tgt_out = out_dir / f"{idx:07d}_target.pt"
        if overwrite or not inp_out.exists():
            inp = normalize_to_scene_linear(_load_tensor(s.input_path).float(), fmt)
            save_tensor_cache(inp_out, inp[0], {"source": s.input_path, "format": s.format_name, "shot_id": s.shot_id, "frame_id": s.frame_id})
        if overwrite or not tgt_out.exists():
            target = normalize_to_scene_linear(_load_tensor(s.target_path).float(), FORMAT_TO_ID["linear"])
            save_tensor_cache(tgt_out, target[0], {"source": s.target_path, "format": "linear", "shot_id": s.shot_id, "frame_id": s.frame_id})
        cached.append(RUDRASample(str(inp_out), str(tgt_out), "linear", s.shot_id, s.frame_id))
    return cached


def write_manifest(samples: Iterable[RUDRASample], path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(RUDRASample.__annotations__.keys()))
        w.writeheader()
        for s in samples:
            w.writerow(asdict(s))
