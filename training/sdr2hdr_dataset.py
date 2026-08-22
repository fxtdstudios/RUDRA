"""Paired pixel datasets for direct SDR-to-HDR image and clip training."""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    return records


def load_rgb(path: str | Path, hdr: bool) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if np.issubdtype(image.dtype, np.integer):
        scale = float(np.iinfo(image.dtype).max)
        image = image.astype(np.float32) / scale
    else:
        image = image.astype(np.float32)
    if not hdr:
        image = np.clip(image, 0.0, 1.0)
    if not np.isfinite(image).all():
        raise ValueError(f"NaN or Inf pixels in {path}")
    return np.ascontiguousarray(image)


def _crop_params(height: int, width: int, size: int, training: bool) -> tuple[int, int, int, int]:
    scale = max(size / height, size / width)
    out_h, out_w = max(size, round(height * scale)), max(size, round(width * scale))
    if training:
        top = random.randint(0, out_h - size)
        left = random.randint(0, out_w - size)
    else:
        top, left = (out_h - size) // 2, (out_w - size) // 2
    return out_h, out_w, top, left


def _resize_crop(image: torch.Tensor, params: tuple[int, int, int, int], size: int) -> torch.Tensor:
    out_h, out_w, top, left = params
    image = F.interpolate(image[None], size=(out_h, out_w), mode="bilinear", align_corners=False)[0]
    return image[:, top:top + size, left:left + size]


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.pow(1 / 2.4) - 0.055)


def degrade_sdr(sdr: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Camera/codec-like augmentation for unknown 8-bit SDR inputs."""
    if strength <= 0:
        return sdr
    x = sdr
    linear = _srgb_to_linear(x)
    ev = random.uniform(-0.65, 0.65) * strength
    linear = linear * (2.0 ** ev)
    x = _linear_to_srgb(linear)
    contrast = 1.0 + random.uniform(-0.22, 0.22) * strength
    x = (x - 0.5) * contrast + 0.5
    gamma = 1.0 + random.uniform(-0.18, 0.18) * strength
    x = x.clamp(0.0, 1.0).pow(gamma)
    gray = x.mean(0, keepdim=True)
    saturation = 1.0 + random.uniform(-0.25, 0.25) * strength
    x = gray + saturation * (x - gray)
    wb = torch.tensor(
        [random.uniform(0.92, 1.08), random.uniform(0.97, 1.03), random.uniform(0.92, 1.08)],
        dtype=x.dtype, device=x.device,
    )[:, None, None]
    x = x * wb

    if random.random() < 0.35 * strength:
        # Approximate 4:2:0 damage by blurring chroma at half resolution.
        y = 0.299 * x[0:1] + 0.587 * x[1:2] + 0.114 * x[2:3]
        chroma = x - y
        chroma = F.interpolate(F.interpolate(chroma[None], scale_factor=0.5, mode="bilinear", align_corners=False),
                               size=x.shape[-2:], mode="bilinear", align_corners=False)[0]
        x = y + chroma
    if random.random() < 0.60 * strength:
        sigma = random.uniform(0.0, 3.0 / 255.0) * strength
        x = x + torch.randn_like(x) * sigma
    if random.random() < 0.45 * strength:
        bits = random.choice((6, 7, 8))
        levels = float(2**bits - 1)
        x = torch.round(x * levels) / levels
    if random.random() < 0.25 * strength:
        # Real JPEG round-trip supplies blocking/ringing that tensor-only noise
        # cannot reproduce. This runs in DataLoader workers, not on the GPU.
        rgb8 = (x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        bgr8 = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
        quality = random.randint(35, 92)
        ok, encoded = cv2.imencode(".jpg", bgr8, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = torch.from_numpy(decoded).permute(2, 0, 1).to(dtype=x.dtype, device=x.device)
    return x.clamp(0.0, 1.0)


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()


class SDRHDRDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        crop_size: int = 256,
        augment: bool | None = None,
        augmentation_strength: float = 1.0,
        degradation_probability: float = 0.65,
        max_items: int | None = None,
    ):
        records = read_jsonl(manifest_path)
        self.records = [r for r in records if r.get("split") == split]
        if max_items:
            self.records = self.records[:max_items]
        if not self.records:
            raise ValueError(f"No {split!r} records in {manifest_path}")
        self.split = split
        self.crop_size = int(crop_size)
        self.augment = split == "train" if augment is None else bool(augment)
        self.augmentation_strength = float(augmentation_strength)
        self.degradation_probability = float(degradation_probability)
        if not 0.0 <= self.degradation_probability <= 1.0:
            raise ValueError("degradation_probability must be between 0 and 1")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        sdr = _to_tensor(load_rgb(record["sdr_path"], hdr=False))
        hdr = _to_tensor(load_rgb(record["hdr_path"], hdr=True))
        if sdr.shape[-2:] != hdr.shape[-2:]:
            raise ValueError(f"Pair geometry mismatch for {record['asset_id']}: {sdr.shape} vs {hdr.shape}")
        params = _crop_params(sdr.shape[-2], sdr.shape[-1], self.crop_size, self.augment)
        sdr = _resize_crop(sdr, params, self.crop_size)
        hdr = _resize_crop(hdr, params, self.crop_size)
        if self.augment and random.random() < 0.5:
            sdr, hdr = torch.flip(sdr, (-1,)), torch.flip(hdr, (-1,))
        clean_sdr = sdr.clone()
        # Keep a substantial clean-input fraction. Otherwise a model trained
        # only on altered tone curves learns to "correct" already valid SDR.
        if self.augment and random.random() < self.degradation_probability:
            sdr = degrade_sdr(sdr, self.augmentation_strength)
        return {
            "sdr": sdr, "clean_sdr": clean_sdr, "hdr": hdr.clamp_min(0.0),
            "asset_id": str(record["asset_id"]), "scene_id": str(record["scene_id"]),
        }


class SDRHDRVideoDataset(Dataset):
    """Real consecutive clips from ``build_video_manifest.py`` output."""

    def __init__(self, manifest_path: str | Path, crop_size: int = 256, augment: bool = True,
                 max_items: int | None = None, split: str | None = None,
                 val_fraction: float = 0.10):
        records = read_jsonl(manifest_path)
        if split in {"train", "val"} and val_fraction > 0:
            scenes = sorted({str(r["scene_id"]) for r in records})
            n_val = max(1, round(len(scenes) * val_fraction))
            val_scenes = set(scenes[-n_val:])
            records = [r for r in records if (str(r["scene_id"]) in val_scenes) == (split == "val")]
        self.records = records
        if max_items:
            self.records = self.records[:max_items]
        if not self.records:
            raise ValueError(f"No video clips selected from {manifest_path} for split={split!r}")
        self.crop_size = int(crop_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        sdr = torch.stack([_to_tensor(load_rgb(p, False)) for p in record["sdr_frames"]])
        hdr = torch.stack([_to_tensor(load_rgb(p, True)) for p in record["hdr_frames"]])
        if sdr.shape != hdr.shape:
            raise ValueError(f"Clip geometry mismatch: {record['clip_id']}")
        params = _crop_params(sdr.shape[-2], sdr.shape[-1], self.crop_size, self.augment)
        sdr = torch.stack([_resize_crop(frame, params, self.crop_size) for frame in sdr])
        hdr = torch.stack([_resize_crop(frame, params, self.crop_size) for frame in hdr])
        if self.augment and random.random() < 0.5:
            sdr, hdr = torch.flip(sdr, (-1,)), torch.flip(hdr, (-1,))
        if self.augment:
            # One tone/codec transform for the whole clip avoids manufacturing flicker.
            state = random.getstate()
            torch_state = torch.random.get_rng_state()
            augmented = []
            for frame in sdr:
                random.setstate(state)
                torch.random.set_rng_state(torch_state)
                augmented.append(degrade_sdr(frame))
            sdr = torch.stack(augmented)
        return {"sdr": sdr, "hdr": hdr.clamp_min(0.0), "clip_id": str(record["clip_id"])}
