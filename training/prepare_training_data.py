#!/usr/bin/env python3
"""
prepare_training_data.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Radiance — Wan LoRA Training Data Preparation Pipeline

Scans D:\\HDR for TIF / MXF / EXR source files and produces paired
SDR + HDR PNG datasets at 1280×720 in G:\\data ready for Wan LoRA training.

Output layout
─────────────
  G:\\data\\
  ├── sdr\\          8-bit  sRGB PNG  (tone-mapped from scene-linear)
  ├── hdr\\          16-bit linear PNG (scene-linear, normalized to [0,1])
  ├── meta\\         per-frame JSON  (source, encoding, peak_nits, stats)
  └── prepare.log   full run log

Encoding auto-detection (directory-name heuristics + EXR header):
  • Folder contains "alexa"  / "arri"      → ARRI LogC3
  • Folder contains "netflix"/ "chimera"   → PQ  ST.2084
  • Folder contains "polyhaven"            → Scene-linear (EXR)
  • .exr files always read as scene-linear unless header says otherwise
  • .tif 16-bit files assumed PQ  unless "linear" or "exr" in path
  • .tif  8-bit files treated as sRGB SDR

Usage
─────
  python prepare_training_data.py [--src D:\\HDR] [--dst G:\\data]
                                  [--fps 1.0] [--workers 4] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import numpy as np

# ── Optional heavy imports (guarded) ─────────────────────────────────────────
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
SRC_DEFAULT = Path(r"D:\HDR")
DST_DEFAULT = Path(r"G:\data")
TARGET_W    = 1280
TARGET_H    = 720
FRAME_RATE  = 1.0          # frames per second extracted from video containers

# Reference peak for HDR normalisation (scene-linear → [0,1])
# 1 unit scene-linear ≈ 203 nits reference white.
# We normalise so that ~10 000-nit peak maps to ~1.0 (generous headroom).
HDR_PEAK_NITS  = 10_000.0
HDR_REF_NITS   = 203.0
HDR_NORM_SCALE = HDR_REF_NITS / HDR_PEAK_NITS   # multiply scene-linear by this

# Tone-map exposure (scene-linear mid-grey at this EV → SDR ~0.5)
TONEMAP_EV_OFFSET = -1.0   # slight underexpose for safety

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
log = logging.getLogger("radiance.prep")
_log_lock = threading.Lock()


def _setup_logging(dst: Path, verbose: bool = False):
    dst.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(dst / "prepare.log", encoding="utf-8"),
    ]
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S", handlers=handlers)


# ─────────────────────────────────────────────────────────────────────────────
# Encoding detection
# ─────────────────────────────────────────────────────────────────────────────
EncodingHint = Literal["pq", "logc3", "logc4", "slog3", "hlg", "linear", "srgb"]


def _detect_encoding(path: Path) -> EncodingHint:
    """Heuristic: directory name + filename keywords → encoding tag."""
    parts = [p.lower() for p in path.parts]
    combined = " ".join(parts)

    # Folder / filename keyword matching (order matters — most specific first)
    if any(k in combined for k in ("logc4",)):
        return "logc4"
    if any(k in combined for k in ("logc3", "logc", "alexa", "arri")):
        return "logc3"
    if any(k in combined for k in ("slog3", "slog", "venice", "fx9", "fx6")):
        return "slog3"
    if any(k in combined for k in ("hlg",)):
        return "hlg"
    if any(k in combined for k in ("pq", "hdr10", "st2084", "netflix", "chimera")):
        return "pq"
    if any(k in combined for k in ("linear", "polyhaven", "hdri", "aces")):
        return "linear"

    # Extension fallback
    if path.suffix.lower() == ".exr":
        return "linear"   # EXR almost always scene-linear in a VFX context
    if path.suffix.lower() in (".tif", ".tiff"):
        # 16-bit TIF in an HDR folder → assume PQ unless flagged linear above
        return "pq"

    return "srgb"   # safe fallback for 8-bit images


# ─────────────────────────────────────────────────────────────────────────────
# EOTFs (NumPy, no torch required)
# ─────────────────────────────────────────────────────────────────────────────

def eotf_pq(v: np.ndarray, peak_nits: float = 1000.0) -> np.ndarray:
    """ST.2084 PQ EOTF: [0,1] code value → scene-linear (ref white = 203 nits)."""
    M1, M2 = 0.1593017578125, 78.84375
    C1, C2, C3 = 0.8359375, 18.8515625, 18.6875
    v   = np.clip(v, 0.0, 1.0).astype(np.float64)
    vp  = v ** (1.0 / M2)
    num = np.maximum(vp - C1, 0.0)
    den = np.maximum(C2 - C3 * vp, 1e-9)
    L   = (num / den) ** (1.0 / M1)
    # L ∈ [0,1] relative to peak_nits; convert to scene-linear (ref 203 nits)
    return (L * peak_nits / HDR_REF_NITS).astype(np.float32)


def eotf_logc3(v: np.ndarray) -> np.ndarray:
    """ARRI LogC3 EI800 → scene-linear float.
    Delegates to color_utils.logc3_to_linear (verified, EI-parameterised).
    """
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from color_utils import logc3_to_linear
        return logc3_to_linear(v.astype(np.float32), ei=800)
    except ImportError:
        # Inline fallback (EI800 constants from ARRI LogC whitepaper)
        v = v.astype(np.float64)
        cut, a, b, c, d, e, f = (0.010591, 5.555556, 0.052272,
                                  0.247190, 0.385537, 5.367655, 0.092809)
        cut_enc = e * cut + f
        lin = np.where(
            v > cut_enc,
            (10.0 ** ((v - d) / c) - b) / a,
            (v - f) / e,
        )
        return np.maximum(lin, 0.0).astype(np.float32)


def eotf_logc4(v: np.ndarray) -> np.ndarray:
    """ARRI LogC4 (ALEXA 35) → scene-linear float.
    Delegates to color_utils.logc4_to_linear (official spec parameters).
    """
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from color_utils import logc4_to_linear
        return logc4_to_linear(v.astype(np.float32))
    except ImportError:
        # Inline fallback — official ARRI LogC4 v1 parameters
        v   = v.astype(np.float64)
        a   = 2231.826309067637
        b   = 0.9071358691330627
        c   = 0.0928641308669373
        s   = 0.1135773173772412
        t   = -0.0180569961199123
        mask = v >= 0.0
        out  = np.empty_like(v, dtype=np.float64)
        out[mask]  = (np.power(2.0, ((v[mask] - c) / b) * 14.0 + 6.0) - 64.0) / a
        out[~mask] = v[~mask] * s + t
        return np.maximum(out, 0.0).astype(np.float32)


def eotf_slog3(v: np.ndarray) -> np.ndarray:
    """Sony S-Log3 → scene-linear float."""
    v   = v.astype(np.float64)
    lin = np.where(
        v >= 171.2102946929 / 1023.0,
        ((10.0 ** ((v - 0.616596) / 0.432699) - 0.037584) * 0.18 /
         (0.028511 / 0.18)),
        (v - 95.0 / 1023.0) / (171.2102946929 / 1023.0 - 95.0 / 1023.0) *
        0.01125000,
    )
    return np.maximum(lin, 0.0).astype(np.float32)


def eotf_hlg(v: np.ndarray) -> np.ndarray:
    """HLG inverse OETF: signal [0,1] → scene-linear."""
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073
    v = v.astype(np.float64)
    lin = np.where(
        v <= 0.5,
        (v ** 2) / 3.0,
        (np.exp((v - c) / a) + b) / 12.0,
    )
    return np.maximum(lin, 0.0).astype(np.float32)


def eotf_srgb(v: np.ndarray) -> np.ndarray:
    """sRGB EOTF (gamma expansion): [0,1] 8-bit SDR → scene-linear."""
    v = v.astype(np.float64)
    lin = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    return lin.astype(np.float32)


_EOTF = {
    "pq":     lambda v: eotf_pq(v),
    "logc3":  eotf_logc3,
    "logc4":  eotf_logc4,
    "slog3":  eotf_slog3,
    "hlg":    eotf_hlg,
    "linear": lambda v: np.clip(v, 0.0, None).astype(np.float32),
    "srgb":   eotf_srgb,
}


def to_scene_linear(data: np.ndarray, encoding: EncodingHint) -> np.ndarray:
    """Apply the appropriate EOTF and return a float32 scene-linear array."""
    fn = _EOTF.get(encoding, eotf_srgb)
    return fn(data)


# ─────────────────────────────────────────────────────────────────────────────
# Tone-mapping (scene-linear → SDR [0,1])
# ─────────────────────────────────────────────────────────────────────────────

def _apply_exposure(lin: np.ndarray, ev: float) -> np.ndarray:
    return lin * (2.0 ** ev)


def tonemap_aces_approx(lin: np.ndarray) -> np.ndarray:
    """
    ACES filmic approximation (Narkowicz 2015) — punchy, industry-familiar look.
    Input:  scene-linear float32 (any range)
    Output: [0,1] display-linear (apply sRGB OETF afterwards)
    """
    lin = _apply_exposure(lin, TONEMAP_EV_OFFSET)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    x   = np.maximum(lin, 0.0)
    out = (x * (a * x + b)) / (x * (c * x + d) + e)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def oetf_srgb(lin: np.ndarray) -> np.ndarray:
    """Display-linear [0,1] → sRGB gamma-encoded [0,1]."""
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308,
                    lin * 12.92,
                    1.055 * lin ** (1.0 / 2.4) - 0.055).astype(np.float32)


def make_sdr(linear: np.ndarray) -> np.ndarray:
    """Scene-linear → 8-bit uint8 sRGB array (H,W,3)."""
    tone = tonemap_aces_approx(linear)
    srgb = oetf_srgb(tone)
    return (srgb * 255.0).clip(0, 255).astype(np.uint8)


def make_hdr(linear: np.ndarray) -> np.ndarray:
    """
    Scene-linear → 16-bit uint16 array (H,W,3).
    Normalised so that HDR_PEAK_NITS → 65535.
    """
    norm = np.clip(linear * HDR_NORM_SCALE, 0.0, 1.0)
    return (norm * 65535.0).astype(np.uint16)


# ─────────────────────────────────────────────────────────────────────────────
# Resize (center-crop to 16:9, then scale to TARGET)
# ─────────────────────────────────────────────────────────────────────────────

def _center_crop_16_9(img: np.ndarray) -> np.ndarray:
    """Crop the largest 16:9 region from the centre of img (H,W,C)."""
    h, w = img.shape[:2]
    target_ratio = TARGET_W / TARGET_H
    src_ratio    = w / h
    if src_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        return img[:, x0:x0 + new_w]
    else:
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        return img[y0:y0 + new_h, :]


def resize_frame(img: np.ndarray) -> np.ndarray:
    """Crop to 16:9 then resize to TARGET_W × TARGET_H."""
    cropped = _center_crop_16_9(img)
    if HAS_CV2:
        return cv2.resize(cropped, (TARGET_W, TARGET_H),
                          interpolation=cv2.INTER_AREA)
    elif HAS_PIL:
        pil = PILImage.fromarray(cropped)
        pil = pil.resize((TARGET_W, TARGET_H), PILImage.LANCZOS)
        return np.asarray(pil)
    else:
        # Nearest-neighbour fallback (numpy only)
        sy = np.linspace(0, cropped.shape[0] - 1, TARGET_H).astype(int)
        sx = np.linspace(0, cropped.shape[1] - 1, TARGET_W).astype(int)
        return cropped[np.ix_(sy, sx)]


# ─────────────────────────────────────────────────────────────────────────────
# PNG save helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_png_8bit(arr: np.ndarray, path: Path):
    """Save uint8 (H,W,3) as 8-bit PNG."""
    if HAS_PIL:
        PILImage.fromarray(arr, mode="RGB").save(path, format="PNG",
                                                  compress_level=6)
    elif HAS_CV2:
        cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    else:
        raise RuntimeError("PIL or cv2 required to save PNG.")


def save_png_16bit(arr: np.ndarray, path: Path):
    """Save uint16 (H,W,3) as 16-bit PNG."""
    if HAS_PIL:
        # PIL needs mode 'I;16' trick for RGB 16-bit
        # Workaround: save each channel then recombine, or use tifffile for 16-bit
        if HAS_TIFFFILE:
            tifffile.imwrite(str(path.with_suffix(".tif")), arr,
                             photometric="rgb", compression="deflate")
            # rename to .png extension (tifffile can write lossless 16-bit PNG)
            tif_path = path.with_suffix(".tif")
            tif_path.rename(path)
            return
        # Fallback: use PIL per-channel
        PILImage.fromarray(arr[:, :, 0], mode="I").save(
            str(path.with_name(path.stem + "_R.png")))
        raise RuntimeError("tifffile required for 16-bit PNG. "
                           "Install with: pip install tifffile")
    elif HAS_TIFFFILE:
        tifffile.imwrite(str(path), arr, photometric="rgb",
                         compression="deflate")
    elif HAS_CV2:
        cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    else:
        raise RuntimeError("tifffile, PIL, or cv2 required to save 16-bit PNG.")


# ─────────────────────────────────────────────────────────────────────────────
# Image readers
# ─────────────────────────────────────────────────────────────────────────────

def read_tif(path: Path) -> tuple[np.ndarray, int]:
    """
    Read a TIFF file.
    Returns (float32 array [0,1], bit_depth).
    """
    if HAS_TIFFFILE:
        data = tifffile.imread(str(path))
        # Ensure (H,W,3)
        if data.ndim == 2:
            data = np.stack([data] * 3, axis=-1)
        if data.shape[-1] == 4:
            data = data[:, :, :3]
        bit = data.dtype.itemsize * 8
        if data.dtype.kind == "u":
            data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
        else:
            data = data.astype(np.float32)
        return data, bit
    elif HAS_PIL:
        img  = PILImage.open(path).convert("RGB")
        arr  = np.asarray(img)
        bit  = 16 if img.mode == "I;16" else 8
        return arr.astype(np.float32) / (65535.0 if bit == 16 else 255.0), bit
    elif HAS_CV2:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        bit = arr.dtype.itemsize * 8
        return arr.astype(np.float32) / float(np.iinfo(arr.dtype).max), bit
    else:
        raise RuntimeError("tifffile, PIL, or cv2 required to read TIF files.")


def read_exr(path: Path) -> np.ndarray:
    """
    Read an OpenEXR file.
    Returns float32 array [H,W,3] in scene-linear (assumed).
    """
    if HAS_OPENEXR:
        exr  = OpenEXR.InputFile(str(path))
        hdr  = exr.header()
        dw   = hdr["dataWindow"]
        w    = dw.max.x - dw.min.x + 1
        h    = dw.max.y - dw.min.y + 1
        pt   = Imath.PixelType(Imath.PixelType.FLOAT)
        r, g, b = [
            np.frombuffer(exr.channel(c, pt), dtype=np.float32).reshape(h, w)
            for c in ("R", "G", "B")
        ]
        return np.stack([r, g, b], axis=-1)
    elif HAS_CV2:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if arr is None:
            raise RuntimeError(f"cv2 could not read EXR: {path}")
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB).astype(np.float32)
    else:
        raise RuntimeError(
            "OpenEXR or cv2[exr] required to read EXR files.\n"
            "  pip install OpenEXR   (Linux: apt install libopenexr-dev first)\n"
            "  or: pip install opencv-python (with EXR support)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MXF extraction (ffmpeg)
# ─────────────────────────────────────────────────────────────────────────────

def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def extract_mxf_frames(mxf_path: Path,
                       tmp_dir: Path,
                       fps: float = FRAME_RATE) -> list[Path]:
    """
    Extract frames from an MXF container using ffmpeg.
    Frames are written as 16-bit TIFF to preserve full bit depth.
    Returns list of extracted TIFF paths.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = tmp_dir / f"{mxf_path.stem}_%06d.tif"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(mxf_path),
        "-vf", f"fps={fps}",
        "-pix_fmt", "rgb48le",    # 16-bit RGB, little-endian
        "-compression_algo", "raw",
        str(out_pattern),
    ]
    log.debug("ffmpeg: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("ffmpeg failed for %s:\n%s", mxf_path.name, result.stderr[-2000:])
        return []

    extracted = sorted(tmp_dir.glob(f"{mxf_path.stem}_*.tif"))
    log.info("  extracted %d frames from %s", len(extracted), mxf_path.name)
    return extracted


# ─────────────────────────────────────────────────────────────────────────────
# Single-frame processing
# ─────────────────────────────────────────────────────────────────────────────

def process_frame(
    raw: np.ndarray,
    encoding: EncodingHint,
    stem: str,
    sdr_dir: Path,
    hdr_dir: Path,
    meta_dir: Path,
    bit_depth: int = 16,
) -> dict:
    """
    Decode raw pixel data → scene-linear → SDR + HDR PNG pair.
    Returns a dict of metadata written to meta_dir.
    """
    # 1. Decode to scene-linear
    linear = to_scene_linear(raw, encoding)

    # 2. Resize both to 720p
    linear_r = resize_frame(linear)

    # 3. Produce SDR (8-bit) and HDR (16-bit) arrays
    sdr_arr = make_sdr(linear_r)
    hdr_arr = make_hdr(linear_r)

    # 4. Save
    sdr_path = sdr_dir / f"{stem}.png"
    hdr_path = hdr_dir / f"{stem}.png"
    save_png_8bit(sdr_arr, sdr_path)
    save_png_16bit(hdr_arr, hdr_path)

    # 5. Compute stats for the metadata sidecar
    scene_mean  = float(linear_r.mean())
    scene_max   = float(linear_r.max())
    peak_nits   = scene_max * HDR_REF_NITS   # rough nit estimate
    meta = {
        "stem":       stem,
        "encoding":   encoding,
        "bit_depth":  bit_depth,
        "scene_mean": round(scene_mean, 6),
        "scene_max":  round(scene_max, 6),
        "peak_nits":  round(peak_nits, 1),
        "sdr":        str(sdr_path.name),
        "hdr":        str(hdr_path.name),
    }
    (meta_dir / f"{stem}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Per-source-file pipeline
# ─────────────────────────────────────────────────────────────────────────────

def handle_tif(path: Path, idx: int,
               sdr_dir: Path, hdr_dir: Path, meta_dir: Path,
               dry_run: bool) -> int:
    """Process one TIF image. Returns 1 on success, 0 on skip/fail."""
    stem      = f"tif_{idx:07d}_{path.stem}"
    encoding  = _detect_encoding(path)
    if dry_run:
        log.info("[DRY] tif %s → encoding=%s", path.name, encoding)
        return 1
    try:
        raw, bit = read_tif(path)
        process_frame(raw, encoding, stem, sdr_dir, hdr_dir, meta_dir, bit)
        log.debug("  ✓ tif %s (%s)", path.name, encoding)
        return 1
    except Exception as exc:
        log.warning("  ✗ tif %s: %s", path.name, exc)
        return 0


def handle_exr(path: Path, idx: int,
               sdr_dir: Path, hdr_dir: Path, meta_dir: Path,
               dry_run: bool) -> int:
    stem     = f"exr_{idx:07d}_{path.stem}"
    encoding = _detect_encoding(path)   # usually "linear"
    if dry_run:
        log.info("[DRY] exr %s → encoding=%s", path.name, encoding)
        return 1
    try:
        linear = read_exr(path)         # EXR assumed scene-linear already
        # For EXRs, skip the EOTF (already linear) but go through the pipeline
        process_frame(linear, "linear", stem, sdr_dir, hdr_dir, meta_dir, 32)
        log.debug("  ✓ exr %s", path.name)
        return 1
    except Exception as exc:
        log.warning("  ✗ exr %s: %s", path.name, exc)
        return 0


def handle_mxf(path: Path, base_idx: int,
               sdr_dir: Path, hdr_dir: Path, meta_dir: Path,
               fps: float, dry_run: bool) -> int:
    encoding = _detect_encoding(path)
    if dry_run:
        log.info("[DRY] mxf %s → encoding=%s, fps=%.2f", path.name, encoding, fps)
        return 1
    if not _ffmpeg_available():
        log.error("ffmpeg not found — cannot extract MXF frames. Install ffmpeg.")
        return 0
    with tempfile.TemporaryDirectory(prefix="radiance_mxf_") as tmp:
        tmp_dir = Path(tmp)
        frames  = extract_mxf_frames(path, tmp_dir, fps)
        count   = 0
        for i, frame_path in enumerate(frames):
            stem = f"mxf_{base_idx + i:07d}_{path.stem}_f{i:04d}"
            try:
                raw, bit = read_tif(frame_path)
                process_frame(raw, encoding, stem,
                              sdr_dir, hdr_dir, meta_dir, bit)
                count += 1
            except Exception as exc:
                log.warning("  ✗ frame %s: %s", frame_path.name, exc)
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Directory scanner
# ─────────────────────────────────────────────────────────────────────────────

def _scan_sources(src: Path) -> dict[str, list[Path]]:
    """Recursively collect TIF, MXF, EXR under src."""
    found: dict[str, list[Path]] = {"tif": [], "mxf": [], "exr": []}
    for p in sorted(src.rglob("*")):
        ext = p.suffix.lower()
        if ext in (".tif", ".tiff"):
            found["tif"].append(p)
        elif ext == ".mxf":
            found["mxf"].append(p)
        elif ext == ".exr":
            found["exr"].append(p)
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Radiance — Wan LoRA training data preparation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src",      type=Path, default=SRC_DEFAULT,
                        help="Source HDR footage directory (default: D:\\HDR)")
    parser.add_argument("--dst",      type=Path, default=DST_DEFAULT,
                        help="Output dataset directory     (default: G:\\data)")
    parser.add_argument("--fps",      type=float, default=FRAME_RATE,
                        help="Frame rate for MXF extraction (default: 1.0)")
    parser.add_argument("--workers",  type=int, default=4,
                        help="Parallel worker threads (default: 4)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Scan and report without writing any files.")
    parser.add_argument("--verbose",  action="store_true")
    parser.add_argument("--skip-tif", action="store_true")
    parser.add_argument("--skip-mxf", action="store_true")
    parser.add_argument("--skip-exr", action="store_true")
    parser.add_argument(
        "--tif-stride", type=int, default=1,
        help="Take every Nth TIF from a sequence (default: 1 = every frame). "
             "Use 24 to sample 1 frame/sec from a 24fps sequence.")
    args = parser.parse_args()

    dst: Path = args.dst
    _setup_logging(dst, args.verbose)

    log.info("━━━━  Radiance Training Data Preparation  ━━━━")
    log.info("  Source : %s", args.src)
    log.info("  Output : %s", dst)
    log.info("  Target : %d×%d  |  %.1f fps (MXF)  |  stride=%d (TIF)",
             TARGET_W, TARGET_H, args.fps, args.tif_stride)
    log.info("  Backend: tifffile=%s  cv2=%s  OpenEXR=%s  PIL=%s",
             HAS_TIFFFILE, HAS_CV2, HAS_OPENEXR, HAS_PIL)

    if not args.src.exists():
        log.error("Source directory not found: %s", args.src)
        sys.exit(1)

    # ── Create output directories ────────────────────────────────────────────
    sdr_dir  = dst / "sdr"
    hdr_dir  = dst / "hdr"
    meta_dir = dst / "meta"
    for d in (sdr_dir, hdr_dir, meta_dir):
        if not args.dry_run:
            d.mkdir(parents=True, exist_ok=True)

    # ── Scan ─────────────────────────────────────────────────────────────────
    log.info("Scanning %s …", args.src)
    sources = _scan_sources(args.src)
    n_tif   = len(sources["tif"])
    n_mxf   = len(sources["mxf"])
    n_exr   = len(sources["exr"])
    log.info("Found  TIF=%d  MXF=%d  EXR=%d", n_tif, n_mxf, n_exr)

    total_written = 0
    idx = 0

    # ── Process TIF ──────────────────────────────────────────────────────────
    if not args.skip_tif and n_tif > 0:
        strided = sources["tif"][:: args.tif_stride]
        log.info("Processing %d TIF files (stride=%d, %d selected) …",
                 n_tif, args.tif_stride, len(strided))

        def _tif_job(item):
            i, p = item
            return handle_tif(p, i, sdr_dir, hdr_dir, meta_dir, args.dry_run)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_tif_job, (idx + i, p)): p
                       for i, p in enumerate(strided)}
            for fut in as_completed(futures):
                total_written += fut.result()
        idx += len(strided)

    # ── Process EXR ──────────────────────────────────────────────────────────
    if not args.skip_exr and n_exr > 0:
        log.info("Processing %d EXR files …", n_exr)

        def _exr_job(item):
            i, p = item
            return handle_exr(p, i, sdr_dir, hdr_dir, meta_dir, args.dry_run)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_exr_job, (idx + i, p)): p
                       for i, p in enumerate(sources["exr"])}
            for fut in as_completed(futures):
                total_written += fut.result()
        idx += n_exr

    # ── Process MXF ──────────────────────────────────────────────────────────
    # MXF extraction is sequential (ffmpeg already multi-threaded internally)
    if not args.skip_mxf and n_mxf > 0:
        log.info("Processing %d MXF files (%.1f fps) …", n_mxf, args.fps)
        for mxf_path in sources["mxf"]:
            n = handle_mxf(mxf_path, idx, sdr_dir, hdr_dir, meta_dir,
                           args.fps, args.dry_run)
            total_written += n
            idx += n

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("━━━━  Done  ━━━━")
    log.info("  Pairs written : %d", total_written)
    log.info("  SDR  →  %s", sdr_dir)
    log.info("  HDR  →  %s", hdr_dir)
    log.info("  Meta →  %s", meta_dir)

    if total_written == 0:
        log.warning("No frames were written — check source paths and dependencies.")
        sys.exit(1)


if __name__ == "__main__":
    main()
