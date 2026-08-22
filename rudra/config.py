"""
RUDRA configuration and format registry.

This implementation contains two compatible paths:
  1. RUDRA-Lite / Global: deterministic global descriptor + MLP projection.
  2. RUDRA-Full / Spatial: per-pixel R(x) map + transformer DRE tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

FORMAT_NAMES = [
    "sdr",          # 0  sRGB display-referred
    "pq",           # 1  SMPTE ST.2084 / HDR10 style PQ-coded RGB
    "hlg",          # 2  BT.2100 HLG-coded RGB, approximate inverse OETF
    "logc3",        # 3  ARRI LogC3, approximate decode
    "logc4",        # 4  ARRI LogC4, approximate decode
    "slog3",        # 5  Sony S-Log3, approximate decode
    "vlog",         # 6  Panasonic V-Log, approximate decode
    "log3g10",      # 7  RED Log3G10, approximate decode
    "linear",       # 8  scene-linear EXR / already linear
]
FORMAT_TO_ID = {name: idx for idx, name in enumerate(FORMAT_NAMES)}
FORMAT_DIM = len(FORMAT_NAMES)

from .radiometry import LUMA_REC709, LUMA_REC2020, LUMA_ACESCG

LUMA_WEIGHTS = {
    "rec709":  LUMA_REC709,
    "rec2020": LUMA_REC2020,
    "acescg":  LUMA_ACESCG,
}

@dataclass
class RUDRAConfig:
    model_type: str = "flux"
    latent_channels: int = 16
    # Spatial compression of the source VAE. Decoder output geometry must be
    # exactly latent H/W multiplied by this factor (8 for Flux/SDXL/Wan,
    # 16 for Flux.2 Klein, 32 for LTX-2/2.3).
    vae_spatial_factor: int = 8
    vae_temporal_factor: int = 1

    # Global RUDRA-Lite path.
    dr_raw_dim: int = 26
    dr_proj_dim: int = 64
    format_dim: int = FORMAT_DIM

    # Spatial RUDRA-Full path.
    spatial_descriptor_channels: int = 5
    dre_embed_dim: int = 512
    dre_depth: int = 12
    dre_heads: int = 8
    dre_patch_size: int = 8

    # Adapter / decoder.
    lora_rank: int = 16
    lora_alpha: float = 1.0
    lora_gate_init: float = -2.0
    decoder_channels: int = 64
    decoder_blocks: int = 4
    full_decoder_channels: int = 128
    full_decoder_blocks: int = 7

    # HDR assumptions.
    color_space: Literal["rec2020", "rec709", "acescg"] = "rec2020"
    y_max_nits: float = 10000.0
    highlight_knee: float = 0.85          # legacy (absolute-nit L knee); see below
    # Scene-linear-relative highlight calibration: a highlight is luminance more
    # than `highlight_ev_threshold` stops above 0.18 grey, with a soft ramp of
    # `highlight_ev_softness` stops. Replaces the unreachable absolute-nit knee
    # for relative scene-linear data (diffuse white ≈ 1).
    highlight_ev_threshold: float = 2.0
    highlight_ev_softness: float = 4.0
    output_domain: Literal["scene_linear_positive", "scene_linear_signed", "log", "raw"] = "scene_linear_positive"

    # Model-specific defaults.
    # VAE latent normalization is z = (raw - shift_factor) * scale_factor. Both
    # must match the backbone the latents are fed back into, or Stage 2/3 inject
    # adapters on mis-scaled latents (review §4.4). shift_factor defaults to 0.
    scale_factor: float = 0.18215
    shift_factor: float = 0.0
    log_curve: str = "ARRI LogC4"
    compression_ratio: float = 0.50


RUDRA_MODEL_CONFIGS = {
    # Flux uses BOTH a scale (0.3611) and a shift (0.1159) — not the SD1.5 0.18215.
    "flux": {"latent_channels": 16, "vae_spatial_factor": 8, "scale_factor": 0.3611, "shift_factor": 0.1159, "log_curve": "ARRI LogC4", "compression_ratio": 0.50},
    # Wan/Hunyuan/Cog VAEs use per-channel mean/std internally; scale_factor here
    # is an approximation applied only where a scalar is required.
    "wan": {"latent_channels": 16, "vae_spatial_factor": 8, "vae_temporal_factor": 4, "scale_factor": 0.18215, "log_curve": "ARRI LogC4", "compression_ratio": 0.60},
    "hunyuanvideo": {"latent_channels": 16, "vae_spatial_factor": 8, "vae_temporal_factor": 4, "scale_factor": 0.476986, "log_curve": "ARRI LogC4", "compression_ratio": 0.60},
    "ltx-video": {"latent_channels": 128, "vae_spatial_factor": 32, "vae_temporal_factor": 8, "scale_factor": 1.0, "log_curve": "Sony S-Log3", "compression_ratio": 0.50, "full_decoder_channels": 256},
    "cogvideox": {"latent_channels": 16, "scale_factor": 0.7, "log_curve": "ARRI LogC4", "compression_ratio": 0.45},
    "sd3": {"latent_channels": 16, "scale_factor": 1.5305, "shift_factor": 0.0609, "log_curve": "ARRI LogC4", "compression_ratio": 0.50},
    "sdxl": {"latent_channels": 4, "scale_factor": 0.13025, "log_curve": "ARRI LogC3", "compression_ratio": 0.40},
    "sd15": {"latent_channels": 4, "scale_factor": 0.18215, "log_curve": "ARRI LogC3", "compression_ratio": 0.35},
    "lumina2": {"latent_channels": 16, "scale_factor": 0.3611, "shift_factor": 0.1159, "log_curve": "ARRI LogC4", "compression_ratio": 0.50},
    "pixart": {"latent_channels": 4, "scale_factor": 0.18215, "log_curve": "ARRI LogC3", "compression_ratio": 0.40},
    "kolors": {"latent_channels": 4, "scale_factor": 0.13025, "log_curve": "ARRI LogC3", "compression_ratio": 0.40},
    "aura_flow": {"latent_channels": 4, "scale_factor": 0.13025, "log_curve": "ARRI LogC3", "compression_ratio": 0.40},
}


def make_rudra_config(model_type: str, **overrides) -> RUDRAConfig:
    cfg = RUDRA_MODEL_CONFIGS.get(model_type, RUDRA_MODEL_CONFIGS["flux"]).copy()
    # The unified registry is authoritative when available. Filter its richer
    # deployment metadata to fields understood by this dataclass.
    try:
        from config.model_map import resolve_model_vae_config
        resolved = resolve_model_vae_config(model_type) or {}
        allowed = {f.name for f in fields(RUDRAConfig)}
        cfg.update({k: v for k, v in resolved.items() if k in allowed})
    except Exception:
        pass
    cfg["model_type"] = model_type
    cfg.update(overrides)
    return RUDRAConfig(**cfg)
