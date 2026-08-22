"""Model URL map and checkpoint presets — single source of truth for model resources.

Every model URL, preset configuration, and architecture heuristic lives here
so loader_utils and other modules never hardcode URLs or heuristics.
"""
from __future__ import annotations

RADIANCE_MODEL_MAP: dict = {
    "flux1-schnell-fp8.safetensors": {
        "url": "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-schnell-fp8.safetensors",
        "type": "diffusion_models",
    },
    "flux1-dev-fp8.safetensors": {
        "url": "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-dev-fp8.safetensors",
        "type": "diffusion_models",
    },
    "sd_xl_base_1.0.safetensors": {
        "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
        "type": "diffusion_models",
    },
    "t5xxl_fp8_e4m3fn.safetensors": {
        "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
        "type": "text_encoders",
    },
    "clip_l.safetensors": {
        "url": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
        "type": "text_encoders",
    },
    "ae.safetensors": {
        "url": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors",
        "type": "vae",
    },
    # LTX-2.3 transformer (46 GB) — only needed for generation, NOT for decoder training.
    "ltx-2.3-22b-dev.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-dev.safetensors",
        "type": "diffusion_models",
    },
    "ltx-2.3-22b-distilled.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled.safetensors",
        "type": "diffusion_models",
    },
    # LTX-2 Video-VAE (AutoencoderKLLTX2Video, 32x/8x/128ch) — this is all the RUDRA
    # decoder training needs to encode HDR -> latents. Shared by LTX-2 and LTX-2.3.
    "ltx2_vae_config.json": {
        "url": "https://huggingface.co/Lightricks/LTX-2/resolve/main/vae/config.json",
        "type": "vae",
    },
    "ltx2_vae.safetensors": {
        "url": "https://huggingface.co/Lightricks/LTX-2/resolve/main/vae/diffusion_pytorch_model.safetensors",
        "type": "vae",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  Unified per-model VAE / training configuration  (single source of truth)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Canonical keys match the user-facing names in RADIANCE_MODEL_PRESETS.
#  All training scripts, decoders, and HDR nodes should resolve through
#  resolve_model_vae_config() rather than hard-coding values.

MODEL_VAE_CONFIG: dict[str, dict] = {
    "ltx-video": {
        "latent_channels":     128,
        "scale_factor":        1.0,
        "log_curve":           "Sony S-Log3",
        "compression_ratio":     0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  32,
        "vae_temporal_factor": 8,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  128,
        "text_embed_hidden":   4096,
        "clip_slots":          ["llm_encoder", "text_projection"],
        "notes":               "LTX-Video / LTX-2.3 Video-VAE (AutoencoderKLLTX2Video): "
                               "spatial_compression_ratio=32, temporal=8, latent_channels=128, "
                               "scaling_factor=1.0. The 32x spatial factor -> log2(32)=5 decoder "
                               "upsample stages (was wrongly 8 -> 3 stages, which broke real-LTX decode).",
    },
    "flux": {
        "latent_channels":     16,
        "scale_factor":        0.3611,
        "shift_factor":        0.1159,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  256,
        "text_embed_hidden":   4096,
        "clip_slots":          ["clip_l", "t5xxl"],
        "notes":               "Flux.1 image model — VAE uses scale 0.3611 AND shift 0.1159 (review §4.4).",
    },
    "zimage": {
        "latent_channels": 16, "scale_factor": 0.3611, "shift_factor": 0.1159,
        "log_curve": "ARRI LogC4", "compression_ratio": 0.50, "vae_spatial_factor": 8,
        "noise_schedule": "flow", "text_embed_hidden": 2560,
        "notes": "Z-Image — uses the FLUX.1 VAE (16ch); flux decoder works directly.",
    },
    "qwen": {
        "latent_channels": 16, "scale_factor": 0.3611,
        "log_curve": "ARRI LogC4", "compression_ratio": 0.50, "vae_spatial_factor": 8,
        "noise_schedule": "flow", "text_embed_hidden": 3584,
        "notes": "Qwen-Image — 16ch VAE (qwen_image_vae.safetensors), 8x.",
    },
    "flux2-klein": {
        "latent_channels": 128, "scale_factor": 0.3611,
        "log_curve": "ARRI LogC4", "compression_ratio": 0.50, "vae_spatial_factor": 16,
        "noise_schedule": "flow", "text_embed_hidden": 7680,
        "notes": "Flux.2 Klein — 128ch latent (post pixel-shuffle), 16x compression -> 4 upsample stages.",
    },
    "cogvideox": {
        "latent_channels":     16,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.45,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 4,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  77,
        "text_embed_hidden":   4096,
        "clip_slots":          ["t5xxl"],
        "notes":               "CogVideoX 3D causal VAE; tighter latent distribution → lower ratio.",
    },
    "wan": {
        "latent_channels":     16,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.60,
        "norm_center":         3.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 4,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  128,
        "text_embed_hidden":   4096,
        "clip_slots":          ["t5xxl"],
        "notes":               "Wan 2.1 Flow-VAE; different latent mean target → higher compression.",
    },
    "hunyuanvideo": {
        "latent_channels":     16,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.60,
        "norm_center":         3.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 4,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  128,
        "text_embed_hidden":   4096,
        "clip_slots":          ["llm_encoder", "clip_l"],
        "notes":               "HunyuanVideo; similar latent conventions to Wan.",
    },
    "sd3": {
        "latent_channels":     16,
        "scale_factor":        1.5305,
        "shift_factor":        0.0609,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  154,
        "text_embed_hidden":   4096,
        "clip_slots":          ["clip_l", "clip_g", "t5xxl"],
        "notes":               "SD 3 / SD 3.5 — scale 1.5305, shift 0.0609 (review §4.4).",
    },
    "sdxl": {
        "latent_channels":     4,
        "scale_factor":        0.13025,
        "log_curve":           "ARRI LogC3",
        "compression_ratio":   0.40,
        "norm_center":         2.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "ddpm",
        "text_embed_seq_len":  77,
        "text_embed_hidden":   2048,
        "clip_slots":          ["clip_l", "clip_g"],
        "notes":               "SDXL classic 4ch VAE; scale_factor 0.13025 (review §4.4).",
    },
    "sd15": {
        "latent_channels":     4,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC3",
        "compression_ratio":   0.35,
        "norm_center":         2.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "ddpm",
        "text_embed_seq_len":  77,
        "text_embed_hidden":   768,
        "clip_slots":          ["clip_l"],
        "notes":               "SD 1.x / 2.x — narrowest latent dynamic range.",
    },
    "lumina2": {
        "latent_channels":     16,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC4",
        "compression_ratio":   0.50,
        "norm_center":         3.0,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "flow",
        "text_embed_seq_len":  77,
        "text_embed_hidden":   4096,
        "clip_slots":          ["t5xxl"],
        "notes":               "Lumina-Next; 16ch latent flow-matching model.",
    },
    "pixart": {
        "latent_channels":     4,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC3",
        "compression_ratio":   0.40,
        "norm_center":         2.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "ddpm",
        "text_embed_seq_len":  77,
        "text_embed_hidden":   4096,
        "clip_slots":          ["t5xxl"],
        "notes":               "PixArt-Σ / PixArt-α; 4ch latent DiT with T5 conditioning.",
    },
    "kolors": {
        "latent_channels":     4,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC3",
        "compression_ratio":   0.40,
        "norm_center":         2.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "ddpm",
        "text_embed_seq_len":  77,
        "text_embed_hidden":   2048,
        "clip_slots":          ["llm_encoder"],
        "notes":               "Kolors; 4ch latent with ChatGLM text encoder.",
    },
    "aura_flow": {
        "latent_channels":     4,
        "scale_factor":        0.18215,
        "log_curve":           "ARRI LogC3",
        "compression_ratio":   0.40,
        "norm_center":         2.5,
        "vae_spatial_factor":  8,
        "vae_temporal_factor": 1,
        "noise_schedule":      "ddpm",
        "text_embed_seq_len":  77,
        "text_embed_hidden":   768,
        "clip_slots":          ["clip_l"],
        "notes":               "AuraFlow; 4ch latent flow model.",
    },
}

# Aliases so that architecture detectors (model/detect.py) and CLI args
# resolve to the same canonical config regardless of naming convention.
_MODEL_VAE_ALIASES: dict[str, str] = {
    # LTX variants
    "ltx":         "ltx-video",
    "ltxv":        "ltx-video",
    "ltx-video":   "ltx-video",
    "ltxav":       "ltx-video",
    # Flux variants
    "flux1":       "flux",
    "flux.1":      "flux",
    "flux-dev":    "flux",
    "flux-schnell":"flux",
    # CogVideoX variants
    "cogvideo":    "cogvideox",
    "cogvideox5b": "cogvideox",
    # Wan variants
    "wanvideo":    "wan",
    "wan2":        "wan",
    "wan2.1":      "wan",
    "wan-2.1":     "wan",
    # Hunyuan variants
    "hunyuan":     "hunyuanvideo",
    "hunyuan_video":"hunyuanvideo",
    "hyvideo":     "hunyuanvideo",
    # SD3 variants
    "sd3.5":       "sd3",
    "sd3":         "sd3",
    "stable-diffusion-3": "sd3",
    # SDXL variants
    "sd_xl":       "sdxl",
    "sdxl-base":   "sdxl",
    # SD1.x variants
    "sd1":         "sd15",
    "sd2":         "sd15",
    "sd1.5":       "sd15",
    "sd-1.5":      "sd15",
    "v1-5":        "sd15",
    # Lumina2
    "lumina":      "lumina2",
    "lumina-next": "lumina2",
    # PixArt
    "pixart_sigma":"pixart",
    "pixart-alpha":"pixart",
    # Kolors
    "kolors":      "kolors",
    # AuraFlow
    "auraflow":    "aura_flow",
    "aura":        "aura_flow",
}


def resolve_model_vae_config(model_hint: str) -> dict | None:
    """
    Look up the unified VAE/training config for a model name or alias.

    Args:
        model_hint: Any model name string (e.g. 'flux', 'wan2.1',
                    'hunyuan_video', 'sd1.5'). Case-insensitive.

    Returns:
        A dict with all VAE/training parameters, or None if not found.
    """
    key = model_hint.strip().lower()
    if key in MODEL_VAE_CONFIG:
        return MODEL_VAE_CONFIG[key]
    if key in _MODEL_VAE_ALIASES:
        canonical = _MODEL_VAE_ALIASES[key]
        return MODEL_VAE_CONFIG.get(canonical)
    # Partial substring match against aliases
    for alias, canonical in _MODEL_VAE_ALIASES.items():
        if alias in key:
            return MODEL_VAE_CONFIG.get(canonical)
    # Partial match against canonical names
    for canonical in MODEL_VAE_CONFIG:
        if canonical in key:
            return MODEL_VAE_CONFIG[canonical]
    return None


def get_model_vae_param(model_hint: str, param: str, default=None):
    """
    Convenience: get a single parameter from the unified config.

    Example:
        scale = get_model_vae_param("flux", "scale_factor", default=0.18215)
    """
    cfg = resolve_model_vae_config(model_hint)
    if cfg is None:
        return default
    return cfg.get(param, default)


CHECKPOINT_PRESETS: dict = {
    "None (Manual)": {},
    "Flux Dev": {
        "model_type": "flux",
        "weight_dtype": "fp8_e4m3fn",
        "clip_dtype": "fp16",
        "offload_mode": "none",
        "clip_slots": {"clip_l": True, "t5xxl": True},
        "vram_gb": 12,
        "unet_hints": ["flux1-dev-fp8", "flux1-dev", "flux-dev"],
        "vae_hints": ["ae.safetensors", "flux_ae", "ae_"],
        "clip_hints": {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "t5xxl": ["t5xxl_fp8_e4m3fn", "t5xxl_fp16", "t5xxl"],
        },
    },
    "Flux Schnell": {
        "model_type": "flux",
        "weight_dtype": "fp8_e4m3fn",
        "clip_dtype": "fp16",
        "offload_mode": "none",
        "clip_slots": {"clip_l": True, "t5xxl": True},
        "vram_gb": 10,
        "unet_hints": ["flux1-schnell-fp8", "flux1-schnell", "flux-schnell"],
        "vae_hints": ["ae.safetensors", "flux_ae", "ae_"],
        "clip_hints": {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "t5xxl": ["t5xxl_fp8_e4m3fn", "t5xxl_fp16", "t5xxl"],
        },
    },
    "Flux Dev (Low VRAM)": {
        "model_type": "flux",
        "weight_dtype": "fp8_e4m3fn",
        "clip_dtype": "fp8_e4m3fn",
        "offload_mode": "cpu_offload",
        "clip_slots": {"clip_l": True, "t5xxl": True},
        "vram_gb": 8,
        "unet_hints": ["flux1-dev-fp8", "flux1-dev", "flux-dev"],
        "vae_hints": ["ae.safetensors", "flux_ae", "ae_"],
        "clip_hints": {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "t5xxl": ["t5xxl_fp8_e4m3fn", "t5xxl_fp16", "t5xxl"],
        },
    },
    "SD3.5 Large": {
        "model_type": "sd3.5",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
        "offload_mode": "none",
        "clip_slots": {"clip_l": True, "clip_g": True, "t5xxl": True},
        "vram_gb": 16,
        "unet_hints": ["sd3.5_large_turbo", "sd3.5_large", "sd3-5_large"],
        "vae_hints": ["sd3_vae", "sd3.5_vae", "sd3"],
        "clip_hints": {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "clip_g": ["clip_g.safetensors", "clip_g"],
            "t5xxl": ["t5xxl_fp8_e4m3fn", "t5xxl_fp16", "t5xxl"],
        },
    },
    "SD3.5 Medium": {
        "model_type": "sd3.5",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
        "offload_mode": "none",
        "clip_slots": {"clip_l": True, "clip_g": True, "t5xxl": True},
        "vram_gb": 10,
        "unet_hints": ["sd3.5_medium", "sd3-5_medium"],
        "vae_hints": ["sd3_vae", "sd3.5_vae", "sd3"],
        "clip_hints": {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "clip_g": ["clip_g.safetensors", "clip_g"],
            "t5xxl": ["t5xxl_fp8_e4m3fn", "t5xxl_fp16", "t5xxl"],
        },
    },
    "SDXL Base": {
        "model_type": "sdxl",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
        "offload_mode": "none",
        "clip_slots": {"clip_l": True, "clip_g": True},
        "vram_gb": 8,
        "unet_hints": ["sd_xl_base", "sdxl_base", "sdxl-base"],
        "vae_hints": ["sdxl_vae", "vae-ft-mse", "xl_vae"],
        "clip_hints": {
            "clip_l": ["clip_l.safetensors", "clip_l"],
            "clip_g": ["clip_g.safetensors", "clip_g"],
        },
    },
    "SD 1.5": {
        "model_type": "sd1.5",
        "weight_dtype": "fp16",
        "clip_dtype": "fp16",
        "offload_mode": "none",
        "clip_slots": {"clip_l": True},
        "vram_gb": 4,
        "unet_hints": ["v1-5", "v1_5", "sd15", "sd-1-5", "sd_1.5"],
        "vae_hints": ["vae-ft-mse", "sd15_vae", "kl-f8"],
        "clip_hints": {
            "clip_l": ["clip_l.safetensors", "clip_l"],
        },
    },
}
