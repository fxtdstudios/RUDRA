"""RUDRA: dynamic-range conditioned adapters for HDR-aware diffusion models.

Version v0.3.2 Delivery Layer Build.

Exports:
  - RUDRA-Lite: global descriptor + projection + FiLM decoder + DR-gated LoRA.
  - RUDRA-Full: spatial R(x) descriptor + transformer DRE tokens + cross-attention injection.
  - Training validation utilities: dataset checks, cache building, stats, sampler,
    augmentation, loss scheduling, metrics, and preview generation.
  - rudra.delivery (torch-free): ACES 2065-1 EXR export, OCIO config, Dolby
    Vision L1 / HDR10+ dynamic metadata analysis, artist grade controls,
    headless CLI, and the Stuttgart/UBC benchmark harness.

The delivery layer is deliberately importable WITHOUT torch so it can run on
render nodes and colorist machines with no CUDA stack. When torch is absent the
package root degrades gracefully: model-side symbols raise an informative
ImportError, TORCH_AVAILABLE is False, and ``import rudra.delivery`` works.
"""

import importlib.util as _ilu

TORCH_AVAILABLE = _ilu.find_spec("torch") is not None

__version__ = "0.3.2"

if TORCH_AVAILABLE:
    try:
        from .config import RUDRAConfig, RUDRA_MODEL_CONFIGS, FORMAT_NAMES, FORMAT_TO_ID, FORMAT_DIM, make_rudra_config
        from .normalization import normalize_to_scene_linear
        from .descriptor import RUDRADescriptor, format_onehot
        from .spatial_descriptor import RUDRASpatialDescriptor
        from .encoder import RUDRAProjection
        from .dre_transformer import RUDRADynamicRangeEncoder
        from .decoder import RUDRADecoder, RUDRAFullDecoder, load_rudra_decoder
        from .adapter import RUDRALoRALinear, inject_rudra_lora, rudra_gate_regularization_loss
        from .cross_attention import RUDRACrossAttentionProjection, RUDRACrossAttentionInjector, inject_rudra_cross_attention
        from .pipeline import RUDRAPipeline, PipelineMode
        from .augment import RUDRAAugmentConfig, RUDRAHDRAugment
        from .losses import (
            highlight_preservation_loss, chromaticity_loss, exposure_consistency_loss,
            perceptual_loss, latent_alignment_loss, rudra_reconstruction_loss,
        )
        from .stats import DescriptorStats, DescriptorNormalizer
        from .sampler import SampleHDRStats, HighlightBalancedSampler, bucket_from_stats
        from .train_modes import RUDRATrainMode, LossWeights, scheduled_loss_weights
        from .metrics import validation_metrics, highlight_reconstruction_accuracy, exposure_ev_error, psnr, ssim, delta_e_2000
        from .hdrvdp import hdr_vdp3_jod, colorvideovdp_available
        from .sdr2hdr import SDR2HDRNet, TemporalHDRRefiner, SDR2HDROutput, canonicalize_sdr, sdr2hdr_loss
    except ImportError:
        from config import RUDRAConfig, RUDRA_MODEL_CONFIGS, FORMAT_NAMES, FORMAT_TO_ID, FORMAT_DIM, make_rudra_config
        from normalization import normalize_to_scene_linear
        from descriptor import RUDRADescriptor, format_onehot
        from spatial_descriptor import RUDRASpatialDescriptor
        from encoder import RUDRAProjection
        from dre_transformer import RUDRADynamicRangeEncoder
        from decoder import RUDRADecoder, RUDRAFullDecoder, load_rudra_decoder
        from adapter import RUDRALoRALinear, inject_rudra_lora, rudra_gate_regularization_loss
        from cross_attention import RUDRACrossAttentionProjection, RUDRACrossAttentionInjector, inject_rudra_cross_attention
        from pipeline import RUDRAPipeline, PipelineMode
        from augment import RUDRAAugmentConfig, RUDRAHDRAugment
        from losses import (
            highlight_preservation_loss, chromaticity_loss, exposure_consistency_loss,
            perceptual_loss, latent_alignment_loss, rudra_reconstruction_loss,
        )
        from stats import DescriptorStats, DescriptorNormalizer
        from sampler import SampleHDRStats, HighlightBalancedSampler, bucket_from_stats
        from train_modes import RUDRATrainMode, LossWeights, scheduled_loss_weights
        from metrics import validation_metrics, highlight_reconstruction_accuracy, exposure_ev_error, psnr, ssim, delta_e_2000
        from hdrvdp import hdr_vdp3_jod, colorvideovdp_available
        from sdr2hdr import SDR2HDRNet, TemporalHDRRefiner, SDR2HDROutput, canonicalize_sdr, sdr2hdr_loss

    __all__ = [
        "TORCH_AVAILABLE",
        # Config
        "RUDRAConfig", "RUDRA_MODEL_CONFIGS", "FORMAT_NAMES", "FORMAT_TO_ID", "FORMAT_DIM", "make_rudra_config",
        # Normalization and descriptors
        "normalize_to_scene_linear", "RUDRADescriptor", "format_onehot", "RUDRASpatialDescriptor",
        # Encoder and DRE
        "RUDRAProjection", "RUDRADynamicRangeEncoder",
        # Decoder
        "RUDRADecoder", "RUDRAFullDecoder", "load_rudra_decoder",
        # Adapters (LoRA-gated + cross-attention)
        "RUDRALoRALinear", "inject_rudra_lora", "rudra_gate_regularization_loss",
        "RUDRACrossAttentionProjection", "RUDRACrossAttentionInjector", "inject_rudra_cross_attention",
        # Pipeline
        "RUDRAPipeline", "PipelineMode",
        # Augmentation
        "RUDRAAugmentConfig", "RUDRAHDRAugment",
        # Losses
        "highlight_preservation_loss", "chromaticity_loss", "exposure_consistency_loss",
        "perceptual_loss", "latent_alignment_loss", "rudra_reconstruction_loss",
        # Stats and sampling
        "DescriptorStats", "DescriptorNormalizer",
        "SampleHDRStats", "HighlightBalancedSampler", "bucket_from_stats",
        # Training
        "RUDRATrainMode", "LossWeights", "scheduled_loss_weights",
        # Metrics
        "validation_metrics", "highlight_reconstruction_accuracy", "exposure_ev_error", "psnr", "ssim", "delta_e_2000",
        "hdr_vdp3_jod", "colorvideovdp_available",
        # Direct pixel SDR-to-HDR recovery
        "SDR2HDRNet", "TemporalHDRRefiner", "SDR2HDROutput", "canonicalize_sdr", "sdr2hdr_loss",
    ]
else:
    __all__ = ["TORCH_AVAILABLE"]

    def __getattr__(name):
        raise ImportError(
            f"rudra.{name} requires torch, which is not installed in this "
            "environment. The torch-free delivery layer is available as "
            "rudra.delivery (ACES/EXR export, HDR metadata, grade controls, CLI)."
        )
