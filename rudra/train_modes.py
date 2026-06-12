"""Explicit RUDRA training modes and loss schedules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RUDRATrainMode(str, Enum):
    DECODER_ONLY = "decoder_only"
    RUDRA_LITE = "rudra_lite"
    LORA_ONLY = "lora_only"
    RUDRA_FULL_DRE = "rudra_full_dre"
    RUDRA_FULL_CROSS_ATTN = "rudra_full_cross_attn"  # Paper §3.5 cross-attention path
    FULL_ADAPTER = "full_adapter"
    VIDEO_CONSISTENCY = "video_consistency"


@dataclass
class LossWeights:
    """Paper §4: L = L_diff + α·L_highlight + β·L_color + γ·L_perceptual + δ·L_exposure + η·L_align."""

    l1: float = 1.0
    highlight: float = 0.0       # α
    chromaticity: float = 0.0    # β
    perceptual: float = 0.0      # γ
    exposure: float = 0.0        # δ
    align: float = 0.0           # η
    temporal: float = 0.0


def scheduled_loss_weights(step: int) -> LossWeights:
    """Conservative staged HDR loss schedule.

    0-10k:  reconstruction only.
    10k-40k: add highlight (α).
    40k-80k: add color (β) and exposure (δ).
    80k-120k: add perceptual (γ) and alignment (η).
    120k+: full RUDRA objective with all terms at paper-recommended weights.
    """
    if step < 10_000:
        return LossWeights(l1=1.0)
    if step < 40_000:
        return LossWeights(l1=1.0, highlight=0.25)
    if step < 80_000:
        return LossWeights(l1=1.0, highlight=0.50, chromaticity=0.15, exposure=0.10)
    if step < 120_000:
        return LossWeights(l1=1.0, highlight=0.50, chromaticity=0.30, exposure=0.20, perceptual=0.05, align=0.05)
    # Paper §4 final weights: α=0.50, β=0.30, γ=0.20, plus δ and η.
    return LossWeights(l1=1.0, highlight=0.50, chromaticity=0.30, exposure=0.20, perceptual=0.10, align=0.10)
