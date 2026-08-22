"""HDR-aware losses for RUDRA training.

Paper §4: L = L_diff + α·L_highlight + β·L_color + γ·L_perceptual + η·L_align

The diffusion denoising loss L_diff is applied externally by the backbone trainer.
This module provides the HDR-specific auxiliary losses.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .radiometry import luma_cf

from .spatial_descriptor import RUDRASpatialDescriptor

_EPS = 1e-8

# ---------------------------------------------------------------------------
# Cached descriptor to avoid re-instantiation on every loss call (BUG-3 fix).
# ---------------------------------------------------------------------------
_LOSS_DESC_CACHE: dict[str, RUDRASpatialDescriptor] = {}


def _get_loss_descriptor(
    device: torch.device,
    dtype: torch.dtype,
    y_max_nits: float = 10000.0,
    color_space: str = "rec2020",
) -> RUDRASpatialDescriptor:
    """Return a cached RUDRASpatialDescriptor for the given device/dtype/space."""
    key = f"{device}_{dtype}_{y_max_nits}_{color_space}"
    if key not in _LOSS_DESC_CACHE:
        _LOSS_DESC_CACHE[key] = RUDRASpatialDescriptor(
            color_space=color_space, y_max_nits=y_max_nits, normalize_input=False
        ).to(device=device, dtype=dtype)
        _LOSS_DESC_CACHE[key].eval()
    return _LOSS_DESC_CACHE[key]


# ---------------------------------------------------------------------------
# Individual loss terms
# ---------------------------------------------------------------------------

def highlight_preservation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_max_nits: float = 10000.0,
    threshold: float = 0.85,
    color_space: str = "rec2020",
) -> torch.Tensor:
    """L_highlight: H(x)-weighted L1 in *log-radiance* (paper §4, review §4.2).

    A raw scene-linear L1 here is unbounded — HDR highlights reach ~1e4, so the
    gradient is dominated by a handful of the brightest pixels and training is
    unstable. We weight by the highlight-energy mask H but measure the error in
    ``log1p`` radiance, which compresses the dynamic range (log1p(1e4) ≈ 9.2)
    while preserving relative highlight ordering.
    """
    desc = _get_loss_descriptor(target.device, target.dtype, y_max_nits, color_space)
    with torch.no_grad():
        r = desc(target, input_is_scene_linear=True)
        H = r[:, 2:3]  # highlight energy channel
    pred_l = torch.log1p(pred.clamp(min=0.0))
    target_l = torch.log1p(target.clamp(min=0.0))
    return (H * (pred_l - target_l).abs()).mean()


def chromaticity_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    y_max_nits: float = 10000.0,
    color_space: str = "rec2020",
) -> torch.Tensor:
    """L_color: MSE in CIE xy chromaticity (paper §4)."""
    desc = _get_loss_descriptor(target.device, target.dtype, y_max_nits, color_space)
    pred_r = desc(pred, input_is_scene_linear=True)
    target_r = desc(target, input_is_scene_linear=True)
    return F.mse_loss(pred_r[:, 3:5], target_r[:, 3:5])


def exposure_consistency_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """L_exposure: L1 in median EV space — measures global exposure drift."""
    def ev(x: torch.Tensor) -> torch.Tensor:
        y = luma_cf(x)
        med = y.flatten(1).median(dim=-1).values.clamp(min=_EPS)
        return torch.log2(med / 0.18 + _EPS)
    return F.l1_loss(ev(pred), ev(target))


def perceptual_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_net: Optional[nn.Module] = None,
    tonemap_fn: Optional[callable] = None,
) -> torch.Tensor:
    """L_perceptual: LPIPS on tone-mapped SDR previews (paper §4).

    If no LPIPS network is provided, falls back to multi-scale L1 on tone-mapped images.
    Standard LPIPS backbones are not HDR-native, so we tone-map first.
    """
    if tonemap_fn is None:
        # Reinhard tone map for SDR preview.
        def tonemap_fn(x: torch.Tensor) -> torch.Tensor:
            x = x.clamp(min=0.0)
            return x / (1.0 + x)

    pred_tm = tonemap_fn(pred).clamp(0.0, 1.0)
    target_tm = tonemap_fn(target).clamp(0.0, 1.0)

    if lpips_net is not None:
        # Standard lpips.LPIPS expects inputs in [-1, 1] (AUDIT_2026-08-10:
        # feeding [0,1] evaluates the VGG features at a shifted operating
        # point; metrics.py already does this correctly).
        return lpips_net(pred_tm * 2.0 - 1.0, target_tm * 2.0 - 1.0).mean()

    # Fallback: multi-scale L1 as a simple perceptual proxy.
    loss = F.l1_loss(pred_tm, target_tm)
    for scale in [2, 4]:
        p = F.avg_pool2d(pred_tm, scale)
        t = F.avg_pool2d(target_tm, scale)
        loss = loss + F.l1_loss(p, t)
    return loss / 3.0


def latent_alignment_loss(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
) -> torch.Tensor:
    """L_align: HDR latent alignment (paper §4).

    Encourages the denoised latent to be close to the VAE-encoded HDR target.
    """
    return F.mse_loss(z_pred, z_target)


# ---------------------------------------------------------------------------
# Combined reconstruction loss (paper §4 formulation)
# ---------------------------------------------------------------------------

def rudra_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha_highlight: float = 0.50,
    beta_color: float = 0.30,
    gamma_perceptual: float = 0.20,
    delta_exposure: float = 0.10,
    eta_align: float = 0.0,
    z_pred: Optional[torch.Tensor] = None,
    z_target: Optional[torch.Tensor] = None,
    lpips_net: Optional[nn.Module] = None,
    recon_domain: str = "log",
    color_space: str = "rec2020",
) -> dict[str, torch.Tensor]:
    """Combined RUDRA loss: L = L1 + α·highlight + β·color + γ·perceptual + δ·exposure + η·align.

    The paper formulation is:
        L = L_diff + α·L_highlight + β·L_color + γ·L_perceptual + η·L_align
    L_diff is the diffusion denoising loss applied externally.

    Args:
        pred: predicted scene-linear HDR RGB (B, 3, H, W)
        target: ground truth scene-linear HDR RGB (B, 3, H, W)
        alpha_highlight: weight for highlight preservation loss
        beta_color: weight for chromaticity loss
        gamma_perceptual: weight for perceptual (LPIPS) loss
        delta_exposure: weight for exposure consistency loss
        eta_align: weight for latent alignment loss
        z_pred: denoised latent (optional, for L_align)
        z_target: VAE-encoded target latent (optional, for L_align)
        lpips_net: optional LPIPS network for perceptual loss
    """
    # Base reconstruction in log-radiance by default (review §4.2): a plain L1 on
    # raw scene-linear HDR is dominated by highlight magnitude. ``recon_domain
    # ="linear"`` restores the old behavior if needed.
    if recon_domain == "log":
        base = F.l1_loss(torch.log1p(pred.clamp(min=0.0)), torch.log1p(target.clamp(min=0.0)))
    else:
        base = F.l1_loss(pred, target)
    # Only compute auxiliary terms when their schedule weight is non-zero. These
    # descriptor-based / perceptual losses are not free and were previously
    # computed every step even while weighted 0 (all of them for the first 10k
    # steps), wasting compute.
    _z = base.detach() * 0.0
    h = highlight_preservation_loss(pred, target, color_space=color_space) if alpha_highlight > 0 else _z
    c = chromaticity_loss(pred, target, color_space=color_space) if beta_color > 0 else _z
    e = exposure_consistency_loss(pred, target) if delta_exposure > 0 else _z
    p = perceptual_loss(pred, target, lpips_net=lpips_net) if gamma_perceptual > 0 else _z

    total = base + alpha_highlight * h + beta_color * c + gamma_perceptual * p + delta_exposure * e

    result = {
        "total": total,
        "l1": base,
        "highlight": h,
        "chromaticity": c,
        "exposure": e,
        "perceptual": p,
    }

    if eta_align > 0 and z_pred is not None and z_target is not None:
        align = latent_alignment_loss(z_pred, z_target)
        result["total"] = total + eta_align * align
        result["align"] = align

    return result
