"""Direct 8-bit SDR to scene-linear HDR image and video models.

Unlike the latent decoders, these modules consume decoded SDR RGB pixels.  The
image network predicts a residual over a deterministic inverse-tone-map
baseline plus explicit highlight and shadow recovery masks.  The temporal
network is deliberately small and refines image-model predictions across a
short clip without requiring a multi-billion-parameter video backbone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .radiometry import ACES_A, ACES_B, ACES_C, ACES_D, ACES_E, LUMA_REC709


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.pow(1.0 / 2.4) - 0.055)


def canonicalize_sdr(sdr: torch.Tensor, transfer: str = "srgb", value_range: str = "full") -> torch.Tensor:
    """Convert common 8-bit SDR encodings to the model's canonical sRGB code."""
    x = sdr.float()
    if value_range == "limited":
        x = (x - 16.0 / 255.0) / (219.0 / 255.0)
    elif value_range != "full":
        raise ValueError("value_range must be 'full' or 'limited'")
    x = x.clamp(0.0, 1.0)
    if transfer == "srgb":
        return x
    if transfer == "rec709":
        linear = torch.where(x < 0.081, x / 4.5, ((x + 0.099) / 1.099).pow(1.0 / 0.45))
    elif transfer == "gamma22":
        linear = x.pow(2.2)
    elif transfer == "gamma24":
        linear = x.pow(2.4)
    else:
        raise ValueError("transfer must be srgb, rec709, gamma22, or gamma24")
    return linear_to_srgb(linear)


def inverse_aces_approx(display_linear: torch.Tensor) -> torch.Tensor:
    """Invert the ACES approximation used by ``prepare_training_data.py``.

    Values at exactly one are clipped and unknowable; clamping them just below
    one produces a finite baseline which the learned residual can extend.
    """
    y = display_linear.clamp(0.0, 0.995)
    a, b, c, d, e = ACES_A, ACES_B, ACES_C, ACES_D, ACES_E
    qa = y * c - a
    qb = y * d - b
    qc = y * e
    disc = (qb.square() - 4.0 * qa * qc).clamp_min(0.0)
    root_a = (-qb - torch.sqrt(disc)) / (2.0 * qa).clamp(max=-1e-7)
    root_b = (-qb + torch.sqrt(disc)) / (2.0 * qa).clamp(max=-1e-7)
    return torch.maximum(root_a, root_b).clamp_min(0.0)


def sdr_to_baseline_hdr(sdr: torch.Tensor) -> torch.Tensor:
    """Map sRGB SDR to the normalized scene-linear convention used by G:\\data.

    Preparation applies -1 EV before the ACES curve, then stores HDR as scene
    linear * 203/10000.  Therefore inverse-ACES output is multiplied by
    ``2 * 203/10000``.  The learned network handles other camera/tone curves.
    """
    display_linear = srgb_to_linear(sdr)
    return inverse_aces_approx(display_linear) * (2.0 * 203.0 / 10000.0)


def luminance(x: torch.Tensor) -> torch.Tensor:
    w0, w1, w2 = LUMA_REC709
    return w0 * x[:, 0:1] + w1 * x[:, 1:2] + w2 * x[:, 2:3]


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


@dataclass
class SDR2HDROutput:
    hdr: torch.Tensor
    baseline: torch.Tensor
    log_residual: torch.Tensor
    highlight_mask: torch.Tensor
    shadow_mask: torch.Tensor
    highlight_logits: torch.Tensor
    shadow_logits: torch.Tensor


class SDR2HDRNet(nn.Module):
    """Compact U-Net for direct SDR pixel to HDR radiance recovery."""

    def __init__(self, base_channels: int = 32, log_scale: float = 16.0, max_hdr: float = 4.0):
        super().__init__()
        c = base_channels
        self.log_scale = float(log_scale)
        self.max_hdr = float(max_hdr)
        self.stem = nn.Conv2d(6, c, 3, padding=1)
        self.enc1 = nn.Sequential(ResidualBlock(c), ResidualBlock(c))
        self.down1 = nn.Conv2d(c, c * 2, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(ResidualBlock(c * 2), ResidualBlock(c * 2))
        self.down2 = nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1)
        self.mid = nn.Sequential(ResidualBlock(c * 4), ResidualBlock(c * 4))
        self.up2 = nn.Conv2d(c * 6, c * 2, 3, padding=1)
        self.dec2 = nn.Sequential(ResidualBlock(c * 2), ResidualBlock(c * 2))
        self.up1 = nn.Conv2d(c * 3, c, 3, padding=1)
        self.dec1 = nn.Sequential(ResidualBlock(c), ResidualBlock(c))
        self.head = nn.Conv2d(c, 5, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        sdr: torch.Tensor,
        preserve_outside: bool = False,
        recovery_mode: str = "all",
        residual_strength: float | torch.Tensor = 1.0,
    ) -> SDR2HDROutput:
        if sdr.ndim != 4 or sdr.shape[1] != 3:
            raise ValueError(f"Expected SDR tensor (B,3,H,W), got {tuple(sdr.shape)}")
        sdr = sdr.float().clamp(0.0, 1.0)
        baseline = sdr_to_baseline_hdr(sdr)
        x = torch.cat((sdr, baseline), dim=1)
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))
        u2 = F.interpolate(m, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.dec2(self.up2(torch.cat((u2, e2), dim=1)))
        u1 = F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        features = self.dec1(self.up1(torch.cat((u1, e1), dim=1)))
        raw = self.head(features)

        residual = raw[:, :3]
        highlight_logits = raw[:, 3:4]
        shadow_logits = raw[:, 4:5]
        highlight = torch.sigmoid(highlight_logits)
        shadow = torch.sigmoid(shadow_logits)
        base_log = torch.log1p(baseline * self.log_scale)
        sdr_y = luminance(sdr)
        highlight_prior = torch.sigmoid((sdr_y - 0.82) * 24.0)
        shadow_prior = torch.sigmoid((0.10 - sdr_y) * 24.0)
        # Preserve the physically strong inverse-tone-map baseline through
        # ordinary midtones. Most learned capacity is applied where clipping or
        # crushed shadows made the SDR mapping non-invertible.
        if recovery_mode == "all":
            residual_gate = torch.maximum(highlight_prior, shadow_prior)
        elif recovery_mode == "highlights":
            residual_gate = highlight_prior
        elif recovery_mode == "shadows":
            residual_gate = shadow_prior
        elif recovery_mode == "off":
            residual_gate = torch.zeros_like(highlight_prior)
        else:
            raise ValueError("recovery_mode must be all, highlights, shadows, or off")
        if isinstance(residual_strength, torch.Tensor):
            # Per-pixel ITM strength (mask-driven local inverse tone mapping):
            # (H,W), (1,1,H,W) or (B,1,H,W) in [0, 2]. Built from artist masks
            # via rudra.delivery.controls.itm_strength_map.
            strength = residual_strength.to(device=sdr.device, dtype=residual_gate.dtype)
            while strength.ndim < 4:
                strength = strength.unsqueeze(0)
            if strength.shape[-2:] != residual_gate.shape[-2:]:
                strength = F.interpolate(strength, size=residual_gate.shape[-2:],
                                         mode="bilinear", align_corners=False)
            residual_gate = residual_gate * strength.clamp(0.0, 2.0)
        else:
            residual_gate = residual_gate * float(residual_strength)
        pred_log = (base_log + residual * residual_gate).clamp(0.0, torch.log1p(torch.tensor(
            self.max_hdr * self.log_scale, device=sdr.device, dtype=sdr.dtype
        )))
        pred = torch.expm1(pred_log) / self.log_scale
        if preserve_outside:
            recovery = torch.maximum(highlight, shadow)
            pred = baseline + recovery * (pred - baseline)
        return SDR2HDROutput(
            pred, baseline, residual, highlight, shadow,
            highlight_logits, shadow_logits,
        )


class TemporalHDRRefiner(nn.Module):
    """Small residual 3D CNN for short-clip temporal consistency.

    Input layouts are ``(B,T,3,H,W)``.  The frozen or jointly trained image
    model supplies the initial HDR frames; this module only learns a bounded
    log-radiance correction using neighboring frames.
    """

    def __init__(self, channels: int = 24, log_scale: float = 16.0):
        super().__init__()
        self.log_scale = float(log_scale)
        self.net = nn.Sequential(
            nn.Conv3d(6, channels, (3, 3, 3), padding=1), nn.SiLU(),
            nn.Conv3d(channels, channels, (3, 3, 3), padding=1), nn.SiLU(),
            nn.Conv3d(channels, channels, (3, 3, 3), padding=1), nn.SiLU(),
            nn.Conv3d(channels, 3, (3, 3, 3), padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, sdr: torch.Tensor, image_hdr: torch.Tensor) -> torch.Tensor:
        if sdr.shape != image_hdr.shape or sdr.ndim != 5 or sdr.shape[2] != 3:
            raise ValueError("Expected matching SDR/HDR tensors with shape (B,T,3,H,W)")
        x = torch.cat((sdr, image_hdr), dim=2).permute(0, 2, 1, 3, 4)
        correction = self.net(x).permute(0, 2, 1, 3, 4)
        base_log = torch.log1p(image_hdr.clamp_min(0.0) * self.log_scale)
        return torch.expm1((base_log + 0.25 * torch.tanh(correction)).clamp_min(0.0)) / self.log_scale


def recovery_masks(sdr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft luminance masks used as supervision for recovery regions."""
    y = luminance(sdr)
    highlight = torch.sigmoid((y - 0.82) * 24.0)
    shadow = torch.sigmoid((0.10 - y) * 24.0)
    return highlight, shadow


def _gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    px, tx = pred[..., :, 1:] - pred[..., :, :-1], target[..., :, 1:] - target[..., :, :-1]
    py, ty = pred[..., 1:, :] - pred[..., :-1, :], target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(px, tx) + F.l1_loss(py, ty)


# How far above a censored pixel's grading ceiling the model may reconstruct
# for free. Beyond this the excess is charged, so a one-sided loss cannot just
# run every clipped highlight to the network's max_hdr clamp. Three stops takes
# a 4,000-nit graded clip up to 32,000 nits, which covers real specular
# highlights without licensing invention.
CENSORED_HEADROOM_STOPS = 3.0


def sdr2hdr_loss(
    output: SDR2HDROutput,
    sdr: torch.Tensor,
    target: torch.Tensor,
    shadow_chroma_weight: float = 0.15,
    shadow_smoothness_weight: float = 0.02,
    target_ceiling: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Stable HDR recovery objective in log-radiance and masked regions.

    ``target_ceiling`` (B,) or (B,1,1,1), in the same units as ``target``, is
    the delivery ceiling the source was graded to -- 4,000 nits for HdM-HDR-2014,
    ~1,000 for Rec2100-PQ-1K, 10,000 for Chimera. Pixels sitting on it are
    CENSORED: the grade says ">= ceiling", not "= ceiling". 84% of the v3 corpus
    comes from such sources, and training plain L1 against those pixels teaches
    the model to cap -- measured on 28 Aug 2026, the model trained on that mix
    reconstructed 1.23 stops LESS highlight than its predecessor.

    Where a pixel is censored the error becomes one-sided: predicting ABOVE the
    ceiling is free, predicting below is penalised as usual. Every term that
    reads the per-pixel error inherits it.
    """
    scale = 16.0
    pred_log = torch.log1p(output.hdr.clamp_min(0.0) * scale)
    target_log = torch.log1p(target.clamp_min(0.0) * scale)
    hi, sh = recovery_masks(sdr)

    error = (pred_log - target_log).abs()
    censored_fraction = torch.zeros((), device=pred_log.device, dtype=pred_log.dtype)
    if target_ceiling is not None:
        ceiling = target_ceiling.to(device=target.device, dtype=target.dtype)
        while ceiling.ndim < target.ndim:
            ceiling = ceiling.unsqueeze(-1)
        # 1e-3 relative slack: the target went through a uint16 log2 round trip,
        # so a pixel graded at exactly 4,000 nits comes back a hair off it.
        censored = (target >= ceiling * (1.0 - 1e-3)) & torch.isfinite(ceiling)
        under = (target_log - pred_log).clamp_min(0.0)
        # A purely one-sided loss has no upper anchor, so the cheapest thing a
        # censored pixel can do is run to the network's max_hdr clamp. The grade
        # says ">= ceiling", not ">= ceiling and arbitrarily far above it":
        # allow CENSORED_HEADROOM_STOPS of free reconstruction, then charge for
        # the excess so highlights stay physical.
        allowance = torch.log1p(
            (ceiling * (2.0 ** CENSORED_HEADROOM_STOPS)).clamp_max(1e6) * scale)
        over = (pred_log - allowance).clamp_min(0.0)
        error = torch.where(censored, under + over, error)
        censored_fraction = censored.float().mean()

    base = error.mean()
    highlight = (error * hi).sum() / (hi.sum() * 3.0 + 1e-6)
    shadow = (error * sh).sum() / (sh.sum() * 3.0 + 1e-6)
    mask = F.binary_cross_entropy_with_logits(output.highlight_logits, hi) + F.binary_cross_entropy_with_logits(output.shadow_logits, sh)
    edge = _gradient_loss(pred_log, target_log)
    pred_chroma = output.hdr / (output.hdr.sum(1, keepdim=True) + 1e-4)
    target_chroma = target / (target.sum(1, keepdim=True) + 1e-4)
    chroma = F.l1_loss(pred_chroma, target_chroma)
    # Chromaticity ratios are poorly conditioned close to black. Opponent
    # channels remain stable there and directly penalize colored speckle in
    # crushed shadows without forcing legitimate shadows toward neutral gray.
    pred_opponent = torch.cat((output.hdr[:, 0:1] - output.hdr[:, 1:2],
                               output.hdr[:, 2:3] - output.hdr[:, 1:2]), dim=1)
    target_opponent = torch.cat((target[:, 0:1] - target[:, 1:2],
                                 target[:, 2:3] - target[:, 1:2]), dim=1)
    shadow_chroma = ((pred_opponent - target_opponent).abs() * sh).sum() / (
        sh.sum() * 2.0 + 1e-6
    )
    residual_dx = output.log_residual[..., :, 1:] - output.log_residual[..., :, :-1]
    residual_dy = output.log_residual[..., 1:, :] - output.log_residual[..., :-1, :]
    shadow_dx = torch.minimum(sh[..., :, 1:], sh[..., :, :-1])
    shadow_dy = torch.minimum(sh[..., 1:, :], sh[..., :-1, :])
    shadow_smoothness = (
        (residual_dx.abs() * shadow_dx).sum() / (shadow_dx.sum() * 3.0 + 1e-6)
        + (residual_dy.abs() * shadow_dy).sum() / (shadow_dy.sum() * 3.0 + 1e-6)
    )
    recovery = torch.maximum(hi, sh)
    outside = (error * (1.0 - recovery)).mean()
    residual_outside = (output.log_residual.abs() * (1.0 - recovery)).mean()
    total = (base + 0.50 * highlight + 0.20 * shadow + 0.10 * chroma +
             float(shadow_chroma_weight) * shadow_chroma +
             float(shadow_smoothness_weight) * shadow_smoothness +
             0.10 * edge + 0.05 * mask + 0.05 * outside + 0.10 * residual_outside)
    return {
        "total": total, "log_l1": base, "highlight": highlight,
        "shadow": shadow, "chroma": chroma,
        "shadow_chroma": shadow_chroma,
        "shadow_smoothness": shadow_smoothness, "edge": edge,
        "mask": mask, "outside": outside, "residual_outside": residual_outside,
        "censored_fraction": censored_fraction,
    }


def temporal_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match adjacent-frame log-radiance changes without assuming optical flow."""
    p = torch.log1p(pred.clamp_min(0.0) * 16.0)
    t = torch.log1p(target.clamp_min(0.0) * 16.0)
    return F.l1_loss(p[:, 1:] - p[:, :-1], t[:, 1:] - t[:, :-1])
