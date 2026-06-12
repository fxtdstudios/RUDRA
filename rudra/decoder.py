"""RUDRA Stage 6: DR-conditioned decoder with explicit HDR output domain."""

from __future__ import annotations

import os
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from radiance.config.model_map import resolve_model_vae_config
except Exception:  # standalone package fallback
    resolve_model_vae_config = None

from .config import RUDRAConfig, make_rudra_config

OutputDomain = Literal["scene_linear_positive", "scene_linear_signed", "log", "raw"]


class FiLMLayer(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, channels)
        self.beta = nn.Linear(cond_dim, channels)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim != 2:
            raise ValueError(f"cond must be (B, cond_dim), got {tuple(cond.shape)}")
        g = self.gamma(cond.to(dtype=x.dtype)).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(cond.to(dtype=x.dtype)).unsqueeze(-1).unsqueeze(-1)
        return (1.0 + g) * x + b


class RUDRADecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dr_dim: int = 64, n_conv: int = 3):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.film_in = FiLMLayer(dr_dim, out_channels)
        convs = []
        for _ in range(max(n_conv - 1, 0)):
            convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            convs.append(nn.SiLU(inplace=True))
        self.residual = nn.Sequential(*convs) if convs else nn.Identity()
        self.film_out = FiLMLayer(dr_dim, out_channels)

    def forward(self, x: torch.Tensor, dr_proj: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        h = self.film_in(h, dr_proj)
        h = F.silu(h, inplace=True)
        h = h + self.residual(h)
        return self.film_out(h, dr_proj)


def apply_hdr_output_transform(raw: torch.Tensor, output_domain: OutputDomain = "scene_linear_positive") -> torch.Tensor:
    if output_domain == "raw":
        return raw
    if output_domain == "log":
        # Signed log-radiance: preserves sign for ACES/VFX, compresses dynamic range.
        return torch.sign(raw) * torch.log1p(raw.abs())
    if output_domain == "scene_linear_positive":
        # Smooth positive activation, appropriate for non-negative radiance.
        return F.softplus(raw).clamp(max=1e4)
    if output_domain == "scene_linear_signed":
        # Allows small negative ACES-style values while avoiding extreme overflow.
        return torch.sign(raw) * torch.expm1(raw.abs().clamp(max=9.21))
    raise ValueError(f"Unknown output_domain={output_domain}")


class RUDRADecoder(nn.Module):
    """Production RUDRA-Lite turbo decoder with FiLM conditioning."""

    def __init__(
        self,
        latent_channels: int = 16,
        output_channels: int = 3,
        dr_dim: int = 64,
        channels: int = 64,
        output_domain: OutputDomain = "scene_linear_positive",
        upsample_mode: Literal["bilinear", "nearest"] = "bilinear",
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.dr_dim = dr_dim
        self.output_domain = output_domain
        self.layers = nn.ModuleList([
            RUDRADecoderBlock(latent_channels, channels, dr_dim=dr_dim),
            RUDRADecoderBlock(channels, channels, dr_dim=dr_dim),
            RUDRADecoderBlock(channels, channels, dr_dim=dr_dim),
            RUDRADecoderBlock(channels, channels, dr_dim=dr_dim),
        ])
        if upsample_mode == "bilinear":
            self.upsample = nn.ModuleList([nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) for _ in range(3)])
        else:
            self.upsample = nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest") for _ in range(3)])
        self.out_conv = nn.Conv2d(channels, output_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, dr_proj: torch.Tensor, return_raw: bool = False) -> torch.Tensor:
        h = self.layers[0](x, dr_proj)
        for i in range(1, len(self.layers)):
            h = self.upsample[i - 1](h)
            h = self.layers[i](h, dr_proj)
        raw = self.out_conv(h)
        return raw if return_raw else apply_hdr_output_transform(raw, self.output_domain)


class RUDRAFullDecoder(nn.Module):
    """Higher-capacity RUDRA decoder for research and VFX-quality experiments."""

    def __init__(
        self,
        latent_channels: int = 16,
        output_channels: int = 3,
        dr_dim: int = 64,
        channels: int = 128,
        output_domain: OutputDomain = "scene_linear_positive",
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.dr_dim = dr_dim
        self.output_domain = output_domain
        self.layers = nn.ModuleList([
            RUDRADecoderBlock(latent_channels, channels, dr_dim=dr_dim, n_conv=6),
            RUDRADecoderBlock(channels, channels, dr_dim=dr_dim, n_conv=6),
            RUDRADecoderBlock(channels, channels, dr_dim=dr_dim, n_conv=6),
            RUDRADecoderBlock(channels, channels, dr_dim=dr_dim, n_conv=6),
            RUDRADecoderBlock(channels, channels, dr_dim=dr_dim, n_conv=6),
            RUDRADecoderBlock(channels, channels, dr_dim=dr_dim, n_conv=6),
            RUDRADecoderBlock(channels, channels, dr_dim=dr_dim, n_conv=6),
        ])
        self.upsample = nn.ModuleList([nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) for _ in range(3)])
        self.out_conv = nn.Conv2d(channels, output_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, dr_proj: torch.Tensor, return_raw: bool = False) -> torch.Tensor:
        h = self.layers[0](x, dr_proj)
        h = self.layers[1](h, dr_proj)
        h = self.upsample[0](h)
        h = self.layers[2](h, dr_proj)
        h = self.layers[3](h, dr_proj)
        h = self.upsample[1](h)
        h = self.layers[4](h, dr_proj)
        h = self.layers[5](h, dr_proj)
        h = self.upsample[2](h)
        h = self.layers[6](h, dr_proj)
        raw = self.out_conv(h)
        return raw if return_raw else apply_hdr_output_transform(raw, self.output_domain)


def load_rudra_decoder(
    model_type: str = "flux",
    model_size: str = "turbo",
    checkpoint_path: str | None = None,
    dr_dim: int = 64,
    output_domain: OutputDomain = "scene_linear_positive",
) -> nn.Module:
    cfg = make_rudra_config(model_type)
    latent_channels = cfg.latent_channels
    if resolve_model_vae_config is not None:
        resolved = resolve_model_vae_config(model_type) or {}
        latent_channels = resolved.get("latent_channels", latent_channels)

    if model_size == "full":
        model = RUDRAFullDecoder(latent_channels=latent_channels, dr_dim=dr_dim, output_domain=output_domain)
    else:
        model = RUDRADecoder(latent_channels=latent_channels, dr_dim=dr_dim, output_domain=output_domain)

    if checkpoint_path and os.path.exists(checkpoint_path):
        if checkpoint_path.endswith(".safetensors"):
            import safetensors.torch
            state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")
        else:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"[RUDRA] Unexpected decoder keys: {unexpected[:8]}")
        if missing:
            print(f"[RUDRA] Missing decoder keys: {missing[:8]}")
    model.eval()
    return model
