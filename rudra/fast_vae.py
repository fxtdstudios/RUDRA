"""Standalone distilled HDR decoder architectures used by the trainer.

This module deliberately contains no ComfyUI imports. It is the reproducible
training counterpart of the production Radiance decoder implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, depth: int = 3):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(nn.Conv2d(channels, channels, 3, padding=1))
            if i != depth - 1:
                layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class RadianceTurboDecoder(nn.Module):
    """Small log-domain decoder with a configurable VAE spatial factor."""

    def __init__(self, latent_channels: int = 16, output_channels: int = 3, n_upsample: int = 3):
        super().__init__()
        if n_upsample < 1:
            raise ValueError("n_upsample must be >= 1")
        layers: list[nn.Module] = [
            nn.Conv2d(latent_channels, 64, 3, padding=1),
            _ResidualBlock(64, depth=3),
        ]
        for _ in range(n_upsample):
            layers.extend([
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(64, 64, 3, padding=1),
                _ResidualBlock(64, depth=3),
            ])
        layers.append(nn.Conv2d(64, output_channels, 3, padding=1))
        self.latent_channels = latent_channels
        self.n_upsample = n_upsample
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class RadianceFullDecoder(nn.Module):
    """Higher-capacity log-domain decoder with configurable upsampling."""

    def __init__(self, latent_channels: int = 16, output_channels: int = 3, n_upsample: int = 3):
        super().__init__()
        if n_upsample < 1:
            raise ValueError("n_upsample must be >= 1")
        layers: list[nn.Module] = [
            nn.Conv2d(latent_channels, 128, 3, padding=1),
            _ResidualBlock(128, depth=5),
        ]
        for _ in range(n_upsample):
            layers.extend([
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(128, 128, 3, padding=1),
                _ResidualBlock(128, depth=5),
                _ResidualBlock(128, depth=5),
            ])
        layers.append(nn.Conv2d(128, output_channels, 3, padding=1))
        self.latent_channels = latent_channels
        self.n_upsample = n_upsample
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
