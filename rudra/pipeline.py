"""RUDRA end-to-end pipeline — wires all components into a unified forward pass.

Paper architecture (§3.1, Figure 1):

    Input (any format)
      → Format Normalization → scene-linear radiance
      → Path A (RUDRA-Lite): Global descriptor (26-d) → MLP projection (64-d) → FiLM decoder
      → Path B (RUDRA-Full): Spatial descriptor R(x) (5-ch) → DRE transformer → Z_R
          → Cross-attention projection → C_R → concat with C_text → inject into backbone
      → HDR Decoder → scene-linear HDR RGB → OpenEXR export

This module provides:
    - RUDRAPipeline: orchestrates the full forward pass with mode selection
    - Helper functions for extracting DR conditioning and decoding HDR output
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Literal

import torch
import torch.nn as nn

from .config import RUDRAConfig, FORMAT_DIM, FORMAT_TO_ID, make_rudra_config
from .normalization import normalize_to_scene_linear
from .descriptor import RUDRADescriptor, format_onehot
from .spatial_descriptor import RUDRASpatialDescriptor
from .encoder import RUDRAProjection
from .dre_transformer import RUDRADynamicRangeEncoder
from .decoder import RUDRADecoder, RUDRAFullDecoder, apply_hdr_output_transform
from .cross_attention import RUDRACrossAttentionProjection


class PipelineMode(str, Enum):
    """Supported pipeline operating modes."""
    LITE_DESCRIPTOR = "lite_descriptor"    # Extract global DR descriptor only
    LITE_DECODER = "lite_decoder"          # Descriptor → projection → FiLM decoder
    LITE_LORA = "lite_lora"               # + DR-gated LoRA on backbone
    FULL_DRE = "full_dre"                 # Spatial descriptor → DRE → cross-attention
    INFERENCE = "inference"                # Full pipeline for inference
MODE_MAP = {
    "decoder_only": PipelineMode.LITE_DECODER,
    "rudra_lite": PipelineMode.LITE_DECODER,
    "lora_only": PipelineMode.LITE_LORA,
    "rudra_full_dre": PipelineMode.FULL_DRE,
    "rudra_full_cross_attn": PipelineMode.FULL_DRE,
    "full_adapter": PipelineMode.INFERENCE,
}

def resolve_pipeline_mode(mode):
    if isinstance(mode, PipelineMode):
        return mode
    if isinstance(mode, str):
        if mode in MODE_MAP:
            return MODE_MAP[mode]
        return PipelineMode(mode)
    raise ValueError(f"Unknown pipeline mode: {mode}")

@dataclass
class DRConditioning:
    """Container for dynamic-range conditioning outputs."""
    # Global (RUDRA-Lite)
    dr_raw: Optional[torch.Tensor] = None         # (B, 26)
    format_onehot: Optional[torch.Tensor] = None   # (B, FORMAT_DIM)
    dr_proj: Optional[torch.Tensor] = None          # (B, dr_proj_dim)

    # Spatial (RUDRA-Full)
    r_map: Optional[torch.Tensor] = None            # (B, 5, H, W)
    z_r: Optional[torch.Tensor] = None              # (B, N, embed_dim) DRE tokens
    c_r: Optional[torch.Tensor] = None              # (B, N, text_embed_dim) cross-attn tokens


class RUDRAPipeline(nn.Module):
    """Unified RUDRA pipeline orchestrating all components.

    Supports both RUDRA-Lite (global descriptor + FiLM decoder + LoRA) and
    RUDRA-Full (spatial descriptor + DRE transformer + cross-attention injection).

    Args:
        config: RUDRA configuration dataclass.
        mode: Pipeline operating mode.
        decoder_size: 'turbo' or 'full' for decoder capacity.
        text_embed_dim: Dimension of text conditioning (for cross-attention projection).
            Set to match the backbone's text embedding dimension.
    """

    def __init__(
        self,
        config: Optional[RUDRAConfig] = None,
        mode: PipelineMode | str = PipelineMode.LITE_DECODER,
        decoder_size: Literal["turbo", "full"] = "turbo",
        text_embed_dim: int = 768,
    ):
        super().__init__()
        self.config = config or RUDRAConfig()
        self.mode = resolve_pipeline_mode(mode)

        # ── Global descriptor (RUDRA-Lite) — always available ─────────────
        self.descriptor = RUDRADescriptor(
            color_space=self.config.color_space,
            y_max_nits=self.config.y_max_nits,
            normalize_input=True,
            highlight_ev_threshold=self.config.highlight_ev_threshold,
            highlight_ev_softness=self.config.highlight_ev_softness,
        )

        self.projection = RUDRAProjection(
            dr_raw_dim=self.config.dr_raw_dim,
            format_dim=self.config.format_dim,
            proj_dim=self.config.dr_proj_dim,
        )

        # ── Spatial descriptor + DRE (RUDRA-Full) ─────────────────────────
        self.spatial_descriptor = RUDRASpatialDescriptor(
            color_space=self.config.color_space,
            y_max_nits=self.config.y_max_nits,
            normalize_input=True,
            highlight_ev_threshold=self.config.highlight_ev_threshold,
            highlight_ev_softness=self.config.highlight_ev_softness,
        )

        self.dre = RUDRADynamicRangeEncoder(
            in_channels=self.config.spatial_descriptor_channels,
            embed_dim=self.config.dre_embed_dim,
            depth=self.config.dre_depth,
            num_heads=self.config.dre_heads,
            patch_size=self.config.dre_patch_size,
        )

        self.cross_attn_proj = RUDRACrossAttentionProjection(
            dre_embed_dim=self.config.dre_embed_dim,
            text_embed_dim=text_embed_dim,
        )

        # ── Decoder ───────────────────────────────────────────────────────
        if decoder_size == "full":
            self.decoder = RUDRAFullDecoder(
                latent_channels=self.config.latent_channels,
                dr_dim=self.config.dr_proj_dim,
                channels=self.config.full_decoder_channels,
                output_domain=self.config.output_domain,
                n_upsample=max(1, (self.config.vae_spatial_factor).bit_length() - 1),
            )
        else:
            self.decoder = RUDRADecoder(
                latent_channels=self.config.latent_channels,
                dr_dim=self.config.dr_proj_dim,
                channels=self.config.decoder_channels,
                output_domain=self.config.output_domain,
                n_upsample=max(1, (self.config.vae_spatial_factor).bit_length() - 1),
            )

    # ─── DR Conditioning Extraction ───────────────────────────────────────

    @torch.no_grad()
    def extract_global_conditioning(
        self,
        image: torch.Tensor,
        format_id: torch.Tensor | int | None = None,
    ) -> DRConditioning:
        """Extract RUDRA-Lite global descriptor and projection.

        Args:
            image: (B, 3, H, W) image in any supported format.
            format_id: Format identifier for normalization.

        Returns:
            DRConditioning with dr_raw, format_onehot, dr_proj populated.
        """
        dr_raw = self.descriptor(image, format_id=format_id)
        fmt_oh = self._format_onehot(format_id, image.shape[0], image.device)
        dr_proj = self.projection(dr_raw, fmt_oh)
        return DRConditioning(
            dr_raw=dr_raw,
            format_onehot=fmt_oh,
            dr_proj=dr_proj,
        )

    def extract_global_conditioning_trainable(
        self,
        image: torch.Tensor,
        format_id: torch.Tensor | int | None = None,
    ) -> DRConditioning:
        """Same as extract_global_conditioning but with gradients through projection."""
        with torch.no_grad():
            dr_raw = self.descriptor(image, format_id=format_id)
        fmt_oh = self._format_onehot(format_id, image.shape[0], image.device)
        dr_proj = self.projection(dr_raw, fmt_oh)  # Gradient flows here
        return DRConditioning(
            dr_raw=dr_raw,
            format_onehot=fmt_oh,
            dr_proj=dr_proj,
        )

    def extract_spatial_conditioning(
        self,
        image: torch.Tensor,
        format_id: torch.Tensor | int | None = None,
        channel_mask: list | None = None,
    ) -> DRConditioning:
        """Extract RUDRA-Full spatial tokens for cross-attention injection.

        Args:
            image: (B, 3, H, W) image in any supported format.
            format_id: Format identifier for normalization.
            channel_mask: Optional 5 floats/bools multiplying the descriptor
                channels [L, E, H, x, y] — used for the paper's descriptor
                channel ablation (§7.2). None = all channels.

        Returns:
            DRConditioning with r_map, z_r, c_r, and also global dr_proj.
        """
        # Spatial path: descriptor → DRE → cross-attention projection
        r_map = self.spatial_descriptor(image, format_id=format_id)
        if channel_mask is not None:
            m = torch.tensor(channel_mask, dtype=r_map.dtype, device=r_map.device).view(1, -1, 1, 1)
            r_map = r_map * m
        z_r = self.dre(r_map)           # (B, N, embed_dim)
        c_r = self.cross_attn_proj(z_r)  # (B, N, text_embed_dim)

        # Also compute global conditioning for the decoder
        global_cond = self.extract_global_conditioning_trainable(image, format_id)

        return DRConditioning(
            dr_raw=global_cond.dr_raw,
            format_onehot=global_cond.format_onehot,
            dr_proj=global_cond.dr_proj,
            r_map=r_map,
            z_r=z_r,
            c_r=c_r,
        )

    def extract_conditioning(
        self,
        image: torch.Tensor,
        format_id: torch.Tensor | int | None = None,
    ) -> DRConditioning:
        """Extract DR conditioning based on current pipeline mode."""
        if self.mode in (PipelineMode.FULL_DRE, PipelineMode.INFERENCE):
            return self.extract_spatial_conditioning(image, format_id)
        return self.extract_global_conditioning_trainable(image, format_id)

    # ─── Decode ───────────────────────────────────────────────────────────

    def decode_hdr(
        self,
        latent: torch.Tensor,
        conditioning: DRConditioning,
        return_raw: bool = False,
    ) -> torch.Tensor:
        """Decode a latent tensor into scene-linear HDR RGB.

        Args:
            latent: (B, C, H_l, W_l) latent from VAE or diffusion backbone.
            conditioning: DRConditioning from extract_conditioning().
            return_raw: If True, skip HDR output transform.

        Returns:
            (B, 3, H, W) scene-linear HDR RGB.
        """
        if conditioning.dr_proj is None:
            raise ValueError("DRConditioning.dr_proj is required for decoding")
        return self.decoder(latent, conditioning.dr_proj, return_raw=return_raw)

    # ─── Full Forward Pass ────────────────────────────────────────────────

    def forward(
        self,
        latent: torch.Tensor,
        image: torch.Tensor,
        format_id: torch.Tensor | int | None = None,
    ) -> tuple[torch.Tensor, DRConditioning]:
        """Full RUDRA forward pass: extract conditioning → decode.

        Args:
            latent: (B, C, H_l, W_l) latent tensor.
            image: (B, 3, H, W) source image for conditioning extraction.
            format_id: Source format identifier.

        Returns:
            Tuple of (HDR prediction, DRConditioning).
        """
        cond = self.extract_conditioning(image, format_id)
        pred = self.decode_hdr(latent, cond)
        return pred, cond

    # ─── Training Utilities ───────────────────────────────────────────────

    def get_trainable_params(self, mode: Optional[PipelineMode | str] = None) -> list[nn.Parameter]:
        """Return the parameters that should be trained for a given mode.

        Descriptor is always frozen. What's trained depends on the stage:
        - lite_decoder: projection + decoder
        - lite_lora: projection + LoRA params (decoder frozen or separate)
        - full_dre: DRE + cross_attn_proj + projection + decoder
        """
        mode = resolve_pipeline_mode(mode) if mode is not None else self.mode
        params: list[nn.Parameter] = []

        if mode == PipelineMode.LITE_DECODER:
            params.extend(self.projection.parameters())
            params.extend(self.decoder.parameters())
        elif mode == PipelineMode.LITE_LORA:
            params.extend(self.projection.parameters())
            # LoRA params are on the backbone, not managed here
        elif mode == PipelineMode.FULL_DRE:
            params.extend(self.dre.parameters())
            params.extend(self.cross_attn_proj.parameters())
            params.extend(self.projection.parameters())
            params.extend(self.decoder.parameters())
        elif mode in (PipelineMode.INFERENCE, PipelineMode.LITE_DESCRIPTOR):
            pass  # Nothing to train
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return params

    def freeze_for_training(self, mode: Optional[PipelineMode | str] = None) -> int:
        """Freeze all parameters, then unfreeze only what the mode needs.

        Returns the count of trainable parameters.
        """
        for p in self.parameters():
            p.requires_grad = False

        trainable = self.get_trainable_params(mode)
        for p in trainable:
            p.requires_grad = True

        return sum(p.numel() for p in trainable)

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _format_onehot(
        self,
        format_id: torch.Tensor | int | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Convert format_id to one-hot, defaulting to linear if None."""
        if format_id is None:
            format_id = FORMAT_TO_ID["linear"]
        if isinstance(format_id, int):
            format_id = torch.full((batch_size,), format_id, dtype=torch.long, device=device)
        return format_onehot(format_id, device=device)

    @classmethod
    def from_model_type(
        cls,
        model_type: str = "flux",
        mode: PipelineMode | str = PipelineMode.LITE_DECODER,
        decoder_size: Literal["turbo", "full"] = "turbo",
        text_embed_dim: Optional[int] = None,
        **overrides,
    ) -> "RUDRAPipeline":
        """Create a pipeline from a model type string with sensible defaults.

        Looks up latent_channels, scale_factor, etc. from the model config registry.
        """
        config = make_rudra_config(model_type, **overrides)

        # Infer text_embed_dim from model type if not specified
        if text_embed_dim is None:
            try:
                try:
                    from config.model_map import resolve_model_vae_config
                except Exception:
                    from radiance.config.model_map import resolve_model_vae_config
                vae_cfg = resolve_model_vae_config(model_type)
                if vae_cfg:
                    text_embed_dim = vae_cfg.get("text_embed_hidden", 768)
                else:
                    text_embed_dim = 768
            except Exception:
                text_embed_dim = 768

        return cls(
            config=config,
            mode=mode,
            decoder_size=decoder_size,
            text_embed_dim=text_embed_dim,
        )

    def load_checkpoint(self, path: str, strict: bool = False) -> tuple[list[str], list[str]]:
        """Load a RUDRA checkpoint (decoder + projection weights).

        Handles both .safetensors and .pth formats. Projection keys
        are expected with 'projection.' prefix.
        """
        import os
        if path.endswith(".safetensors"):
            import safetensors.torch
            state_dict = safetensors.torch.load_file(path, device="cpu")
        else:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                if "state_dict" in ckpt:
                    state_dict = ckpt["state_dict"]
                elif "model" in ckpt:
                    state_dict = ckpt["model"]
                else:
                    state_dict = ckpt
            else:
                state_dict = ckpt

        # Separate projection keys from decoder keys
        proj_keys = {k.replace("projection.", ""): v for k, v in state_dict.items()
                     if k.startswith("projection.")}
        decoder_keys = {k: v for k, v in state_dict.items()
                        if not k.startswith("projection.")}

        missing_d, unexpected_d = [], []
        missing_p, unexpected_p = [], []

        if decoder_keys:
            m, u = self.decoder.load_state_dict(decoder_keys, strict=strict)
            missing_d.extend(m)
            unexpected_d.extend(u)
        if proj_keys:
            m, u = self.projection.load_state_dict(proj_keys, strict=strict)
            missing_p.extend([f"projection.{k}" for k in m])
            unexpected_p.extend([f"projection.{k}" for k in u])

        return missing_d + missing_p, unexpected_d + unexpected_p
