"""RUDRA-Full cross-attention conditioning adapter.

Paper §3.5: Radiometric tokens are injected into selected U-Net cross-attention
blocks via key/value concatenation:

    C_total = concat(C_text, λ · C_R)

where C_R are projected dynamic-range tokens from Z_R, and λ controls radiometric
conditioning strength.

This module provides the cross-attention injection mechanism described in the paper,
complementing the LoRA-gating approach in adapter.py (RUDRA-Lite).
"""

from __future__ import annotations

import math
from typing import List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


class RUDRACrossAttentionProjection(nn.Module):
    """Projects DRE spatial tokens Z_R into cross-attention compatible tokens C_R.

    Takes the output of RUDRADynamicRangeEncoder (B, N, embed_dim) and projects it
    into the text conditioning dimension so it can be concatenated with C_text.
    """

    def __init__(
        self,
        dre_embed_dim: int = 512,
        text_embed_dim: int = 768,
        num_projection_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dre_embed_dim = dre_embed_dim
        self.text_embed_dim = text_embed_dim

        layers: list[nn.Module] = []
        in_dim = dre_embed_dim
        for i in range(num_projection_layers - 1):
            layers.extend([
                nn.Linear(in_dim, text_embed_dim),
                nn.LayerNorm(text_embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = text_embed_dim
        layers.append(nn.Linear(in_dim, text_embed_dim))
        self.projection = nn.Sequential(*layers)

        # Near-zero initialization for the final projection layer so that the
        # adapter starts as near-identity for the diffusion backbone.
        nn.init.normal_(self.projection[-1].weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.projection[-1].bias)

        self.norm = nn.LayerNorm(text_embed_dim)

    def forward(self, z_r: torch.Tensor) -> torch.Tensor:
        """Project DRE tokens into cross-attention token space.

        Args:
            z_r: DRE output tokens, shape (B, N, dre_embed_dim).

        Returns:
            C_R: Projected radiometric tokens, shape (B, N, text_embed_dim).
        """
        if z_r.ndim != 3 or z_r.shape[-1] != self.dre_embed_dim:
            raise ValueError(
                f"Expected (B, N, {self.dre_embed_dim}), got {tuple(z_r.shape)}"
            )
        return self.norm(self.projection(z_r))


class RUDRACrossAttentionInjector(nn.Module):
    """Injects radiometric tokens into cross-attention via key/value concatenation.

    Paper §3.5: C_total = concat(C_text, λ · C_R)

    This wraps an existing cross-attention module and augments its key/value
    inputs with projected radiometric tokens.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        text_embed_dim: int = 768,
        lambda_init: float = 0.8,
        learnable_lambda: bool = True,
    ):
        super().__init__()
        self.original_attn = original_attn
        self.text_embed_dim = text_embed_dim

        # λ controls radiometric conditioning strength (paper §3.5).
        if learnable_lambda:
            self.lambda_param = nn.Parameter(
                torch.tensor(lambda_init, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "lambda_param",
                torch.tensor(lambda_init, dtype=torch.float32),
            )

    @property
    def lambda_value(self) -> torch.Tensor:
        """Return the current conditioning strength, sigmoid-bounded to [0, 2]."""
        # Allow lambda to be tuned but keep it in a reasonable range.
        return self.lambda_param.clamp(0.0, 2.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        dr_tokens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with optional radiometric token injection.

        When dr_tokens is provided, concatenates them to the conditioning sequence:
            C_total = concat(C_text, λ · C_R)
        Then runs the original attention with the augmented sequence.

        Args:
            hidden_states: Query states from the U-Net (B, S_q, D).
            encoder_hidden_states: Text conditioning tokens C_text (B, S_text, D).
            dr_tokens: Projected radiometric tokens C_R (B, S_r, D), or None.
            **kwargs: Passed through to the original attention module.
        """
        if dr_tokens is None:
            dr_tokens = getattr(self, "current_dr_tokens", None)
        if dr_tokens is not None and encoder_hidden_states is not None:
            lam = self.lambda_value.to(dr_tokens.dtype)
            scaled_dr = lam * dr_tokens
            encoder_hidden_states = torch.cat(
                [encoder_hidden_states, scaled_dr], dim=1
            )
        return self.original_attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )


# Default cross-attention module names in common diffusion backbones.
_DEFAULT_CROSS_ATTN_MODULES: Set[str] = {
    "attn2",             # Stable Diffusion / SDXL U-Net
    "cross_attn",        # Generic
    "attn.cross_attn",   # Some transformer architectures
}


def _get_parent_module(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    """Navigate to parent module given a dotted path."""
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def inject_rudra_cross_attention(
    model: nn.Module,
    text_embed_dim: int = 768,
    lambda_init: float = 0.8,
    learnable_lambda: bool = True,
    target_modules: Optional[Set[str]] = None,
) -> tuple[list[tuple[str, RUDRACrossAttentionInjector]], int]:
    """Replace cross-attention modules with RUDRA radiometric token injectors.

    This implements the paper's conditioning pathway (§3.5) by wrapping existing
    cross-attention layers to accept and concatenate radiometric tokens.

    Args:
        model: The diffusion backbone (e.g. U-Net).
        text_embed_dim: Dimension of text conditioning embeddings.
        lambda_init: Initial λ value for radiometric conditioning strength.
        learnable_lambda: If True, λ is a learnable parameter.
        target_modules: Set of module leaf names to replace. Defaults to common
            cross-attention names.

    Returns:
        Tuple of (list of (name, injector) pairs, trainable parameter count).
    """
    if target_modules is None:
        target_modules = _DEFAULT_CROSS_ATTN_MODULES

    replacements: list[tuple[str, RUDRACrossAttentionInjector]] = []
    for name, module in list(model.named_modules()):
        leaf = name.split(".")[-1]
        if leaf not in target_modules:
            continue
        # Cast to same device/dtype as the original module
        first_param = next(module.parameters(), None)
        device = first_param.device if first_param is not None else torch.device("cpu")
        dtype = first_param.dtype if first_param is not None else torch.float32
        
        injector = RUDRACrossAttentionInjector(
            original_attn=module,
            text_embed_dim=text_embed_dim,
            lambda_init=lambda_init,
            learnable_lambda=learnable_lambda,
        ).to(device=device, dtype=dtype)
        replacements.append((name, injector))

    for name, injector in replacements:
        parent, leaf = _get_parent_module(model, name)
        if leaf.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent[int(leaf)] = injector
        else:
            setattr(parent, leaf, injector)

    # Freeze base model, unfreeze ONLY injector-owned parameters.
    #
    # The previous implementation compared a Parameter to a generator object
    # (`p is not injector.original_attn.parameters()`), which is always True, so
    # it re-enabled grad on the wrapped frozen backbone attention weights too —
    # silently training the "frozen" backbone and inflating the trainable count
    # (review §3.3). Here we build an explicit id-set of the wrapped attention's
    # parameters and skip exactly those.
    for p in model.parameters():
        p.requires_grad = False

    trainable_params = 0
    for _, injector in replacements:
        frozen_ids = {id(p) for p in injector.original_attn.parameters()}
        for p in injector.parameters():
            if id(p) in frozen_ids:
                p.requires_grad = False
                continue
            p.requires_grad = True
            trainable_params += p.numel()

    return replacements, trainable_params
