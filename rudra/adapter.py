"""RUDRA Stage 4: DR-gated LoRA adapter."""

from __future__ import annotations

import math
import warnings
from typing import Iterable, List, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


class RUDRALoRALinear(nn.Module):
    """LoRA linear layer whose contribution is gated by RUDRA conditioning.

    Because the gate is sample-conditioned, the adapter is not generally mergeable
    into a static base weight without removing dynamic-range behavior.
    """

    def __init__(
        self,
        original_linear: nn.Module,
        rank: int = 16,
        alpha: float = 1.0,
        dr_proj_dim: int = 64,
        gate_init: float = -2.0,
    ):
        super().__init__()
        if hasattr(original_linear, "in_features"):
            in_features = int(original_linear.in_features)
            out_features = int(original_linear.out_features)
            has_bias = original_linear.bias is not None
        elif hasattr(original_linear, "weight") and original_linear.weight.ndim == 2:
            out_features, in_features = original_linear.weight.shape
            has_bias = hasattr(original_linear, "bias") and original_linear.bias is not None
        else:
            raise ValueError(f"Cannot determine linear dimensions from {type(original_linear)}")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / max(self.rank, 1)
        self.dr_proj_dim = int(dr_proj_dim)

        self.lora_down = nn.Linear(in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, out_features, bias=False)
        self.dr_gate = nn.Linear(dr_proj_dim, self.rank, bias=True)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        nn.init.zeros_(self.dr_gate.weight)
        nn.init.constant_(self.dr_gate.bias, gate_init)  # -2 starts mostly closed.

        self.original_linear = original_linear
        self.last_gate: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, dr_proj: torch.Tensor | None = None) -> torch.Tensor:
        is_comfy_op = not isinstance(self.original_linear, nn.Module) or type(self.original_linear).__name__ != "Linear"
        is_fp8 = False
        if hasattr(self.original_linear, "weight") and self.original_linear.weight is not None:
            w_dtype = getattr(self.original_linear.weight, "dtype", torch.float32)
            if w_dtype in [getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)]:
                is_fp8 = True

        if (is_comfy_op or is_fp8) and x.requires_grad:
            cast_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.bfloat16
            w = self.original_linear.weight
            w_cast = w.to(cast_dtype)
            scale = getattr(self.original_linear, "weight_scale", None)
            if scale is not None:
                w_cast = w_cast * scale.to(cast_dtype)
            b = getattr(self.original_linear, "bias", None)
            b_cast = b.to(cast_dtype) if b is not None else None
            y = F.linear(x, w_cast, b_cast)
        else:
            try:
                y = self.original_linear(x)
            except Exception as e:
                # Fallback for special layers
                w = self.original_linear.weight
                b = getattr(self.original_linear, "bias", None)
                y = F.linear(x, w, b)

        lora_down = self.lora_down(x.to(self.lora_down.weight.dtype))
        if dr_proj is None:
            dr_proj = getattr(self, "current_dr_proj", None)
        if dr_proj is not None:
            if dr_proj.ndim != 2 or dr_proj.shape[-1] != self.dr_proj_dim:
                raise ValueError(f"dr_proj must be (B, {self.dr_proj_dim}), got {tuple(dr_proj.shape)}")
            gate = torch.sigmoid(self.dr_gate(dr_proj.to(dtype=lora_down.dtype)))
            self.last_gate = gate.detach()
            while gate.ndim < lora_down.ndim:
                gate = gate.unsqueeze(1)
            lora_down = lora_down * gate
        else:
            self.last_gate = None
        return (y + self.scale * self.lora_up(lora_down).to(dtype=y.dtype)).clone()

    def lora_parameters(self) -> List[nn.Parameter]:
        return list(self.lora_down.parameters()) + list(self.lora_up.parameters()) + list(self.dr_gate.parameters())

    def merge_gate_open_unsafe(self) -> nn.Linear:
        """Merge as if the DR gate is fully open. This removes DR gating."""
        base_bias = getattr(self.original_linear, "bias", None)
        merged = nn.Linear(self.in_features, self.out_features, bias=base_bias is not None)
        merged = merged.to(device=self.original_linear.weight.device, dtype=self.original_linear.weight.dtype)
        merged.weight.data.copy_(self.original_linear.weight + self.scale * (self.lora_up.weight @ self.lora_down.weight))
        if base_bias is not None:
            merged.bias.data.copy_(base_bias)
        return merged

    def merge_and_unload(self) -> nn.Linear:
        return self.merge_gate_open_unsafe()

    def merge_andunload(self) -> nn.Linear:
        """Deprecated typo alias for merge_and_unload. Will be removed in v0.5."""
        warnings.warn(
            "merge_andunload() is deprecated, use merge_and_unload()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.merge_gate_open_unsafe()


class FP8LinearWrapper(nn.Module):
    """Pass-through wrapper for remaining ComfyUI FP8 layers that aren't replaced
    by RUDRALoRALinear. Bypasses the cuBLAS/dlpack CUDA errors when activations
    have requires_grad=True during autograd backprop.
    """
    def __init__(self, original_linear: nn.Module):
        super().__init__()
        self.original_linear = original_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_comfy_op = not isinstance(self.original_linear, nn.Linear) or type(self.original_linear).__name__ != "Linear"
        is_fp8 = False
        if hasattr(self.original_linear, "weight") and self.original_linear.weight is not None:
            w_dtype = getattr(self.original_linear.weight, "dtype", torch.float32)
            if w_dtype in [getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)]:
                is_fp8 = True

        if (is_comfy_op or is_fp8) and x.requires_grad:
            cast_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.bfloat16
            w = self.original_linear.weight
            w_cast = w.to(cast_dtype)
            scale = getattr(self.original_linear, "weight_scale", None)
            if scale is not None:
                w_cast = w_cast * scale.to(cast_dtype)
            b = getattr(self.original_linear, "bias", None)
            b_cast = b.to(cast_dtype) if b is not None else None
            return F.linear(x, w_cast, b_cast)
        else:
            return self.original_linear(x)


# Attention projection layers — the meaningful default target for DR-gated LoRA.
_ATTENTION_TARGET_MODULES: Set[str] = {
    "to_q", "to_k", "to_v", "to_out", "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
    "query", "key", "value", "to_qkv", "qkv", "qkv_proj",
}
# MLP / feed-forward layers — opt-in via include_mlp (review §4.6). The old default
# also wrapped these plus generic names like "proj"/"dense"/"linear1", injecting LoRA
# almost everywhere and inflating the trainable count.
_MLP_TARGET_MODULES: Set[str] = {
    "gate_proj", "up_proj", "down_proj", "fc1", "fc2",
}
# Back-compat alias (attention-only by default now).
_DEFAULT_TARGET_MODULES: Set[str] = set(_ATTENTION_TARGET_MODULES)


def _get_parent_module(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)) else getattr(parent, part)
    return parent, parts[-1]


def inject_rudra_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 1.0,
    dr_proj_dim: int = 64,
    target_modules: Set[str] | None = None,
    gate_init: float = -2.0,
    include_mlp: bool = False,
) -> tuple[list[tuple[str, RUDRALoRALinear]], int]:
    """Replace target linear layers with DR-gated LoRA and freeze the base model.

    By default only attention projection layers are targeted (review §4.6). Pass
    ``include_mlp=True`` to also wrap feed-forward layers, or an explicit
    ``target_modules`` set to override entirely. Targeting MLP layers is an
    ablatable choice, not a silent default.
    """
    if target_modules is None:
        target_modules = set(_ATTENTION_TARGET_MODULES)
        if include_mlp:
            target_modules |= _MLP_TARGET_MODULES

    replacements: list[tuple[str, RUDRALoRALinear]] = []

    def _replace(module: nn.Module, prefix: str = ""):
        for name, child in list(module.named_children()):
            full_path = f"{prefix}.{name}" if prefix else name

            # Check if this is a linear-like module (nn.Linear or ComfyUI ops)
            is_linear = isinstance(child, nn.Linear)
            if not is_linear:
                classname = type(child).__name__.lower()
                if "linear" in classname or "cast" in classname:
                    if hasattr(child, "weight") and child.weight is not None and child.weight.ndim == 2:
                        is_linear = True

            if is_linear:
                leaf = name.split(".")[-1]
                in_features = getattr(child, "in_features", child.weight.shape[1] if hasattr(child, "weight") and child.weight is not None else 0)

                # Check if it matches target modules and meets features count criteria
                if (leaf in target_modules or any(t in full_path for t in target_modules)) and in_features >= 32:
                    lora_layer = RUDRALoRALinear(child, rank=rank, alpha=alpha, dr_proj_dim=dr_proj_dim, gate_init=gate_init)

                    if hasattr(child, "weight") and child.weight is not None:
                        base_dtype = child.weight.dtype
                        _NO_GRAD_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}
                        if base_dtype in _NO_GRAD_DTYPES:
                            lora_layer = lora_layer.to(device=child.weight.device)
                        else:
                            lora_layer = lora_layer.to(device=child.weight.device, dtype=base_dtype)

                    setattr(module, name, lora_layer)
                    replacements.append((full_path, lora_layer))
                else:
                    # Wrap remaining FP8 / ComfyUI linear layers to avoid cuBLAS errors during autograd backprop
                    wrapper = FP8LinearWrapper(child)
                    setattr(module, name, wrapper)
            else:
                _replace(child, full_path)

    # 1. Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # 2. Inject LoRA and wrappers recursively
    _replace(model)

    # 3. Unfreeze only LoRA parameters
    trainable_params = 0
    for _, lora_layer in replacements:
        for p in lora_layer.lora_parameters():
            p.requires_grad = True
            trainable_params += p.numel()
    return replacements, trainable_params


def rudra_gate_regularization_loss(
    lora_modules: Iterable[RUDRALoRALinear],
    target_gate: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """Optional regularizer that aligns learned gates with a normalized HDR score."""
    gates = [m.last_gate.mean(dim=-1) for m in lora_modules if m.last_gate is not None]
    if not gates:
        return torch.zeros((), device=target_gate.device, dtype=target_gate.dtype)
    gate = torch.stack(gates, dim=0).mean(dim=0).to(target_gate.device, target_gate.dtype)
    target = target_gate.view_as(gate).detach().clamp(0.0, 1.0)
    return weight * F.mse_loss(gate, target)
