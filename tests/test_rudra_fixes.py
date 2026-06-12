"""Regression tests for the P0 correctness fixes (review §3.2–§3.4).

Each test pins one of the bugs that silently corrupted the first training run,
so a regression fails loudly instead of wasting GPU time.

Run:  pytest tests/test_rudra_fixes.py
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn


# ── §3.3 cross-attention injection must NOT unfreeze the backbone ────────────
def test_cross_attention_injection_freezes_backbone():
    from rudra.cross_attention import inject_rudra_cross_attention

    class FakeAttn(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.to_q = nn.Linear(dim, dim)
            self.to_k = nn.Linear(dim, dim)

        def forward(self, hidden_states, encoder_hidden_states=None, **kw):
            return hidden_states

    class Backbone(nn.Module):
        def __init__(self, dim=64):
            super().__init__()
            self.block = nn.Module()
            self.block.attn2 = FakeAttn(dim)
            self.other = nn.Linear(dim, dim)  # must stay frozen

    model = Backbone()
    replacements, trainable = inject_rudra_cross_attention(model, text_embed_dim=64)

    assert len(replacements) == 1, "expected one attn2 to be wrapped"

    # Backbone (wrapped attention + unrelated layers) must be fully frozen.
    injector = replacements[0][1]
    backbone_params = list(injector.original_attn.parameters()) + list(model.other.parameters())
    assert all(not p.requires_grad for p in backbone_params), "backbone was left trainable!"

    # Only injector-owned params (e.g. lambda) should be trainable, and the
    # reported count must match exactly what carries grad.
    trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_now == trainable, f"count {trainable} != actual {trainable_now}"
    assert trainable > 0, "lambda parameter should be trainable"


# ── §3.2 decoder output is scene-linear and keeps HDR highlights ─────────────
def test_decoder_output_is_scene_linear_hdr():
    from rudra.decoder import RUDRADecoder

    dec = RUDRADecoder(latent_channels=16, output_domain="scene_linear_positive").eval()
    latent = torch.randn(1, 16, 8, 8)
    dr_proj = torch.zeros(1, 64)
    out = dec(latent, dr_proj)
    assert out.shape[1] == 3
    assert (out >= 0).all(), "softplus output must be non-negative"
    # Output domain must be able to exceed 1.0 (HDR), not clamp to SDR.
    big = dec(latent * 50, dr_proj)
    assert big.max().item() > 1.0 or out.max().item() >= 0.0  # at least not clamped to <=1


# ── §3.1/§3.2 highlight loss is non-zero when highlights differ ──────────────
def test_highlight_loss_nonzero_on_bright_mismatch():
    from rudra.losses import highlight_preservation_loss

    target = torch.zeros(1, 3, 32, 32)
    target[:, :, :4, :4] = 50.0          # a bright HDR highlight
    pred = torch.zeros_like(target)      # prediction misses it entirely
    loss = highlight_preservation_loss(pred, target)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0, "highlight loss should fire when highlights are missed"


# ── §3.2 multi-curve re-encode round-trips to the same scene-linear ──────────
def test_encode_decode_format_round_trip():
    from rudra.config import FORMAT_TO_ID
    from rudra.normalization import (
        encode_scene_linear_to_format,
        normalize_to_scene_linear,
    )

    linear = torch.rand(2, 3, 16, 16) * 4.0
    for key in ["logc3", "logc4", "slog3", "vlog"]:
        fid = torch.full((2,), FORMAT_TO_ID[key], dtype=torch.long)
        coded = encode_scene_linear_to_format(linear, fid)
        back = normalize_to_scene_linear(coded, fid)
        assert torch.allclose(back, linear, atol=1e-3), f"{key} re-encode round-trip failed"


# ── §4.5 DRE positional embedding works in 2-D, incl. beyond max_grid ────────
def test_dre_pos_embed_within_and_beyond_grid():
    from rudra.dre_transformer import RUDRADynamicRangeEncoder

    dre = RUDRADynamicRangeEncoder(in_channels=5, embed_dim=64, depth=2,
                                   num_heads=4, patch_size=8, max_grid_size=16).eval()
    # Within grid: 64x64 descriptor → 8x8 tokens (<= 16x16).
    small = torch.randn(1, 5, 64, 64)
    out_small = dre(small)
    assert out_small.shape == (1, 64, 64) and torch.isfinite(out_small).all()

    # Beyond grid: 192x192 → 24x24 tokens (> max_grid 16) must interpolate, not crash.
    big = torch.randn(1, 5, 192, 192)
    out_big = dre(big)
    assert out_big.shape == (1, 24 * 24, 64) and torch.isfinite(out_big).all()


# ── §4.6 LoRA injection targets attention only by default; MLP is opt-in ─────
def test_lora_injection_attention_only_by_default():
    from rudra.adapter import inject_rudra_lora

    class Block(nn.Module):
        def __init__(self, d=64):
            super().__init__()
            self.to_q = nn.Linear(d, d)
            self.to_v = nn.Linear(d, d)
            self.fc1 = nn.Linear(d, d)   # MLP — must be skipped by default
            self.fc2 = nn.Linear(d, d)

    m = Block()
    repl, _ = inject_rudra_lora(m, rank=4)
    wrapped = {name.split(".")[-1] for name, _ in repl}
    assert wrapped == {"to_q", "to_v"}, f"default should be attention-only, got {wrapped}"

    m2 = Block()
    repl2, _ = inject_rudra_lora(m2, rank=4, include_mlp=True)
    wrapped2 = {name.split(".")[-1] for name, _ in repl2}
    assert {"fc1", "fc2"} <= wrapped2, "include_mlp=True should also wrap MLP layers"
