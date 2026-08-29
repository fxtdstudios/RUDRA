"""The per-frame residual-scale head, and the compatibility it must not break.

Why it exists: the benchmark on 29 Aug 2026 found v5 losing 3.0 dB to its own
analytic baseline on clean input, concentrated entirely in low-dynamic-range
frames -- the 60 worst average -10.8 dB at a median reference peak of 238 nits,
the 60 best +3.4 dB at 19 590 nits. The gate deciding how much to reconstruct
was a per-pixel luminance sigmoid that cannot see the frame. An oracle global
scale is worth +5.84 dB clean and +0.29 dB hard simultaneously, and no constant
achieves it: clean wants ~0.125, hard wants ~1.1.

The tests that matter here are the compatibility ones. A head that is enabled
by default, or that changes the state dict when disabled, would break every
checkpoint and every loader in the project.
"""
import math

import pytest
import torch

from rudra.sdr2hdr import (ConditionGate, SDR2HDRNet, frame_conditioning_stats,
                           luminance)


def _frame(batch=2, size=64):
    torch.manual_seed(20260829)
    return torch.rand(batch, 3, size, size)


def test_disabled_by_default_and_state_dict_is_unchanged():
    """Every checkpoint written before this head existed must still load."""
    plain = SDR2HDRNet(base_channels=8)
    assert plain.gate is None
    gated = SDR2HDRNet(base_channels=8, gate_conditioning=True)
    assert set(plain.state_dict()) < set(gated.state_dict())
    extra = set(gated.state_dict()) - set(plain.state_dict())
    assert extra and all(k.startswith("gate.") for k in extra)
    # the load a checkpoint from before 29 Aug 2026 performs
    SDR2HDRNet(base_channels=8).load_state_dict(plain.state_dict(), strict=True)


def test_a_fresh_head_emits_exactly_one():
    """Enabling the head must change nothing until it is trained."""
    gated = SDR2HDRNet(base_channels=8, gate_conditioning=True).eval()
    with torch.no_grad():
        out = gated(_frame())
    assert out.residual_scale is not None
    assert torch.allclose(out.residual_scale, torch.ones_like(out.residual_scale), atol=1e-5)


def test_a_fresh_head_reproduces_the_ungated_composite():
    frame = _frame()
    plain = SDR2HDRNet(base_channels=8).eval()
    gated = SDR2HDRNet(base_channels=8, gate_conditioning=True).eval()
    gated.load_state_dict(plain.state_dict(), strict=False)
    with torch.no_grad():
        a, b = plain(frame), gated(frame)
    assert torch.allclose(a.hdr, b.hdr, atol=1e-6)
    assert a.residual_scale is None


def test_from_config_round_trips_the_flag():
    for flag in (False, True):
        net = SDR2HDRNet.from_config({"base_channels": 8, "gate_conditioning": flag})
        assert (net.gate is not None) is flag
    assert SDR2HDRNet.from_config({}).gate is None          # a pre-flag config
    assert SDR2HDRNet.from_config(None).gate is None


def test_the_scale_is_one_per_frame_not_per_pixel():
    gate = ConditionGate(channels=4)
    torch.manual_seed(1)
    alpha = gate(torch.rand(3, 4, 8, 8), _frame(3, 16), luminance(_frame(3, 16)))
    assert alpha.shape == (3, 1, 1, 1)
    assert (alpha >= 0).all() and (alpha <= gate.alpha_max).all()


def test_the_scale_actually_scales():
    """A hand-set scale must move the composite the way the alpha sweep did."""
    net = SDR2HDRNet(base_channels=8, gate_conditioning=True).eval()
    torch.nn.init.normal_(net.head.weight, std=0.05)
    frame = _frame()
    with torch.no_grad():
        full = net(frame, residual_strength=1.0)
        none = net(frame, residual_strength=0.0)
    assert not torch.allclose(full.hdr, none.hdr)
    with torch.no_grad():
        assert torch.allclose(none.hdr, none.baseline, atol=1e-6), \
            "scale 0 must fall back exactly to the analytic baseline"


def test_stats_separate_headroom_from_condition():
    """Blown highlights move the headroom stat; noise moves the condition stat."""
    flat = torch.full((1, 3, 32, 32), 0.5)
    blown = flat.clone(); blown[..., :8, :] = 1.0
    torch.manual_seed(0)
    noisy = (flat + torch.randn_like(flat) * 0.05).clamp(0, 1)

    s_flat = frame_conditioning_stats(flat, luminance(flat))[0]
    s_blown = frame_conditioning_stats(blown, luminance(blown))[0]
    s_noisy = frame_conditioning_stats(noisy, luminance(noisy))[0]

    assert s_blown[0] > s_flat[0] + 0.1, "clipped fraction must rise on blown highlights"
    assert s_noisy[4] > s_flat[4], "high-frequency energy must rise on a degraded frame"
    assert s_blown[4] < s_noisy[4], "a clean blown frame is not a degraded one"


def test_stats_are_differentiable():
    """Hard thresholds would leave the head with no gradient at all."""
    frame = _frame().requires_grad_(True)
    frame_conditioning_stats(frame, luminance(frame)).sum().backward()
    assert frame.grad is not None and torch.isfinite(frame.grad).all()
    assert frame.grad.abs().sum() > 0


def test_gradient_reaches_the_head():
    net = SDR2HDRNet(base_channels=8, gate_conditioning=True)
    torch.nn.init.normal_(net.head.weight, std=0.05)
    net(_frame()).hdr.square().mean().backward()
    grads = [p.grad for n, p in net.named_parameters()
             if n.startswith("gate.") and p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_alpha_max_bias_solves_to_one():
    for alpha_max in (1.25, 1.5, 2.0):
        gate = ConditionGate(channels=4, alpha_max=alpha_max)
        bias = gate.mlp[-1].bias.detach()
        assert math.isclose(torch.sigmoid(bias).item() * alpha_max, 1.0, rel_tol=1e-5)


def test_tiles_share_one_whole_frame_scale():
    """A tile must not predict its own scale from its own window.

    Left alone, a patch of bright sky inside a dim interior reads as a
    high-headroom frame and reconstructs hard -- the exact failure the head
    exists to prevent, reintroduced one tile at a time.
    """
    from training.infer_sdr2hdr import predict_image

    net = SDR2HDRNet(base_channels=8, gate_conditioning=True).eval()
    torch.nn.init.normal_(net.head.weight, std=0.05)
    torch.nn.init.normal_(net.gate.mlp[-1].weight, std=0.5)   # make it opinionated

    torch.manual_seed(7)
    frame = torch.rand(1, 3, 160, 160) * 0.15
    frame[..., :24, :] = 1.0                                   # a bright band

    seen = []
    original = net.forward

    def spy(sdr, **kwargs):
        seen.append(kwargs.get("residual_scale"))
        return original(sdr, **kwargs)

    net.forward = spy
    try:
        predict_image(net, frame, preserve_outside=False, tile_size=64, overlap=16)
    finally:
        net.forward = original

    passed = [s for s in seen if s is not None]
    assert len(passed) > 1, "expected several tiles"
    first = passed[0]
    assert all(torch.equal(first, s) for s in passed), \
        "every tile must be given the same whole-frame scale"


def test_fields_carry_the_scale_so_the_viewer_stays_correct():
    """RUDRA Studio composes from the fields alone and knows nothing of a head."""
    from training.infer_sdr2hdr import predict_fields

    net = SDR2HDRNet(base_channels=8, gate_conditioning=True).eval()
    torch.nn.init.normal_(net.head.weight, std=0.05)
    frame = _frame(1, 64)

    with torch.no_grad():
        plain = net(frame, preserve_outside=False, recovery_mode="all")
        fields = predict_fields(net, frame, tile_size=0, overlap=0)
    expected = plain.log_residual * plain.residual_scale
    assert torch.allclose(fields["residual"], expected, atol=1e-5), \
        "the per-frame scale must be folded into the residual the page receives"
