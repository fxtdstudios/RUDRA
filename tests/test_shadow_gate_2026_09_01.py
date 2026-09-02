"""The shadow-arm switch, and the endpoints it has to reproduce exactly.

The 1 Sep 2026 ablation measured two configurations on 429 held-out frames:
`recovery_mode="all"` (clean -3.00 dB, hard +1.43 dB against the analytic
baseline) and `recovery_mode="highlights"` (clean +0.51, hard +0.33). The gate
interpolates between them, so weight 1.0 and weight 0.0 must land on those two
configurations exactly -- otherwise the dial measures something nobody scored.
"""
import pytest
import torch

from rudra.sdr2hdr import SDR2HDRNet, ShadowGate


def _net(shadow=True, seed=0):
    torch.manual_seed(seed)
    net = SDR2HDRNet(base_channels=8, shadow_conditioning=shadow).eval()
    torch.nn.init.normal_(net.head.weight, std=0.05)
    torch.nn.init.normal_(net.head.bias, std=0.05)
    return net


def _frame(b=2, n=48):
    torch.manual_seed(7)
    return torch.rand(b, 3, n, n)


def test_weight_one_is_recovery_mode_all():
    net, f = _net(shadow=False), _frame()
    with torch.no_grad():
        a = net(f, recovery_mode="all")
        b = net(f, recovery_mode="all", shadow_weight=1.0)
    assert torch.allclose(a.hdr, b.hdr, atol=1e-6)


def test_weight_zero_is_recovery_mode_highlights():
    """The other measured endpoint. If this drifts, the ablation no longer bounds it."""
    net, f = _net(shadow=False), _frame()
    with torch.no_grad():
        a = net(f, recovery_mode="highlights")
        b = net(f, recovery_mode="all", shadow_weight=0.0)
    assert torch.allclose(a.hdr, b.hdr, atol=1e-6)


def test_the_dial_is_monotone_between_the_endpoints():
    net, f = _net(shadow=False), _frame(1, 64)
    with torch.no_grad():
        outs = [net(f, recovery_mode="all", shadow_weight=w).hdr
                for w in (0.0, 0.25, 0.5, 0.75, 1.0)]
    diffs = [float((outs[i + 1] - outs[i]).abs().sum()) for i in range(4)]
    assert all(d > 0 for d in diffs), "every step of the dial must change the output"


def test_preserve_masks_are_untouched():
    """The ablation weighted the residual gate only, never the learned masks."""
    net, f = _net(shadow=False), _frame()
    with torch.no_grad():
        a = net(f, recovery_mode="all", shadow_weight=1.0)
        b = net(f, recovery_mode="all", shadow_weight=0.0)
    assert torch.allclose(a.shadow_mask, b.shadow_mask, atol=1e-7)
    assert torch.allclose(a.highlight_mask, b.highlight_mask, atol=1e-7)


def test_off_by_default_and_state_dict_unchanged():
    plain = SDR2HDRNet(base_channels=8)
    assert plain.shadow_gate is None
    gated = SDR2HDRNet(base_channels=8, shadow_conditioning=True)
    extra = set(gated.state_dict()) - set(plain.state_dict())
    assert extra and all(k.startswith("shadow_gate.") for k in extra)
    SDR2HDRNet(base_channels=8).load_state_dict(plain.state_dict(), strict=True)


def test_a_fresh_head_is_within_one_percent_of_the_shipped_behaviour():
    """Not identical, and the docstring says so.

    A fresh switch emits sigmoid(4) = 0.982, so enabling the flag moves the
    composite slightly before any training. A bias large enough to make it
    exactly 1.0 would leave the sigmoid with a ~1e-6 derivative, and a head that
    cannot receive gradient at initialisation is precisely what wasted two
    8,000-step runs on the residual-scale gate. This pins the trade.
    """
    f = _frame()
    plain, gated = _net(shadow=False, seed=3), _net(shadow=True, seed=3)
    gated.load_state_dict(plain.state_dict(), strict=False)
    with torch.no_grad():
        a, b = plain(f, recovery_mode="all"), gated(f, recovery_mode="all")
    assert b.shadow_weight is not None
    assert float(b.shadow_weight.mean()) == pytest.approx(0.9820, abs=1e-3)
    rel = ((a.hdr - b.hdr).abs().sum() / a.hdr.abs().sum().clamp_min(1e-9))
    assert float(rel) < 0.01, f"a fresh switch moved the composite by {100*float(rel):.2f}%"


def test_the_switch_has_usable_gradient_at_initialisation():
    """The failure mode that wasted two runs on the other gate."""
    gate = ShadowGate(channels=4)
    bias = gate.mlp[-1].bias.detach()
    w = torch.sigmoid(bias)
    assert float(w * (1 - w)) > 1e-3, "initialised into the flat tail of the sigmoid"


def test_from_config_round_trips_the_flag():
    for flag in (False, True):
        net = SDR2HDRNet.from_config({"base_channels": 8, "shadow_conditioning": flag})
        assert (net.shadow_gate is not None) is flag
    assert SDR2HDRNet.from_config({}).shadow_gate is None


def test_predict_shadow_weight_matches_the_head():
    net = _net(shadow=True)
    frame = torch.rand(1, 3, 720, 1280)
    from rudra.sdr2hdr import luminance, sdr_to_baseline_hdr
    from training.gate_oracle import gate_view
    with torch.no_grad():
        view = gate_view(frame, 512)
        _, _, mid = net.encode(view, sdr_to_baseline_hdr(view))
        direct = net.shadow_gate(mid, frame, luminance(frame))
        viaapi = net.predict_shadow_weight(frame, max_side=512)
    assert torch.allclose(direct, viaapi, atol=1e-5)


def test_gradient_reaches_the_switch():
    net = _net(shadow=True)
    w = net.shadow_gate(*(lambda s: (net.encode(s, __import__("rudra.sdr2hdr", fromlist=["x"])
                                     .sdr_to_baseline_hdr(s))[2], s,
                                     __import__("rudra.sdr2hdr", fromlist=["x"]).luminance(s)))(_frame()))
    w.sum().backward()
    grads = [p.grad for n, p in net.named_parameters()
             if n.startswith("shadow_gate.") and p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_a_batch_gets_one_weight_per_frame():
    net = _net(shadow=True)
    with torch.no_grad():
        out = net(_frame(3, 48), recovery_mode="all")
    assert out.shadow_weight.shape == (3, 1, 1, 1)
    assert ((out.shadow_weight >= 0) & (out.shadow_weight <= 1)).all()
