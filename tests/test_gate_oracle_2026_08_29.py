"""Oracle targets for the residual scale, and the composite they are built on.

The value of a supervised target is that it is exactly the quantity the
benchmark measures. That only holds if three things are true, and each is
pinned here: the torch PU21 matches the numpy one `rudra bench` uses, the
reconstructed composite matches what the network itself produces, and the
search recovers a scale that was planted in the data.
"""
import numpy as np
import pytest
import torch

from rudra.delivery.bench import pu21_encode as pu21_numpy, pu_psnr as pu_psnr_numpy
from rudra.sdr2hdr import SDR2HDRNet
from training.gate_oracle import (NETWORK_PEAK_NITS, composite_at, effective_residual,
                                  gate_view, oracle_alpha, pu21_encode, pu_psnr)


def test_pu21_matches_the_numpy_benchmark():
    """One definition. A drift here would silently retarget the whole run."""
    nits = np.geomspace(0.001, 20000.0, 512)
    got = pu21_encode(torch.from_numpy(nits)).numpy()
    assert np.allclose(got, pu21_numpy(nits), rtol=1e-10, atol=1e-10)


def test_pu_psnr_matches_the_numpy_benchmark():
    rng = np.random.default_rng(20260829)
    ref = rng.uniform(0.01, 4000.0, size=(1, 16, 16, 3))
    test = ref * rng.uniform(0.5, 2.0, size=ref.shape)
    got = float(pu_psnr(torch.from_numpy(test), torch.from_numpy(ref))[0])
    assert got == pytest.approx(pu_psnr_numpy(test[0], ref[0]), rel=1e-9)


def _model_and_frame(seed=0):
    torch.manual_seed(seed)
    net = SDR2HDRNet(base_channels=8).eval()
    torch.nn.init.normal_(net.head.weight, std=0.05)
    torch.nn.init.normal_(net.head.bias, std=0.05)
    return net, torch.rand(2, 3, 48, 48)


def test_composite_at_one_reproduces_the_network():
    """The search is only meaningful if alpha=1 is what the model actually does."""
    net, frame = _model_and_frame()
    with torch.no_grad():
        raw = net(frame, preserve_outside=False, recovery_mode="all")
        blended = net(frame, preserve_outside=True, recovery_mode="all")
    effective = effective_residual(raw.hdr, raw.baseline)
    recovery = torch.maximum(raw.highlight_mask, raw.shadow_mask)
    ones = torch.ones(frame.shape[0])
    assert torch.allclose(composite_at(raw.baseline, effective, None, ones),
                          raw.hdr, atol=1e-5)
    assert torch.allclose(composite_at(raw.baseline, effective, recovery, ones),
                          blended.hdr, atol=1e-5)


def test_alpha_zero_is_exactly_the_analytic_baseline():
    """Gains are quoted against alpha zero, so it must be doing nothing at all."""
    net, frame = _model_and_frame()
    with torch.no_grad():
        raw = net(frame, preserve_outside=False, recovery_mode="all")
    effective = effective_residual(raw.hdr, raw.baseline)
    zeros = torch.zeros(frame.shape[0])
    assert torch.allclose(composite_at(raw.baseline, effective, None, zeros),
                          raw.baseline, atol=1e-6)


def test_the_search_recovers_a_planted_scale():
    """Build the target FROM a known alpha; the search must find it back."""
    net, frame = _model_and_frame(seed=3)
    with torch.no_grad():
        raw = net(frame, preserve_outside=False, recovery_mode="all")
    effective = effective_residual(raw.hdr, raw.baseline)
    recovery = torch.maximum(raw.highlight_mask, raw.shadow_mask)
    planted = torch.tensor([0.25, 1.0])
    target = composite_at(raw.baseline, effective, recovery, planted)
    found, gain = oracle_alpha(raw.baseline, effective, recovery, target)
    assert torch.allclose(found, planted, atol=0.0626), f"found {found}, planted {planted}"
    assert (gain >= -1e-6).all(), "the best scale can never be worse than doing nothing"


def test_the_search_never_loses_to_doing_nothing():
    net, frame = _model_and_frame(seed=5)
    with torch.no_grad():
        raw = net(frame, preserve_outside=False, recovery_mode="all")
    effective = effective_residual(raw.hdr, raw.baseline)
    torch.manual_seed(9)
    target = torch.rand_like(raw.baseline) * 0.02          # a low-headroom frame
    _, gain = oracle_alpha(raw.baseline, effective, None, target)
    assert (gain >= 0.0).all()


def test_gate_view_matches_inference():
    """Train and inference must downscale identically or the head sees two worlds."""
    net = SDR2HDRNet(base_channels=8, gate_conditioning=True).eval()
    frame = torch.rand(1, 3, 720, 1280)
    view = gate_view(frame, 512)
    assert view.shape[-1] == 512 and view.shape[-2] == 288, view.shape
    assert gate_view(torch.rand(1, 3, 100, 200), 512).shape[-2:] == (100, 200), \
        "a frame already under the cap must pass through untouched"
    # predict_residual_scale pools features from the view but reads statistics
    # from the NATIVE frame, so the two halves come from two resolutions.
    with torch.no_grad():
        from rudra.sdr2hdr import luminance, sdr_to_baseline_hdr
        _, _, mid = net.encode(view, sdr_to_baseline_hdr(view))
        native_stats = net.gate(mid, frame, luminance(frame))
        view_stats = net.gate(mid, view, luminance(view))
        viaapi = net.predict_residual_scale(frame, max_side=512)
    assert torch.allclose(native_stats, viaapi, atol=1e-5), \
        "statistics must be read at native resolution, where degradation is visible"
    assert view_stats.shape == viaapi.shape


def test_oracle_prefers_a_small_scale_on_a_low_headroom_frame():
    """The finding in one test: where there is no headroom, do less.

    A target that barely exceeds the baseline cannot be improved by adding a
    large residual, so the search must come back near zero.
    """
    torch.manual_seed(11)
    baseline = torch.full((1, 3, 32, 32), 0.02)            # ~200 nits
    effective = torch.full_like(baseline, 1.5)             # the model wants to add a lot
    target = baseline * 1.05                               # the truth barely moves
    alpha, _ = oracle_alpha(baseline, effective, None, target)
    assert float(alpha[0]) <= 0.25, f"expected a small scale, got {float(alpha[0])}"
