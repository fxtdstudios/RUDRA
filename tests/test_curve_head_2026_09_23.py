"""N3: the baseline stops assuming the corpus's own tone curve.

The shipped model lost to its analytic baseline on SDR from a curve and codec
it never saw (bench/oog, 23 Sep 2026). CurveHead estimates the curve per
frame. These pin: identity at init, no effect on old checkpoints, one curve
per frame under tiling, the viewer's shader doing the same maths, and that
the head can actually learn a curve it was not built around.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.sdr2hdr import CurveHead, SDR2HDRNet, sdr2hdr_loss, sdr_to_baseline_hdr  # noqa: E402


def _sdr(b=1, h=48, w=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, 3, h, w, generator=g)


def test_fresh_curve_head_is_exactly_the_analytic_baseline():
    torch.manual_seed(0)
    model = SDR2HDRNet(base_channels=8, curve_head=True, corpus_ev=0.0).eval()
    x = _sdr()
    with torch.no_grad():
        out = model(x)
    assert torch.equal(out.baseline, out.analytic_baseline)
    assert torch.allclose(out.analytic_baseline, sdr_to_baseline_hdr(x, 0.0))
    assert out.curve_params.shape == (1, 9) and float(out.curve_params.abs().max()) == 0.0


def test_a_checkpoint_without_the_head_is_untouched():
    torch.manual_seed(0)
    old = SDR2HDRNet(base_channels=8, corpus_ev=0.0)
    assert not any(k.startswith("curve.") for k in old.state_dict())
    new = SDR2HDRNet.from_config({"base_channels": 8, "corpus_ev": 0.0})
    new.load_state_dict(old.state_dict(), strict=True)
    out = new.eval()(_sdr())
    assert out.curve_params is None and out.analytic_baseline is out.baseline
    assert SDR2HDRNet.from_config({"base_channels": 8, "curve_head": True}).curve is not None


def test_correction_is_piecewise_linear_in_the_code_value():
    head = CurveHead()
    params = torch.zeros(1, 9)
    params[0, 0] = 1.0                      # +1 stop everywhere
    params[0, 1:] = torch.linspace(-1.0, 1.0, 8)
    codes = torch.linspace(0, 1, 15).view(1, 3, 1, 5)
    corr = head.correction_log2(codes, params)
    expected = 1.0 + (codes * 7.0) * (2.0 / 7.0) - 1.0   # linear knots -> linear in code
    assert torch.allclose(corr, expected, atol=1e-5)


def test_tiled_inference_uses_one_curve_per_frame():
    from training.infer_sdr2hdr import predict_image
    torch.manual_seed(1)
    model = SDR2HDRNet(base_channels=8, curve_head=True, corpus_ev=0.0).eval()
    with torch.no_grad():
        model.curve.mlp[-1].bias.copy_(torch.linspace(-0.5, 0.5, 9))
    x = _sdr(h=96, w=128, seed=3)
    whole = predict_image(model, x, True, 0, 16, "all")
    tiled = predict_image(model, x, True, 64, 16, "all")
    # With the curve fixed per frame, tiling moves only the network's own
    # receptive field, not the baseline: far from the seams they agree.
    inner = (slice(None), slice(None), slice(20, 40), slice(20, 40))
    assert torch.allclose(whole[inner], tiled[inner], rtol=2e-2, atol=1e-5)


def test_viewer_shader_applies_the_same_curve():
    src = (REPO / "ui" / "compositor.js").read_text(encoding="utf-8")
    assert "uniform float uCurve[9];" in src
    assert "exp2(curveLog2(sdr))" in src
    # numerical parity of the hat-sum the shader uses with correction_log2
    head = CurveHead()
    rng = np.random.default_rng(0)
    params = torch.tensor(rng.uniform(-1, 1, (1, 9)), dtype=torch.float32)
    codes = torch.rand(1, 3, 4, 4)
    ref = head.correction_log2(codes, params).numpy()
    pos = codes.numpy().clip(0, 1) * 7.0
    k = params.numpy()[0]
    hat = sum(k[i + 1] * np.maximum(0.0, 1.0 - np.abs(pos - i)) for i in range(8)) + k[0]
    assert np.allclose(hat, ref, atol=1e-5)
    server = (REPO / "ui" / "server.py").read_text(encoding="utf-8")
    assert '"curve":' in server


def test_the_head_learns_an_exposure_it_was_not_built_around():
    """Targets rendered one stop brighter than the analytic inverse assumes:
    a model with the head closes most of that stop; the baseline cannot."""
    torch.manual_seed(0)
    model = SDR2HDRNet(base_channels=8, curve_head=True, corpus_ev=0.0)
    opt = torch.optim.Adam(model.curve.parameters(), lr=3e-3)
    x = _sdr(b=4, h=32, w=32, seed=5) * 0.8
    target = sdr_to_baseline_hdr(x, 0.0) * 2.0
    for _ in range(300):
        out = model(x)
        loss = sdr2hdr_loss(out, x, target, baseline_weight=1.0)["curve_baseline"]
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        out = model(x)
        # exposure and knots can share the stop between them; what matters is
        # the correction the pixels receive
        stops = float(model.curve.correction_log2(x, out.curve_params).mean())
    assert 0.8 < stops < 1.2, stops


def test_trainer_wires_the_flag_and_scores_gain_against_the_analytic_baseline():
    src = (REPO / "training" / "train_sdr2hdr.py").read_text(encoding="utf-8")
    assert '"--curve-head"' in src and "curve_head=args.curve_head" in src
    assert "output.analytic_baseline" in src and "curve_baseline_psnr_log" in src
    assert re.search(r'startswith\(\("gate\.", "curve\."\)\)', src)
