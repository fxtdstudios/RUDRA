"""Cross-implementation radiometric equivalence tests (refactor audit 2026-08-22).

The audit found the same math implemented in parallel — torch (rudra core),
numpy (rudra.hdr10 / delivery), and the self-contained storage codec
(pipeline/hdr_io.py). These tests pin them to each other so drift between
implementations becomes a red test instead of corrupted training supervision.

Numpy-side tests always run; torch-side tests importorskip.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra import radiometry  # noqa: E402
from rudra.hdr10 import pq_eotf, pq_oetf  # noqa: E402


# ── constants agree across the packages (always run) ────────────────────────

def test_config_luma_weights_come_from_radiometry():
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        pytest.skip("rudra.config requires torch at package level")
    from rudra.config import LUMA_WEIGHTS
    assert LUMA_WEIGHTS["rec2020"] == radiometry.LUMA_REC2020
    assert LUMA_WEIGHTS["rec709"] == radiometry.LUMA_REC709
    assert LUMA_WEIGHTS["acescg"] == radiometry.LUMA_ACESCG


def test_hdr10_luma_matches_radiometry():
    from rudra.hdr10 import REC2020_LUMA
    assert np.allclose(REC2020_LUMA, radiometry.LUMA_REC2020)


def test_luma_weights_sum_to_one():
    for w in (radiometry.LUMA_REC709, radiometry.LUMA_REC2020, radiometry.LUMA_ACESCG):
        assert abs(sum(w) - 1.0) < 1e-5


# ── PQ: numpy (hdr10) vs storage codec (pipeline/hdr_io) ────────────────────

def test_pq_hdr10_vs_storage_codec():
    sys.path.insert(0, str(REPO / "pipeline"))
    import hdr_io
    codes = np.linspace(0.0, 1.0, 513, dtype=np.float64)
    ours = pq_eotf(codes)
    theirs = hdr_io.pq_eotf(codes) if hasattr(hdr_io, "pq_eotf") else None
    if theirs is None:
        pytest.skip("hdr_io exposes no public pq_eotf")
    # hdr10 computes in float32, the storage codec in float64 — allow
    # float32 rounding; a wrong constant shows up at the percent level.
    assert np.allclose(ours, theirs, rtol=2e-3, atol=1e-2)


def test_pq_round_trip_and_anchors():
    nits = np.array([0.0, 0.005, 0.18 * 203.0, 100.0, 203.0, 1000.0, 10000.0])
    codes = pq_oetf(nits)
    assert np.all(np.diff(codes) > 0), "PQ OETF must be monotonic"
    back = pq_eotf(codes)
    assert np.allclose(back, nits, rtol=2e-4, atol=1e-3)
    # ST 2084 anchor: 100 nits ≈ code 0.508
    assert abs(float(pq_oetf(np.array(100.0))) - 0.5081) < 2e-3


# ── forward ACES fit is the one from radiometry everywhere ──────────────────

def test_aces_tonemap_shape():
    x = np.linspace(0.0, 4.0, 100)
    y = radiometry.aces_tonemap(x)
    assert y[0] == 0.0 and np.all(np.diff(y) > 0) and y[-1] < 1.10
    # 18% grey with the -1 EV convention lands mid-tone-ish
    assert 0.2 < radiometry.aces_tonemap(0.18) < 0.35


# ── torch vs numpy vs each other (torch-gated) ──────────────────────────────

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

needs_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


def _grid(n=1025):
    return torch.linspace(0.0, 1.0, n, dtype=torch.float64).view(1, 1, 1, -1).repeat(1, 3, 1, 1)


@needs_torch
def test_pq_torch_vs_numpy():
    """normalization._pq_eotf is diffuse-white-relative; hdr10.pq_eotf is nits.
    They must be the same curve up to the 203-nit anchor."""
    from rudra.normalization import _pq_eotf
    codes = _grid()
    rel = _pq_eotf(codes)                                # diffuse-white-relative
    nits = pq_eotf(codes.numpy()[0, 0, 0])               # absolute nits
    assert np.allclose(rel.numpy()[0, 0, 0] * radiometry.PQ_REF_WHITE_NITS,
                       nits, rtol=1e-5, atol=1e-3)


@needs_torch
def test_pq_encode_decode_inverse_torch():
    from rudra.normalization import _pq_eotf, _pq_inverse_eotf
    codes = _grid()
    assert torch.allclose(_pq_inverse_eotf(_pq_eotf(codes)), codes, rtol=1e-5, atol=1e-5)


@needs_torch
def test_srgb_torch_impls_agree():
    from rudra.normalization import _linear_to_srgb, _srgb_to_linear
    from rudra.sdr2hdr import linear_to_srgb, srgb_to_linear
    codes = _grid()
    assert torch.allclose(_srgb_to_linear(codes), srgb_to_linear(codes), rtol=1e-6, atol=1e-7)
    lin = _grid()
    assert torch.allclose(_linear_to_srgb(lin), linear_to_srgb(lin), rtol=1e-6, atol=1e-6)


@needs_torch
def test_srgb_matches_radiometry_constants():
    from rudra.sdr2hdr import srgb_to_linear
    x = _grid()
    r = radiometry
    expected = torch.where(
        x <= r.SRGB_CODE_KNEE, x / r.SRGB_SLOPE,
        ((x + r.SRGB_OFFSET) / (1.0 + r.SRGB_OFFSET)) ** r.SRGB_GAMMA)
    assert torch.allclose(srgb_to_linear(x), expected, rtol=1e-6, atol=1e-7)


@needs_torch
def test_inverse_aces_inverts_forward():
    from rudra.sdr2hdr import inverse_aces_approx
    x = torch.linspace(0.0, 3.0, 800, dtype=torch.float64)
    y = torch.as_tensor(radiometry.aces_tonemap(x))
    back = inverse_aces_approx(y.clamp(0.0, 1.0).float())
    keep = y < 0.99  # values at the clip ceiling are unknowable by design
    assert torch.allclose(back[keep].double(), x[keep], rtol=5e-3, atol=1e-3)


@needs_torch
def test_hlg_round_trip_torch():
    from rudra.normalization import _hlg_inverse_oetf, _hlg_oetf
    signal = _grid()
    lin = _hlg_inverse_oetf(signal)
    assert torch.allclose(_hlg_oetf(lin), signal, rtol=1e-4, atol=1e-4)
    # BT.2408: 75% signal is diffuse white == 1.0 after the rescale
    white = _hlg_inverse_oetf(torch.full((1, 3, 1, 1), 0.75))
    assert torch.allclose(white, torch.ones_like(white), rtol=1e-4, atol=1e-4)


@needs_torch
def test_luma_cf_equals_literal():
    x = torch.rand(2, 3, 8, 8, dtype=torch.float64)
    lit = 0.2627 * x[:, 0:1] + 0.6780 * x[:, 1:2] + 0.0593 * x[:, 2:3]
    assert torch.allclose(radiometry.luma_cf(x), lit, rtol=0, atol=1e-12)
