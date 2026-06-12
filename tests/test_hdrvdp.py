"""Tests for the real HDR perceptual metric integration (review §4.1).

These run without ColorVideoVDP installed (they exercise the labeled proxy
fallback path) and also validate the backend wiring when cvvdp is present.

Run:  pytest tests/test_hdrvdp.py
"""

import pytest

torch = pytest.importorskip("torch")

from rudra.hdrvdp import hdr_vdp3_jod, colorvideovdp_available
from rudra.metrics import validation_metrics


def test_returns_jod_and_backend():
    pred = torch.rand(1, 3, 64, 64)
    target = pred.clone()
    jod, backend = hdr_vdp3_jod(pred, target)
    assert backend in ("colorvideovdp", "proxy")
    assert isinstance(jod, float)
    # Identical inputs → near-perfect quality (proxy returns ~10; cvvdp ~10).
    assert jod > 7.0


def test_distortion_lowers_jod():
    target = torch.rand(1, 3, 64, 64)
    pred = (target + 0.5 * torch.rand_like(target)).clamp(0, 4)
    jod_same, _ = hdr_vdp3_jod(target, target)
    jod_diff, _ = hdr_vdp3_jod(pred, target)
    assert jod_diff <= jod_same + 1e-4


def test_validation_metrics_reports_backend():
    pred = torch.rand(1, 3, 48, 48)
    target = torch.rand(1, 3, 48, 48)
    m = validation_metrics(pred, target, color_space="rec2020")
    assert "hdr_vdp3" in m
    assert "hdr_vdp_backend" in m
    assert m["hdr_vdp_backend"] in ("colorvideovdp", "proxy")


def test_color_space_conversion_paths_run():
    pred = torch.rand(1, 3, 32, 32)
    target = torch.rand(1, 3, 32, 32)
    for cs in ("rec2020", "rec709", "acescg"):
        jod, _ = hdr_vdp3_jod(pred, target, color_space=cs)
        assert isinstance(jod, float)


def test_availability_flag_is_bool():
    assert isinstance(colorvideovdp_available(), bool)
