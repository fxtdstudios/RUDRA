"""Smoke tests for the four previously zero-coverage modules (audit 2026-08-22):
rudra.sampler, rudra.spatial_descriptor, rudra.train_modes, rudra.exr_io.

These modules feed the training pipeline; the retrain queue exercises them
heavily, so a construction/shape/round-trip failure should surface here, not
mid-run. Torch-gated (they are all torch modules)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")

from rudra.exr_io import load_tensor_cache, save_exr, save_tensor_cache  # noqa: E402
from rudra.sampler import HighlightBalancedSampler, SampleHDRStats, bucket_from_stats  # noqa: E402
from rudra.spatial_descriptor import RUDRASpatialDescriptor  # noqa: E402
from rudra.train_modes import LossWeights, RUDRATrainMode, scheduled_loss_weights  # noqa: E402


# ── sampler ─────────────────────────────────────────────────────────────────

def _stats(peak, highlight_fraction):
    return SampleHDRStats(index=0, peak=peak, p95=peak * 0.5, mean=peak * 0.1,
                          highlight_fraction=highlight_fraction)


def test_bucket_ordering():
    lo = bucket_from_stats(_stats(0.5, 0.0))
    hi = bucket_from_stats(_stats(50.0, 0.4))
    assert lo != hi, "SDR-ish and highlight-heavy samples must land in different buckets"


def test_highlight_balanced_sampler_covers_dataset():
    stats = [_stats(0.5, 0.0), _stats(2.0, 0.05), _stats(50.0, 0.4), _stats(8.0, 0.2)]
    for i, s in enumerate(stats):
        s.index = i
    sampler = HighlightBalancedSampler(stats)
    idx = list(iter(sampler))
    assert len(idx) == len(sampler) > 0
    assert set(idx) <= set(range(len(stats)))


# ── spatial descriptor ──────────────────────────────────────────────────────

def test_spatial_descriptor_shapes_and_finiteness():
    desc = RUDRASpatialDescriptor()
    img = torch.rand(2, 3, 32, 32) * 8.0        # scene-linear with highlights
    out = desc(img)
    assert out.shape[0] == 2 and out.ndim == 4 and out.shape[-2:] == (32, 32)
    assert torch.isfinite(out).all()
    # black input must not produce NaNs (log/chroma denominators)
    out0 = desc(torch.zeros(1, 3, 16, 16))
    assert torch.isfinite(out0).all()


# ── train modes ─────────────────────────────────────────────────────────────

def test_scheduled_loss_weights_progression():
    for mode in RUDRATrainMode:
        w0 = scheduled_loss_weights(mode, step=0, total_steps=1000)
        w1 = scheduled_loss_weights(mode, step=999, total_steps=1000)
        assert isinstance(w0, LossWeights) and isinstance(w1, LossWeights)
        for w in (w0, w1):
            for f in w.__dataclass_fields__:
                v = getattr(w, f)
                assert v >= 0.0 and v == v, f"{mode} {f} invalid: {v}"


# ── exr_io ──────────────────────────────────────────────────────────────────

def test_tensor_cache_round_trip(tmp_path):
    t = torch.rand(3, 8, 8)
    path = tmp_path / "cache.pt"
    save_tensor_cache(path, t, {"note": "smoke"})
    back, meta = load_tensor_cache(path)
    assert torch.equal(back, t) and meta["note"] == "smoke"


def test_save_exr_writes_readable_file(tmp_path):
    from rudra.delivery.exr import read_exr
    t = torch.rand(3, 6, 10) * 100.0
    out = save_exr(tmp_path / "t.exr", t, {"source": "smoke"})
    img, attrs = read_exr(out)
    assert img.shape == (6, 10, 3)
    assert attrs["rudra:source"] == "smoke"
    import numpy as np
    assert np.allclose(img, t.permute(1, 2, 0).numpy(), rtol=1e-3, atol=5e-2)
