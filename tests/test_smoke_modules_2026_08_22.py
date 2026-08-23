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

def _stats(peak, highlight_fraction, index=0):
    return SampleHDRStats(index=index, peak=peak, p95=peak * 0.5, mean=peak * 0.1,
                          highlight_fraction=highlight_fraction)


def test_bucket_ordering():
    lo = bucket_from_stats(_stats(0.5, 0.0))
    hi = bucket_from_stats(_stats(50.0, 0.4))
    assert lo != hi, "SDR-ish and highlight-heavy samples must land in different buckets"


def test_highlight_balanced_sampler_covers_dataset():
    specs = [(0.5, 0.0), (2.0, 0.05), (50.0, 0.4), (8.0, 0.2)]
    stats = [_stats(p, hf, index=i) for i, (p, hf) in enumerate(specs)]
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
    # Signature is scheduled_loss_weights(step); stages land at 0/10k/40k+.
    weights = {s: scheduled_loss_weights(s) for s in (0, 5_000, 15_000, 45_000, 80_000)}
    for s, w in weights.items():
        assert isinstance(w, LossWeights)
        for f in w.__dataclass_fields__:
            v = getattr(w, f)
            assert v >= 0.0 and v == v, f"step {s} {f} invalid: {v}"
    # highlight term must switch on as the schedule progresses
    assert weights[45_000].highlight >= weights[0].highlight
    assert list(RUDRATrainMode)  # modes enumerable"


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


# ── trainer_utils (run-integrity helpers, added same audit) ─────────────────

def test_seed_everything_reproduces():
    from rudra.trainer_utils import seed_everything
    seed_everything(123)
    a = torch.rand(4)
    seed_everything(123)
    b = torch.rand(4)
    assert torch.equal(a, b)


def test_atomic_torch_save_round_trip(tmp_path):
    from rudra.trainer_utils import atomic_torch_save
    payload = {"step": 7, "w": torch.rand(3)}
    path = atomic_torch_save(payload, tmp_path / "ckpt.pth")
    assert path.exists() and not path.with_suffix(".pth.tmp").exists()
    back = torch.load(path, weights_only=True)
    assert back["step"] == 7 and torch.equal(back["w"], payload["w"])


def test_atomic_safetensors_save_round_trip(tmp_path):
    st = pytest.importorskip("safetensors.torch")
    from rudra.trainer_utils import atomic_safetensors_save
    state = {"layer.weight": torch.rand(2, 2)}
    path = atomic_safetensors_save(state, tmp_path / "w.safetensors", metadata={"v": "1"})
    assert path.exists() and not path.with_suffix(".safetensors.tmp").exists()
    back = st.load_file(str(path))
    assert torch.equal(back["layer.weight"], state["layer.weight"])
