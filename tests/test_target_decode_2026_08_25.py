"""HDR targets must reach the trainer in the network's own radiance units.

The 24 Aug 2026 image run trained for 50,000 steps against targets that were
never decoded: ``load_rgb`` divided the stored uint16 by 65535 and handed the
raw log2 transfer-function code to the loss, while the model's inverse-ACES
baseline spoke linear nits/10000.  Mid grey: target 0.465, baseline 0.020.
Result: loss 1.83, psnr_log 6.12 dB.

These tests pin the decode so that cannot silently return.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.hdr_io import HDRStorage, encode_hdr_u16  # noqa: E402
from training.sdr2hdr_dataset import (  # noqa: E402
    DEFAULT_TARGET_CEILING, NETWORK_PEAK_NITS, load_rgb, storage_for,
)

LOG2 = HDRStorage(mode="log2_extended", diffuse_white_nits=203.0,
                  floor_nits=0.005, ceiling_nits=1_000_000.0)
PQ = HDRStorage(mode="pq_10000", diffuse_white_nits=203.0)


def _write_corpus(root: Path, nits: np.ndarray, storage: HDRStorage | None) -> Path:
    """A miniature pairs dir: hdr/<file>.png plus the ingest sentinel."""
    pairs = root / "pairs"
    (pairs / "hdr").mkdir(parents=True, exist_ok=True)
    if storage is not None:
        (pairs / "_ingest_config.json").write_text(
            json.dumps({"hdr_io_version": storage.version,
                        "storage": storage.as_dict()}), encoding="utf-8")
        scene_linear = nits / storage.diffuse_white_nits
        code, _ = encode_hdr_u16(scene_linear, storage)
    else:
        # Legacy August convention: clip(linear * 203/10000) * 65535.
        code = np.rint(np.clip(nits / NETWORK_PEAK_NITS, 0.0, 1.0) * 65535.0).astype(np.uint16)
    path = pairs / "hdr" / "sample.png"
    cv2.imwrite(str(path), cv2.cvtColor(code, cv2.COLOR_RGB2BGR))
    return path


@pytest.mark.parametrize("storage", [LOG2, PQ], ids=["log2_extended", "pq_10000"])
def test_known_nits_round_trip_into_network_units(tmp_path, storage):
    # 0.01 nit (deep shadow), diffuse white, a 1,000-nit specular, 8,000 nits.
    nits = np.array([[[0.01, 203.0, 1000.0]], [[8000.0, 203.0, 0.01]]], dtype=np.float64)
    path = _write_corpus(tmp_path, nits, storage)

    loaded = load_rgb(path, hdr=True)

    expected = nits / NETWORK_PEAK_NITS
    assert loaded.dtype == np.float32
    # 16-bit quantisation, not algebra, sets the tolerance.
    assert np.allclose(loaded, expected, rtol=2e-3, atol=1e-6), (
        f"decoded {loaded.ravel()} vs expected {expected.ravel()}")


def test_diffuse_white_matches_the_models_own_baseline(tmp_path):
    """The number the regression is really about.

    Diffuse white is 203 nits, i.e. 0.0203 in network units.  Undecoded, the
    log2 code for 203 nits reads ~0.465 -- a 23x scale error, applied to every
    pixel of every sample.
    """
    nits = np.full((2, 2, 3), 203.0)
    path = _write_corpus(tmp_path, nits, LOG2)

    decoded = load_rgb(path, hdr=True)
    raw_code = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0

    assert np.allclose(decoded, 203.0 / NETWORK_PEAK_NITS, rtol=2e-3)
    assert raw_code.mean() > 0.4, "sanity: the undecoded code really is ~0.465"
    assert decoded.mean() < 0.05 < raw_code.mean(), "the two conventions must not be confused"


def test_targets_are_clamped_to_the_network_ceiling(tmp_path):
    """SDR2HDRNet cannot emit past max_hdr; targets past it are unlearnable."""
    nits = np.full((2, 2, 3), 900_000.0)  # a sun, inside log2_extended's range
    path = _write_corpus(tmp_path, nits, LOG2)

    loaded = load_rgb(path, hdr=True)

    assert np.all(loaded <= DEFAULT_TARGET_CEILING + 1e-6)
    assert np.allclose(loaded, DEFAULT_TARGET_CEILING, rtol=1e-5)


def test_legacy_corpus_without_a_sentinel_is_read_the_old_way(tmp_path):
    nits = np.array([[[50.0, 203.0, 4000.0]]], dtype=np.float64)
    path = _write_corpus(tmp_path, nits, storage=None)

    assert storage_for(path) is None
    loaded = load_rgb(path, hdr=True)
    assert np.allclose(loaded, nits / NETWORK_PEAK_NITS, rtol=1e-3, atol=1e-6)


def test_sdr_is_untouched_by_the_hdr_path(tmp_path):
    sdr = (np.random.default_rng(0).random((4, 4, 3)) * 255).astype(np.uint8)
    path = tmp_path / "sdr.png"
    cv2.imwrite(str(path), cv2.cvtColor(sdr, cv2.COLOR_RGB2BGR))

    loaded = load_rgb(path, hdr=False)

    assert np.allclose(loaded, sdr.astype(np.float32) / 255.0, atol=1e-6)
    assert loaded.max() <= 1.0
