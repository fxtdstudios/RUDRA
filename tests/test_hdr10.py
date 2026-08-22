"""Tests for HDR10 mastering and PQ encoding."""

import numpy as np

from rudra.hdr10 import master_to_peak, pq_eotf, pq_oetf


def test_pq_reference_points_round_trip():
    nits = np.asarray([0.0, 0.1, 100.0, 203.0, 1000.0, 10000.0], dtype=np.float32)
    decoded = pq_eotf(pq_oetf(nits))
    assert np.allclose(decoded, nits, rtol=2e-4, atol=2e-4)


def test_mastering_preserves_values_below_knee_and_limits_peak():
    # Normalized RUDRA values: 0.01 = 100 nits, 0.2 = 2000 nits.
    image = np.asarray([[[0.01, 0.01, 0.01], [0.2, 0.1, 0.05]]], dtype=np.float32)
    mastered = master_to_peak(image, peak_nits=1000.0, knee_nits=750.0)
    assert np.allclose(mastered[0, 0], 100.0, atol=1e-3)
    assert float(mastered.max()) <= 1000.0
    assert np.all(np.isfinite(mastered))


def test_mastering_keeps_neutral_pixels_neutral():
    image = np.full((2, 2, 3), 0.5, dtype=np.float32)
    mastered = master_to_peak(image, peak_nits=1000.0)
    assert np.allclose(mastered[..., 0], mastered[..., 1])
    assert np.allclose(mastered[..., 1], mastered[..., 2])
