"""rudra.grain.settle_highlight_grain: grain settled in flat highlights only.

The tester's report (25 Sep 2026): after the chroma fix, a colour-neutral grain
in recovered highlights, peaking where the source sits at 0.95-0.98. These pin
what the fix may and may not touch.
"""
import numpy as np
import pytest

from rudra.grain import LUMA_REC2020, settle_highlight_grain, source_flatness


def _plate(seed=7, h=40, w=96):
    rng = np.random.default_rng(seed)
    sdr = np.zeros((h, w, 3))
    nits = np.zeros((h, w, 3))
    sdr[:, :32] = 0.5                                   # below the knee
    sdr[:, 32:80] = 0.97 + rng.integers(-1, 2, (h, 48, 1)) / 255.0   # one-code grain
    sdr[:, 80:] = 0.6                                   # an edge at x = 80
    nits[:, :32] = 50.0
    nits[:, 32:80] = 1000.0 * (1.0 + 0.1 * rng.standard_normal((h, 48, 1)))
    nits[:, 80:] = 120.0
    nits *= np.array([1.0, 0.8, 0.5])                   # a warm hue to keep
    return np.clip(sdr, 0, 1), nits


def _luma(n):
    return n @ LUMA_REC2020


def test_flat_highlight_grain_is_settled():
    sdr, nits = _plate()
    out = settle_highlight_grain(nits, sdr)
    before = _luma(nits)[4:-4, 40:72]
    after = _luma(out)[4:-4, 40:72]
    assert after.std() < 0.3 * before.std()
    assert after.mean() == pytest.approx(before.mean(), rel=0.01)


def test_below_the_knee_and_across_an_edge_nothing_moves():
    sdr, nits = _plate()
    out = settle_highlight_grain(nits, sdr)
    np.testing.assert_array_equal(out[:, :32], nits[:, :32])
    np.testing.assert_array_equal(out[:, 84:], nits[:, 84:])


def test_hue_is_kept():
    sdr, nits = _plate()
    out = settle_highlight_grain(nits, sdr)
    np.testing.assert_allclose(out[..., 0] / out[..., 1], nits[..., 0] / nits[..., 1], rtol=1e-12)
    np.testing.assert_allclose(out[..., 2] / out[..., 1], nits[..., 2] / nits[..., 1], rtol=1e-12)


def test_structure_in_the_source_is_not_flat():
    sdr = np.full((20, 20, 3), 0.97)
    sdr[:, 10:] = 0.93                                  # ten codes: structure
    flat = source_flatness(sdr)
    assert flat[:, 9:11].max() == 0.0
    assert flat[:, :5].min() == 1.0


def test_a_glint_that_clipped_one_channel_is_structure():
    sdr = np.full((20, 20, 3), [1.0, 0.6, 0.4])
    sdr[8:12, 8:12] = [1.0, 0.95, 0.9]                  # max channel flat, luma not
    assert source_flatness(sdr)[10, 10] == 0.0


def test_nothing_above_the_knee_returns_the_input():
    sdr = np.full((8, 8, 3), 0.4)
    nits = np.full((8, 8, 3), 30.0)
    assert settle_highlight_grain(nits, sdr) is nits


def test_arguments_are_checked():
    sdr, nits = _plate()
    with pytest.raises(ValueError):
        settle_highlight_grain(nits[:, :10], sdr)
    with pytest.raises(ValueError):
        settle_highlight_grain(nits, sdr, knee=1.0)
    with pytest.raises(ValueError):
        settle_highlight_grain(nits, sdr, sigma=0.0)
