"""The exposure anchor: does unclipped picture come back unchanged?

The failure this guards against is silent. ``sdr_to_baseline_hdr`` carries a
factor of two -- the ``-1 EV`` that prepare_training_data.py applies before the
ACES curve -- which is right for corpus frames and wrong for every other SDR.
Measured on the shipped checkpoint, mid-grey came back +1.43 stops and half of
a real frame moved by more than a stop, with nothing reporting an error.

The second test here is the one that already caught a bug: keyed on LUMA, a
saturated red clipped only in the red channel reads as ordinary picture and
gets pushed back down, which cost the clipped region most of its headroom
(mean gain 1.14x where it should have been about 2x). The knee is keyed on the
max channel for exactly that reason.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.anchor import (DEFAULT_KNEE, DIFFUSE_WHITE_NITS, anchor_gain,  # noqa: E402
                          anchor_to_sdr, srgb_to_linear)


def ramp(width: int = 256) -> np.ndarray:
    v = np.linspace(0.0, 1.0, width, dtype=np.float64)
    return np.repeat(v[None, :, None], 3, axis=2).repeat(8, axis=0)


def lifted(sdr: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """A stand-in for the baseline's behaviour: everything lifted."""
    return srgb_to_linear(sdr) * DIFFUSE_WHITE_NITS * factor


def test_unclipped_picture_returns_to_the_source():
    sdr = ramp()
    out = anchor_to_sdr(lifted(sdr), sdr)
    want = srgb_to_linear(sdr) * DIFFUSE_WHITE_NITS
    below = sdr.max(axis=-1) < DEFAULT_KNEE - 0.04
    # Absolute nits, not a ratio: in black both sides are zero and a ratio
    # there measures the guard constant rather than the picture.
    err = np.abs(out[below] - want[below])
    assert err.max() < 0.5, float(err.max())          # half a nit


def test_a_single_clipped_channel_keeps_its_headroom():
    """255, 40, 30 -- a clipped red. Its LUMA is 0.29, well under the knee.

    Keyed on luma this reads as ordinary picture and the recovery is discarded.
    The patch sits inside an ordinary ramp so the knee band has real pixels to
    measure from: a frame that is nothing but clipped red has no knee, and
    testing on one would prove nothing about a plate.
    """
    sdr = ramp(256)
    sdr[:4, 100:120, :] = [1.0, 40/255, 30/255]
    hdr = lifted(sdr)
    hdr[:4, 100:120, 0] *= 5.0                  # what recovery put back
    out = anchor_to_sdr(hdr, sdr)
    kept = out[:4, 100:120, 0].mean() / hdr[:4, 100:120, 0].mean()
    # The whole frame is scaled down by roughly the baseline's lift; the red
    # must come through with the rest of the highlights, not be singled out.
    assert kept > 0.35, kept


def test_the_gain_is_one_scalar_per_pixel_so_hue_cannot_move():
    rng = np.random.default_rng(20260910)
    sdr = rng.random((16, 16, 3))
    out = anchor_to_sdr(lifted(sdr, 3.0), sdr)
    hdr = lifted(sdr, 3.0)
    per_channel = out / np.maximum(hdr, 1e-9)
    spread = per_channel.max(axis=-1) - per_channel.min(axis=-1)
    assert spread.max() < 1e-9, float(spread.max())


def test_the_output_is_continuous_across_the_knee():
    """The picture must not step, which is not the same as the gain not stepping.

    The gain does jump at absolute black -- where both target and actual are
    zero, its value is whatever the guard says -- but it is multiplied by zero
    there, so nothing reaches the picture. Measuring gain instead of output
    fails on that harmless spike and says nothing about the knee, which is
    where a step would actually be visible: a hard edge along a gradient sky.
    """
    sdr = ramp(4096)
    out = anchor_to_sdr(lifted(sdr), sdr)[0, :, 0]
    d = np.abs(np.diff(out))
    near = np.abs(ramp(4096)[0, :-1, 0] - DEFAULT_KNEE) < 0.08
    # Across the knee, no single code step may move more than a few times the
    # median step of the ramp as a whole.
    assert d[near].max() < 6.0 * np.median(d[d > 0]), (d[near].max(), np.median(d[d > 0]))


def test_above_the_knee_the_reconstruction_survives():
    sdr = ramp()
    hdr = lifted(sdr)
    hdr[:, -20:, :] *= 6.0                      # a reconstructed specular
    out = anchor_to_sdr(hdr, sdr)
    plain = anchor_to_sdr(lifted(sdr), sdr)
    assert out[:, -1, :].mean() > 3.0 * plain[:, -1, :].mean()


def test_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match="shape mismatch"):
        anchor_to_sdr(np.zeros((4, 4, 3)), np.zeros((4, 5, 3)))


@pytest.mark.parametrize("knee", [0.0, 1.0, -0.2, 1.4])
def test_a_knee_outside_the_range_is_refused(knee):
    sdr = ramp(64)
    with pytest.raises(ValueError, match="knee"):
        anchor_to_sdr(lifted(sdr), sdr, knee=knee)


def test_black_input_does_not_produce_nan():
    sdr = np.zeros((8, 8, 3))
    out = anchor_to_sdr(np.zeros((8, 8, 3)), sdr)
    assert np.isfinite(out).all()


def test_order_is_preserved():
    sdr = ramp(512)
    out = anchor_to_sdr(lifted(sdr), sdr) @ np.array([0.2126, 0.7152, 0.0722])
    row = out[0]
    assert np.all(np.diff(row) >= -1e-6), "anchoring must not invert the ramp"
