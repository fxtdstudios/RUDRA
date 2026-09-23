"""The out-of-generator condition: SDR made with a Hable tone curve and a real
H.264 round trip, neither of which the model's training ever saw.

The headline +1.43 dB / +0.44 JOD on "hard" measures how well the network
undoes its own synthetic corruption (``degrade_sdr``). This condition exists
to ask whether any of that survives a tone curve and a codec the model was not
trained on.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


ffmpeg_only = pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not installed")


def test_hable_is_a_different_curve_from_aces():
    """The out-of-generator tone curve must differ from ACES in shape, not just
    by a constant scale -- otherwise the condition tests nothing new."""
    from training.prepare_training_data import hable_tonemap, tonemap_aces_approx

    lin = np.linspace(0.0, 16.0, 4096, dtype=np.float32)
    aces = tonemap_aces_approx(lin, 0.0)
    hable = hable_tonemap(lin)

    assert hable[0] == 0.0, "black must stay black"
    assert float(hable.max()) <= 1.0, "output must be bounded"
    assert np.all(np.diff(hable) >= 0.0), "Hable must be monotonic"
    ratio = hable / np.clip(aces, 1e-6, None)
    assert float(ratio.std()) > 1e-3, "Hable and ACES differ only by a scale"


@ffmpeg_only
def test_codec_round_trip_single_damages_the_frame():
    """A round trip that changes nothing would make the condition a no-op."""
    from training.export_bench_pairs import codec_round_trip_single

    rng = np.random.default_rng(0)
    frame = rng.random((64, 64, 3), dtype=np.float32)
    out = codec_round_trip_single(frame, crf=28)

    assert out.shape == frame.shape
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0
    assert float(np.abs(out - frame).mean()) > 1e-3


@ffmpeg_only
def test_out_of_generator_sdr_is_bounded_and_shaped():
    """The assembled condition returns a [1, 3, H, W] float tensor in [0, 1]."""
    from training.export_bench_pairs import out_of_generator_sdr

    rng = np.random.default_rng(1)
    # A reference in network units (nits / 10 000); 0.02 is ~diffuse white.
    hdr_np = (rng.random((32, 48, 3), dtype=np.float32) * 0.2).astype(np.float32)
    sdr = out_of_generator_sdr(hdr_np, crf=28)

    assert sdr.shape == (1, 3, 32, 48)
    assert float(sdr.min()) >= 0.0 and float(sdr.max()) <= 1.0
