"""The corpus renders its SDR side at an exposure offset, and everything
downstream has to undo the same one.

Two defects live here, and only one of them is arithmetic.

The arithmetic one: the analytic baseline multiplied inverse-ACES by a literal
``2 * 203/10000``. The 2 is not a constant of nature, it is
``2**(-(-1 EV))`` -- the corpus render's exposure, undone. Written as a literal
it was correct and unexplained, and it silently became wrong the moment anyone
changed the render.

The real one: -1 EV before a curve that saturates near 1.0 means the SDR side
almost never clips. The corpus this produced has a median clipped fraction of
0.000% and 52.6% of frames with no clipped pixel at all. An inverse tone mapper
is a machine for saying what was above a blown highlight, and it was being
trained on a set that has almost none. No test caught it because no test, and
no part of the pipeline, ever measured whether the SDR side clipped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")

from rudra.sdr2hdr import (CLIPPING_CORPUS_EV, LEGACY_CORPUS_EV,  # noqa: E402
                           SDR2HDRNet, sdr_to_baseline_hdr)


def test_the_default_baseline_is_bit_for_bit_what_it_replaced():
    """Every shipped checkpoint was trained against ``2 * 203/10000``. If this
    drifts by so much as a rounding step those weights are being fed a baseline
    they never saw, which is a whole-stop error dressed as a grade."""
    sdr = torch.rand(1, 3, 16, 16)
    from rudra.sdr2hdr import inverse_aces_approx, srgb_to_linear
    expected = inverse_aces_approx(srgb_to_linear(sdr)) * (2.0 * 203.0 / 10000.0)
    got = sdr_to_baseline_hdr(sdr)
    assert torch.equal(got, expected), (got - expected).abs().max().item()


def test_the_scale_is_derived_from_the_exposure_not_written_down():
    """One stop of render exposure is one stop of baseline, in the opposite
    direction. The relationship is the point; a literal hides it."""
    sdr = torch.rand(1, 3, 8, 8)
    legacy = sdr_to_baseline_hdr(sdr, LEGACY_CORPUS_EV)
    clipping = sdr_to_baseline_hdr(sdr, CLIPPING_CORPUS_EV)
    ratio = (legacy / clipping.clamp_min(1e-12)).median().item()
    assert ratio == pytest.approx(2.0, rel=1e-5), ratio


def test_a_checkpoint_without_the_field_is_legacy():
    """Every checkpoint that existed when this was added has no corpus_ev in
    its config. Defaulting to anything else would silently re-baseline them."""
    net = SDR2HDRNet.from_config({"base_channels": 8})
    assert net.corpus_ev == LEGACY_CORPUS_EV
    newer = SDR2HDRNet.from_config({"base_channels": 8, "corpus_ev": 0.0})
    assert newer.corpus_ev == CLIPPING_CORPUS_EV


def test_the_model_uses_its_own_convention_for_the_baseline():
    """A network carrying the new convention must not be handed the old
    baseline, and the only way to be sure is that the two differ."""
    sdr = torch.rand(1, 3, 32, 32)
    legacy = SDR2HDRNet(base_channels=8, corpus_ev=LEGACY_CORPUS_EV)
    clipping = SDR2HDRNet(base_channels=8, corpus_ev=CLIPPING_CORPUS_EV)
    clipping.load_state_dict(legacy.state_dict())
    with torch.no_grad():
        a = legacy(sdr).baseline
        b = clipping(sdr).baseline
    assert not torch.allclose(a, b)
    assert torch.allclose(a, b * 2.0, atol=1e-6)


# -- the render, and whether it clips at all ------------------------------
def _prep():
    sys.path.insert(0, str(REPO / "training"))
    import importlib
    return importlib.import_module("prepare_training_data")


def _ramp(top: float = 16.0, n: int = 8192):
    """Scene-linear ramp from black to ``top``, where 1.0 is diffuse white."""
    r = np.linspace(0.0, top, n, dtype=np.float32)
    return np.repeat(r[None, :, None], 3, axis=2)


def test_the_top_code_is_reachable_at_all():
    """The defect underneath the defect.

    oetf_srgb(1.0) is 0.99999994 in float32, so ``* 255`` is 254.99998 and
    ``astype(uint8)`` floored it to 254. The corpus SDR could not contain 255
    at any exposure: every blown pixel was stored one code below full, nothing
    testing for clipping by ``== 255`` ever found any, and a model trained on
    it never saw the top code that real delivered SDR is full of.
    """
    prep = _prep()
    blown = prep.make_sdr(np.full((4, 4, 3), 64.0, np.float32), 0.0)
    assert blown.max() == 255, f"a 64x-over-white patch stores as {blown.max()}"


def test_zero_ev_clips_where_minus_one_ev_clips_far_less():
    """The change itself, measured rather than argued.

    -1 EV halves the scene before a curve that saturates near a linear input of
    7.24, so it takes twice the radiance to blow a highlight. On the same ramp
    that is a factor of three in how much of the frame clips.
    """
    prep = _prep()
    linear = _ramp()
    f_legacy = prep.clipped_fraction(prep.make_sdr(linear, prep.LEGACY_TONEMAP_EV))
    f_clipping = prep.clipped_fraction(prep.make_sdr(linear, 0.0))

    assert f_clipping > f_legacy, (f_legacy, f_clipping)
    assert f_clipping > 2.0 * f_legacy, (f_legacy, f_clipping)


def test_the_hdr_side_rounds_too():
    """Truncation there is half a code everywhere, which is a bias and not
    noise: every stored value comes out low."""
    prep = _prep()
    full = prep.make_hdr(np.full((2, 2, 3), prep.HDR_PEAK_NITS / prep.HDR_REF_NITS, np.float32))
    assert full.max() == 65535


def test_the_render_records_the_exposure_it_used():
    """The sidecar has to say which convention produced it. Without that, a
    corpus is a set of files whose baseline nobody can reconstruct."""
    prep = _prep()
    assert prep.TONEMAP_EV_OFFSET == CLIPPING_CORPUS_EV
    assert prep.LEGACY_TONEMAP_EV == LEGACY_CORPUS_EV


def test_clipped_fraction_measures_what_it_says():
    prep = _prep()
    assert prep.clipped_fraction(np.zeros((4, 4, 3), np.uint8)) == 0.0
    assert prep.clipped_fraction(np.full((4, 4, 3), 255, np.uint8)) == 1.0
    half = np.zeros((2, 4, 3), np.uint8)
    half[0] = 255
    assert prep.clipped_fraction(half) == pytest.approx(0.5)


def test_measure_clipping_scores_the_baseline_at_the_corpus_exposure():
    """measure_clipping.py --score compares RUDRA against the analytic inverse
    on clipped pixels. That baseline must invert the exposure the corpus
    actually applied -- ``model.corpus_ev`` -- not the legacy -1 EV default.
    On a 0 EV corpus the default is one stop off, which corrupts exactly the
    "RUDRA minus baseline" number the tool exists to produce."""
    src = (REPO / "training" / "measure_clipping.py").read_text(encoding="utf-8")
    assert "sdr_to_baseline_hdr(x, model.corpus_ev)" in src, \
        "measure_clipping.py must score its baseline contender at model.corpus_ev"
