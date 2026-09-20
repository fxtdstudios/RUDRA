"""The reconstruction QC gate.

Section 4 of the QC practice is explicit: test that the gate FAILS on a
deliberately broken render before trusting it to pass on a good one. Every
check below breaks one thing and asserts the gate notices. The gate passing a
good frame is one test out of many, and the least interesting.

The rule the last test guards is the one that matters most. UNMEASURED is not
a pass. A metric whose precondition failed -- no flat region, nothing clipped
-- has told you nothing, and a gate that reads nothing as approval is worse
than no gate, because it produces a green tick.
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

from rudra.anchor import anchor_to_sdr                              # noqa: E402
from rudra.chroma import carry_source_chroma                        # noqa: E402
from rudra.qc import (DEFAULT_THRESHOLDS, FAIL, PASS, UNMEASURED,   # noqa: E402
                      check_frame, format_report, load_thresholds, srgb_to_linear)

DW = 203.0


def source(size: int = 320, seed: int = 20260910) -> np.ndarray:
    """A gradient with a blown patch and 8-bit dither: flat area and clipping."""
    import cv2
    rng = np.random.default_rng(seed)
    x = np.linspace(0.15, 0.98, size)
    field = np.repeat(np.repeat((x[None, :] * 255)[..., None], 3, axis=2), size, axis=0)
    # Real texture, not just quantisation. A pure ramp with +/-1 code of dither
    # is the worst case this metric can see -- the only high-frequency content
    # IS the quantisation, so the predicted noise floor is near zero and the
    # excess ratio explodes (5.2x on a frame with nothing wrong with it). A
    # photograph carries detail the curve amplifies proportionally. The
    # threshold is calibrated on one; the fixture has to be one.
    texture = cv2.GaussianBlur(rng.normal(0.0, 1.0, (size, size, 3)), (0, 0), 1.4)
    codes = (field + texture * 6.0).round()
    codes += rng.integers(-1, 2, size=codes.shape)
    codes[40:240, 40:240] = 255                                     # a large blown patch
    return np.clip(codes, 0, 255) / 255.0


def reconstructed(sdr: np.ndarray) -> np.ndarray:
    """The real baseline curve, then the two corrections, as shipped."""
    import torch
    from rudra.sdr2hdr import sdr_to_baseline_hdr
    x = torch.from_numpy(sdr.astype(np.float32)).permute(2, 0, 1)[None]
    hdr = sdr_to_baseline_hdr(x)[0].permute(1, 2, 0).numpy().astype(np.float64) * 10_000.0
    # Recovered specular WITH structure in it: a uniform scale of a blown
    # area is not recovery, and the gate is supposed to say so.
    yy, xx = np.mgrid[0:200, 0:200]
    hdr[40:240, 40:240] *= (2.0 + 1.4 * np.sin(xx / 9.0) * np.cos(yy / 11.0))[..., None]
    return carry_source_chroma(anchor_to_sdr(hdr, sdr), sdr)


@pytest.fixture(scope="module")
def good():
    sdr = source()
    return sdr, reconstructed(sdr)


def status_of(report, name):
    return next(c.status for c in report.checks if c.name == name)


# -- the gate must pass a good frame, once --------------------------------
def test_a_good_reconstruction_passes(good):
    sdr, hdr = good
    report = check_frame(hdr, sdr)
    assert report.verdict == PASS, format_report(report, "good", "320x320")


# -- and fail every broken one --------------------------------------------
def test_a_nan_fails(good):
    sdr, hdr = good
    broken = hdr.copy()
    broken[10, 10, 0] = np.nan
    assert check_frame(broken, sdr).verdict == FAIL


def test_a_negative_fails(good):
    sdr, hdr = good
    broken = hdr.copy()
    broken[10, 10, 2] = -1.0
    assert check_frame(broken, sdr).verdict == FAIL


def test_pinning_the_ceiling_fails(good):
    sdr, hdr = good
    broken = hdr.copy()
    broken[:60, :] = 40_000.0
    r = check_frame(broken, sdr)
    assert status_of(r, "at_ceiling_pct") == FAIL


def test_an_unanchored_lift_fails_do_no_harm(good):
    """The defect that started all of this: unclipped picture re-exposed."""
    sdr, hdr = good
    r = check_frame(hdr * 2.0, sdr)
    assert status_of(r, "do_no_harm_median") == FAIL
    assert r.verdict == FAIL


def test_per_channel_flecking_fails_the_chroma_check(good):
    sdr, hdr = good
    rng = np.random.default_rng(3)
    broken = hdr * (1.0 + rng.normal(0.0, 0.12, size=hdr.shape))
    assert status_of(check_frame(broken, sdr), "chroma_shift_uv") == FAIL


def test_a_reconstruction_that_recovered_nothing_fails(good):
    """Output equal to the source is an expensive copy, not a reconstruction."""
    sdr, _ = good
    r = check_frame(srgb_to_linear(sdr) * DW, sdr)
    assert status_of(r, "clip_gain") == FAIL


def test_a_uniform_scale_of_the_blown_area_fails_structure(good):
    sdr, hdr = good
    broken = hdr.copy()
    clipped = sdr.max(axis=-1) >= 254 / 255
    broken[clipped] = np.array([900.0, 900.0, 900.0])               # flat, no detail
    assert status_of(check_frame(broken, sdr), "clip_structure_pct") == FAIL


def test_manufactured_noise_fails(good):
    sdr, hdr = good
    rng = np.random.default_rng(11)
    y = hdr @ np.array([0.2627, 0.6780, 0.0593])
    broken = hdr + rng.normal(0.0, 0.30, size=hdr.shape) * y[..., None]
    assert status_of(check_frame(broken, sdr), "hf_excess") == FAIL


# -- unmeasured must never read as approval -------------------------------
def test_unmeasured_counts_as_a_failure():
    """A frame too small to measure has told the gate nothing."""
    sdr = np.full((8, 8, 3), 0.5)
    hdr = srgb_to_linear(sdr) * DW
    report = check_frame(hdr, sdr)
    assert any(c.status == UNMEASURED for c in report.checks)
    assert report.verdict == FAIL
    assert all(not c.ok for c in report.checks if c.status == UNMEASURED)


def test_no_clipping_in_the_source_is_unmeasured_not_passed():
    sdr = np.clip(np.repeat(np.linspace(0.1, 0.7, 256)[None, :, None], 3, axis=2)
                  .repeat(256, axis=0), 0, 1)
    hdr = srgb_to_linear(sdr) * DW
    r = check_frame(hdr, sdr)
    assert status_of(r, "clip_gain") == UNMEASURED
    assert r.verdict == FAIL


# -- thresholds live in config, with a reason each ------------------------
def test_every_threshold_carries_a_reason():
    raw = json.loads(DEFAULT_THRESHOLDS.read_text(encoding="utf-8"))
    for key, entry in raw.items():
        if key.startswith("_"):
            continue
        assert "value" in entry and "reason" in entry, key
        assert len(entry["reason"]) > 40, f"{key}: a reason nobody can use is not a reason"


def test_thresholds_are_not_hard_coded(good):
    """Passing a stricter set must change the verdict, or they are decoration."""
    sdr, hdr = good
    strict = load_thresholds()
    strict["do_no_harm_within_1pct_min"] = 0.999
    assert check_frame(hdr, sdr, thresholds=strict).verdict == FAIL
