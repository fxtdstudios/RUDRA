"""Delivery-graded targets are censored observations, not ground truth.

Measured 28 Aug 2026: 84% of the v3 corpus comes from sources with a hard
grading ceiling (HdM-HDR-2014 at 4,000 nits, Rec2100-PQ-1K at ~991, Chimera at
10,000), and 8,528 records peak exactly on it. A pixel sitting there means
">= ceiling", not "= ceiling". Training plain L1 against it teaches the model
to cap -- and it did: the v4 model, trained on that mix, reconstructed 1.23
stops LESS highlight than v3b on a controlled specular (2.18 -> 0.95 stops
inside the highlight mask), despite six times the data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.build_manifests import annotate_ceilings, detect_grading_ceiling  # noqa: E402


# --------------------------------------------------------------------------
# ceiling detection (no torch)
# --------------------------------------------------------------------------
def test_a_repeated_exact_peak_is_a_grading_ceiling():
    """Natural scene peaks never repeat bit-for-bit; a clip does."""
    graded = [4000.0] * 30 + [1201.4, 883.2, 3999.1, 2140.7]
    assert detect_grading_ceiling(graded) == 4000.0


def test_scene_referred_peaks_are_not_flagged():
    """963 Poly Haven stills, zero false positives, on the real corpus."""
    hdri = [171332.4, 8821.9, 40233.1, 612.5, 99812.3, 1204.8, 55231.0, 7781.2,
            310.4, 24488.9]
    assert detect_grading_ceiling(hdri) is None


def test_too_few_frames_is_not_evidence_of_a_ceiling():
    assert detect_grading_ceiling([4000.0, 4000.0, 4000.0]) is None


def test_a_dark_repeated_peak_is_not_a_delivery_ceiling():
    """A scene that simply never gets bright is not a graded clip."""
    assert detect_grading_ceiling([12.0] * 40) is None


def test_annotate_marks_only_records_sitting_on_the_ceiling():
    records = (
        [{"scene_id": "graded", "asset_id": f"g{i}", "peak_nits": 4000.0} for i in range(12)]
        + [{"scene_id": "graded", "asset_id": "dim", "peak_nits": 900.0}]
        + [{"scene_id": "hdri", "asset_id": f"h{i}", "peak_nits": 1000.0 * (i + 3)}
           for i in range(12)]
    )

    censored = annotate_ceilings(records)

    assert censored == 12
    graded = [r for r in records if r["scene_id"] == "graded"]
    assert all(r["ceiling_nits"] == 4000.0 for r in graded)
    assert sum(r["peak_at_ceiling"] for r in graded) == 12
    assert [r for r in graded if r["asset_id"] == "dim"][0]["peak_at_ceiling"] is False
    assert all(r["ceiling_nits"] is None for r in records if r["scene_id"] == "hdri")
    assert all(r["peak_at_ceiling"] is False for r in records if r["scene_id"] == "hdri")


# --------------------------------------------------------------------------
# the loss itself (needs torch; runs on the training box)
# --------------------------------------------------------------------------
# Imported lazily so the ceiling-detection tests above still run on a machine
# without torch -- they are the half that guards the corpus, and they should
# never be silently skipped just because this box has no CUDA stack.
try:
    import torch

    from rudra.sdr2hdr import CENSORED_HEADROOM_STOPS, SDR2HDRNet, sdr2hdr_loss
    HAS_TORCH = True
except Exception:  # pragma: no cover - exercised on the training box
    HAS_TORCH = False
    CENSORED_HEADROOM_STOPS = 3.0

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="loss test needs torch")

CEILING = 4000.0 / 10_000.0  # network units


def _case(pred_nits: float, target_nits: float = 4000.0):
    """One flat frame at a censored level, with a controllable prediction."""
    sdr = torch.full((1, 3, 32, 32), 0.99)          # clipped SDR -> highlight mask fires
    target = torch.full((1, 3, 32, 32), target_nits / 10_000.0)
    model = SDR2HDRNet(base_channels=8)
    with torch.no_grad():
        output = model(sdr)
        output = output.__class__(
            torch.full_like(target, pred_nits / 10_000.0), output.baseline,
            output.log_residual, output.highlight_mask, output.shadow_mask,
            output.highlight_logits, output.shadow_logits)
    return sdr, target, output


def _loss(pred_nits, ceiling):
    sdr, target, output = _case(pred_nits)
    tensor = None if ceiling is None else torch.tensor([ceiling])
    return sdr2hdr_loss(output, sdr, target, target_ceiling=tensor)


@needs_torch
def test_over_predicting_a_censored_pixel_is_free():
    """">= 4,000 nits" must not be punished for saying 12,000."""
    at = _loss(4000.0, CEILING)
    above = _loss(12000.0, CEILING)

    assert float(above["highlight"]) == pytest.approx(float(at["highlight"]), abs=1e-6)
    assert float(above["log_l1"]) == pytest.approx(float(at["log_l1"]), abs=1e-6)


@needs_torch
def test_under_predicting_a_censored_pixel_still_costs():
    below = _loss(1000.0, CEILING)
    at = _loss(4000.0, CEILING)
    assert float(below["highlight"]) > float(at["highlight"]) + 0.1


@needs_torch
def test_without_a_ceiling_over_prediction_is_punished_as_before():
    """Scene-referred targets keep plain L1 -- no behaviour change for Poly Haven."""
    above = _loss(12000.0, None)
    at = _loss(4000.0, None)
    assert float(above["highlight"]) > float(at["highlight"]) + 0.1


@needs_torch
def test_runaway_past_the_allowance_is_charged():
    """A one-sided loss with no upper anchor would run to the max_hdr clamp."""
    allowed = 4000.0 * (2.0 ** CENSORED_HEADROOM_STOPS)
    inside = _loss(allowed * 0.9, CEILING)
    outside = _loss(allowed * 4.0, CEILING)
    assert float(outside["highlight"]) > float(inside["highlight"]) + 0.05


@needs_torch
def test_censored_fraction_is_reported():
    assert float(_loss(4000.0, CEILING)["censored_fraction"]) == pytest.approx(1.0)
    assert float(_loss(4000.0, None)["censored_fraction"]) == 0.0


@needs_torch
def test_an_infinite_ceiling_censors_nothing():
    sdr, target, output = _case(12000.0)
    with_inf = sdr2hdr_loss(output, sdr, target, target_ceiling=torch.tensor([float("inf")]))
    plain = sdr2hdr_loss(output, sdr, target)
    assert float(with_inf["log_l1"]) == pytest.approx(float(plain["log_l1"]), abs=1e-6)
