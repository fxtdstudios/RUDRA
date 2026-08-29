"""The temporal refiner has to censor graded highlights, like the image loss does.

It did not. `--mode temporal` trained on a plain log-L1 that never saw a
ceiling, while 84% of the video clips carry one -- 562 graded to 4,000 nits,
223 to 991. A refiner trained that way learns to pull the image model's
reconstructed highlights back down to the cap, undoing the single thing that
makes v5 worth shipping. Nothing caught it because nothing tested that the
ceiling reached the loss at all.

So that is what these test: the shared error function, and the fact that both
paths use it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")

from rudra.sdr2hdr import (CENSORED_HEADROOM_STOPS, SDR2HDRNet,   # noqa: E402
                           censored_log_error, sdr2hdr_loss,
                           temporal_spatial_loss)

SCALE = 16.0
CEILING = 0.4          # 4,000 nits in network units


def _log(x):
    return float(np.log1p(np.asarray(x, dtype=np.float64) * SCALE))


def test_without_a_ceiling_it_is_plain_log_l1():
    pred = torch.rand(2, 3, 5, 7)
    target = torch.rand(2, 3, 5, 7)
    error, _, _, fraction = censored_log_error(pred, target, None)
    expect = (torch.log1p(pred * SCALE) - torch.log1p(target * SCALE)).abs()
    assert torch.allclose(error, expect)
    assert float(fraction) == 0.0


def test_a_censored_pixel_may_reconstruct_upward_for_free():
    """The grade says ">= ceiling". Going above it is the whole point."""
    target = torch.full((1, 3, 1, 1), CEILING)
    ceiling = torch.tensor([CEILING])
    for factor in (1.0, 2.0, 4.0, 2.0 ** CENSORED_HEADROOM_STOPS):
        pred = torch.full((1, 3, 1, 1), CEILING * factor)
        error, _, _, fraction = censored_log_error(pred, target, ceiling)
        assert float(error.max()) == pytest.approx(0.0, abs=1e-6), (
            f"charged for reconstructing {factor}x the ceiling")
        assert float(fraction) == 1.0


def test_a_censored_pixel_is_still_charged_for_falling_short():
    """One-sided, not free. Capping below the grade is the failure mode."""
    target = torch.full((1, 1, 1, 1), CEILING)
    pred = torch.full((1, 1, 1, 1), CEILING / 4.0)
    error, _, _, _ = censored_log_error(pred, target, torch.tensor([CEILING]))
    assert float(error) == pytest.approx(_log(CEILING) - _log(CEILING / 4.0), rel=1e-5)


def test_runaway_highlights_are_charged_past_the_allowance():
    """Without an upper anchor the cheapest move is to run to max_hdr."""
    target = torch.full((1, 1, 1, 1), CEILING)
    allowance = CEILING * (2.0 ** CENSORED_HEADROOM_STOPS)
    pred = torch.full((1, 1, 1, 1), allowance * 4.0)
    error, _, _, _ = censored_log_error(pred, target, torch.tensor([CEILING]))
    assert float(error) == pytest.approx(_log(allowance * 4.0) - _log(allowance), rel=1e-5)


def test_uncensored_pixels_in_a_censored_frame_are_untouched():
    target = torch.tensor([[[[CEILING]], [[CEILING / 8.0]], [[CEILING / 2.0]]]])
    pred = torch.full_like(target, CEILING / 16.0)
    error, _, _, fraction = censored_log_error(pred, target, torch.tensor([CEILING]))
    plain = (torch.log1p(pred * SCALE) - torch.log1p(target * SCALE)).abs()
    assert torch.allclose(error[:, 1:], plain[:, 1:])      # below the cap: unchanged
    assert float(fraction) == pytest.approx(1 / 3)


def test_an_infinite_ceiling_censors_nothing():
    """Sources with no recorded grade ceiling must behave like plain L1."""
    pred, target = torch.rand(2, 3, 4, 4), torch.rand(2, 3, 4, 4)
    error, _, _, fraction = censored_log_error(
        pred, target, torch.tensor([float("inf"), float("inf")]))
    plain, _, _, _ = censored_log_error(pred, target, None)
    assert torch.allclose(error, plain)
    assert float(fraction) == 0.0


def test_it_broadcasts_over_a_five_dimensional_clip():
    """(B, T, C, H, W) with a (B,) ceiling -- the temporal batch shape."""
    target = torch.rand(2, 4, 3, 5, 5) * 0.2
    target[0, :, :, 0, 0] = CEILING              # a censored pixel in every frame
    pred = torch.rand(2, 4, 3, 5, 5) * 0.2
    ceiling = torch.tensor([CEILING, float("inf")])
    error, _, _, fraction = censored_log_error(pred, target, ceiling)
    assert error.shape == target.shape
    # clip 1 has no finite ceiling, so nothing in it is censored
    assert float(fraction) == pytest.approx(4 * 3 / target.numel())


def test_temporal_spatial_loss_uses_the_ceiling():
    """The regression guard: a refiner that ignores the cap scores the same
    whether it reconstructs the highlight or flattens it."""
    target = torch.full((1, 2, 3, 4, 4), CEILING)
    ceiling = torch.tensor([CEILING])
    reconstructed = torch.full_like(target, CEILING * 4.0)   # +2 stops, allowed
    flattened = torch.full_like(target, CEILING / 4.0)       # capped, penalised
    good, fraction = temporal_spatial_loss(reconstructed, target, ceiling)
    bad, _ = temporal_spatial_loss(flattened, target, ceiling)
    assert float(fraction) == 1.0
    assert float(good) == pytest.approx(0.0, abs=1e-6)
    assert float(bad) > 0.3, "flattening a censored highlight went unpunished"
    # and without the ceiling the two are indistinguishable in the wrong direction
    naive_good, _ = temporal_spatial_loss(reconstructed, target, None)
    assert float(naive_good) > float(bad), (
        "plain L1 prefers the flattened highlight -- this is the bug")


def test_the_eval_path_actually_reads_the_ceiling_off_the_batch():
    """Plumbing, not maths. The bug was never in the loss -- it was that the
    ceiling in the batch never reached it, so that is what gets checked."""
    sys.path.insert(0, str(REPO / "training"))
    from train_sdr2hdr import evaluate_temporal
    from rudra.sdr2hdr import TemporalHDRRefiner

    torch.manual_seed(3)
    image_model = SDR2HDRNet(base_channels=8).eval()
    refiner = TemporalHDRRefiner(channels=8).eval()
    target = torch.rand(1, 3, 3, 16, 16) * 0.1
    target[..., :8, :8] = CEILING                      # a quarter of every frame
    batch = {"sdr": torch.rand(1, 3, 3, 16, 16),
             "hdr": target,
             "ceiling": torch.tensor([CEILING])}
    metrics = evaluate_temporal(image_model, refiner, [batch],
                                torch.device("cpu"), max_batches=1)
    assert "censored_fraction" in metrics, "eval never looked at the ceiling"
    assert metrics["censored_fraction"] == pytest.approx(0.25, abs=0.02)
    # Equal here, and that is right: an untrained refiner under-predicts, and
    # censoring only ever forgives over-prediction. It must never cost more.
    assert metrics["censored_log_l1"] <= metrics["log_l1"] + 1e-6


def test_censoring_never_costs_more_than_plain_l1():
    """The invariant that makes this safe to switch on mid-project: for any
    prediction, censored error <= plain error, elementwise. Falling short is
    charged identically; only over-reconstruction is forgiven."""
    torch.manual_seed(5)
    for _ in range(8):
        target = torch.rand(2, 3, 8, 8) * 0.5
        target[:, :, :2, :2] = CEILING
        pred = torch.rand(2, 3, 8, 8) * 2.0        # deliberately over the top
        ceiling = torch.tensor([CEILING, CEILING])
        censored, _, _, _ = censored_log_error(pred, target, ceiling)
        plain, _, _, _ = censored_log_error(pred, target, None)
        assert torch.all(censored <= plain + 1e-6)


def test_the_image_loss_still_agrees_with_the_shared_helper():
    """sdr2hdr_loss was rewritten to call it; its numbers must not have moved."""
    torch.manual_seed(11)
    net = SDR2HDRNet(base_channels=8).eval()
    sdr = torch.rand(2, 3, 32, 32)
    target = torch.rand(2, 3, 32, 32) * 0.5
    target[:, :, :4, :4] = CEILING
    ceiling = torch.tensor([CEILING, CEILING])
    with torch.inference_mode():
        out = net(sdr)
        losses = sdr2hdr_loss(out, sdr, target, target_ceiling=ceiling)
        error, _, _, fraction = censored_log_error(out.hdr, target, ceiling)
    assert float(losses["censored_fraction"]) == pytest.approx(float(fraction))
    assert float(losses["log_l1"]) == pytest.approx(float(error.mean()), rel=1e-6)


def test_checkpoint_selection_follows_the_censored_objective():
    """The loss and the selector have to agree, or the fix undoes itself.

    The refiner trains on a censored objective but `log_l1` is reported plain,
    for comparability with image mode. Selecting best.pt on that plain number
    would have scored a refiner that correctly reconstructs ABOVE a grading
    ceiling as worse than one that flattens to it -- because plain L1 measures
    against a target that was capped. So the model that does the right thing
    loses the checkpoint race, and the censored loss achieves nothing.
    """
    sys.path.insert(0, str(REPO / "training"))
    from train_sdr2hdr import temporal_score

    weight = 0.5
    # identical temporal consistency; the two differ only in the highlights
    flattens = {"log_l1": 0.0036, "censored_log_l1": 0.0036, "temporal": 0.0019}
    reconstructs = {"log_l1": 0.0052, "censored_log_l1": 0.0021, "temporal": 0.0019}

    naive = lambda m: m["log_l1"] + weight * m["temporal"]
    assert naive(flattens) < naive(reconstructs), (
        "premise check: plain L1 really does prefer the flattened highlight")
    assert temporal_score(reconstructs, weight) < temporal_score(flattens, weight), (
        "selection still prefers flattening -- the censored loss is being undone "
        "at checkpoint time")


def test_selection_falls_back_for_logs_written_before_censoring():
    """Runs from before censored_log_l1 existed must still score, not crash."""
    sys.path.insert(0, str(REPO / "training"))
    from train_sdr2hdr import temporal_score
    legacy = {"log_l1": 0.0036, "temporal": 0.0019}
    assert temporal_score(legacy, 0.5) == pytest.approx(0.0036 + 0.5 * 0.0019)
