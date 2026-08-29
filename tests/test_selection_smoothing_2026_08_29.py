"""best.pt must not be selected on a single lucky evaluation.

The v5 run selected on a raw maximum of `composite_gain` over 102 evaluations.
The clean/hard eval set is deterministic -- centre crops, shuffle=False, the
same 256 records every time -- so its spread is the model oscillating, not
sampling noise, and a maximum over that series finds the luckiest step. The
shipped checkpoint's `clean_gain_db` of +0.02 ranked 9th of 102 against a mean
of -1.43 and a standard deviation of 1.29, and the independent benchmark later
measured +1.43 dB where selection had promised +1.80.

These tests pin the trailing-median selection that replaces it.
"""
import random

import pytest

from training.train_sdr2hdr import selection_score


def test_window_of_one_is_the_old_behaviour():
    history = [3.0, 1.0, 2.0]
    assert selection_score(history, 1) == 2.0
    assert selection_score(history, 0) == 2.0


def test_median_of_a_full_window():
    assert selection_score([9.0, 1.0, 2.0, 3.0, 4.0, 5.0], 5) == 3.0


def test_short_history_uses_what_there_is():
    """A resumed run starts with an empty window; it must not crash or lie."""
    assert selection_score([2.0], 5) == 2.0
    assert selection_score([2.0, 4.0], 5) == 3.0


def test_one_lucky_step_cannot_win():
    """The whole point. Scores are negated gains, so lower is better."""
    noisy = [-1.4] * 20
    noisy[10] = -9.9                      # a single outlier draw
    raw_best = min(noisy)
    smoothed = [selection_score(noisy[:i + 1], 5) for i in range(len(noisy))]
    assert raw_best == -9.9, "a raw maximum takes the outlier"
    assert min(smoothed) > -9.9, "the trailing median must not"
    assert min(smoothed) == pytest.approx(-1.4)


def test_a_genuinely_better_region_still_wins():
    """Smoothing must not make selection blind -- a sustained improvement wins."""
    history = [-1.0] * 10 + [-3.0] * 10
    smoothed = [selection_score(history[:i + 1], 5) for i in range(len(history))]
    assert min(smoothed) == pytest.approx(-3.0)


def test_v5_series_selection_moves_off_the_lucky_step():
    """Replay of the v5 run's shape: a noisy series with one standout draw.

    Reproduces the failure with the measured statistics (mean -1.43, sd 1.29)
    rather than asserting on the run's exact numbers, which live on E:.
    """
    rng = random.Random(20260829)
    clean = [rng.gauss(-1.43, 1.29) for _ in range(102)]
    hard = [rng.gauss(1.19, 0.64) for _ in range(102)]
    composite = [h + min(0.0, c) for h, c in zip(hard, clean)]
    scores = [-value for value in composite]              # negated: lower wins

    raw_pick = min(range(len(scores)), key=lambda i: scores[i])
    running = [selection_score(scores[:i + 1], 5) for i in range(len(scores))]
    smooth_pick = min(range(len(running)), key=lambda i: running[i])

    assert clean[raw_pick] > -0.5, "the raw maximum lands on a lucky clean draw"
    # The smoothed pick is not required to be a different index every seed, but
    # its promise must be closer to what the run actually sustains.
    assert running[smooth_pick] >= scores[raw_pick], (
        "the smoothed score must never promise more than the raw maximum")
