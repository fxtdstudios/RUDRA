"""The validation eval must sample the split, not read the front of it.

`DataLoader(val, shuffle=False)` with `evaluate_image(max_batches=N)` evaluates
the alphabetically first N records. On 29 Aug 2026 that made the conditioning
head's eval 32 records over 11 scenes -- every scene name between
"abandoned_factory" and "blau_river" -- while 403 records and 87 scenes were
never measured. Since RUDRA's error correlates with scene headroom (+0.46
against log2 peak nits), a slice chosen by alphabetical position reports a
different answer from the held-out benchmark, and the head trained against it
learned nothing because at that slice there was no defect.

The replacement must satisfy both constraints at once: representative, and
identical at every step so scores stay comparable within a run.
"""
import pytest

from training.train_sdr2hdr import deterministic_eval_order


def test_it_is_a_permutation():
    order = deterministic_eval_order(435)
    assert sorted(order) == list(range(435))


def test_it_is_stable_across_calls():
    """Scores are compared across thousands of steps; the order cannot drift."""
    assert deterministic_eval_order(435) == deterministic_eval_order(435)
    assert deterministic_eval_order(435) != deterministic_eval_order(435, seed=1)


def test_the_front_is_no_longer_the_front():
    order = deterministic_eval_order(435)
    assert order[:32] != list(range(32))


def test_a_prefix_spreads_across_the_split():
    """The failure was 32 records landing in one alphabetical corner.

    Real scenes sit in contiguous runs of 3 frames, so a front-read prefix
    covers prefix/3 scenes. A permuted prefix must reach far more of the split.
    """
    order = deterministic_eval_order(435)
    prefix = order[:32]
    assert max(prefix) > 300, "the prefix must reach the far end of the split"
    scenes = {i // 3 for i in prefix}          # 3 frames per scene, as in the corpus
    assert len(scenes) >= 28, f"only {len(scenes)} scenes covered, front-read gave 11"


def test_degenerate_sizes():
    assert deterministic_eval_order(0) == []
    assert deterministic_eval_order(1) == [0]


@pytest.mark.parametrize("size", [37, 256, 435, 1000])
def test_every_record_is_reachable(size):
    assert set(deterministic_eval_order(size)) == set(range(size))
