"""tools/diagnose_bench_gap.py: the training metric and the bench's, on one frame."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "diagnose_bench_gap", Path(__file__).resolve().parents[1] / "tools" / "diagnose_bench_gap.py")
gap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gap)


def test_the_training_metric_barely_sees_midtones_and_pu21_does():
    rng = np.random.default_rng(1)
    ref = np.exp(rng.uniform(np.log(0.5), np.log(20000.0), (32, 32, 3)))
    lifted = np.where(ref < 100.0, ref * 1.5, ref)          # half a stop wrong below 100 nits
    s = gap.frame_stats(lifted, ref)
    assert s["train_psnr"] > 45.0                           # the training metric calls it close
    assert s["pu_psnr"] < 40.0                              # the bench does not
    bias = [b["bias_stops"] for b in s["bands"]]
    assert abs(bias[1] - np.log2(1.5)) < 1e-9 and abs(bias[4]) < 1e-12


def test_bands_partition_the_frame_error():
    rng = np.random.default_rng(2)
    ref = np.exp(rng.uniform(np.log(0.01), np.log(5000.0), (16, 16, 3)))
    test = ref * np.exp(rng.normal(0.0, 0.1, ref.shape))
    s = gap.frame_stats(test, ref)
    total = np.mean((gap.pu21_encode(test) - gap.pu21_encode(ref)) ** 2)
    assert abs(sum(b["mse_part"] for b in s["bands"]) - total) < 1e-9 * total
    assert abs(sum(b["share"] for b in s["bands"]) - 1.0) < 1e-12
