"""CVVDP has to actually run, and say so honestly when it does not.

Between July and 28 Aug 2026 `rudra bench` never produced a single JOD number,
on a machine where ColorVideoVDP was installed. Three layers of blanket
`except Exception` turned one small type error -- `_get_cvvdp` was handed the
string "cpu" where pycvvdp later reads `self.device.type` -- into the message
"unavailable (install torch + pycvvdp)". The advice was to install what was
already installed, so nobody looked further, and the benchmark that gates the
paper went unrun.

Two things are worth testing after that: that the real backend is selected,
and that when it is not, the summary says why.
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

from rudra.delivery import bench                                  # noqa: E402
from rudra.delivery.exr import write_exr                          # noqa: E402


def _frame(peak_nits: float, seed: int = 0, size: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = rng.random((size, size, 3)).astype(np.float32) * 0.3
    frame[size // 3:size // 2, size // 3:size // 2] = peak_nits
    return frame


def _tree(root: Path, name: str, frames: dict[str, np.ndarray]) -> None:
    for stem, frame in frames.items():
        path = root / name / "scene" / f"{stem}.exr"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_exr(path, frame, half=True)


def test_a_missing_backend_names_itself(tmp_path, monkeypatch):
    """The message that cost this project a benchmark."""
    ref = {"a": _frame(4.0, 1)}
    _tree(tmp_path, "ref", ref)
    _tree(tmp_path, "test", ref)
    monkeypatch.setattr(bench, "_cvvdp_fn", lambda: None)
    bench._BACKEND_NOTE.clear()
    summary = bench.run_benchmark(tmp_path, nits_scale=203.0)
    assert summary["cvvdp_jod_mean"] is None
    assert summary["cvvdp_backend"].startswith("unavailable")
    # Never again a bare instruction to install what may already be installed.
    assert "install torch + pycvvdp" not in summary["cvvdp_backend"]


def test_test_dir_scores_a_second_method_against_one_reference(tmp_path):
    """A paper needs several rows; the reference frames are the expensive half."""
    ref = {"a": _frame(4.0, 1), "b": _frame(2.0, 2)}
    near = {k: v * 1.01 for k, v in ref.items()}
    far = {k: v * 4.0 for k, v in ref.items()}
    _tree(tmp_path, "ref", ref)
    _tree(tmp_path, "test", near)
    _tree(tmp_path, "baseline", far)
    good = bench.run_benchmark(tmp_path, nits_scale=203.0)
    bad = bench.run_benchmark(tmp_path, nits_scale=203.0, test_dir="baseline")
    assert good["pairs"] == bad["pairs"] == 2
    assert good["pu_psnr_db_mean"] > bad["pu_psnr_db_mean"] + 10
    assert bad["test_dir"] == "baseline"


def test_a_missing_method_directory_is_an_error_not_an_empty_score(tmp_path):
    _tree(tmp_path, "ref", {"a": _frame(4.0, 1)})
    _tree(tmp_path, "test", {"a": _frame(4.0, 1)})
    with pytest.raises(FileNotFoundError):
        bench.run_benchmark(tmp_path, nits_scale=203.0, test_dir="nope")


# ---- the real backend ------------------------------------------------------
pycvvdp = pytest.importorskip("pycvvdp", reason="ColorVideoVDP not installed")

from rudra.hdrvdp import _get_cvvdp, colorvideovdp_available, hdr_vdp3_jod  # noqa: E402


def test_get_cvvdp_tolerates_a_string_device():
    """The exact bug: pycvvdp stores what it is given and later reads
    .device.type, so a string blew up deep inside predict()."""
    metric = _get_cvvdp("cpu", "standard_hdr_linear")
    assert isinstance(metric.device, torch.device)


def test_the_real_backend_is_selected_not_the_proxy():
    assert colorvideovdp_available()
    ref = torch.from_numpy(_frame(4.0, 3)).permute(2, 0, 1)[None] * 203.0
    test = ref * 1.02
    value, backend = hdr_vdp3_jod(test, ref, color_space="rec2020",
                                  diffuse_white_nits=1.0)
    assert backend == "colorvideovdp", "silently fell back to the proxy again"
    assert 0.0 <= value <= 10.0


def test_jod_separates_a_good_reconstruction_from_a_bad_one(tmp_path):
    ref = {"a": _frame(4.0, 1), "b": _frame(2.0, 2)}
    _tree(tmp_path, "ref", ref)
    _tree(tmp_path, "test", {k: v * 1.01 for k, v in ref.items()})
    _tree(tmp_path, "baseline", {k: v * 6.0 for k, v in ref.items()})
    good = bench.run_benchmark(tmp_path, nits_scale=203.0)
    bad = bench.run_benchmark(tmp_path, nits_scale=203.0, test_dir="baseline")
    assert good["cvvdp_backend"] == "colorvideovdp"
    assert good["cvvdp_jod_mean"] > bad["cvvdp_jod_mean"]
