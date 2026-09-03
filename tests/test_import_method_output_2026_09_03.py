"""Importing a third-party method's output into a `rudra bench` tree.

The paper's largest gap is that no published method is scored on our split.
`export_bench_pairs.py --write-sdr` writes the inputs those methods consume and
`training/import_method_output.py` brings their output back; this covers the
half that can go wrong silently.

Two things it must get right. First the exposure fit: published single-image
iTMO methods predict RELATIVE radiance, so a per-frame scalar has to be fitted
before scoring or the number measures their exposure guess. Second the
matching: a method that drops or renames frames must be REPORTED, never
silently paired against the wrong reference -- that would produce a plausible,
wrong table.

The white-noise case is deliberately absent. Median-ratio exposure fitting is
biased on pixel-scale noise (a known 37x offset recovers as ~14x) because the
method's frame is a resampled copy of ours and the valid-pixel mask correlates
with the reference but not with the resampled copy. Real frames are smooth at
the scale a 2x resample acts on, which is what is modelled here.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "training" / "import_method_output.py"

from rudra.delivery.exr import read_exr, write_exr  # noqa: E402

TRUE_K = 37.0          # the exposure offset the importer has to recover


def _smooth(rng, h=128, w=192):
    small = rng.random((8, 12, 3)).astype(np.float32)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    ramp = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    return np.clip(big * 10.0 + ramp * 6.0 + 0.05, 0.02, None)


@pytest.fixture
def tree(tmp_path):
    """A bench tree with ref/ and sdr/, plus a 'method' at half res and 1/37 exposure."""
    rng = np.random.default_rng(7)
    bench, theirs = tmp_path / "bench", tmp_path / "theirs"
    for scene, asset in (("sceneA", "f001"), ("sceneA", "f002"), ("sceneB", "f003")):
        ref = _smooth(rng)
        (bench / "ref" / scene).mkdir(parents=True, exist_ok=True)
        write_exr(bench / "ref" / scene / f"{asset}.exr",
                  np.ascontiguousarray(ref), half=True)

        sdr = np.clip(ref / 4.0, 0, 1) ** (1 / 2.2)     # clips the top ~2 stops
        (bench / "sdr" / scene).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(bench / "sdr" / scene / f"{asset}.png"),
                    (np.clip(sdr, 0, 1) * 255 + 0.5).astype(np.uint8)[..., ::-1])

        half = cv2.resize(ref / TRUE_K, (ref.shape[1] // 2, ref.shape[0] // 2),
                          interpolation=cv2.INTER_AREA)
        (theirs / scene).mkdir(parents=True, exist_ok=True)
        write_exr(theirs / scene / f"{asset}.exr",
                  np.ascontiguousarray(half), half=True)
    return bench, theirs


def _run(bench, theirs, name, *extra):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(bench), "--from", str(theirs),
         "--name", name, *extra],
        cwd=str(REPO), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((bench / f"import_{name}.json").read_text())


def test_recovers_the_exposure_offset(tree):
    bench, theirs = tree
    manifest = _run(bench, theirs, "faux")
    assert manifest["frames"] == 3
    assert not manifest["missing_from_method"] and not manifest["failed"]
    assert manifest["scale_median"] == pytest.approx(TRUE_K, rel=0.05)


def test_aligned_output_matches_the_reference(tree):
    bench, theirs = tree
    _run(bench, theirs, "faux")
    got = read_exr(bench / "faux" / "sceneA" / "f001.exr")[0]
    ref = read_exr(bench / "ref" / "sceneA" / "f001.exr")[0]
    # The floor is the 2x resample, not the fit.
    assert float(np.median(np.abs(got - ref) / np.maximum(ref, 1e-6))) < 0.10


def test_least_squares_agrees_with_median(tree):
    bench, theirs = tree
    manifest = _run(bench, theirs, "faux_ls", "--align", "ls")
    assert manifest["scale_median"] == pytest.approx(TRUE_K, rel=0.10)


def test_align_none_scales_by_exactly_one(tree):
    bench, theirs = tree
    manifest = _run(bench, theirs, "raw", "--align", "none")
    assert manifest["scale_median"] == 1.0


def test_resampling_is_recorded_per_frame(tree):
    bench, theirs = tree
    manifest = _run(bench, theirs, "faux")
    # Judging a method at a resolution it did not produce is a caveat on that
    # table row, so it has to survive into the manifest.
    assert len(manifest["resized"]) == 3


def test_a_dropped_frame_is_reported_not_mispaired(tree):
    bench, theirs = tree
    (theirs / "sceneB" / "f003.exr").unlink()
    manifest = _run(bench, theirs, "partial")
    assert manifest["frames"] == 2
    assert manifest["missing_from_method"] == ["sceneB/f003.exr"]


def test_manifest_paths_are_portable(tree):
    """Forward slashes on every OS.

    The manifest is the provenance for a table in a paper. Written on Windows
    with native separators it would not compare equal to the same manifest
    written on Linux, so "which frames did the method drop?" would depend on
    where you asked. This failed on Windows the first time it ran.
    """
    bench, theirs = tree
    (theirs / "sceneB" / "f003.exr").unlink()
    manifest = _run(bench, theirs, "portable")
    recorded = (manifest["missing_from_method"]
                + [r for r, _ in manifest["resized"]]
                + [r for r, _ in manifest["failed"]])
    assert recorded, "nothing was recorded, so nothing was checked"
    assert not any("\\" in r for r in recorded), recorded
    assert all("/" in r for r in recorded), recorded


def test_refuses_to_overwrite_the_reference(tree):
    bench, theirs = tree
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(bench), "--from", str(theirs),
         "--name", "ref"], cwd=str(REPO), capture_output=True, text=True)
    assert result.returncode != 0
    assert "overwrite" in (result.stdout + result.stderr).lower()
