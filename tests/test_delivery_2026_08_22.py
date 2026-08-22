"""Regression tests for the 2026-08-22 delivery layer (torch-free).

Covers: colorspace matrices, EXR round-trip incl. ACES container attributes,
dynamic metadata (MaxRGB convention, PQ codes, shot detection, sidecar
schemas), grade controls (order, qualifiers, hue preservation), PU21-PSNR,
the benchmark harness end-to-end, the CLI, and package import without torch.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.delivery import aces, bench, colorspace, controls, exr, metadata  # noqa: E402
from rudra.hdr10 import pq_eotf, pq_oetf  # noqa: E402


# ---------------------------------------------------------------- colorspace

def test_white_maps_to_white():
    for src, dst in [("rec709", "ap0"), ("rec2020", "ap0"), ("rec2020", "rec709"),
                     ("ap1", "ap0"), ("p3d65", "rec2020")]:
        out = colorspace.convert(np.ones((1, 1, 3)), src, dst)
        assert np.allclose(out, 1.0, atol=2e-4), (src, dst, out)


def test_bt709_to_ap0_matches_aces_reference():
    # Reference Bradford-adapted BT.709 -> AP0 matrix (ACES TB-2014-004 lineage).
    expected = np.array([
        [0.4397, 0.3830, 0.1774],
        [0.0898, 0.8134, 0.0968],
        [0.0175, 0.1115, 0.8710],
    ])
    got = colorspace.rgb_to_rgb_matrix("rec709", "ap0")
    assert np.allclose(got, expected, atol=2e-3), got


def test_round_trip_matrix_is_identity():
    m = colorspace.rgb_to_rgb_matrix("rec2020", "ap0") @ colorspace.rgb_to_rgb_matrix("ap0", "rec2020")
    assert np.allclose(m, np.eye(3), atol=1e-10)


def test_convert_rejects_bad_shape():
    with pytest.raises(ValueError):
        colorspace.convert(np.ones((4, 4)), "rec709", "ap0")


# ----------------------------------------------------------------------- exr

def test_exr_round_trip_half(tmp_path):
    rng = np.random.default_rng(7)
    img = (rng.random((17, 23, 3)) * 100.0).astype(np.float32)
    path = exr.write_exr(tmp_path / "t.exr", img, half=True)
    back, attrs = exr.read_exr(path)
    assert back.shape == img.shape
    assert attrs["channel_order"] == ["R", "G", "B"]
    # half precision: relative error bounded by 2^-11
    assert np.allclose(back, img, rtol=1e-3, atol=1e-2)


def test_exr_round_trip_float_exact(tmp_path):
    rng = np.random.default_rng(3)
    img = (rng.random((5, 9, 3)) * 1e6).astype(np.float32)  # log2_extended range
    path = exr.write_exr(tmp_path / "f.exr", img, half=False)
    back, _ = exr.read_exr(path)
    assert np.array_equal(back, img)


def test_exr_aces_container_attributes(tmp_path):
    img = np.full((4, 4, 3), 0.18, dtype=np.float32)
    path = aces.write_aces_exr(img, tmp_path / "a.exr", source_space="rec2020")
    back, attrs = exr.read_exr(path)
    assert attrs["chromaticities"] == pytest.approx(colorspace.AP0_CHROMATICITIES, abs=1e-6)
    assert attrs["acesImageContainerFlag"] == "1"
    # 0.18 grey is exposure-invariant under the matrix (white row-sums to 1)
    assert np.allclose(back, 0.18, atol=2e-3)


def test_exr_reader_refuses_compressed(tmp_path):
    img = np.ones((4, 4, 3), dtype=np.float32)
    path = exr.write_exr(tmp_path / "c.exr", img)
    raw = bytearray(path.read_bytes())
    idx = raw.index(b"compression\x00compression\x00")
    raw[idx + len(b"compression\x00compression\x00") + 4] = 3  # claim PIZ
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="compressed"):
        exr.read_exr(path)


# ------------------------------------------------------------------ metadata

def _frame(peak=1000.0, base=50.0, h=16, w=16):
    img = np.full((h, w, 3), base, dtype=np.float64)
    img[0, 0] = (peak, base, base)  # saturated red specular
    return img


def test_maxcll_uses_maxrgb_not_luma():
    stats = [metadata.analyze_frame(_frame(peak=1000.0), 0)]
    max_cll, _ = metadata.maxcll_maxfall(stats)
    assert max_cll == 1000  # luma-weighted would report ~263


def test_pq12_codes_match_hdr10_curve():
    stats = [metadata.analyze_frame(_frame(), 0)]
    shot = metadata.l1_per_shot(stats, [(0, 1)])[0]
    decoded = pq_eotf(shot["max_pq"] / 4095.0)
    assert decoded == pytest.approx(1000.0, rel=0.02)


def test_shot_detection_finds_the_cut():
    dark = [metadata.analyze_frame(np.full((8, 8, 3), 1.0), i) for i in range(5)]
    bright = [metadata.analyze_frame(np.full((8, 8, 3), 4000.0), 5 + i) for i in range(5)]
    shots = metadata.detect_shots(dark + bright)
    assert shots == [(0, 5), (5, 5)]
    stable = metadata.detect_shots(dark)
    assert stable == [(0, 5)]


def test_sidecars_written_and_consistent(tmp_path):
    stats = [metadata.analyze_frame(_frame(peak=800.0 + 10 * i), i) for i in range(4)]
    paths = metadata.write_all_sidecars(stats, tmp_path / "clip")
    dovi = json.loads(paths["dovi"].read_text())
    assert dovi["profile"] == "8.1" and dovi["length"] == 4
    assert dovi["level6"]["max_content_light_level"] == 830
    assert all(0 <= s["metadata_blocks"][0]["max_pq"] <= 4095 for s in dovi["shots"])
    hdr10p = json.loads(paths["hdr10plus"].read_text())
    assert hdr10p["SceneInfo"][0]["LuminanceParameters"]["MaxSCL"][0] == pytest.approx(830.0)
    sidecar = json.loads(paths["rudra"].read_text())
    assert sidecar["max_cll_nits"] == 830 and len(sidecar["frames"]) == 4


def test_analyze_frame_sanitizes_nonfinite():
    img = _frame()
    img[1, 1] = (np.nan, np.inf, -5.0)
    stats = metadata.analyze_frame(img, 0)
    assert np.isfinite(stats.max_nits) and stats.min_nits >= 0.0


# ------------------------------------------------------------------ controls

def test_qualifier_mask_selects_band():
    img = np.zeros((1, 3, 3))
    img[0, 0] = 10.0    # below band
    img[0, 1] = 500.0   # inside band
    img[0, 2] = 5000.0  # above band
    mask = controls.qualifier_mask(img, 100.0, 1000.0, softness_stops=0.5)
    assert mask[0, 1] == pytest.approx(1.0, abs=1e-6)
    assert mask[0, 0] < 0.05 and mask[0, 2] < 0.05


def test_apply_grade_exposure_and_region():
    img = np.full((4, 4, 3), 100.0)
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[:2] = 1.0
    grade = controls.GradeControls(
        exposure_ev=1.0, peak_nits=4000.0,
        regions=[controls.RegionEV(mask=mask, ev=1.0, label="top")])
    out = controls.apply_grade(img, grade)
    assert np.allclose(out[2:], 200.0, rtol=1e-4)   # +1 EV global
    assert np.allclose(out[:2], 400.0, rtol=1e-4)   # +1 EV more in region


def test_grade_shoulder_respects_peak_and_hue():
    img = np.zeros((1, 1, 3))
    img[0, 0] = (8000.0, 4000.0, 2000.0)
    out = controls.apply_grade(img, controls.GradeControls(peak_nits=1000.0))
    assert out.max() <= 1000.0 + 1e-3
    ratios_in = img[0, 0] / img[0, 0].max()
    ratios_out = out[0, 0] / out[0, 0].max()
    assert np.allclose(ratios_in, ratios_out, atol=1e-3)  # hue-preserving


def test_grade_describe_is_json_serializable():
    grade = controls.GradeControls(regions=[
        controls.RegionEV(mask=np.ones((2, 2)), ev=0.5, label="x")])
    json.dumps(grade.describe())


def test_itm_strength_map_shape_and_clamp():
    mask = np.ones((8, 8))
    strength = controls.itm_strength_map((8, 8), base_strength=1.0,
                                         regions=[controls.RegionEV(mask, ev=4.0)])
    assert strength.shape == (1, 1, 8, 8)
    assert strength.max() <= 2.0  # clamped


# --------------------------------------------------------------------- bench

def test_pu21_monotonic_and_range():
    y = np.array([0.005, 0.1, 1.0, 100.0, 1000.0, 10000.0])
    v = bench.pu21_encode(y)
    assert np.all(np.diff(v) > 0)
    assert v[0] == pytest.approx(0.0, abs=0.5)


def test_pu_psnr_identical_is_inf_and_noise_is_finite():
    rng = np.random.default_rng(0)
    ref = rng.random((16, 16, 3)) * 1000.0
    assert bench.pu_psnr(ref, ref) == float("inf")
    noisy = ref * 1.02
    db = bench.pu_psnr(noisy, ref)
    assert 20.0 < db < 60.0


def test_run_benchmark_end_to_end(tmp_path):
    rng = np.random.default_rng(1)
    for clip in ("a", "b"):
        (tmp_path / "ref" / clip).mkdir(parents=True)
        (tmp_path / "test" / clip).mkdir(parents=True)
        for i in range(2):
            ref = (rng.random((8, 8, 3)) * 500.0).astype(np.float32)
            np.save(tmp_path / "ref" / clip / f"{i}.npy", ref)
            np.save(tmp_path / "test" / clip / f"{i}.npy", ref * 1.05)
    summary = bench.run_benchmark(tmp_path, output=tmp_path / "results.json")
    assert summary["pairs"] == 4
    assert summary["pu_psnr_db_mean"] > 20.0
    assert set(summary["pu_psnr_db_per_clip"]) == {"a", "b"}
    assert (tmp_path / "results.json").exists() and (tmp_path / "results.csv").exists()


def test_run_benchmark_rejects_shape_mismatch(tmp_path):
    (tmp_path / "ref").mkdir(); (tmp_path / "test").mkdir()
    np.save(tmp_path / "ref" / "x.npy", np.ones((4, 4, 3)))
    np.save(tmp_path / "test" / "x.npy", np.ones((5, 5, 3)))
    with pytest.raises(ValueError, match="shape mismatch"):
        bench.run_benchmark(tmp_path)


# ---------------------------------------------------------------- aces/ocio

def test_ocio_config_contains_exact_matrix(tmp_path):
    path = aces.generate_ocio_config(tmp_path / "cfg.ocio")
    text = path.read_text()
    assert "ocio_profile_version: 2" in text
    assert "RUDRA Scene Linear (Rec.2020)" in text
    m = colorspace.rgb_to_rgb_matrix("rec2020", "ap0")
    assert f"{m[0, 0]:.10f}" in text  # matrix embedded verbatim


# ----------------------------------------------------------------------- cli

def _run_cli(*argv):
    from rudra.delivery.cli import main
    return main(list(argv))


def test_cli_metadata_and_grade(tmp_path, capsys):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(3):
        np.save(frames_dir / f"{i:04d}.npy", _frame(peak=900.0 + i))
    assert _run_cli("metadata", str(frames_dir), "--nits-scale", "1",
                    "--output", str(tmp_path / "out" / "clip")) == 0
    assert (tmp_path / "out" / "clip_dovi_generate.json").exists()

    assert _run_cli("grade", str(frames_dir / "0000.npy"), "--nits-scale", "1",
                    "--output", str(tmp_path / "graded"), "--ev", "0.5",
                    "--region", "400:2000:1.0:0.5") == 0
    graded = sorted((tmp_path / "graded").glob("*.exr"))
    assert graded, "graded EXR not written"
    img, attrs = exr.read_exr(graded[0])
    assert "rudra:grade" in attrs


def test_cli_bench(tmp_path, capsys):
    (tmp_path / "ref").mkdir(); (tmp_path / "test").mkdir()
    ref = np.full((4, 4, 3), 100.0)
    np.save(tmp_path / "ref" / "f.npy", ref)
    np.save(tmp_path / "test" / "f.npy", ref * 1.01)
    assert _run_cli("bench", str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "pu_psnr_db_mean" in out


# ------------------------------------------------------------------- package

def test_package_importable_without_torch():
    """rudra.delivery must never require torch; rudra root must degrade."""
    import rudra
    if not rudra.TORCH_AVAILABLE:
        with pytest.raises(ImportError, match="delivery"):
            _ = rudra.SDR2HDRNet
    for module in (colorspace, exr, aces, metadata, controls, bench):
        assert "torch" not in getattr(module, "__dict__", {})


def test_cli_help_runs_as_subprocess():
    result = subprocess.run([sys.executable, "-m", "rudra.delivery.cli", "--help"],
                            capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0 and "rudra" in result.stdout
