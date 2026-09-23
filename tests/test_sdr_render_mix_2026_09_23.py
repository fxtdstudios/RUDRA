"""corpus_v4c: the SDR side is drawn from a family of curves, not one.

The shipped model lost to its own analytic baseline on SDR made with a curve
and codec it never saw (bench/oog, 23 Sep 2026: -0.41 dB PU21, 270 of 429
frames worse). These pin the render family and its bookkeeping.
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

from pipeline import sdr_render  # noqa: E402


def _scene(seed=0, shape=(96, 128)):
    rng = np.random.default_rng(seed)
    return (np.exp(rng.normal(0.0, 2.0, (*shape, 3))) * 0.3).astype(np.float32)


def test_aces_render_is_bit_exact_with_the_legacy_make_sdr():
    import training.prepare_training_data as ptd
    old = ptd.TONEMAP_EV_OFFSET
    try:
        for ev in (0.0, -1.0):
            ptd.TONEMAP_EV_OFFSET = ev
            lin = _scene(1)
            assert np.array_equal(sdr_render.render_aces(lin, ev), ptd.make_sdr(lin))
    finally:
        ptd.TONEMAP_EV_OFFSET = old


@pytest.mark.parametrize("name", sorted(sdr_render.CURVES))
def test_every_curve_is_monotone_and_bounded(name):
    x = np.linspace(0.0, 64.0, 4096, dtype=np.float32)
    y = sdr_render.CURVES[name](x, white=8.0, contrast=1.0)
    assert np.all(np.diff(y) >= -1e-6), name
    assert y.min() >= 0.0 and y.max() <= 1.0
    assert y[-1] > 0.95, f"{name} never reaches white"


def test_curves_actually_differ_at_diffuse_white():
    at_white = {n: float(f(np.float32(1.0), white=8.0, contrast=1.0))
                for n, f in sdr_render.CURVES.items()}
    assert max(at_white.values()) - min(at_white.values()) > 0.3, at_white


def test_draw_is_deterministic_and_covers_the_family():
    a = [sdr_render.draw_render(np.random.default_rng(i), codecs=[]) for i in range(400)]
    b = [sdr_render.draw_render(np.random.default_rng(i), codecs=[]) for i in range(400)]
    assert a == b
    assert {r["curve"] for r in a} == set(sdr_render.CURVES)
    assert all(-1.5 <= r["render_ev"] <= 1.5 for r in a)
    assert {r["codec"] for r in a} == {"none", "jpeg"}


def test_apply_render_is_uint8_and_codecs_change_pixels():
    lin = _scene(2)
    recipe = sdr_render.draw_render(np.random.default_rng(3), codec_probability=0.0)
    clean = sdr_render.apply_render(lin, recipe)
    assert clean.dtype == np.uint8 and clean.shape == lin.shape
    coded = sdr_render.apply_render(lin, {**recipe, "codec": "jpeg", "quality": 40})
    assert np.abs(clean.astype(int) - coded.astype(int)).mean() > 0.2


def test_prepare_pairs_mix_records_the_recipe_and_keeps_one_nominal_ev(tmp_path, monkeypatch):
    from pipeline import prepare_pairs as pp
    from training.sdr2hdr_dataset import corpus_ev_of
    from pipeline.build_manifests import build_image_manifest

    inventory = tmp_path / "inv.jsonl"
    rows = [{"path": str(tmp_path / f"scene{i}.exr"), "kind": "image",
             "encoding_guess": "linear", "scene_id": f"x::scene{i}",
             "is_sequence_member": False} for i in range(3)]
    inventory.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(pp, "load_scene_linear", lambda path, enc: _scene(hash(str(path)) % 97, (360, 640)))
    monkeypatch.setattr(sdr_render, "available_video_codecs", lambda: [])
    dst = tmp_path / "pairs"
    monkeypatch.setattr(sys, "argv", ["prepare_pairs.py", "--inventory", str(inventory),
                                      "--dst", str(dst), "--crops", "2", "--crop-size", "256",
                                      "--tonemap-ev", "0", "--sdr-render", "mix"])
    pp.main()
    index = [json.loads(l) for l in (dst / "pairs_index.jsonl").read_text().splitlines()]
    assert len(index) == 6
    assert all({"sdr_curve", "render_ev", "sdr_codec"} <= set(r) for r in index)
    assert all(r["tonemap_ev"] == 0.0 for r in index)
    meta = json.loads(Path(index[0]["metadata_path"]).read_text())
    assert meta["sdr_render"]["render_version"] == sdr_render.RENDER_VERSION
    assert json.loads((dst / "_ingest_config.json").read_text())["tonemap"] == "mix:v1"
    manifest, _ = build_image_manifest(index, 0.0, 0.0, 1, 0.0)
    assert corpus_ev_of("in-memory", manifest) == 0.0
    # an ACES ingest into the same directory is refused
    monkeypatch.setattr(sys, "argv", ["prepare_pairs.py", "--inventory", str(inventory),
                                      "--dst", str(dst), "--tonemap-ev", "0"])
    with pytest.raises(SystemExit):
        pp.main()
