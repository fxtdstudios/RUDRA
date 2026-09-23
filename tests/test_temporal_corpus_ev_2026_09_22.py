"""What the 22 Sep 2026 audit found, pinned so it stays fixed.

`sdr2hdr_temporal_v4` was started from an image model trained on
`G:\\corpus_v4` (old ingest, no sidecars, so the legacy -1 EV baseline) over
`corpus_v4b\\video_manifest_9f.jsonl` (rendered at 0 EV). Two faults let it
start: the video manifest never carried `tonemap_ev`, so `corpus_ev_of` fell
back to -1 EV and agreed with the wrong checkpoint by accident; and nothing
compared the image model's exposure with the corpus's at all.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.build_manifests import build_image_manifest, build_video_manifest  # noqa: E402
from training.sdr2hdr_dataset import corpus_ev_of  # noqa: E402


def _records(n_frames: int, ev: float) -> list[dict]:
    return [{"asset_id": f"take0_{i:04d}_c0", "scene_id": "sceneA", "is_video": True,
             "frame_index": i, "source_take": "take0",
             "sdr_path": f"sdr/{i}.png", "hdr_path": f"hdr/{i}.png",
             "metadata_path": f"meta/{i}.json", "tonemap_ev": ev,
             "sdr_clipped_fraction": 0.03}
            for i in range(n_frames)]


def test_video_manifest_carries_the_render_exposure(tmp_path):
    """A clip row says what exposure its frames were rendered at, so a temporal
    run reads the same corpus_ev as an image run on the same corpus."""
    rows, _ = build_image_manifest(_records(20, 0.0), 0.0, 0.0, 1, 0.0)
    out = tmp_path / "video.jsonl"
    build_video_manifest(rows, clip_length=9, stride=8, out_path=out)
    clips = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert clips, "the fixture should yield at least one clip"
    assert all(c["tonemap_ev"] == 0.0 for c in clips)
    assert all(c["metadata_path"] for c in clips)
    assert corpus_ev_of(out) == 0.0


def test_corpus_ev_of_reads_a_clip_rows_sidecars(tmp_path):
    """Video manifests built before this fix have no `tonemap_ev` but do carry
    `metadata_paths`; the first sidecar that exists speaks for the clip, so
    corpus_v4b's existing video manifest needs no rebuild."""
    side = tmp_path / "meta.json"
    side.write_text(json.dumps({"tonemap_ev": 0.0}), encoding="utf-8")
    manifest = tmp_path / "clips.jsonl"
    manifest.write_text(json.dumps({"clip_id": "x", "metadata_paths": [str(side), None]}) + "\n",
                        encoding="utf-8")
    assert corpus_ev_of(manifest) == 0.0


def test_temporal_training_refuses_an_image_model_at_another_exposure():
    """The refiner learns a residual over the image model's output; if that
    model inverts a different exposure than the clips were rendered at, the
    refiner spends itself undoing a constant stop and every eval reads as a
    gain. The trainer must stop, not warn."""
    src = (REPO / "training" / "train_sdr2hdr.py").read_text(encoding="utf-8")
    temporal = src.split("TemporalHDRRefiner(channels=args.temporal_channels)")[0]
    assert "image_model.corpus_ev" in temporal and "raise SystemExit" in temporal[-2000:], \
        "temporal mode must compare image_model.corpus_ev with the manifest's and exit"


def test_run_outputs_are_not_tracked():
    """570 MB of step checkpoints and 7 GB of bench frames were staged on
    22 Sep 2026 because `!checkpoints/` un-ignored every subdirectory. Shipped
    weights live at checkpoints/*.pt; run directories do not belong in git."""
    rules = [l.strip() for l in (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()]
    assert "checkpoints/*/" in rules
    assert "/bench/" in rules


def test_hold_out_scenes_pin_the_old_bench_to_test(tmp_path):
    """The paper's 429 frames are E:\\RUDRA_v3's test split. corpus_v4b was
    re-ingested from the same sources under a fresh split, so without this the
    acceptance step would score the new model on scenes it trained on. Ids are
    matched on their drive-independent tail, because scan_sources bakes the
    parent path into scene_id."""
    from pipeline.build_manifests import load_hold_out_scenes, scene_tail

    assert scene_tail("E:/source_hdr/HdM/Bar::Bar_0001") == scene_tail(
        r"G:\datasets\sources\hdm_hfr_2017\HdM\Bar::Bar_0001")

    old = tmp_path / "old_manifest.jsonl"
    old.write_text("".join(json.dumps(r) + "\n" for r in [
        {"scene_id": "E:/source_hdr/x/Bar::Bar", "split": "test"},
        {"scene_id": "E:/source_hdr/x/Fire::Fire", "split": "train"},
    ]), encoding="utf-8")
    hold = load_hold_out_scenes(old)
    assert hold == {scene_tail("x/Bar::Bar")}

    records = []
    for i in range(40):
        name = "Bar" if i < 10 else f"s{i}"
        records += [{"asset_id": f"{name}_{c}", "scene_id": f"G:/sources/x/{name}::{name}",
                     "is_video": False, "tonemap_ev": 0.0, "sdr_clipped_fraction": 0.03}
                    for c in range(3)]
    rows, stats = build_image_manifest(records, 0.1, 0.1, 1, 0.0, 0.0, hold)
    assert stats["held_out_scenes"] == 1
    assert {r["split"] for r in rows if "Bar" in r["scene_id"]} == {"test"}
    # A list that matches nothing is a mistake, not a no-op.
    _, stats = build_image_manifest(records, 0.1, 0.1, 1, 0.0, 0.0, {"nope::nope"})
    assert any("hold-out" in p for p in stats["problems"])


def test_holdout_overlap_reports_old_bench_scenes_in_train(tmp_path):
    from pipeline.check_holdout_overlap import overlap
    from pipeline.build_manifests import scene_tail
    old = tmp_path / "old.jsonl"
    old.write_text("".join(json.dumps(r) + "\n" for r in [
        {"scene_id": "E:/source_hdr/x/Bar::Bar", "split": "test"},
        {"scene_id": "E:/source_hdr/x/Sky::Sky", "split": "test"},
        {"scene_id": "E:/source_hdr/x/Fire::Fire", "split": "train"},
    ]), encoding="utf-8")
    new = tmp_path / "new.jsonl"
    new.write_text("".join(json.dumps(r) + "\n" for r in [
        {"scene_id": "G:/sources/x/Bar::Bar", "split": "train"},
        {"scene_id": "G:/sources/x/Bar::Bar", "split": "train"},
        {"scene_id": "G:/sources/x/Sky::Sky", "split": "test"},
        {"scene_id": "G:/sources/x/Fire::Fire", "split": "test"},
    ]), encoding="utf-8")
    result = overlap(old, new)
    assert result["old_test_scenes"] == 2 and result["found_here"] == 2
    assert result["in_train"] == [scene_tail("x/Bar::Bar")]
    assert result["records_by_split"] == {"train": 2, "test": 1}


def test_a_float_container_under_a_pq_dataset_name_is_still_linear():
    """corpus_v4b dropped 1,799 Sparks pairs for peaking below 1 nit: the ACES
    EXRs sit under netflix_sparks/ and "netflix" is a PQ keyword, so scene-linear
    values were decoded as PQ codes. Dataset names say what a TIFF is; a float
    container says what it is itself."""
    from pipeline.scan_sources import guess_encoding
    assert guess_encoding(Path("G:/datasets/sources/netflix_sparks/ACES/sparks_0001.exr"))[0] == "linear"
    assert guess_encoding(Path("G:/datasets/sources/netflix_chimera/tif/chimera_0001.tif"))[0] == "pq"
    assert guess_encoding(Path("G:/x/alexa35_logc4/plate_0001.exr"))[0] == "logc4"
    assert guess_encoding(Path("G:/x/hdm-hfr/take_0001.tif"))[0] == "pq"


def test_swscale_is_given_the_matrix_name_it_knows():
    """FFmpeg 7.x parses scale's out_color_matrix as a named constant and has no
    "bt2020nc" (that is the tag's name); every encode failed "Invalid argument"
    on the Windows box on 23 Sep 2026 while 4.4 in CI let it through."""
    from rudra.delivery import video
    assert video.SWS_MATRIX == "bt2020" and video.MATRIX == "bt2020nc"
    src = (REPO / "rudra" / "delivery" / "video.py").read_text(encoding="utf-8")
    assert "out_color_matrix={SWS_MATRIX}" in src and "out_color_matrix={MATRIX}" not in src
