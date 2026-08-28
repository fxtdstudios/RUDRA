"""Guards on what actually reaches the trainer, added 25 Aug 2026.

Three defects the 24 Aug corpus carried, none of which raised an error:

  * 4,071 pairs (21%) came from ``08_Research/data/hdr`` -- ALREADY NORMALISED
    August output, re-ingested as if it were scene-linear source, so every
    frame landed ~5,000x too dark and each became its own single-frame "scene".
  * two Poly Haven crops carried NaN pixels; NaN encodes to an undefined uint16
    code that decodes back as ordinary finite radiance.
  * whole-scene holdout plus four huge video scenes made val 95.6% Bar-Scene:
    every held-out number described one bar interior.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.build_manifests import (  # noqa: E402
    build_image_manifest, cap_scene_share, load_records,
)


def _index(tmp_path: Path, records: list[dict]) -> Path:
    pairs = tmp_path / "pairs"
    pairs.mkdir(parents=True, exist_ok=True)
    with (pairs / "pairs_index.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return pairs


def _record(asset: str, scene: str, peak: float, *, video=False, frame=None) -> dict:
    return {"asset_id": asset, "scene_id": scene, "peak_nits": peak,
            "is_video": video, "frame_index": frame, "source_take": "t"}


def test_dark_and_non_finite_records_never_reach_the_manifest(tmp_path):
    pairs = _index(tmp_path, [
        _record("good", "/src/PolyHaven::sunset", 4000.0),
        _record("derived", "/data/hdr::tif_0000000", 0.036),   # the 21% defect
        _record("nan", "/src/PolyHaven::venice", float("nan")),
    ])

    kept = load_records(pairs, min_peak_nits=1.0)

    assert [r["asset_id"] for r in kept] == ["good"]


def test_min_peak_nits_zero_keeps_everything_dark(tmp_path):
    pairs = _index(tmp_path, [_record("dim", "/src::a", 0.2),
                              _record("bright", "/src::b", 900.0)])

    kept = load_records(pairs, min_peak_nits=0.0)

    assert {r["asset_id"] for r in kept} == {"dim", "bright"}


def test_one_scene_cannot_own_an_eval_split():
    rows = ([_record(f"v{i}", "bar", 500.0, video=True, frame=i) for i in range(1000)]
            + [_record(f"s{i}", f"still{i}", 500.0) for i in range(100)])

    capped, trimmed = cap_scene_share(rows, 0.35)

    counts = {}
    for row in capped:
        counts[row["scene_id"]] = counts.get(row["scene_id"], 0) + 1
    assert max(counts.values()) / len(capped) <= 0.35 + 1e-9
    assert trimmed and trimmed[0][0] == "bar"
    assert counts["still1"] == 1, "other scenes must be untouched"


def test_thinning_keeps_a_uniform_frame_stride():
    """Clips need consecutive frames at one step; random sampling would kill them."""
    rows = ([_record(f"v{i}", "bar", 500.0, video=True, frame=i) for i in range(600)]
            + [_record(f"s{i}", f"still{i}", 500.0) for i in range(100)])

    capped, _ = cap_scene_share(rows, 0.35)

    frames = sorted(r["frame_index"] for r in capped if r["scene_id"] == "bar")
    steps = {b - a for a, b in zip(frames, frames[1:])}
    assert len(steps) == 1, f"non-uniform frame steps after thinning: {sorted(steps)[:5]}"


def test_training_split_is_never_thinned():
    """Its sampler is scene-balanced, so record counts do not set influence."""
    records = ([_record(f"v{i}", "bar", 500.0, video=True, frame=i) for i in range(900)]
               + [_record(f"s{i}", f"still{i}", 500.0) for i in range(60)])

    rows, stats = build_image_manifest(records, 0.10, 0.10, seed=1,
                                       min_video_share=0.0, max_eval_scene_share=0.35)

    train_bar = sum(1 for r in rows if r["split"] == "train" and r["scene_id"] == "bar")
    assert train_bar == 900, "the whole video scene landed in train and must survive intact"
    for split in ("val", "test"):
        subset = [r for r in rows if r["split"] == split]
        if subset:
            top = max(sum(1 for r in subset if r["scene_id"] == s)
                      for s in {r["scene_id"] for r in subset})
            assert top / len(subset) <= 0.35 + 1e-9


def test_scan_skips_rudras_own_output_directories(tmp_path):
    """Derived pairs must never be re-ingested as source footage."""
    from pipeline.scan_sources import walk

    (tmp_path / "Source_HDR" / "PolyHaven").mkdir(parents=True)
    (tmp_path / "data" / "hdr").mkdir(parents=True)
    (tmp_path / "Source_HDR" / "PolyHaven" / "sunset.exr").write_bytes(b"")
    (tmp_path / "data" / "hdr" / "frame_0000.png").write_bytes(b"")

    found = walk(tmp_path, follow_links=False)
    assert [p.name for p in found] == ["sunset.exr"]

    everything = walk(tmp_path, follow_links=False, include_derived=True)
    assert len(everything) == 2


def test_hdm2014_framings_of_one_setup_collapse_to_one_scene():
    """cars_closeshot / _fullshot / _longshot are one location, one lighting setup.

    Treating them as three scenes would let whole-scene holdout put two of them
    in train and one in val and still report "no scenes straddle a boundary".
    """
    from pipeline.build_manifests import normalize_scene_id

    groups = {
        "cars": ["cars_closeshot", "cars_fullshot", "cars_longshot"],
        "fishing": ["fishing_closeshot", "fishing_longshot"],
        "poker": ["poker_fullshot", "poker_travelling_slowmotion"],
        "showgirl": ["showgirl_01", "showgirl_02"],
    }
    for expected, folders in groups.items():
        collapsed = {normalize_scene_id(f"E:/src/{f}::{f}") for f in folders}
        assert collapsed == {f"E:/src/{expected}::{expected}"}, collapsed

    # Distinct subjects must NOT be merged, however similar the names look.
    distinct = ["beerfest_lightshow", "bistro", "carousel_fireworks", "fireplace",
                "smith_hammering"]
    ids = {normalize_scene_id(f"E:/src/{f}::{f}") for f in distinct}
    assert len(ids) == len(distinct)


def test_hdm_tiffs_are_classified_pq_not_unknown():
    """HdM-HDR-2014 carries no transfer function in its paths or containers.

    Measured 27 Aug 2026: every frame of every sequence tops out at code
    59,150, which is exactly the ST-2084 code for 4,000 nits -- the grading
    ceiling the dataset documents. Classified UNKNOWN, all 11,007 frames are
    skipped in silence; forced through with --include-unknown-encoding they
    decode as sRGB, which is the data/hdr poisoning all over again.
    """
    from pipeline.scan_sources import guess_encoding

    stuttgart = Path("E:/source_hdr/Stuttgart_HDR_2014/carousel_fireworks/f_094903.tif")
    assert guess_encoding(stuttgart)[0] == "pq"
    assert guess_encoding(Path("E:/x/HdM-HFR-2017_Color-Graded/Bar/A004C006.tif"))[0] == "pq"
    # An explicit declaration outranks every heuristic.
    assert guess_encoding(stuttgart, "linear") == ("linear", "declared:cli")


def test_unknown_encoding_is_refused_not_silently_treated_as_srgb():
    import numpy as np
    import pytest as _pytest

    from training.prepare_training_data import to_scene_linear

    with _pytest.raises(ValueError, match="unknown source encoding"):
        to_scene_linear(np.zeros((2, 2, 3), dtype=np.float32), "UNKNOWN")


def test_pq_code_59150_decodes_to_the_documented_4000_nit_ceiling():
    import numpy as np

    from training.prepare_training_data import to_scene_linear

    linear = to_scene_linear(np.full((1, 1, 3), 59150 / 65535.0, dtype=np.float32), "pq")
    assert abs(float(linear.max()) * 203.0 - 4000.0) < 5.0


def test_a_scene_barely_over_the_cap_is_not_halved():
    """The 27 Aug 2026 regression, in miniature.

    Thinning strides over whole frames, so it lands where the stride lands and
    can leave a scene a hair OVER the cap. Re-entering to shave that remainder
    has only stride 2 available, which halves the scene: test's fireplace went
    1,392 -> 156 -> 78 records and fell under --min-video-share, failing a
    corpus that was fine.
    """
    from pipeline.build_manifests import cap_scene_share

    rows = ([_record(f"v{i}", "fireplace", 500.0, video=True, frame=i // 3)
             for i in range(1392)]
            + [_record(f"s{i}", f"still{i}", 500.0) for i in range(288)])

    capped, trimmed = cap_scene_share(rows, 0.35)

    kept = sum(1 for r in capped if r["scene_id"] == "fireplace")
    share = kept / len(capped)
    assert share <= 0.35 + 1e-9, share
    # Under the cap, but not by a landslide: halving would land near 0.21.
    assert share > 0.25, f"over-thinned to {share:.1%} -- the halving bug is back"
    assert len(trimmed) == 1, "a scene must be thinned at most once"


def test_thinning_never_returns_more_than_asked_for():
    from pipeline.build_manifests import _thin_scene

    rows = [_record(f"v{i}", "s", 500.0, video=True, frame=i // 3) for i in range(1392)]
    for target in (1391, 700, 155, 113, 40, 7):
        kept = _thin_scene(rows, target)
        assert len(kept) <= target, f"target {target} -> {len(kept)}"
        frames = sorted(r["frame_index"] for r in kept)
        steps = {b - a for a, b in zip(frames, frames[1:]) if b != a}
        assert len(steps) <= 1, f"non-uniform frame step at target {target}: {steps}"
