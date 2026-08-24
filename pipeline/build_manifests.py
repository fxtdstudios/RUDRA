"""Build scene-safe train/val/test manifests with guaranteed video coverage.

What went wrong in the August 2026 corpus, and what this fixes:

  * val (96) and test (88) contained ZERO video frames while 92% of training
    was video. The held-out metric described HDRI panoramas, not footage.
    -> ``--min-video-share`` forces moving content into every split, holding out
       WHOLE scenes.
  * the temporal dataset computed its own split as
    ``n_val = max(1, round(n_scenes * 0.1))``, which at n=2 gives a one-clip
    training set.
    -> the video manifest now carries an explicit per-clip ``split`` field, and
       the loader is told to honour it (see patch_video_split.py).
  * nothing checked that a scene stayed on one side of the boundary.
    -> asserted here, and again in verify_dataset.py.

    python pipeline/build_manifests.py --pairs-dir hdrdata/pairs_v3 \\
        --out-dir hdrdata --clip-length 9
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# A temporal refiner trained on fewer than this many independent scenes learns
# the clip, not the task. The August run had an effective training set of ONE.
MIN_TEMPORAL_TRAIN_SCENES = 6
MIN_TEMPORAL_HELDOUT_SCENES = 2

# Production footage arrives as scene/shot/camera-take hierarchies. Splitting
# by take treats nine shots of the same bar as nine "scenes" -- the same set,
# lighting and actors then sit on both sides of the split boundary (leakage,
# found 23 Aug 2026: HdM-HFR-2017 produced 13 "scenes" from 3 physical ones).
_TAKE_SEG = re.compile(r"^[A-Z]\d{3}C\d{3}_\d{6}(_\w+)?$")   # A004C006_170212_R3W4
_SHOT_SUFFIX = re.compile(r"[_-]Shot[_-]?\d+$", re.IGNORECASE)


def normalize_scene_id(scene_id: str) -> str:
    """Collapse shot/take structure so a scene id names a PHYSICAL scene."""
    path_part, _, stem = scene_id.partition("::")
    segments = [s for s in path_part.split("/") if not _TAKE_SEG.match(s)]
    if segments:
        segments[-1] = _SHOT_SUFFIX.sub("", segments[-1])
    if stem and not _TAKE_SEG.match(stem):
        stem = _SHOT_SUFFIX.sub("", stem)
        return "/".join(segments) + "::" + stem
    return "/".join(segments)


def load_records(pairs_dir: Path) -> list[dict]:
    index = pairs_dir / "pairs_index.jsonl"
    if not index.exists():
        raise SystemExit(f"error: {index} not found -- run prepare_pairs.py first")
    records = [json.loads(line) for line in index.open(encoding="utf-8") if line.strip()]
    raw = {r["scene_id"] for r in records}
    for record in records:
        record["scene_id"] = normalize_scene_id(record["scene_id"])
    merged = {r["scene_id"] for r in records}
    if len(merged) != len(raw):
        print(f"  scene normalization: {len(raw)} raw scene ids -> "
              f"{len(merged)} physical scenes (shot/take structure collapsed)")
    return records


def split_scenes(scenes: list[str], val_frac: float, test_frac: float,
                 seed: int) -> dict[str, str]:
    """Assign whole scenes to splits. Deterministic for a given seed."""
    ordered = sorted(scenes)
    random.Random(seed).shuffle(ordered)
    n = len(ordered)
    n_val = max(1, round(n * val_frac)) if n > 2 else 0
    n_test = max(1, round(n * test_frac)) if n > 2 else 0
    if n_val + n_test >= n:
        n_val = max(0, n // 3)
        n_test = max(0, n // 3)
    assignment = {}
    for i, scene in enumerate(ordered):
        if i < n_val:
            assignment[scene] = "val"
        elif i < n_val + n_test:
            assignment[scene] = "test"
        else:
            assignment[scene] = "train"
    return assignment


def build_image_manifest(records: list[dict], val_frac: float, test_frac: float,
                         seed: int, min_video_share: float) -> tuple[list[dict], dict]:
    by_kind: dict[bool, set[str]] = defaultdict(set)
    for record in records:
        by_kind[bool(record.get("is_video", False))].add(record["scene_id"])

    # Split each population independently so both land in every split. This is
    # the single change that makes the held-out numbers mean something.
    assignment: dict[str, str] = {}
    for is_video, scenes in by_kind.items():
        offset = 1 if is_video else 0
        assignment.update(split_scenes(sorted(scenes), val_frac, test_frac, seed + offset))

    out = []
    for record in records:
        row = dict(record)
        row["split"] = assignment[record["scene_id"]]
        out.append(row)

    stats = {}
    for split in ("train", "val", "test"):
        rows = [r for r in out if r["split"] == split]
        video = [r for r in rows if r.get("is_video")]
        stats[split] = {
            "records": len(rows),
            "scenes": len({r["scene_id"] for r in rows}),
            "video_records": len(video),
            "video_scenes": len({r["scene_id"] for r in video}),
            "video_share": round(len(video) / max(len(rows), 1), 4),
        }

    problems = []
    has_video = bool(by_kind[True])
    for split in ("val", "test"):
        if has_video and stats[split]["video_scenes"] == 0:
            problems.append(f"{split} contains no video scenes -- this is the August 2026 defect")
        if has_video and stats[split]["video_share"] < min_video_share:
            problems.append(
                f"{split} video share {stats[split]['video_share']:.2%} is below "
                f"--min-video-share {min_video_share:.0%}"
            )
    straddle = defaultdict(set)
    for row in out:
        straddle[row["scene_id"]].add(row["split"])
    leaked = [s for s, splits in straddle.items() if len(splits) > 1]
    if leaked:
        problems.append(f"{len(leaked)} scenes straddle a split boundary: {leaked[:5]}")

    stats["problems"] = problems
    return out, stats


def build_video_manifest(image_rows: list[dict], clip_length: int, stride: int,
                         out_path: Path) -> dict:
    """Consecutive clips, each carrying the split its scene was assigned."""
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in image_rows:
        if row.get("is_video") and row.get("frame_index") is not None:
            by_scene[row["scene_id"]].append(row)

    clips = []
    for scene, rows in sorted(by_scene.items()):
        # Multi-crop ingest emits several records per frame (…_c0/_c1/_c2) with
        # the same frame_index; keeping them all made every window non-
        # consecutive and produced 0 clips (23 Aug 2026). One record per frame:
        # prefer the _c0 crop so a clip is spatially consistent when the ingest
        # used scene-seeded crop origins.
        per_frame: dict[tuple, dict] = {}
        for r in rows:
            key = (r.get("source_take") or "", r["frame_index"])
            best = per_frame.get(key)
            if best is None or (str(r["asset_id"]).endswith("_c0")
                                and not str(best["asset_id"]).endswith("_c0")):
                per_frame[key] = r
        rows = sorted(per_frame.values(), key=lambda r: (r.get("source_take") or "", r["frame_index"]))
        for start in range(0, len(rows) - clip_length + 1, stride):
            window = rows[start:start + clip_length]
            if len({w.get("source_take") or "" for w in window}) > 1:
                continue  # never span a cut between camera takes
            frames = [w["frame_index"] for w in window]
            steps = {b - a for a, b in zip(frames, frames[1:])}
            if len(steps) != 1 or next(iter(steps)) <= 0:
                continue  # a gap or irregular stride -- skip rather than fabricate
            step = next(iter(steps))
            clips.append({
                "clip_id": f"{scene}:{frames[0]}-{frames[-1]}",
                "scene_id": scene,
                "split": window[0]["split"],
                "frame_numbers": frames,
                "frame_step": step,
                "sdr_frames": [w["sdr_path"] for w in window],
                "hdr_frames": [w["hdr_path"] for w in window],
                "metadata_paths": [w.get("metadata_path") for w in window],
            })

    with out_path.open("w", encoding="utf-8") as handle:
        for clip in clips:
            handle.write(json.dumps(clip) + "\n")

    per_split = defaultdict(lambda: {"clips": 0, "scenes": set()})
    for clip in clips:
        per_split[clip["split"]]["clips"] += 1
        per_split[clip["split"]]["scenes"].add(clip["scene_id"])
    stats = {k: {"clips": v["clips"], "scenes": len(v["scenes"])} for k, v in per_split.items()}

    train_scenes = stats.get("train", {}).get("scenes", 0)
    heldout = stats.get("val", {}).get("scenes", 0) + stats.get("test", {}).get("scenes", 0)
    problems = []
    if train_scenes < MIN_TEMPORAL_TRAIN_SCENES:
        problems.append(
            f"only {train_scenes} independent training scenes (need >= {MIN_TEMPORAL_TRAIN_SCENES}). "
            "Temporal training on this will diverge, exactly as the August run did."
        )
    if heldout < MIN_TEMPORAL_HELDOUT_SCENES:
        problems.append(f"only {heldout} held-out scenes (need >= {MIN_TEMPORAL_HELDOUT_SCENES})")
    stats["problems"] = problems
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("hdrdata"))
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--min-video-share", type=float, default=0.25,
                        help="Minimum share of val/test records that must be video frames")
    parser.add_argument("--clip-length", type=int, default=9)
    parser.add_argument("--clip-stride", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--allow-problems", action="store_true",
                        help="Write manifests even when guards fail (you must say why)")
    args = parser.parse_args()

    records = load_records(args.pairs_dir)
    rows, image_stats = build_image_manifest(
        records, args.val_frac, args.test_frac, args.seed, args.min_video_share)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.out_dir / "sdr_hdr_manifest.jsonl"
    video_path = args.out_dir / f"video_manifest_{args.clip_length}f.jsonl"

    with image_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    video_stats = build_video_manifest(rows, args.clip_length, args.clip_stride, video_path)

    print("=" * 66)
    print("IMAGE MANIFEST")
    print("=" * 66)
    for split in ("train", "val", "test"):
        s = image_stats[split]
        print(f"  {split:<6} {s['records']:>7,} records  {s['scenes']:>5,} scenes  "
              f"video {s['video_records']:>7,} ({s['video_share']:.1%}) in {s['video_scenes']} scenes")
    print("\nVIDEO MANIFEST")
    for split in ("train", "val", "test"):
        s = video_stats.get(split, {"clips": 0, "scenes": 0})
        print(f"  {split:<6} {s['clips']:>7,} clips    {s['scenes']:>5,} scenes")

    problems = image_stats["problems"] + video_stats["problems"]
    if problems:
        print("\n" + "!" * 66)
        for problem in problems:
            print(f"  FAIL  {problem}")
        print("!" * 66)
        if not args.allow_problems:
            print("\nManifests were written but SHOULD NOT be trained on. "
                  "Fix the corpus, or re-run with --allow-problems if you accept this.")
            return 1

    print(f"\nwrote {image_path}\nwrote {video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
