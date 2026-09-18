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
import math
import random
import re
import sys
from collections import Counter, defaultdict
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

# HdM-HDR-2014 ships several framings of ONE physical setup as sibling folders:
# cars_closeshot / cars_fullshot / cars_longshot, fishing_closeshot /
# fishing_longshot, poker_fullshot / poker_travelling_slowmotion, showgirl_01 /
# showgirl_02. Same location, same lighting, same dynamic range -- so putting
# two of them on opposite sides of a split is leakage dressed as scene-safety.
# Collapse the framing qualifier and the trailing take number; keep the subject.
_FRAMING_SUFFIX = re.compile(
    r"[_-](close ?shot|full ?shot|long ?shot|medium ?shot|travelling(_slowmotion)?|"
    r"slowmotion|closeup|wide)$", re.IGNORECASE)
_TAKE_NUMBER = re.compile(r"[_-]\d{1,2}$")


def _strip_shot(name: str) -> str:
    """Remove shot/framing/take qualifiers, leaving the physical scene's name."""
    name = _SHOT_SUFFIX.sub("", name)
    previous = None
    while previous != name:
        previous = name
        name = _FRAMING_SUFFIX.sub("", name)
        name = _TAKE_NUMBER.sub("", name)
    return name or previous


def normalize_scene_id(scene_id: str) -> str:
    """Collapse shot/take structure so a scene id names a PHYSICAL scene."""
    path_part, _, stem = scene_id.partition("::")
    segments = [s for s in path_part.split("/") if not _TAKE_SEG.match(s)]
    if segments:
        segments[-1] = _strip_shot(segments[-1])
    if stem and not _TAKE_SEG.match(stem):
        stem = _strip_shot(stem)
        return "/".join(segments) + "::" + stem
    return "/".join(segments)


def load_records(pairs_dir: Path, min_peak_nits: float = 0.0) -> list[dict]:
    index = pairs_dir / "pairs_index.jsonl"
    if not index.exists():
        raise SystemExit(f"error: {index} not found -- run prepare_pairs.py first")
    records = [json.loads(line) for line in index.open(encoding="utf-8") if line.strip()]

    # A source frame carrying NaN/Inf pixels encodes to undefined uint16 codes,
    # and the decode side cannot tell them from real radiance -- the stored
    # target is finite garbage.  prepare_pairs records peak_nits from the same
    # array, so a non-finite peak is the tell.  Two Poly Haven crops
    # (venice_dawn_1, venice_sunrise) tripped this on 23 Aug 2026.
    poisoned = [r for r in records if not math.isfinite(float(r.get("peak_nits", 0.0)))]
    if poisoned:
        names = ", ".join(r["asset_id"] for r in poisoned[:4])
        print(f"  dropped {len(poisoned)} record(s) with non-finite source pixels: {names}"
              + (" ..." if len(poisoned) > 4 else ""))
        keep = {id(r) for r in poisoned}
        records = [r for r in records if id(r) not in keep]

    if min_peak_nits > 0:
        dark = [r for r in records if float(r.get("peak_nits", 0.0)) < min_peak_nits]
        if dark:
            roots = Counter(r["scene_id"].rsplit("::", 1)[0] for r in dark)
            print(f"  dropped {len(dark):,} pair(s) peaking below {min_peak_nits:g} nit "
                  f"({len(dark) / len(records):.1%} of the index) -- no HDR content to learn:")
            for root, count in roots.most_common(4):
                print(f"      {count:>7,}  {root}")
            drop = {id(r) for r in dark}
            records = [r for r in records if id(r) not in drop]
            if not records:
                raise SystemExit("error: --min-peak-nits removed every record")

    raw = {r["scene_id"] for r in records}
    for record in records:
        record["scene_id"] = normalize_scene_id(record["scene_id"])
    annotate_ceilings(records)

    merged = {r["scene_id"] for r in records}
    if len(merged) != len(raw):
        print(f"  scene normalization: {len(raw)} raw scene ids -> "
              f"{len(merged)} physical scenes (shot/take structure collapsed)")

    return records


# A grading ceiling is a peak value that MANY FRAMES LAND ON EXACTLY. Natural
# scene peaks never repeat bit-for-bit; a clip does. Measured 28 Aug 2026 on the
# v3 corpus this flags 12 scenes (HdM 4,000 nits, Rec2100-PQ-1K ~991, Chimera
# 10,000) and none of the 963 Poly Haven stills.
#
# Why it matters: a pixel at exactly 4,000 nits in a 4,000-nit graded source is
# a CENSORED observation -- it means ">= 4,000", not "= 4,000". 84% of this
# corpus comes from such sources and 8,234 records peak right at their ceiling.
# Training L1 against them teaches the model to cap, which is measurable: the
# v4 model, trained on this mix, reconstructs 1.23 stops LESS highlight than
# v3b did on a controlled specular.
CEILING_MIN_REPEATS = 8
CEILING_MIN_NITS = 100.0


def detect_grading_ceiling(peaks: list[float],
                           min_repeats: int = CEILING_MIN_REPEATS,
                           floor: float = CEILING_MIN_NITS) -> float | None:
    """The scene's delivery ceiling, or None if its peaks look scene-referred."""
    finite = [p for p in peaks if math.isfinite(p)]
    if len(finite) < min_repeats:
        return None
    value, count = Counter(finite).most_common(1)[0]
    if count < min_repeats or value < floor:
        return None
    return float(value)


def annotate_ceilings(records: list[dict]) -> int:
    """Tag every record with its scene's grading ceiling in nits (or None)."""
    peaks: dict[str, list[float]] = defaultdict(list)
    for record in records:
        peaks[record["scene_id"]].append(float(record.get("peak_nits", 0.0)))
    ceilings = {scene: detect_grading_ceiling(values) for scene, values in peaks.items()}
    censored = 0
    for record in records:
        ceiling = ceilings.get(record["scene_id"])
        record["ceiling_nits"] = ceiling
        at_ceiling = ceiling is not None and float(record.get("peak_nits", 0.0)) >= ceiling
        record["peak_at_ceiling"] = bool(at_ceiling)
        censored += at_ceiling
    limited = sum(1 for c in ceilings.values() if c)
    if limited:
        print(f"  grading ceilings: {limited} of {len(ceilings)} scenes are delivery-graded; "
              f"{censored:,} record(s) peak AT their ceiling (censored highlights)")
    return censored


def _thin_scene(rows: list[dict], target: int) -> list[dict]:
    """Keep ~target rows of one scene by dropping whole SOURCE FRAMES on a stride.

    Not a random sample: build_video_manifest needs a uniform frame step to form
    clips, and all crops of a kept frame must survive together or the per-frame
    dedup picks a different crop for neighbouring frames. Dropping whole frames
    on a stride multiplies the ingest stride and leaves both properties intact.
    """
    if target >= len(rows) or target <= 0:
        return rows if target >= len(rows) else []
    frames = sorted({(r.get("source_take") or "", r.get("frame_index")) for r in rows})
    # Walk the stride up until the result is genuinely at or under target.
    # A single ceil() is not enough: it can leave the scene ONE record over,
    # and the caller then re-thins with the only stride left to it -- 2 --
    # halving a scene that was 0.1 percentage points too big. That is what
    # took the test split's fireplace scene 1,392 -> 156 -> 78 on 27 Aug 2026
    # and dropped its video share under the --min-video-share floor.
    step = max(1, -(-len(rows) // target))
    while step <= len(frames):
        allowed = set(frames[::step])
        kept = [r for r in rows
                if (r.get("source_take") or "", r.get("frame_index")) in allowed]
        if len(kept) <= target:
            return kept
        step += 1
    return rows[:target]


def cap_scene_share(rows: list[dict], max_share: float) -> tuple[list[dict], list[tuple]]:
    """Thin whichever scene dominates until no scene exceeds max_share of rows.

    Self-tuning rather than a fixed record cap: to leave a scene at share s of
    the total, keep s/(1-s) times the records every OTHER scene contributes.
    Repeated because thinning the leader shrinks the total and can promote the
    runner-up.
    """
    trimmed: list[tuple] = []
    done: set[str] = set()
    for _ in range(8):
        counts = Counter((r["scene_id"] for r in rows if r["scene_id"] not in done))
        if not counts:
            break
        total = sum(Counter(r["scene_id"] for r in rows).values())
        scene, count = counts.most_common(1)[0]
        if total == 0 or count / total <= max_share:
            break
        # One pass per scene, ever. Thinning is a stride over frames, so it
        # lands where the stride lands; re-entering to shave a rounding
        # remainder is how a scene gets halved.
        done.add(scene)
        others = total - count
        target = max(1, int(max_share * others / (1.0 - max_share)))
        scene_rows = [r for r in rows if r["scene_id"] == scene]
        kept = _thin_scene(scene_rows, target)
        trimmed.append((scene, len(scene_rows), len(kept)))
        rows = [r for r in rows if r["scene_id"] != scene] + kept
    return rows, trimmed


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


def scene_tail(scene_id: str) -> str:
    """The drive-independent part of a scene id.

    scan_sources.scene_key() builds ``<parent path>::<stem>``, so the same
    footage ingested from E:\\source_hdr and from G:\\datasets\\sources gets
    two ids. Hold-out matching uses the parent's last component plus the stem,
    which is what actually names the shot.
    """
    parent, _, stem = str(scene_id).replace("\\", "/").rpartition("::")
    return f"{parent.rstrip('/').rsplit('/', 1)[-1]}::{stem}".lower()


def load_hold_out_scenes(path: Path) -> set[str]:
    """Scenes that must land in *test*, as tails (see scene_tail).

    Accepts a manifest (.jsonl: the scene_ids of its ``split == "test"`` rows,
    or of every row when it has no split) or a text file with one scene id
    per line. Pointing it at the previous corpus's manifest keeps that
    corpus's benchmark scenes out of this corpus's training set, which is the
    only way a number on the old bench still means "held out".
    """
    tails: set[str] = set()
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".jsonl":
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split", "test") == "test" and row.get("scene_id"):
                tails.add(scene_tail(row["scene_id"]))
    else:
        tails.update(scene_tail(l.strip()) for l in text.splitlines() if l.strip())
    return tails


def build_image_manifest(records: list[dict], val_frac: float, test_frac: float,
                         seed: int, min_video_share: float,
                         max_eval_scene_share: float = 0.0,
                         hold_out: set[str] | None = None) -> tuple[list[dict], dict]:
    by_kind: dict[bool, set[str]] = defaultdict(set)
    for record in records:
        by_kind[bool(record.get("is_video", False))].add(record["scene_id"])

    # Split each population independently so both land in every split. This is
    # the single change that makes the held-out numbers mean something.
    assignment: dict[str, str] = {}
    for is_video, scenes in by_kind.items():
        offset = 1 if is_video else 0
        assignment.update(split_scenes(sorted(scenes), val_frac, test_frac, seed + offset))

    # Scenes the previous corpus benchmarked on stay out of training here, or
    # the acceptance step scores a model on frames it trained on (22 Sep 2026:
    # the paper's 429 frames come from E:\\RUDRA_v3's test split, and the same
    # sources were re-ingested into corpus_v4b under a fresh split).
    held = 0
    if hold_out:
        for scene in list(assignment):
            if scene_tail(scene) in hold_out:
                assignment[scene] = "test"
                held += 1

    out = []
    for record in records:
        row = dict(record)
        row["split"] = assignment[record["scene_id"]]
        out.append(row)

    # Thin over-represented scenes in val/test ONLY.
    #
    # Training is immune: the trainer draws with a scene-balanced weighted
    # sampler, so a scene's record count does not set its influence there, and
    # extra frames from a shot are free diversity. Evaluation is a plain mean
    # over records, so record counts ARE the weighting -- and because whole
    # scenes are held out, one big scene does not merely dominate a split, it
    # becomes the split. On 23 Aug 2026 the Bar scene was 95.6% of val: every
    # "held-out" number described one bar interior.
    if max_eval_scene_share > 0:
        kept = [r for r in out if r["split"] == "train"]
        for split in ("val", "test"):
            rows = [r for r in out if r["split"] == split]
            rows, trimmed = cap_scene_share(rows, max_eval_scene_share)
            for scene, before, after in trimmed:
                print(f"  {split} scene share cap {max_eval_scene_share:.0%}: "
                      f"{before:,} -> {after:,} records  "
                      f"{scene.rsplit('/', 1)[-1][:60]}")
            kept.extend(rows)
        out = kept

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

    if hold_out:
        stats["held_out_scenes"] = held
        stats["hold_out_requested"] = len(hold_out)
        if held == 0:
            problems.append(f"--hold-out-scenes listed {len(hold_out)} scenes and none of them "
                            f"exist in this corpus -- wrong file, or the ids do not line up")
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
                # Carried through so the temporal loss censors graded highlights
                # the same way the image loss does; one scene, one ceiling.
                "ceiling_nits": window[0].get("ceiling_nits"),
                # The exposure the render applied, carried per clip so
                # corpus_ev_of() reads the same value from a video manifest
                # as from the image manifest. Without it a temporal run fell
                # back to the legacy -1 EV over a 0 EV corpus (22 Sep 2026).
                "tonemap_ev": window[0].get("tonemap_ev"),
                "metadata_path": window[0].get("metadata_path"),
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
    # 0.15 matches verify_dataset.py check 6. Setting this to 0.25 while
    # max-eval-scene-share is also 0.25 creates an impossible knife-edge whenever
    # val or test holds only 1 video scene: capping that scene to <= 25% via
    # integer stride thinning lands it at 24.8% (e.g. 95/383 records), failing
    # the builder check while passing dataset verification.
    parser.add_argument("--min-video-share", type=float, default=0.15,
                        help="Minimum share of val/test records that must be video frames (default %(default)s)")
    parser.add_argument("--clip-length", type=int, default=9)
    parser.add_argument("--clip-stride", type=int, default=9)
    parser.add_argument("--min-peak-nits", type=float, default=1.0,
                        help="Drop pairs whose HDR target never reaches this peak. "
                             "Default 1.0 nit: a target that dark carries no HDR "
                             "information to learn, and on 23 Aug 2026 this was "
                             "exactly the 4,071 pairs (21%% of the corpus) that the "
                             "scan pulled out of 08_Research/data/hdr -- ALREADY "
                             "NORMALISED August output, re-ingested as if it were "
                             "scene-linear source, so every frame landed ~5,000x too "
                             "dark and each became its own single-frame 'scene'. "
                             "0 disables.")
    # 0.25, because that is what verify_dataset.py's check 7 demands. This
    # defaulted to 0.35 until 18 Sep 2026, which meant the builder thinned a
    # dominant scene down to a share the gate then rejected: corpus v4b came
    # out at 34.7% of test and 34.2% of val -- sitting exactly on the old cap
    # -- and failed verification for it. A builder whose default cannot pass
    # the default gate is a builder that wastes a render.
    parser.add_argument("--max-eval-scene-share", type=float, default=0.25,
                        help="No single scene may exceed this share of val or of test. "
                             "Training is untouched -- its sampler is already "
                             "scene-balanced -- but evaluation is a plain mean over "
                             "records, so a scene's share IS its weight in every number "
                             "you quote. On 23 Aug 2026 the Bar scene was 95.6%% of val "
                             "and the held-out metric described one bar interior. Frames "
                             "are dropped on a uniform stride so clips still form. "
                             "0 disables.")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--hold-out-scenes", type=Path, default=None,
                        help="A previous manifest (.jsonl; its test rows) or a text file of "
                             "scene ids. Those scenes are forced into TEST here, matched on "
                             "the drive-independent tail of the id, so a model trained on "
                             "this corpus can still be scored on the old benchmark as held "
                             "out. Point it at E:\\RUDRA_v3_20260822\\sdr_hdr_manifest.jsonl "
                             "before training anything that will be compared with the paper.")
    parser.add_argument("--allow-problems", action="store_true",
                        help="Write manifests even when guards fail (you must say why)")
    parser.add_argument("--require-temporal", action="store_true",
                        help="Fail the whole build when the temporal corpus is too small. "
                             "Off by default: an under-sized VIDEO corpus must not block "
                             "IMAGE training, which is what happened on 23 Aug 2026 -- a "
                             "healthy 19,275-pair image corpus never reached the trainer "
                             "because temporal was 3 scenes short.")
    args = parser.parse_args()

    records = load_records(args.pairs_dir, args.min_peak_nits)
    hold_out = load_hold_out_scenes(args.hold_out_scenes) if args.hold_out_scenes else None
    rows, image_stats = build_image_manifest(
        records, args.val_frac, args.test_frac, args.seed, args.min_video_share,
        args.max_eval_scene_share, hold_out)
    if hold_out is not None:
        print(f"  hold-out: {image_stats.get('held_out_scenes', 0)} of {len(hold_out)} listed "
              f"scenes found in this corpus and pinned to test")

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

    image_problems = image_stats["problems"]
    video_problems = video_stats["problems"]

    if image_problems:
        print("\n" + "!" * 66)
        for problem in image_problems:
            print(f"  FAIL  {problem}")
        print("!" * 66)
        if not args.allow_problems:
            print("\nThe IMAGE manifest is not trainable. Fix the corpus, or re-run "
                  "with --allow-problems if you accept this.")
            return 1

    if video_problems:
        # Mark the video manifest so the temporal trainer can refuse it, and say
        # plainly that this does not stop the image model.
        marker = video_path.with_suffix(video_path.suffix + ".GATED")
        marker.write_text("\n".join(video_problems) + "\n", encoding="utf-8")
        print("\n" + "-" * 66)
        print("  TEMPORAL GATED -- do not start temporal training:")
        for problem in video_problems:
            print(f"    {problem}")
        print(f"  marker: {marker.name}")
        print("  IMAGE training is unaffected and may proceed.")
        print("-" * 66)
        if args.require_temporal and not args.allow_problems:
            return 1

    print(f"\nwrote {image_path}\nwrote {video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
