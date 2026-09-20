"""Build a deterministic, scene-safe manifest for direct SDR-to-HDR training.

Pairs are matched by relative filename.  All frames belonging to one extracted
video sequence are assigned to the same split, so adjacent frames can never
leak between train, validation, and test sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


# NOTE: prepare_training_data.py names MXF frames "mxf_{idx}_{stem}_f{i:04d}"
# (an "f" before the frame digits) while TIF frames are "tif_{idx}_{stem}_{frame}".
# The optional "f" below is required — without it no MXF frame ever matched,
# so adjacent video frames leaked across train/val/test and build_video_manifest
# dropped all MXF content as stills (AUDIT_2026-08-10 NEW-6).
SEQUENCE_RE = re.compile(
    r"^(?P<kind>tif|mxf)_(?P<ingest>\d+)_(?P<sequence>.+?)_f?(?P<frame>\d+)$",
    re.IGNORECASE,
)
INGEST_RE = re.compile(r"^(?P<kind>exr|hdr|tif|mxf)_(?P<ingest>\d+)_(?P<name>.+)$", re.IGNORECASE)


def scene_identity(stem: str) -> tuple[str, int | None, bool]:
    """Return stable scene id, source frame number, and whether this is video."""
    match = SEQUENCE_RE.match(stem)
    if match:
        return (
            f"{match.group('kind').lower()}:{match.group('sequence')}",
            int(match.group("frame")),
            True,
        )
    match = INGEST_RE.match(stem)
    if match:
        return f"{match.group('kind').lower()}:{match.group('name')}", None, False
    return f"still:{stem}", None, False


def split_for_scene(scene_id: str, val_fraction: float, test_fraction: float, seed: int) -> str:
    """Map a scene to a split using a stable hash, independent of file order."""
    digest = hashlib.sha256(f"{seed}:{scene_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < test_fraction:
        return "test"
    if value < test_fraction + val_fraction:
        return "val"
    return "train"


def balanced_scene_splits(scene_sizes: dict[str, int], val_fraction: float,
                          test_fraction: float, seed: int) -> dict[str, str]:
    """Assign whole scenes without letting dominant sequences consume a split.

    Very large scenes are placed first.  This matters for the current corpus,
    where two long Chimera sequences dominate the frame count; a pure hash can
    accidentally put most of the entire dataset in validation.
    """
    total = sum(scene_sizes.values())
    targets = {
        "train": total * (1.0 - val_fraction - test_fraction),
        "val": total * val_fraction,
        "test": total * test_fraction,
    }
    heldout_capacity = max(targets["val"], targets["test"])
    result: dict[str, str] = {}
    for scene, size in scene_sizes.items():
        # A scene larger than an entire requested held-out split would destroy
        # pair balance if hashed there. Keep dominant sequences in training;
        # hash the remaining independent scenes for broad validation coverage.
        if size > heldout_capacity:
            result[scene] = "train"
        else:
            result[scene] = split_for_scene(scene, val_fraction, test_fraction, seed)
    return result


def _index(folder: Path, extensions: tuple[str, ...]) -> tuple[dict[str, Path], int]:
    result: dict[str, Path] = {}
    duplicates = 0
    priority = {extension: index for index, extension in enumerate(extensions)}
    paths = sorted(folder.rglob("*"), key=lambda p: (p.as_posix().lower(), priority.get(p.suffix.lower(), 999)))
    for path in paths:
        if path.is_file() and path.suffix.lower() in extensions:
            key = path.relative_to(folder).with_suffix("").as_posix().lower()
            if key in result:
                duplicates += 1
                # Legacy preparation runs may leave a .tif beside the preferred
                # .png with the same stem. Keep the earliest extension in the
                # caller-provided priority list and report the duplicate.
                if priority[path.suffix.lower()] >= priority[result[key].suffix.lower()]:
                    continue
            result[key] = path.resolve()
    return result, duplicates


def build_manifest(
    sdr_dir: Path,
    hdr_dir: Path,
    output: Path,
    metadata_dir: Path | None = None,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
    seed: int = 20260715,
) -> dict[str, object]:
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be >= 0 and sum to less than 1")
    sdr, sdr_duplicates = _index(sdr_dir, (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"))
    hdr, hdr_duplicates = _index(hdr_dir, (".png", ".tif", ".tiff", ".exr", ".hdr"))
    common = sorted(set(sdr).intersection(hdr))
    if not common:
        raise ValueError(f"No matching SDR/HDR filenames under {sdr_dir} and {hdr_dir}")

    pair_info = []
    scene_sizes: Counter[str] = Counter()
    for key in common:
        scene_id, frame_number, is_video = scene_identity(sdr[key].stem)
        pair_info.append((key, scene_id, frame_number, is_video))
        scene_sizes[scene_id] += 1
    scene_splits = balanced_scene_splits(dict(scene_sizes), val_fraction, test_fraction, seed)

    records = []
    for key, scene_id, frame_number, is_video in pair_info:
        split = scene_splits[scene_id]
        record: dict[str, object] = {
            "asset_id": key,
            "scene_id": scene_id,
            "split": split,
            "sdr_path": str(sdr[key]),
            "hdr_path": str(hdr[key]),
            "sdr_encoding": "srgb",
            "hdr_encoding": "linear_normalized_10000_nits",
            "is_video": is_video,
        }
        if frame_number is not None:
            record["frame_number"] = frame_number
        if metadata_dir:
            candidate = metadata_dir / Path(key).with_suffix(".json")
            if candidate.exists():
                record["metadata_path"] = str(candidate.resolve())
                # The exposure the pair was rendered at travels with the row,
                # so a model can be built with the right baseline without
                # opening thirty thousand sidecars (sdr2hdr_dataset.corpus_ev_of).
                try:
                    tonemap_ev = json.loads(candidate.read_text(encoding="utf-8")).get("tonemap_ev")
                except (OSError, ValueError):
                    tonemap_ev = None
                if tonemap_ev is not None:
                    record["tonemap_ev"] = float(tonemap_ev)
        records.append(record)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    counts = Counter(str(r["split"]) for r in records)
    scene_counts = Counter(scene_splits.values())
    return {
        "pairs": len(records),
        "sdr_unpaired": len(sdr) - len(common),
        "hdr_unpaired": len(hdr) - len(common),
        "sdr_duplicate_stems_ignored": sdr_duplicates,
        "hdr_duplicate_stems_ignored": hdr_duplicates,
        "scenes": len(scene_splits),
        "pairs_by_split": dict(sorted(counts.items())),
        "scenes_by_split": dict(sorted(scene_counts.items())),
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdr-dir", required=True, type=Path)
    parser.add_argument("--hdr-dir", required=True, type=Path)
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    stats = build_manifest(
        args.sdr_dir, args.hdr_dir, args.output, args.metadata_dir,
        args.val_fraction, args.test_fraction, args.seed,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
