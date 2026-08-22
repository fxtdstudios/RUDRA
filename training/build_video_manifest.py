"""Build deterministic paired SDR/HDR clip manifests from extracted sequences.

The preparation pipeline names sequence frames like::

    tif_0000123_Chimera_DCI4k2398p_HDR_P3PQ_00456.png

The first number is an ingestion counter and the final number is the source
frame. This tool groups by the middle sequence name, de-duplicates source frame
numbers, finds consecutive runs, and writes fixed-length JSONL clips.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


# Optional "f" before the frame digits: MXF frames are named
# "mxf_{idx}_{stem}_f{i:04d}" by prepare_training_data.py (AUDIT_2026-08-10
# NEW-6 — without this, every MXF frame was misclassified as a still and no
# MXF clip ever reached temporal training).
SEQUENCE_RE = re.compile(
    r"^(?P<kind>tif|mxf)_(?P<ingest>\d+)_(?P<sequence>.+?)_f?(?P<frame>\d+)$",
    re.IGNORECASE,
)


def _index_frames(folder: Path, extensions=(".png", ".exr", ".hdr")):
    grouped: dict[str, dict[int, Path]] = defaultdict(dict)
    duplicates = 0
    stills = 0
    for path in sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in extensions):
        match = SEQUENCE_RE.match(path.stem)
        if not match:
            stills += 1
            continue
        sequence = f"{match.group('kind').lower()}:{match.group('sequence')}"
        frame = int(match.group("frame"))
        if frame in grouped[sequence]:
            duplicates += 1
            continue
        grouped[sequence][frame] = path.resolve()
    return grouped, duplicates, stills


def build_manifest(
    hdr_dir: Path,
    output: Path,
    sdr_dir: Path | None = None,
    clip_length: int = 9,
    frame_step: int = 1,
    clip_stride: int | None = None,
) -> dict[str, int]:
    if clip_length < 2:
        raise ValueError("clip_length must be >= 2 for a video manifest")
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")
    clip_stride = clip_stride or clip_length

    hdr, hdr_dupes, hdr_stills = _index_frames(hdr_dir)
    sdr, sdr_dupes, sdr_stills = _index_frames(sdr_dir) if sdr_dir else ({}, 0, 0)
    clips = []
    paired_frames = 0

    for sequence in sorted(hdr):
        frames = hdr[sequence]
        if sdr_dir:
            common = sorted(set(frames).intersection(sdr.get(sequence, {})))
        else:
            common = sorted(frames)
        paired_frames += len(common)
        common_set = set(common)
        starts = [
            frame for frame in common
            if frame - frame_step not in common_set
        ]
        for run_start in starts:
            run = []
            frame = run_start
            while frame in common_set:
                run.append(frame)
                frame += frame_step
            for offset in range(0, len(run) - clip_length + 1, clip_stride):
                ids = run[offset: offset + clip_length]
                item = {
                    "clip_id": f"{sequence}:{ids[0]}-{ids[-1]}:{frame_step}",
                    "scene_id": sequence,
                    "frame_numbers": ids,
                    "frame_step": frame_step,
                    "hdr_frames": [str(frames[i]) for i in ids],
                }
                if sdr_dir:
                    item["sdr_frames"] = [str(sdr[sequence][i]) for i in ids]
                clips.append(item)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for item in clips:
            handle.write(json.dumps(item, separators=(",", ":")) + "\n")

    return {
        "clips": len(clips),
        "sequences": len(hdr),
        "paired_frames": paired_frames,
        "hdr_duplicates_ignored": hdr_dupes,
        "sdr_duplicates_ignored": sdr_dupes,
        "hdr_stills_excluded": hdr_stills,
        "sdr_stills_excluded": sdr_stills,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdr-dir", required=True, type=Path)
    parser.add_argument("--sdr-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clip-length", type=int, default=9)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--clip-stride", type=int)
    args = parser.parse_args()
    stats = build_manifest(
        args.hdr_dir, args.output, args.sdr_dir,
        args.clip_length, args.frame_step, args.clip_stride,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
