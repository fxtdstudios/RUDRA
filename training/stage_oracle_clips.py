#!/usr/bin/env python3
"""Materialise a scene-balanced sample of rendered clips for the temporal gate.

`gate_temporal_oracle.py` wants clip DIRECTORIES -- `sdr/`, `hdr/`, `meta/`,
one frame per file -- because it needs each frame's camera pose, and the pose
lives in the meta JSON the renderer wrote. The moves corpus is stored flat:
17 874 frames in three directories, with `video_manifest_moves.jsonl` saying
which of them form a clip. This turns the second into the first.

It samples ONE CLIP PER SCENE by default. The corpus renders two clips from
every panorama, and two clips of the same panorama are two camera moves
through the same light -- as a sample they count once, not twice, and letting
both in would make 50 clips look like 50 scenes when it is 25.

    python training/stage_oracle_clips.py \\
        --manifest D:\\A.I\\Devlopments\\RUDRA_v02\\video_manifest_moves.jsonl \\
        --meta-dir D:\\A.I\\Devlopments\\RUDRA_v02\\pairs_moves\\meta \\
        --dest D:\\A.I\\Devlopments\\RUDRA_v02\\_oracle_clips_40 --count 40

Without --count it stages every clip it finds.

Deterministic: the same --seed and --count give the same clips, so a gate
number can be re-measured on exactly the sample that produced it.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath


def basename(recorded: str) -> str:
    """The file name out of a manifest path, whichever OS wrote it.

    The moves manifest was written on Windows and holds backslash paths.
    `Path("D:\\...\\x.png").stem` on Linux is the WHOLE string -- silently,
    with no error until something tries to open it. Frames are resolved
    against --pairs-dir rather than against the recorded directory for the
    same reason: the corpus is reachable under a different root from every
    machine that has ever read it.
    """
    name = PureWindowsPath(recorded).name if "\\" in recorded \
        else PurePosixPath(recorded).name
    return name


def read_manifest(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def choose(records: list[dict], count: int, seed: int,
           per_scene: int) -> list[dict]:
    by_scene: dict[str, list[dict]] = {}
    for record in records:
        by_scene.setdefault(record["scene_id"], []).append(record)
    rng = random.Random(seed)
    scenes = sorted(by_scene)
    rng.shuffle(scenes)
    picked: list[dict] = []
    for scene in scenes:
        clips = sorted(by_scene[scene], key=lambda r: r["clip_id"])
        picked.extend(clips[:per_scene])
        if len(picked) >= count:
            break
    if len(picked) < count:
        print(f"   only {len(picked)} clip(s) available at {per_scene} per "
              f"scene across {len(scenes)} scene(s); asked for {count}")
    return picked[:count]


def stage(record: dict, meta_dir: Path, pairs_dir: Path, dest: Path,
          skip_existing: bool) -> tuple[str, bool]:
    """One clip directory. Returns (name, copied).

    Resumable, like the fetcher and the renderer: a clip whose directory
    already holds the right number of frames in all three trees is left alone.
    Copying 40 clips is 1 080 files and well over a gigabyte, which is more
    than one shell call gets on some of the machines this runs from.
    """
    # tif:aarfontein_dirt_road_4k:0-8:1 -> aarfontein_dirt_road_4k_c0
    stem = Path(basename(record["hdr_frames"][0])).stem
    scene = record["scene_id"].split(":")[-1]
    clip_index = json.loads((meta_dir / f"{stem}.json").read_text(
        encoding="utf-8"))["clip"]
    name = f"{scene}_c{clip_index}"
    out = dest / name
    expected = len(record["hdr_frames"])
    if skip_existing and all(
            (out / sub).is_dir() and len(list((out / sub).iterdir())) == expected
            for sub in ("sdr", "hdr", "meta")):
        return name, False
    for sub in ("sdr", "hdr", "meta"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for sdr_rec, hdr_rec in zip(record["sdr_frames"], record["hdr_frames"]):
        sdr_path = pairs_dir / "sdr" / basename(sdr_rec)
        hdr_path = pairs_dir / "hdr" / basename(hdr_rec)
        if sdr_path.stem != hdr_path.stem:
            raise SystemExit(f"error: {name} pairs {sdr_path.stem} with "
                             f"{hdr_path.stem}; the manifest is misaligned")
        shutil.copy2(sdr_path, out / "sdr" / sdr_path.name)
        shutil.copy2(hdr_path, out / "hdr" / hdr_path.name)
        shutil.copy2(meta_dir / f"{sdr_path.stem}.json",
                     out / "meta" / f"{sdr_path.stem}.json")
    return name, True


def group_render(meta_dir: Path, frames_per_clip: int) -> list[list[Path]]:
    """Clips straight out of a render directory, no manifest in between.

    `build_video_manifest.py` exists and is the right tool when the corpus is
    going into training. For a pilot render whose only consumer is the gate,
    it is a step that can disagree with the meta JSONs it was built from --
    and the meta JSONs are what the gate reads the poses out of anyway. This
    groups them directly: one clip per (source panorama, clip index), frames
    in frame order.
    """
    clips: dict[tuple[str, str, int], list[tuple[int, Path]]] = {}
    for path in sorted(meta_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        scene = Path(basename(record["source"])).stem
        # The renderer's `tif_NNNNNNN_` prefix is the panorama's position in
        # the batch it was rendered in, not an identity -- so a scene rendered
        # twice (a batch that died mid-panorama and got re-run) lands under
        # two prefixes. Keying on the prefix keeps those apart instead of
        # merging them into one 16-frame "clip", and the length check below
        # then drops the partial one.
        prefix = path.stem.rsplit("_", 1)[0]
        clips.setdefault((prefix, scene, int(record["clip"])), []).append(
            (int(record["frame"]), path))
    out, dropped = [], []
    for (prefix, scene, index), frames in sorted(clips.items()):
        if len(frames) != frames_per_clip:
            dropped.append(f"{scene}_c{index} ({len(frames)} frames)")
            continue
        out.append([path for _, path in sorted(frames)])
    if dropped:
        print(f"   incomplete : dropped {len(dropped)} partial clip(s): "
              + ", ".join(dropped[:6])
              + (" ..." if len(dropped) > 6 else ""))
    if not out:
        raise SystemExit(f"error: no complete {frames_per_clip}-frame clip in "
                         f"{meta_dir}")
    return out


def stage_render(meta_paths: list[Path], pairs_dir: Path, dest: Path) -> str:
    record = json.loads(meta_paths[0].read_text(encoding="utf-8"))
    scene = Path(basename(record["source"])).stem
    name = f"{scene}_c{int(record['clip'])}"
    out = dest / name
    for sub in ("sdr", "hdr", "meta"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for meta_path in meta_paths:
        stem = meta_path.stem
        shutil.copy2(pairs_dir / "sdr" / f"{stem}.png", out / "sdr" / f"{stem}.png")
        shutil.copy2(pairs_dir / "hdr" / f"{stem}.png", out / "hdr" / f"{stem}.png")
        shutil.copy2(meta_path, out / "meta" / f"{stem}.json")
    return name


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="video_manifest_*.jsonl. Omit with --from-render.")
    ap.add_argument("--frames", type=int, default=9,
                    help="frames a complete clip has. With --from-render, "
                         "anything else is a partial render and is dropped.")
    ap.add_argument("--from-render", action="store_true",
                    help="group clips from the meta JSONs in --meta-dir "
                         "instead of from a manifest. For a pilot render "
                         "whose only consumer is the gate.")
    ap.add_argument("--meta-dir", required=True, type=Path)
    ap.add_argument("--pairs-dir", type=Path, default=None,
                    help="root holding sdr/ and hdr/. Defaults to the parent "
                         "of --meta-dir. Frames are resolved here by NAME, "
                         "never at the path the manifest recorded.")
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--count", type=int, default=0,
                    help="how many clips to stage. 0 (the default) stages "
                         "them all. It defaulted to 40 until 6 Sep 2026, "
                         "which silently truncated a 50-scene seed sweep to "
                         "40 in every arm -- consistently, so the pairing "
                         "survived, but not the sample the scene list named. "
                         "A number this script was not given should not be "
                         "able to decide how big a measurement is.")
    ap.add_argument("--per-scene", type=int, default=1,
                    help="clips taken from any one panorama. 1 keeps the "
                         "sample size honest -- see the module docstring.")
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--no-skip", action="store_true",
                    help="re-copy clips that are already complete")
    ap.add_argument("--ingest-config", type=Path, default=None,
                    help="_ingest_config.json to copy in beside the clips. "
                         "Defaults to the one next to --meta-dir; the gate "
                         "refuses to run without it, because it is what says "
                         "how the 16-bit HDR PNGs decode.")
    args = ap.parse_args()

    pairs_dir = args.pairs_dir or args.meta_dir.parent
    args.dest.mkdir(parents=True, exist_ok=True)
    config = args.ingest_config or (args.meta_dir.parent / "_ingest_config.json")
    if not config.is_file():
        raise SystemExit(f"error: no _ingest_config.json at {config}")
    shutil.copy2(config, args.dest / "_ingest_config.json")

    if args.from_render:
        groups = group_render(args.meta_dir, args.frames)
        if args.count:
            groups = groups[:args.count]
            print(f"   limited    : {args.count} of the clips found")
        print(f"   staging    : {len(groups)} clip(s) -> {args.dest}")
        for index, group in enumerate(groups, 1):
            print(f"   [{index:3d}/{len(groups)}] "
                  f"{stage_render(group, pairs_dir, args.dest)}")
        print(f"   done: {args.dest}")
        return 0

    if args.manifest is None:
        raise SystemExit("error: --manifest is required without --from-render")
    records = read_manifest(args.manifest)
    picked = choose(records, args.count or len(records), args.seed,
                    args.per_scene)
    print(f"   manifest   : {len(records)} clip(s)")
    print(f"   staging    : {len(picked)} clip(s) -> {args.dest}")
    copied = 0
    for index, record in enumerate(picked, 1):
        name, did = stage(record, args.meta_dir, pairs_dir, args.dest,
                          not args.no_skip)
        copied += did
        print(f"   [{index:3d}/{len(picked)}] {name}"
              + ("" if did else "   (already staged)"))
    print(f"   copied {copied}, skipped {len(picked) - copied}")
    print(f"   done: {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
