#!/usr/bin/env python3
"""Render virtual camera moves from equirectangular HDRIs, as SDR/HDR clip pairs.

The temporal refiner cannot be evaluated. Not for want of clips -- there are 935
-- but for want of SCENES: the whole video corpus is 13, split 11 train / 1 val
/ 1 test. A number measured on one held-out scene describes that scene. So §9
reports the refiner as unevaluated, and no amount of GPU time changes that.

Public HDR *video* is scarce. Public HDR *panoramas* are not: the corpus already
carries 963 scene-referred Poly Haven HDRIs, CC0, with real suns above 100,000
nits and no grade ceiling. A virtual camera moving through one of those produces
a sequence with genuine temporal structure -- parallax-free, but with the
translation-free motion that dominates plate photography anyway -- and, more to
the point, with EXACT ground truth at every frame, because both halves of every
pair are rendered from the same radiance.

That turns 13 scenes into as many as you have panoramas.

    python pipeline/render_hdri_moves.py --hdri-dir <polyhaven> --dst work/pairs_moves
    python training/build_video_manifest.py --hdr-dir work/pairs_moves/hdr \\
        --sdr-dir work/pairs_moves/sdr --output work/video_manifest_moves.jsonl \\
        --clip-length 9 --frame-step 1

Frames are named for `build_video_manifest.py`'s own regex, and the HDR halves
go through `pipeline/hdr_io.encode_hdr_u16` with the same storage block as the
rest of the corpus, so the output drops into the existing pipeline unchanged.
The ingest sentinel is written too: a directory prepared at one setting refuses
to accept frames prepared at another.

WHAT THIS IS NOT
----------------
A panorama has no parallax, so nothing occludes anything as the camera turns.
Pans, tilts, rolls and zooms are real; dollies are not, and neither is anything
moving in the scene. That covers a large share of real plates and none of the
hardest ones. Treat a temporal result measured only here as a lower bound on
difficulty, and say so wherever it is reported.

Resolution is the other honest caveat. Poly Haven's 2k panoramas are 2048x1024,
so a 75-degree horizontal field of view samples about 427 source pixels across
a 1280-wide frame -- a 3x upscale. `--max-upscale` refuses by default rather
than quietly training on softened sources. Fetch the 4k or 8k panoramas (also
CC0, same URL pattern) for anything you intend to report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.hdr_io import (  # noqa: E402
    HDR_IO_VERSION, HDRStorage, encode_hdr_u16,
)
from training.prepare_training_data import (  # noqa: E402
    make_sdr, read_exr, save_png_8bit, save_png_16bit,
)


# Gap between clips in the frame numbering. Large enough that no clip length
# can bridge two clips into one run.
CLIP_NUMBER_STRIDE = 1000


def ray_grid(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Unit ray per output pixel for a pinhole camera looking down +Z."""
    half = math.tan(math.radians(hfov_deg) * 0.5)
    aspect = height / width
    xs = (np.arange(width, dtype=np.float64) + 0.5 - width * 0.5) / (width * 0.5) * half
    ys = (np.arange(height, dtype=np.float64) + 0.5 - height * 0.5) / (height * 0.5) \
        * half * aspect
    gx, gy = np.meshgrid(xs, ys)
    dirs = np.stack([gx, -gy, np.ones_like(gx)], axis=-1)
    return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)


def rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    y, p, r = (math.radians(v) for v in (yaw_deg, pitch_deg, roll_deg))
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return ry @ rx @ rz


def project(pano: np.ndarray, dirs: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Sample an equirectangular panorama along rotated rays. Linear light in,
    linear light out -- interpolating after a tone curve would soften exactly
    the highlights this corpus exists to preserve."""
    h, w = pano.shape[:2]
    world = dirs @ rot.T
    lon = np.arctan2(world[..., 0], world[..., 2])
    lat = np.arcsin(np.clip(world[..., 1], -1.0, 1.0))
    # +1 for the wrap column padded on the left below.
    u = ((lon / (2.0 * math.pi)) + 0.5) * w + 1.0
    v = (0.5 - lat / math.pi) * h
    padded = np.concatenate([pano[:, -1:], pano, pano[:, :1]], axis=1)
    return cv2.remap(padded, u.astype(np.float32), v.astype(np.float32),
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def move(rng: np.random.Generator, frames: int, hfov: float) -> list[dict]:
    """One camera path. Pans dominate, tilts and rolls are small, zoom is slow --
    the distribution of moves a plate actually contains."""
    yaw = float(rng.uniform(0.0, 360.0))
    pitch = float(rng.uniform(-22.0, 22.0))
    roll = float(rng.normal(0.0, 1.2))
    d_yaw = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.25, 1.6))
    d_pitch = float(rng.normal(0.0, 0.18))
    d_roll = float(rng.normal(0.0, 0.05))
    zoom = float(rng.normal(0.0, 0.25))          # degrees of hfov per frame
    path = []
    for i in range(frames):
        path.append({"yaw": yaw + d_yaw * i, "pitch": max(-80.0, min(80.0, pitch + d_pitch * i)),
                     "roll": roll + d_roll * i,
                     "hfov": max(20.0, min(140.0, hfov + zoom * i))})
    return path


def select_sources(hdri_dir: Path,
                   done_file: Path | None) -> tuple[list[Path], set[str]]:
    """Panoramas still to render, and the set already finished.

    Two situations that look alike and must not share an exit code:

    An EMPTY directory is a mistake worth stopping for, so this raises.

    Every panorama present being ALREADY RENDERED is not a mistake. It is
    the normal state of the last batches of a batched run, and of any
    re-run. This returns an empty list for it, and the caller exits 0. On
    4 Sep 2026 the two shared one code and the corpus build aborted at
    batch 0, having done nothing wrong.

    The skip itself matters as much: nothing removes a rendered panorama
    from ``hdri_dir`` unless ``--drop-source`` is on, so without it the
    batched driver re-renders every earlier batch on every pass -- 993
    panoramas in 20 batches becomes 10,500 renders.
    """
    hdris = sorted(p for p in hdri_dir.rglob("*")
                   if p.suffix.lower() in (".exr", ".hdr"))
    if not hdris:
        raise SystemExit(f"error: no .exr/.hdr under {hdri_dir}")

    already: set[str] = set()
    if done_file and done_file.is_file():
        already = {line.strip() for line in
                   done_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        before = len(hdris)
        hdris = [p for p in hdris if p.name not in already]
        if before != len(hdris):
            print(f"   resuming: {before - len(hdris)} panorama(s) already rendered")
    if not hdris:
        print(f"   nothing to do: all {len(already)} panorama(s) in "
              f"{hdri_dir} are already rendered")
    return hdris, already


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdri-dir", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--clips-per-hdri", type=int, default=2)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--hfov", type=float, default=75.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-upscale", type=float, default=2.0,
                        help="refuse sources that would be upscaled more than this "
                             "across the frame. 0 disables the check.")
    parser.add_argument("--ceiling-nits", type=float, default=1_000_000.0)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--done-file", type=Path, default=None,
                        help="append each panorama's name here once its clips "
                             "are written. Pass the same path to "
                             "fetch_polyhaven.py --exclude-file so a "
                             "fetch/render/drop loop does not re-fetch what it "
                             "already consumed. Without it, --drop-source "
                             "deletes the evidence that the work was done.")
    parser.add_argument("--drop-source", action="store_true",
                        help="delete each panorama once its clips are written. "
                             "A full 4k Poly Haven set is ~48 GB parked and the "
                             "rendered pairs are ~30 GB on top; fetching a batch, "
                             "rendering it and dropping it keeps peak disk to the "
                             "pairs plus one file. Never deletes a source it "
                             "skipped, and never in --dry-run.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    hdris, already = select_sources(args.hdri_dir, args.done_file)
    if not hdris:
        return 0
    if args.limit:
        hdris = hdris[:args.limit]

    storage = HDRStorage(mode="log2_extended", ceiling_nits=args.ceiling_nits)
    sentinel = {
        "hdr_io_version": HDR_IO_VERSION,
        "storage": storage.as_dict(),
        "target_size": [args.width, args.height],
        "tonemap": "aces_approx_narkowicz2015",
        "sdr_encoding": "srgb",
        "source": "equirectangular HDRI virtual camera moves",
        "hfov_deg": args.hfov,
        "frames_per_clip": args.frames,
        "clips_per_hdri": args.clips_per_hdri,
        "seed": args.seed,
        "note": ("Rendered by pipeline/render_hdri_moves.py. No parallax: pans, "
                 "tilts, rolls and zooms only, and nothing in the scene moves."),
    }
    sdr_dir, hdr_dir, meta_dir = (args.dst / d for d in ("sdr", "hdr", "meta"))

    def open_destination() -> None:
        """Check the ingest sentinel, and write it. Called before the FIRST frame
        rather than up front: a run that renders nothing -- every panorama
        skipped by --max-upscale, say -- must not leave a config behind that
        then refuses the settings you meant to use."""
        for d in (sdr_dir, hdr_dir, meta_dir):
            d.mkdir(parents=True, exist_ok=True)
        path = args.dst / "_ingest_config.json"
        if not path.exists():
            path.write_text(json.dumps(sentinel, indent=2), encoding="utf-8")
            return
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key in ("hdr_io_version", "storage", "target_size", "tonemap",
                    "sdr_encoding"):
            if existing.get(key) != sentinel.get(key):
                raise SystemExit(
                    f"error: {args.dst} was prepared with a different config "
                    f"({key}: on disk {existing.get(key)!r}, requested "
                    f"{sentinel.get(key)!r}). Use a fresh --dst; mixing "
                    f"conventions is how the August corpus became unreadable.")

    opened = False

    print("\n" + "=" * 72)
    print("   HDRI camera moves")
    print("=" * 72)
    print(f"   panoramas  : {len(hdris)}")
    print(f"   clips      : {args.clips_per_hdri} per panorama x {args.frames} frames"
          f"  -> {len(hdris) * args.clips_per_hdri} clips, "
          f"{len(hdris) * args.clips_per_hdri * args.frames} frames")
    print(f"   output     : {args.width}x{args.height} at {args.hfov:.0f} deg hfov")

    dirs = ray_grid(args.width, args.height, args.hfov)
    written, skipped, clips_written = 0, [], 0
    started = time.time()

    for index, src in enumerate(hdris):
        try:
            pano = read_exr(src) if src.suffix.lower() == ".exr" else None
            if pano is None:
                raise RuntimeError("unsupported source")
        except Exception as exc:                                  # noqa: BLE001
            skipped.append((src.name, f"{type(exc).__name__}: {exc}"))
            continue
        pano = np.asarray(pano, dtype=np.float32)[..., :3]
        src_px = pano.shape[1] * args.hfov / 360.0
        upscale = args.width / max(src_px, 1e-6)
        if args.max_upscale and upscale > args.max_upscale:
            skipped.append((src.name, f"{upscale:.1f}x upscale from {pano.shape[1]}px "
                                      f"equirect; raise --max-upscale or --hfov, or "
                                      f"fetch a larger panorama"))
            continue

        scene = src.stem
        for clip in range(args.clips_per_hdri):
            rng = np.random.default_rng(abs(hash((args.seed, scene, clip))) % (2**32))
            path = move(rng, args.frames, args.hfov)
            for frame, pose in enumerate(path):
                grid = (dirs if abs(pose["hfov"] - args.hfov) < 1e-9
                        else ray_grid(args.width, args.height, pose["hfov"]))
                linear = project(pano, grid,
                                 rotation(pose["yaw"], pose["pitch"], pose["roll"]))
                linear = np.maximum(linear, 0.0)
                # The clip index goes in the FRAME NUMBER, not the sequence
                # name. build_video_manifest.py groups by sequence and then
                # splits on gaps, so clip 0 numbered 0..8 and clip 1 numbered
                # 1000..1008 come back as two clips of ONE scene. Putting the
                # clip in the name instead would make every clip its own scene,
                # and a scene-held-out split could then put two views of the
                # same panorama on both sides of it -- the leakage the image
                # path is careful to avoid.
                stem = f"tif_{index:07d}_{scene}_{clip * CLIP_NUMBER_STRIDE + frame:05d}"
                if not args.dry_run:
                    if not opened:
                        open_destination()
                        opened = True
                    code, stats = encode_hdr_u16(linear, storage)
                    save_png_8bit(make_sdr(linear), sdr_dir / f"{stem}.png")
                    save_png_16bit(code, hdr_dir / f"{stem}.png")
                    (meta_dir / f"{stem}.json").write_text(json.dumps({
                        "stem": stem, "source": str(src.resolve()), "clip": clip,
                        "frame": frame, "pose": pose, **stats,
                    }, indent=2), encoding="utf-8")
                written += 1
            clips_written += 1
        if args.done_file and not args.dry_run and src.name not in already:
            already.add(src.name)
            with open(args.done_file, "a", encoding="utf-8") as done:
                done.write(src.name + "\n")

        if args.drop_source and not args.dry_run:
            # Only reached when every clip for this panorama was written: the
            # skip paths above `continue` before here. Deleting a source we
            # could not render would silently shrink the corpus and leave no
            # way to find out which scenes went missing.
            try:
                src.unlink()
            except OSError as exc:                                # noqa: BLE001
                print(f"   could not drop {src.name}: {exc}")

        if (index + 1) % 10 == 0 or index + 1 == len(hdris):
            rate = (index + 1) / max(time.time() - started, 1e-9)
            print(f"   {index + 1:>5}/{len(hdris)} panoramas  {rate:5.2f}/s  "
                  f"{written} frames")

    print("=" * 72)
    print(f"   wrote {written} frame(s) in {clips_written} clip(s) "
          f"across {len(hdris) - len(skipped)} scene(s)")
    if skipped:
        print(f"   skipped {len(skipped)}:")
        for name, why in skipped[:5]:
            print(f"     {name}: {why}")
        if len(skipped) > 5:
            print(f"     ... and {len(skipped) - 5} more")
    print(f"\n   python training/build_video_manifest.py --hdr-dir {hdr_dir} "
          f"--sdr-dir {sdr_dir} \\\n       --output <manifest>.jsonl --clip-length "
          f"{args.frames} --frame-step 1\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
