"""Ingest HDR sources into SDR/HDR training pairs -- with correct target storage.

This deliberately does NOT fork ``training/prepare_training_data.py``. It imports
that module's readers, EOTFs and tone-mapper (all of which got the August 2026
fixes) and replaces exactly one thing: how the HDR target is written.

    old:  clip(scene_linear * 203/10000, 0, 1) * 65535   -> uint16 PNG
          78% of stills lost their highlights; shadows kept ~8 bits.
    new:  pipeline.hdr_io.encode_hdr_u16(...)            -> uint16 PNG
          nothing clips under log2_extended; ~29x the shadow codes.

It also writes what the old ingest did not record: the true source peak in nits,
the clipped fraction, and a ``_ingest_config.json`` sentinel so a directory can
never silently mix two conventions.

    python pipeline/prepare_pairs.py --src Z:\\08_Research --dst hdrdata/pairs_v3 \\
        --inventory hdrdata/source_inventory.jsonl --mode log2_extended --crops 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.hdr_io import (  # noqa: E402
    HDR_IO_VERSION, HDRStorage, encode_hdr_u16, selftest,
)

# Readers, EOTFs and the tone-map curve come from the existing (fixed) module.
from pipeline import sdr_render  # noqa: E402
import training.prepare_training_data as ptd  # noqa: E402
from training.prepare_training_data import (  # noqa: E402
    TARGET_H, TARGET_W, clipped_fraction as sdr_clipped_fraction, make_sdr,
    read_exr, read_tif, resize_frame, save_png_8bit, save_png_16bit,
    to_scene_linear,
)


def sentinel_payload(storage: HDRStorage, args: argparse.Namespace) -> dict:
    return {
        "hdr_io_version": HDR_IO_VERSION,
        "storage": storage.as_dict(),
        "target_size": [TARGET_W, TARGET_H],
        "crops_per_source": args.crops,
        "crop_size": args.crop_size,
        # "mix:v1" when --sdr-render mix: the curve is drawn per pair and
        # recorded in each record (pipeline/sdr_render.py). A directory never
        # mixes the two, because check_sentinel compares this key.
        "tonemap": ("aces_approx_narkowicz2015" if args.sdr_render == "aces"
                    else sdr_render.RENDER_VERSION),
        "ev_jitter": float(args.ev_jitter) if args.sdr_render == "mix" else 0.0,
        "codec_probability": (float(args.codec_probability)
                              if args.sdr_render == "mix" else 0.0),
        # The exposure applied before the curve. It decides whether the SDR
        # side ever clips, so a directory cannot mix two values of it, and it
        # rides on every pair so the model built from this corpus inverts the
        # same render (rudra.sdr2hdr.sdr_to_baseline_hdr, corpus_ev).
        "tonemap_ev": float(args.tonemap_ev),
        "sdr_encoding": "srgb",
        "seed": args.seed,
        "note": ("Targets are stored via pipeline/hdr_io.py. Decode with "
                 "decode_hdr_u16() using the storage block above -- never assume "
                 "the old linear/10000 convention."),
    }


def check_sentinel(dst: Path, payload: dict) -> None:
    """Refuse to write into a directory prepared with different settings."""
    path = dst / "_ingest_config.json"
    if not path.exists():
        dst.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    keys = ("hdr_io_version", "storage", "target_size", "tonemap", "tonemap_ev",
            "sdr_encoding")
    for key in keys:
        if existing.get(key) != payload.get(key):
            raise SystemExit(
                f"error: {dst} was prepared with a different ingest config.\n"
                f"       {key}: on disk {existing.get(key)!r} vs requested {payload.get(key)!r}\n"
                f"       Use a fresh --dst. Mixing conventions is how the August corpus "
                f"ended up unreadable."
            )


def crops_for(image: np.ndarray, count: int, size: int, rng: np.random.Generator
              ) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """Multi-crop. The old ingest emitted one frame per source, which is why
    SDXL / Qwen / Klein were stuck on 963 pairs while Flux and Wan had 12-13k."""
    h, w = image.shape[:2]
    if count <= 1 or size <= 0 or h < size or w < size:
        return [(image, (0, 0))]
    out = []
    for _ in range(count):
        y = int(rng.integers(0, h - size + 1))
        x = int(rng.integers(0, w - size + 1))
        out.append((image[y:y + size, x:x + size], (y, x)))
    return out


def load_scene_linear(path: Path, encoding: str, retries: int = 3) -> np.ndarray | None:
    """Read + decode one source. Retries with backoff: a transient SMB drop on
    the NAS made EVERY read fail for the rest of a run (23 Aug 2026 — 3,677
    sources skipped that probed fine minutes later)."""
    import time

    suffix = path.suffix.lower()
    data = None
    for attempt in range(retries):
        try:
            if suffix == ".exr":
                data = read_exr(path)
            elif suffix in (".tif", ".tiff"):
                data, _ = read_tif(path)
            else:
                return None
            break
        except Exception as exc:
            if attempt + 1 < retries:
                print(f"  retry {attempt + 1}/{retries - 1} {path.name}: {exc}",
                      file=sys.stderr)
                time.sleep(2.0 * (attempt + 1))
            else:
                print(f"  skip {path.name}: {exc}", file=sys.stderr)
                return None
    if data is None:
        return None
    return np.asarray(to_scene_linear(data, encoding), dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inventory", type=Path, required=True,
                        help="source_inventory.jsonl from scan_sources.py")
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--mode", choices=("log2_extended", "pq_10000"), default="log2_extended",
                        help="log2_extended keeps the full scene range (recommended); "
                             "pq_10000 is display-referred and clips at 10,000 nits")
    parser.add_argument("--ceiling-nits", type=float, default=1_000_000.0)
    parser.add_argument("--crops", type=int, default=1, help="Random crops per source image")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--include-unknown-encoding", action="store_true",
                        help="Ingest sources whose EOTF scan_sources could not identify")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--video-stride", type=int, default=1,
                        help="Ingest every Nth frame of image sequences. High-frame-rate "
                             "footage (192fps HFR-2017) yields near-duplicate pairs at "
                             "stride 1 and drowns the stills; 8 gives 24fps-equivalent "
                             "motion, which is also better temporal supervision.")
    parser.add_argument("--append", action="store_true",
                        help="Add to an existing pairs dir: index is appended, sources "
                             "whose outputs already exist are skipped. Sentinel must match.")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--tonemap-ev", type=float, default=ptd.TONEMAP_EV_OFFSET,
                        help=f"Exposure applied before the tone curve (default "
                             f"{ptd.TONEMAP_EV_OFFSET:+.1f}). {ptd.LEGACY_TONEMAP_EV:+.1f} "
                             f"reproduces every corpus up to v3, whose SDR side almost "
                             f"never clipped; 0 clips the way delivered SDR clips.")
    parser.add_argument("--sdr-render", choices=("aces", "mix"), default="aces",
                        help="aces: every pair through the one Narkowicz curve (every corpus "
                             "up to v4b). mix: the curve, exposure, contrast, saturation, "
                             "OETF and a real codec round trip are drawn per pair (per shot "
                             "for sequences) from pipeline/sdr_render.py and recorded. v4c "
                             "onward: a model trained on one curve learns that curve.")
    parser.add_argument("--ev-jitter", type=float, default=1.5,
                        help="mix only: +/- stops drawn around --tonemap-ev per pair")
    parser.add_argument("--codec-probability", type=float, default=0.5,
                        help="mix only: share of pairs put through JPEG/H.264/HEVC/AV1")
    args = parser.parse_args()
    # make_sdr reads the module constant; this is the one place it is set.
    ptd.TONEMAP_EV_OFFSET = float(args.tonemap_ev)

    storage = HDRStorage(mode=args.mode, ceiling_nits=args.ceiling_nits)  # type: ignore[arg-type]
    print(f"storage: {storage.describe()}")
    print(f"round-trip selftest: {selftest(storage)}")

    payload = sentinel_payload(storage, args)
    check_sentinel(args.dst, payload)

    sdr_dir, hdr_dir, meta_dir = args.dst / "sdr", args.dst / "hdr", args.dst / "meta"
    for directory in (sdr_dir, hdr_dir, meta_dir):
        directory.mkdir(parents=True, exist_ok=True)

    sources = [json.loads(line) for line in args.inventory.open(encoding="utf-8") if line.strip()]
    sources = [s for s in sources if s.get("kind") == "image" and "error" not in s]
    if not args.include_unknown_encoding:
        dropped = [s for s in sources if s.get("encoding_guess") == "UNKNOWN"]
        if dropped:
            print(f"  dropping {len(dropped):,} sources with UNKNOWN encoding "
                  f"(pass --include-unknown-encoding to ingest anyway, at your risk)")
        sources = [s for s in sources if s.get("encoding_guess") != "UNKNOWN"]
    if args.video_stride > 1:
        before = len(sources)
        sources = [s for s in sources
                   if not s.get("is_sequence_member")
                   or ((_frame_index(Path(s["path"])) or 0) % args.video_stride == 0)]
        print(f"  video stride {args.video_stride}: {before:,} -> {len(sources):,} sources")
    if args.limit:
        sources = sources[: args.limit]

    rng = np.random.default_rng(args.seed)
    # Sequence members get crop origins seeded by SCENE, not the global stream:
    # per-frame random crops made consecutive frames spatially unaligned, which
    # is useless for temporal training (found 23 Aug 2026).
    scene_rngs: dict[str, np.random.Generator] = {}
    scene_origins: dict[tuple, list] = {}
    index_path = args.dst / "pairs_index.jsonl"
    written, skipped, clipped_records, unsupported = 0, 0, 0, 0
    sdr_clipped_records = 0

    with index_path.open("a" if args.append else "w", encoding="utf-8") as index:
        for position, source in enumerate(sources):
            path = Path(source["path"])
            if path.suffix.lower() not in (".exr", ".tif", ".tiff"):
                unsupported += 1
                continue
            if args.append and (meta_dir / f"{position:07d}_{path.stem}.json").exists():
                skipped += 1
                continue
            linear = load_scene_linear(path, source["encoding_guess"])
            if linear is None:
                skipped += 1
                continue

            if source.get("is_sequence_member"):
                scene = source["scene_id"]
                key = (scene, linear.shape[0], linear.shape[1])
                if key not in scene_origins:
                    srng = scene_rngs.setdefault(scene, np.random.default_rng(
                        (args.seed * 1_000_003) ^ (hash(scene) & 0x7FFFFFFF)))
                    scene_origins[key] = [o for _, o in crops_for(
                        linear, args.crops, args.crop_size, srng)]
                h, w = linear.shape[:2]
                size = args.crop_size
                fixed = []
                for (y, x) in scene_origins[key]:
                    fixed.append((linear[y:y + size, x:x + size]
                                  if args.crops > 1 and size > 0 and h >= size and w >= size
                                  else linear, (y, x)))
                crop_iter = fixed
            else:
                crop_iter = crops_for(linear, args.crops, args.crop_size, rng)

            for crop_idx, (crop, origin) in enumerate(crop_iter):
                frame = resize_frame(crop)
                recipe = None
                if args.sdr_render == "mix":
                    # One grade per shot: every frame (and crop) of a sequence
                    # shares a recipe, as a real clip would. Stills draw per crop.
                    key = (source["scene_id"] if source.get("is_sequence_member")
                           else f"{position}:{crop_idx}")
                    rrng = np.random.default_rng(
                        [args.seed, int(hashlib.sha256(str(key).encode()).hexdigest()[:8], 16)])
                    recipe = sdr_render.draw_render(
                        rrng, ev_jitter=args.ev_jitter,
                        codec_probability=args.codec_probability)
                    sdr = sdr_render.apply_render(frame, recipe, float(args.tonemap_ev))
                else:
                    sdr = make_sdr(frame)
                hdr, stats = encode_hdr_u16(frame, storage)
                # Two different clip statistics live in this record and they
                # must not be confused. stats["clipped_fraction"] is the HDR
                # TARGET above the storage ceiling -- wanted near zero.
                # sdr_clipped_fraction is the SDR INPUT at the top code -- the
                # thing an inverse tone mapper exists to undo, and the thing
                # the -1 EV corpus had none of. verify_dataset puts a floor on
                # the second and a ceiling on the first.
                stats["sdr_clipped_fraction"] = sdr_clipped_fraction(sdr)
                stats["tonemap_ev"] = float(args.tonemap_ev)

                stem = f"{position:07d}_{path.stem}"
                if args.crops > 1:
                    stem = f"{stem}_c{crop_idx}"
                save_png_8bit(sdr, sdr_dir / f"{stem}.png")
                save_png_16bit(hdr, hdr_dir / f"{stem}.png")

                meta = {
                    "stem": stem,
                    "source_path": str(path),
                    "sdr_render": recipe,
                    "source_encoding": source["encoding_guess"],
                    "encoding_reason": source.get("encoding_reason"),
                    "crop_origin": list(origin),
                    "storage": storage.as_dict(),
                    **stats,
                }
                (meta_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

                if stats["clipped_fraction"] > 0.0001:
                    clipped_records += 1
                if stats["sdr_clipped_fraction"] > 0.0001:
                    sdr_clipped_records += 1

                index.write(json.dumps({
                    "asset_id": stem,
                    "scene_id": source["scene_id"],
                    "source_take": path.parent.name if source.get("is_sequence_member") else None,
                    "is_video": bool(source.get("is_sequence_member")),
                    "frame_index": _frame_index(path) if source.get("is_sequence_member") else None,
                    "sdr_path": str((sdr_dir / f"{stem}.png").resolve()),
                    "hdr_path": str((hdr_dir / f"{stem}.png").resolve()),
                    "metadata_path": str((meta_dir / f"{stem}.json").resolve()),
                    "sdr_encoding": "srgb",
                    "hdr_encoding": storage.mode,
                    "peak_nits": stats["peak_nits"],
                    "clipped_fraction": stats["clipped_fraction"],
                    "sdr_clipped_fraction": stats["sdr_clipped_fraction"],
                    "tonemap_ev": float(args.tonemap_ev),
                    "sdr_curve": recipe["curve"] if recipe else "aces",
                    "render_ev": recipe["render_ev"] if recipe else 0.0,
                    "sdr_codec": recipe["codec"] if recipe else "none",
                }) + "\n")
                written += 1

            if (position + 1) % 200 == 0:
                print(f"  {position + 1:,}/{len(sources):,} sources -> {written:,} pairs", flush=True)

    digest = hashlib.sha256(index_path.read_bytes()).hexdigest()
    payload["pairs_index_sha256"] = digest
    payload["pairs_written"] = written
    (args.dst / "_ingest_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n  pairs written        {written:,}")
    print(f"  sources skipped      {skipped:,} (load failures / already present)")
    print(f"  unsupported suffix   {unsupported:,} (only .exr/.tif ingest; .png sources are not HDR)")
    print(f"  records that clip    {clipped_records:,} ({clipped_records / max(written, 1):.2%})"
          f"   <- August corpus was 77.8%")
    print(f"  SDR records that clip {sdr_clipped_records:,} "
          f"({sdr_clipped_records / max(written, 1):.2%})   <- the -1 EV corpus was 0.57%; "
          f"verify_dataset wants this ABOVE its floor")
    print(f"  index sha256         {digest[:16]}...")
    print(f"\nwrote {index_path}\nnext: pipeline/build_manifests.py --pairs-dir {args.dst}")
    return 0


def _frame_index(path: Path) -> int | None:
    import re

    match = re.search(r"(\d{3,8})$", path.stem)
    return int(match.group(1)) if match else None


if __name__ == "__main__":
    raise SystemExit(main())
