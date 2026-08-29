#!/usr/bin/env python3
"""Export a held-out split as ref/test EXR pairs, so `rudra bench` can run.

`rudra bench` scores paired directories, and `training/benchmark_hdr.py` says
outright that it does not run inference. Between a trained checkpoint and a
publishable CVVDP number there was therefore nothing at all -- which is why, as
of 28 Aug 2026, the benchmark had never once been run. This is the missing step.

It writes, at native resolution unless told otherwise:

    <out>/ref/<scene>/<asset>.exr        ground truth, decoded and UNCLAMPED
    <out>/test/<scene>/<asset>.exr       the checkpoint's reconstruction
    <out>/baseline/<scene>/<asset>.exr   the analytic inverse-ACES baseline

Everything is scene-linear with diffuse white at 1.0 -- the convention the
Master EXR path already uses -- so the benchmark is told `--nits-scale 203`.

    python training/export_bench_pairs.py --checkpoint <ckpt> \
        --manifest <manifest> --split test --out <dir> --condition clean
    rudra bench <dir> --nits-scale 203
    rudra bench <dir> --nits-scale 203 --test-dir baseline

A second model reuses the reference rather than writing 429 more frames of it:

    python training/export_bench_pairs.py --checkpoint <other> \
        --manifest <manifest> --split test --out <dir> --condition clean \
        --only-test --test-name v6
    rudra bench <dir> --nits-scale 203 --test-dir v6

`--condition hard` applies the same degradation model as the training eval,
seeded per record the same way, so the condition is reproducible run to run.
It is NOT pixel-identical to the eval's hard frames: the eval degrades a
384-pixel crop and this degrades the whole frame, so the same seed lands on a
different realisation. Treat the two as the same *condition*, not the same
images -- the numbers here are not directly comparable with `hard_gain_db`.

The reference is decoded WITHOUT the network's max_hdr clamp. Clamping it would
score the model against a ground truth cropped to the model's own ceiling,
which flatters it for free.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from hashlib import sha1
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.delivery.exr import write_exr                          # noqa: E402
from rudra.sdr2hdr import SDR2HDRNet, sdr_to_baseline_hdr         # noqa: E402
from training.infer_sdr2hdr import predict_image                  # noqa: E402
from training.sdr2hdr_dataset import (NETWORK_PEAK_NITS, announce_storage,  # noqa: E402
                                      degrade_sdr, load_rgb, read_jsonl)

DIFFUSE_WHITE_NITS = 203.0
# Well past log2_extended's 1e6-nit top end, so the reference is effectively
# unclamped while still catching a decode that has gone haywire.
REFERENCE_CEILING = 1000.0
UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def slug(text: str, keep: int = 40) -> str:
    """A filesystem-safe directory name that stays unique.

    scene_id is a UNC path, or a drive path with '::' in it; several scenes
    collapse to the same tail, so the hash is not decoration.
    """
    tail = re.split(r"[\\/]|::", str(text))[-1] or "scene"
    return f"{UNSAFE.sub('_', tail)[:keep]}_{sha1(str(text).encode()).hexdigest()[:6]}"


def to_scene_linear(network_units) -> np.ndarray:
    """network units (nits / 10 000) -> scene-linear, diffuse white = 1.0."""
    return (np.asarray(network_units, dtype=np.float32)
            * (NETWORK_PEAK_NITS / DIFFUSE_WHITE_NITS)).astype(np.float32)


def degrade_like_eval(sdr: torch.Tensor, index: int) -> torch.Tensor:
    """The eval's degradation model, seeded exactly as it seeds it.

    Same model and same seed, but the eval feeds it a 384-pixel crop and this
    feeds it a whole frame, so the realisations differ. Reproducible, not
    identical -- see the module docstring.
    """
    py_state, torch_state = random.getstate(), torch.random.get_rng_state()
    random.seed(24_082_600 + index)
    torch.manual_seed(24_082_600 + index)
    try:
        return degrade_sdr(sdr, 1.0)
    finally:
        random.setstate(py_state)
        torch.random.set_rng_state(torch_state)


def load_model(checkpoint: Path, device: torch.device) -> SDR2HDRNet:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config", {}) or {}
    model = SDR2HDRNet.from_config(config)
    model.load_state_dict(payload.get("model", payload), strict=True)
    step = payload.get("step")
    print(f"   checkpoint : {checkpoint}"
          + (f"  step {step}" if step is not None else "")
          + f"  base_channels {config.get('base_channels', 32)}")
    return model.to(device).eval()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--condition", choices=("clean", "hard"), default="clean",
                        help="hard applies the eval's seeded camera/codec degradation")
    parser.add_argument("--max-side", type=int, default=0,
                        help="0 keeps native resolution")
    parser.add_argument("--tile-size", type=int, default=0,
                        help="0 runs the frame in one pass; falls back on OOM")
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--test-name", default="test",
                        help="directory the prediction goes into (default: test). "
                             "Name a second model here to score it against a "
                             "reference that already exists, e.g. --test-name v6.")
    parser.add_argument("--only-test", action="store_true",
                        help="write only the prediction tree, reusing the ref/ and "
                             "baseline/ an earlier export already produced")
    parser.add_argument("--no-baseline", action="store_true",
                        help="skip the analytic baseline tree")
    parser.add_argument("--preserve-outside", action="store_true", default=True)
    parser.add_argument("--raw", dest="preserve_outside", action="store_false",
                        help="export the unblended prediction")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    records = [r for r in read_jsonl(args.manifest) if r.get("split") == args.split]
    if not records:
        raise SystemExit(f"error: no {args.split!r} records in {args.manifest}")
    # The eval seeds its degradation by position in the unfiltered split, so
    # keep the index from before --limit or the hard frames stop matching.
    indexed = list(enumerate(records))
    if args.limit:
        indexed = indexed[:args.limit]

    device = torch.device(args.device)
    print("\n" + "=" * 66)
    print("   export bench pairs")
    print("=" * 66)
    model = load_model(Path(args.checkpoint), device)
    print(f"   split      : {args.split}  ({len(indexed)} of {len(records)} records)")
    print(f"   condition  : {args.condition}")
    print(f"   device     : {device}")
    announce_storage(f"image/{args.split}", records[0]["hdr_path"])

    out = Path(args.out)
    # The reference frames are the expensive half of an export and they do not
    # depend on the checkpoint, so a second model reuses them: --only-test
    # --test-name v6 adds one tree beside the first model's.
    trees = ([] if args.only_test else ["ref"]) + [args.test_name]
    if not (args.no_baseline or args.only_test):
        trees.append("baseline")
    for tree in trees:
        (out / tree).mkdir(parents=True, exist_ok=True)

    started, written, skipped = time.time(), 0, []
    for position, (index, record) in enumerate(indexed, 1):
        asset = UNSAFE.sub("_", str(record["asset_id"]))
        scene = slug(record.get("scene_id", "scene"))
        try:
            sdr_np = load_rgb(record["sdr_path"], hdr=False)
            hdr_np = load_rgb(record["hdr_path"], hdr=True, ceiling=REFERENCE_CEILING)
        except Exception as exc:                      # noqa: BLE001
            skipped.append((asset, f"{type(exc).__name__}: {exc}"))
            continue
        if sdr_np.shape[:2] != hdr_np.shape[:2]:
            skipped.append((asset, f"geometry {sdr_np.shape} vs {hdr_np.shape}"))
            continue

        sdr = torch.from_numpy(sdr_np).permute(2, 0, 1)[None]
        reference = hdr_np
        if args.max_side and max(sdr.shape[-2:]) > args.max_side:
            scale = args.max_side / max(sdr.shape[-2:])
            size = (max(1, round(sdr.shape[-2] * scale)),
                    max(1, round(sdr.shape[-1] * scale)))
            sdr = torch.nn.functional.interpolate(sdr, size=size, mode="area")
            reference = torch.nn.functional.interpolate(
                torch.from_numpy(hdr_np).permute(2, 0, 1)[None], size=size,
                mode="area")[0].permute(1, 2, 0).numpy()
        if args.condition == "hard":
            sdr = degrade_like_eval(sdr[0], index)[None]
        sdr = sdr.to(device)

        def predict(tile_size: int) -> torch.Tensor:
            return predict_image(model, sdr, preserve_outside=args.preserve_outside,
                                 tile_size=tile_size, overlap=args.tile_overlap,
                                 recovery_mode="all", recovery_strength=1.0)

        try:
            hdr = predict(args.tile_size)
        except Exception as exc:                      # noqa: BLE001
            if "out of memory" not in str(exc).lower():
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            hdr = predict(512)

        prediction = hdr[0].permute(1, 2, 0).float().cpu().numpy()
        written_here = [(args.test_name, prediction)]
        if not args.only_test:
            written_here.insert(0, ("ref", reference))
        for tree, frame in written_here:
            path = out / tree / scene / f"{asset}.exr"
            path.parent.mkdir(parents=True, exist_ok=True)
            write_exr(path, to_scene_linear(frame), half=True)
        if not (args.no_baseline or args.only_test):
            base = sdr_to_baseline_hdr(sdr)[0].permute(1, 2, 0).float().cpu().numpy()
            path = out / "baseline" / scene / f"{asset}.exr"
            path.parent.mkdir(parents=True, exist_ok=True)
            write_exr(path, to_scene_linear(base), half=True)
        written += 1
        if position % 25 == 0 or position == len(indexed):
            rate = position / max(time.time() - started, 1e-6)
            print(f"   {position:>5}/{len(indexed)}  {rate:5.2f} frames/s  "
                  f"{reference.shape[1]}x{reference.shape[0]}")

    manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "split": args.split, "condition": args.condition,
        "test_name": args.test_name,
        "preserve_outside": bool(args.preserve_outside),
        "frames": written, "skipped": skipped,
        "units": "scene-linear, diffuse white = 1.0",
        "nits_scale_for_bench": DIFFUSE_WHITE_NITS,
        "reference_clamped_at_network_units": REFERENCE_CEILING,
        "trees": trees,
    }
    # A second model exports with --only-test into the same directory; if it
    # wrote export.json it would erase the first run's provenance.
    stem = "export" if args.test_name == "test" else f"export_{args.test_name}"
    (out / f"{stem}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 66)
    print(f"   wrote {written} frame(s) to {out}")
    if skipped:
        print(f"   skipped {len(skipped)}: " + ", ".join(a for a, _ in skipped[:5])
              + (" ..." if len(skipped) > 5 else ""))
    suffix = "" if args.test_name == "test" else f" --test-dir {args.test_name}"
    print(f"\n   rudra bench {out} --nits-scale {DIFFUSE_WHITE_NITS:.0f}{suffix}")
    if not (args.no_baseline or args.only_test):
        print(f"   rudra bench {out} --nits-scale {DIFFUSE_WHITE_NITS:.0f} --test-dir baseline")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
