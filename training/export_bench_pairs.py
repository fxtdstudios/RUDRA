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

`--condition out-of-generator` re-tone-maps the reference with a Hable curve
and a real H.264 round trip -- neither of which the model's training ever saw.
It is the one condition that asks whether the model generalises past its own
training augmentation, rather than how well it undoes it.

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

import cv2
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


def codec_round_trip_single(rgb01: np.ndarray, crf: int) -> np.ndarray:
    """One frame through a real H.264 encoder and back out.

    8-bit sRGB in, 8-bit sRGB out. A single frame is an I-frame, so this is a
    real 4:2:0 subsample + quantise + block-codec round trip rather than the
    synthetic corruption ``degrade_sdr`` applies. The training augmentation
    never saw H.264, which is what makes it out-of-generator.
    """
    import subprocess
    import tempfile

    eight = np.clip(rgb01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        if not cv2.imwrite(str(work / "in.png"), eight[..., ::-1]):
            raise RuntimeError("could not stage frame for H.264")
        encoded = work / "frame.mp4"
        encode = ["ffmpeg", "-y", "-loglevel", "error",
                  "-framerate", "24", "-i", str(work / "in.png"),
                  "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
                  str(encoded)]
        decode = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(encoded),
                  str(work / "out_%05d.png")]
        for command in (encode, decode):
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed ({command[0]}): {result.stderr}")
        image = cv2.imread(str(work / "out_00001.png"), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError("ffmpeg returned no frame")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def out_of_generator_sdr(hdr_np: np.ndarray, crf: int) -> torch.Tensor:
    """SDR the model never saw: a Hable tone curve + a real H.264 round trip.

    ``hdr_np`` is the reference in network units (nits / 10 000). Convert to
    scene-linear (diffuse white = 1.0), tone-map with Hable instead of the
    ACES curve the corpus used, sRGB-encode, then put it through a real H.264
    round trip. Returns a [1, 3, H, W] float tensor in [0, 1].

    Deterministic: the Hable curve and a single-frame H.264 encode are both
    pure functions of the reference, so this condition is reproducible without
    a seed.
    """
    from training.prepare_training_data import hable_tonemap, oetf_srgb

    scene_linear = hdr_np * (NETWORK_PEAK_NITS / DIFFUSE_WHITE_NITS)
    tone = hable_tonemap(scene_linear)
    srgb = oetf_srgb(tone)
    coded = codec_round_trip_single(srgb, crf)
    return torch.from_numpy(coded.astype(np.float32)).permute(2, 0, 1)[None]


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
    parser.add_argument("--condition", choices=("clean", "hard", "out-of-generator"),
                        default="clean",
                        help="hard applies the eval's seeded camera/codec degradation; "
                             "out-of-generator re-tone-maps the reference with a Hable "
                             "curve and a real H.264 round trip -- SDR the model never saw")
    parser.add_argument("--oog-crf", type=int, default=28,
                        help="H.264 CRF for --condition out-of-generator (default 28)")
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
    parser.add_argument("--write-sdr", action="store_true",
                        help="also write <out>/sdr/<scene>/<asset>.png: the exact "
                             "8-bit frame the model was given, after --condition "
                             "and --max-side. This is what a third-party inverse "
                             "tone mapper consumes, so it is the input half of any "
                             "comparison against published work. Feed this tree to "
                             "their code, then bring their output back with "
                             "training/import_method_output.py.")
    parser.add_argument("--recovery-mode", default="all",
                        choices=("all", "highlights", "shadows", "off"),
                        help="Which arm of the per-pixel gate is allowed to fire. The "
                             "default 'all' is what ships. 'highlights' disables the "
                             "SHADOW arm, which is the ablation the 1 Sep 2026 error "
                             "analysis calls for: on clean frames below 400 nits the "
                             "worst 1%% of pixels sit at a median true luminance of 4 "
                             "nits against 21 frame-wide, so the damage looks like the "
                             "shadow prior firing on content that needs no "
                             "reconstruction. If that is right, --recovery-mode "
                             "highlights should recover most of the clean deficit while "
                             "leaving the hard gain intact.")
    parser.add_argument("--recovery-strength", type=float, default=1.0,
                        help="Global scale on the gate. The oracle sweep wanted ~0.125 on "
                             "clean and ~1.1 on hard; this exposes that dial to the "
                             "benchmark so a constant can be scored honestly.")
    parser.add_argument("--preserve-outside", action="store_true", default=True)
    parser.add_argument("--raw", dest="preserve_outside", action="store_false",
                        help="export the unblended prediction")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16",
                        help="CUDA precision of the model's forward pass. bf16 is what "
                             "every bench before 24 Sep 2026 used; the baseline tree is "
                             "fp32 either way (see predict_image).")
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
    print(f"   precision  : {args.precision}")
    print(f"   device     : {device}")
    announce_storage(f"image/{args.split}", records[0]["hdr_path"])

    out = Path(args.out)
    # The reference frames are the expensive half of an export and they do not
    # depend on the checkpoint, so a second model reuses them: --only-test
    # --test-name v6 adds one tree beside the first model's.
    trees = ([] if args.only_test else ["ref"]) + [args.test_name]
    if not (args.no_baseline or args.only_test):
        trees.append("baseline")
    if args.write_sdr:
        trees.append("sdr")
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
        elif args.condition == "out-of-generator":
            sdr = out_of_generator_sdr(reference, args.oog_crf)
        sdr = sdr.to(device)

        def predict(tile_size: int) -> torch.Tensor:
            return predict_image(model, sdr, preserve_outside=args.preserve_outside,
                                 tile_size=tile_size, overlap=args.tile_overlap,
                                 recovery_mode=args.recovery_mode,
                                 recovery_strength=args.recovery_strength,
                                 bf16=args.precision == "bf16")

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
            # The baseline scored against the model is the one the model was
            # trained over -- same exposure -- or the comparison is between two
            # different renders of the input.
            base = sdr_to_baseline_hdr(sdr, model.corpus_ev)[0].permute(1, 2, 0).float().cpu().numpy()
            path = out / "baseline" / scene / f"{asset}.exr"
            path.parent.mkdir(parents=True, exist_ok=True)
            write_exr(path, to_scene_linear(base), half=True)
        if args.write_sdr:
            # 8-bit sRGB PNG, exactly the pixels the network saw. cv2 wants BGR.
            eight = (sdr[0].permute(1, 2, 0).float().cpu().numpy() * 255.0 + 0.5)
            eight = np.clip(eight, 0, 255).astype(np.uint8)[..., ::-1]
            path = out / "sdr" / scene / f"{asset}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), eight):
                raise RuntimeError(f"failed to write {path}")
        written += 1
        if position % 25 == 0 or position == len(indexed):
            rate = position / max(time.time() - started, 1e-6)
            print(f"   {position:>5}/{len(indexed)}  {rate:5.2f} frames/s  "
                  f"{reference.shape[1]}x{reference.shape[0]}")

    manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "split": args.split, "condition": args.condition,
        "oog_crf": args.oog_crf,
        "test_name": args.test_name,
        "preserve_outside": bool(args.preserve_outside),
        "recovery_mode": args.recovery_mode,
        "recovery_strength": args.recovery_strength,
        "frames": written, "skipped": skipped,
        "units": "scene-linear, diffuse white = 1.0",
        "nits_scale_for_bench": DIFFUSE_WHITE_NITS,
        "reference_clamped_at_network_units": REFERENCE_CEILING,
        "trees": trees,
        "sdr_written": bool(args.write_sdr),
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
