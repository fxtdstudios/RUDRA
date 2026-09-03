#!/usr/bin/env python3
"""Bring a third-party inverse tone mapper's output into a `rudra bench` tree.

This closes the loop that `export_bench_pairs.py --write-sdr` opens:

    export_bench_pairs.py --write-sdr   ->  <out>/sdr/<scene>/<asset>.png
    <their code>                        ->  <their dir>/.../<asset>.hdr
    import_method_output.py             ->  <out>/<name>/<scene>/<asset>.exr
    rudra bench <out> --nits-scale 203 --test-dir <name>

so a published method is scored on our split, with our reference, by the same
two metrics. As of 3 Sep 2026 that comparison is the paper's largest gap.

WHY THERE IS AN ALIGNMENT STEP, AND WHAT IT COSTS
-------------------------------------------------
Most published single-image iTMO methods (ExpandNet, HDRCNN, and the reverse
tone mapping line generally) predict RELATIVE radiance. They are trained on
corpora with no absolute-luminance anchor and their output carries no nit
scale, so scoring them directly against a reference in cd/m^2 measures their
exposure guess, not their reconstruction, and the number is meaningless.

So we fit ONE global scalar per frame before scoring. That is standard, and it
is also a free parameter that our own model does not get: RUDRA predicts
absolute nits and is scored as it stands. A comparison table that does not say
so is misleading. Score our own tree through this same script to get the
symmetric row:

    import_method_output.py --out <dir> --from <dir>/test --name test_aligned
    rudra bench <dir> --nits-scale 203 --test-dir test_aligned

and report both -- RUDRA unaligned (as deployed) and RUDRA aligned (as
compared). If the aligned row is much better than the unaligned one, the gap
is exposure, and the paper should say that plainly rather than bury it.

ALIGNMENT MODES
---------------
  median  (default)  k = median(ref / method) over valid pixels. Robust to the
                     clipped highlights that dominate a least-squares fit and
                     are exactly where the two disagree most.
  ls                 k = <method, ref> / <method, method>. Minimises squared
                     linear error, which weights the brightest pixels heavily.
  none               No scaling. Correct only for a method that genuinely
                     predicts absolute luminance -- say so when you use it.

Valid pixels are those where the SDR input was neither clipped nor crushed
(all channels within 5..250 of 255), read from <out>/sdr/. Those are the pixels
where the input actually determined the answer, so they are where an exposure
match is meaningful. Without an sdr/ tree the mask falls back to every finite
pixel and the manifest records that it did.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.delivery.exr import read_exr, write_exr                # noqa: E402

HDR_SUFFIXES = (".exr", ".hdr", ".npy", ".pfm", ".tif", ".tiff")
DIFFUSE_WHITE_NITS = 203.0


def read_hdr(path: Path) -> np.ndarray:
    """Read a linear HDR frame as float32 RGB. Never clamps.

    EXR goes through the repo's own reader, not OpenCV: OpenCV is frequently
    built without OpenEXR, and when it is, cv2.imread returns None and imageio
    guesses a wrong plugin rather than failing -- which reads as "your export
    is corrupt" when nothing is wrong with it.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        image = np.load(path)
    elif suffix == ".exr":
        image, _ = read_exr(path)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if image is None:
            try:
                import imageio.v3 as iio
                image = iio.imread(path)
            except Exception as exc:                              # noqa: BLE001
                raise RuntimeError(f"cannot read {path}: {exc}") from exc
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] > 3:
        image = image[..., :3]
    if not np.isfinite(image).all():
        # A single Inf from a method's own overflow should not kill the import,
        # but it must not silently become a finite number either.
        raise ValueError(f"non-finite pixels in {path}")
    return np.ascontiguousarray(image)


def valid_mask(sdr_path: Path | None, shape: tuple[int, int]) -> np.ndarray | None:
    """Pixels the SDR input actually determined: not clipped, not crushed."""
    if sdr_path is None or not sdr_path.exists():
        return None
    sdr = cv2.imread(str(sdr_path), cv2.IMREAD_UNCHANGED)
    if sdr is None:
        return None
    if sdr.ndim == 2:
        sdr = np.repeat(sdr[..., None], 3, axis=2)
    sdr = sdr[..., :3]
    if sdr.shape[:2] != shape:
        sdr = cv2.resize(sdr, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    lo, hi = sdr.min(axis=2), sdr.max(axis=2)
    return (hi < 250) & (lo > 5)


def fit_scale(method: np.ndarray, ref: np.ndarray,
              mask: np.ndarray | None, mode: str) -> float:
    if mode == "none":
        return 1.0
    m, r = method, ref
    keep = np.isfinite(m) & np.isfinite(r) & (m > 1e-8) & (r > 1e-8)
    if mask is not None:
        keep &= mask[..., None]
    if keep.sum() < 64:
        # Too little agreement to fit on; fall back to every positive pixel
        # rather than inventing a scale from a handful of them.
        keep = np.isfinite(m) & np.isfinite(r) & (m > 1e-8) & (r > 1e-8)
    if keep.sum() < 64:
        return 1.0
    if mode == "median":
        # RATIO OF MEDIANS, not median of ratios. The two agree on smooth
        # content and diverge badly on textured content: if the method's frame
        # is a resampled or slightly blurred version of ours -- and several
        # published methods only run at a fixed resolution, so it usually is --
        # then median(r/m) is a median of per-pixel ratios between a signal and
        # a locally averaged copy of itself, which is biased low by a factor
        # that depends on the texture. A synthetic white-noise frame at a known
        # exposure offset of 37x recovers 14.4 that way and 37 this way.
        denom = float(np.median(m[keep]))
        return float(np.median(r[keep]) / denom) if denom > 1e-12 else 1.0
    if mode == "ls":
        denom = float(np.sum(m[keep] * m[keep]))
        return float(np.sum(m[keep] * r[keep]) / denom) if denom > 0 else 1.0
    raise ValueError(f"unknown alignment mode {mode!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True,
                        help="the bench directory holding ref/ (and ideally sdr/)")
    parser.add_argument("--from", dest="src", required=True,
                        help="directory of the method's HDR output")
    parser.add_argument("--name", required=True,
                        help="tree to write, e.g. expandnet. Becomes --test-dir.")
    parser.add_argument("--align", default="median",
                        choices=("median", "ls", "none"))
    parser.add_argument("--input-nits-scale", type=float, default=1.0,
                        help="multiply their pixels by this before aligning, for a "
                             "method whose units are known (e.g. 1/203 if they emit "
                             "absolute nits and we store diffuse-white-relative). "
                             "Independent of --align.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out, src = Path(args.out), Path(args.src)
    ref_root, sdr_root = out / "ref", out / "sdr"
    if not ref_root.is_dir():
        raise SystemExit(f"error: no reference tree at {ref_root}")
    if args.name in ("ref", "sdr"):
        raise SystemExit("error: --name would overwrite the reference or the inputs")

    # Their output is matched to ours by FILE STEM, not by position: a method
    # that drops or reorders frames must not silently score against the wrong
    # reference. Anything unmatched is reported, never guessed at.
    theirs: dict[str, Path] = {}
    collisions: list[str] = []
    for path in sorted(src.rglob("*")):
        if path.suffix.lower() not in HDR_SUFFIXES or not path.is_file():
            continue
        if path.stem in theirs:
            collisions.append(path.stem)
        theirs[path.stem] = path
    if not theirs:
        raise SystemExit(f"error: no HDR files ({', '.join(HDR_SUFFIXES)}) under {src}")

    refs = [p for p in sorted(ref_root.rglob("*")) if p.suffix.lower() in (".exr", ".npy")]
    if args.limit:
        refs = refs[:args.limit]

    print("\n" + "=" * 66)
    print("   import third-party output")
    print("=" * 66)
    print(f"   reference  : {len(refs)} frames under {ref_root}")
    print(f"   method     : {len(theirs)} frames under {src}")
    print(f"   alignment  : {args.align}"
          + ("" if sdr_root.is_dir() else "   (no sdr/ tree -- mask falls back to all pixels)"))

    started, written, scales = time.time(), 0, []
    missing, resized, failed = [], [], []
    for position, ref_path in enumerate(refs, 1):
        rel = ref_path.relative_to(ref_root)
        # Record paths with forward slashes whatever the OS. These manifests are
        # the provenance for a table in a paper: one written on Windows and read
        # on Linux has to compare equal, or "which frames did the method drop?"
        # gets a different answer depending on where you ask.
        rel_id = rel.as_posix()
        their_path = theirs.get(ref_path.stem)
        if their_path is None:
            missing.append(rel_id)
            continue
        try:
            ref = read_hdr(ref_path)
            method = read_hdr(their_path) * args.input_nits_scale
        except Exception as exc:                                  # noqa: BLE001
            failed.append((rel_id, f"{type(exc).__name__}: {exc}"))
            continue
        if method.shape[:2] != ref.shape[:2]:
            # Several published methods only run at a fixed size. Resampling is
            # recorded per frame, because it is a caveat on that row, not a
            # detail: their result is being judged at a resolution they did not
            # produce.
            resized.append((rel_id, f"{method.shape[:2]} -> {ref.shape[:2]}"))
            method = cv2.resize(method, (ref.shape[1], ref.shape[0]),
                                interpolation=cv2.INTER_AREA)

        mask = valid_mask(sdr_root / rel.parent / f"{ref_path.stem}.png"
                          if sdr_root.is_dir() else None, ref.shape[:2])
        k = fit_scale(method, ref, mask, args.align)
        scales.append(k)

        dest = out / args.name / rel.parent / f"{ref_path.stem}.exr"
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_exr(dest, np.ascontiguousarray(method * k, dtype=np.float32), half=True)
        written += 1
        if position % 25 == 0 or position == len(refs):
            rate = position / max(time.time() - started, 1e-6)
            print(f"   {position:>5}/{len(refs)}  {rate:5.2f} frames/s")

    manifest = {
        "name": args.name,
        "source": str(src.resolve()),
        "reference": str(ref_root.resolve()),
        "align": args.align,
        "input_nits_scale": args.input_nits_scale,
        "mask": "sdr 5..250" if sdr_root.is_dir() else "all finite pixels",
        "frames": written,
        "scale_median": float(np.median(scales)) if scales else None,
        "scale_p10": float(np.percentile(scales, 10)) if scales else None,
        "scale_p90": float(np.percentile(scales, 90)) if scales else None,
        "missing_from_method": missing,
        "resized": resized,
        "failed": failed,
        "stem_collisions": sorted(set(collisions)),
        "units": "scene-linear, diffuse white = 1.0",
        "nits_scale_for_bench": DIFFUSE_WHITE_NITS,
    }
    (out / f"import_{args.name}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 66)
    print(f"   wrote {written} frame(s) to {out / args.name}")
    if scales:
        print(f"   per-frame scale: median {np.median(scales):.4g}, "
              f"p10 {np.percentile(scales, 10):.4g}, p90 {np.percentile(scales, 90):.4g}")
        spread = np.percentile(scales, 90) / max(np.percentile(scales, 10), 1e-12)
        if spread > 100:
            print("   NOTE: the fitted scale spans more than two decades across "
                  "frames.\n         That is a method with no stable exposure, "
                  "not a small correction;\n         say so beside the row.")
    for label, items in (("missing from method", missing),
                         ("resized", [f"{r} ({w})" for r, w in resized]),
                         ("failed", [f"{r}: {w}" for r, w in failed]),
                         ("stem collisions", sorted(set(collisions)))):
        if items:
            print(f"   {label}: {len(items)} -- " + ", ".join(items[:3])
                  + (" ..." if len(items) > 3 else ""))
    print(f"\n   rudra bench {out} --nits-scale {DIFFUSE_WHITE_NITS:.0f} "
          f"--test-dir {args.name}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
