"""``rudra`` — the headless delivery CLI (torch-free).

Escapes ComfyUI: everything the delivery layer does is scriptable from a
shell or a render farm. Model inference still lives in the training scripts
and the ComfyUI node; this CLI covers what happens AFTER radiance exists —
grade, master, measure, export, benchmark.

Subcommands:
  info       print frame statistics (nits, PQ codes, percentiles)
  grade      apply GradeControls (EV / regions / knee / peak) to linear frames
  aces       write ACES 2065-1 container EXR(s) + optional OCIO config
  metadata   analyze frames -> DoVi L1 generate-JSON, HDR10+ scenes, sidecar
  bench      run the PU21-PSNR (+CVVDP when available) paired benchmark

Frame formats: .exr (uncompressed scanline; rudra.delivery.exr) and .npy.
Values are interpreted as linear; --nits-scale converts to absolute cd/m²
(203 for RUDRA's diffuse-white-relative convention, 1 if already nits).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import aces as aces_mod
from . import bench as bench_mod
from . import controls as controls_mod
from . import metadata as metadata_mod
from .exr import write_exr

__all__ = ["main"]


def _frames(path: Path) -> list[Path]:
    if path.is_dir():
        found = sorted(p for p in path.iterdir() if p.suffix.lower() in (".exr", ".npy"))
        if not found:
            raise FileNotFoundError(f"no .exr/.npy frames in {path}")
        return found
    if not path.exists():
        raise FileNotFoundError(path)
    return [path]


def _load(path: Path, nits_scale: float) -> np.ndarray:
    return bench_mod.load_frame(path, nits_scale)


def _cmd_info(args) -> int:
    stats = [metadata_mod.analyze_frame(_load(p, args.nits_scale), i)
             for i, p in enumerate(_frames(args.input))]
    max_cll, max_fall = metadata_mod.maxcll_maxfall(stats)
    print(json.dumps({
        "frames": len(stats),
        "max_cll_nits": max_cll, "max_fall_nits": max_fall,
        "shots": metadata_mod.detect_shots(stats),
        "per_frame": [{"index": s.index, "min": s.min_nits, "avg": s.avg_nits,
                       "max": s.max_nits} for s in stats],
    }, indent=2))
    return 0


def _cmd_metadata(args) -> int:
    stats = [metadata_mod.analyze_frame(_load(p, args.nits_scale), i)
             for i, p in enumerate(_frames(args.input))]
    paths = metadata_mod.write_all_sidecars(
        stats, args.output, mastering_peak_nits=args.peak_nits,
        shot_threshold=args.shot_threshold)
    for kind, path in paths.items():
        print(f"{kind}: {path}")
    return 0


def _parse_regions(specs: list[str], shape_hw, sample_nits) -> list[controls_mod.RegionEV]:
    """--region low_nits:high_nits:ev[:softness] (luminance qualifier) or
       --region-mask path.npy:ev"""
    regions = []
    for spec in specs or []:
        parts = spec.split(":")
        if len(parts) < 3:
            raise SystemExit(f"bad --region spec {spec!r}: need low:high:ev[:softness]")
        low, high, ev = float(parts[0]), float(parts[1]), float(parts[2])
        soft = float(parts[3]) if len(parts) > 3 else 1.0
        mask = controls_mod.qualifier_mask(sample_nits, low, high, soft)
        regions.append(controls_mod.RegionEV(mask=mask, ev=ev, label=spec))
    return regions


def _cmd_grade(args) -> int:
    out_dir = Path(args.output)
    for i, path in enumerate(_frames(args.input)):
        nits = _load(path, args.nits_scale)
        regions = _parse_regions(args.region, nits.shape[:2], nits)
        for mask_spec in args.region_mask or []:
            mask_path, _, ev = mask_spec.rpartition(":")
            mask = np.load(mask_path).astype(np.float32)
            regions.append(controls_mod.RegionEV(mask=mask, ev=float(ev), label=mask_path))
        grade = controls_mod.GradeControls(
            exposure_ev=args.ev, peak_nits=args.peak_nits,
            knee_nits=args.knee_nits, highlight_desat=args.highlight_desat,
            regions=regions)
        graded = controls_mod.apply_grade(nits, grade)
        out = out_dir / (path.stem + "_graded.exr")
        write_exr(out, graded / args.nits_scale, half=not args.float32,
                  attributes={"rudra:grade": json.dumps(grade.describe())})
        print(out)
    return 0


def _cmd_aces(args) -> int:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in _frames(args.input):
        linear = _load(path, 1.0)  # keep stored convention; matrix-only conversion
        out = out_dir / (path.stem + "_aces2065-1.exr")
        aces_mod.write_aces_exr(linear, out, source_space=args.source_space,
                                exposure_scale=args.exposure_scale,
                                provenance={"source": str(path)})
        print(out)
    if args.ocio:
        print(aces_mod.generate_ocio_config(out_dir / "rudra-delivery.ocio"))
    return 0


def _cmd_bench(args) -> int:
    summary = bench_mod.run_benchmark(args.root, nits_scale=args.nits_scale,
                                      output=args.output, limit=args.limit)
    printable = {k: v for k, v in summary.items() if k != "results"}
    print(json.dumps(printable, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rudra", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("input", type=Path, help="frame file or directory (.exr/.npy)")
        p.add_argument("--nits-scale", type=float, default=203.0,
                       help="stored 1.0 equals this many nits (default 203)")

    p = sub.add_parser("info", help="frame statistics")
    common(p)
    p.set_defaults(fn=_cmd_info)

    p = sub.add_parser("metadata", help="DoVi L1 / HDR10+ / sidecar analysis")
    common(p)
    p.add_argument("--output", required=True, help="output stem for sidecars")
    p.add_argument("--peak-nits", type=float, default=1000.0)
    p.add_argument("--shot-threshold", type=float, default=0.35)
    p.set_defaults(fn=_cmd_metadata)

    p = sub.add_parser("grade", help="apply artist grade controls")
    common(p)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--ev", type=float, default=0.0)
    p.add_argument("--peak-nits", type=float, default=1000.0)
    p.add_argument("--knee-nits", type=float, default=None)
    p.add_argument("--highlight-desat", type=float, default=0.0)
    p.add_argument("--region", action="append",
                   help="luminance qualifier low:high:ev[:softness_stops]; repeatable")
    p.add_argument("--region-mask", action="append",
                   help="mask.npy:ev — explicit soft mask; repeatable")
    p.add_argument("--float32", action="store_true", help="write FLOAT instead of HALF")
    p.set_defaults(fn=_cmd_grade)

    p = sub.add_parser("aces", help="export ACES 2065-1 container EXR")
    p.add_argument("input", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--source-space", default="rec2020", choices=["rec2020", "rec709", "p3d65"])
    p.add_argument("--exposure-scale", type=float, default=1.0)
    p.add_argument("--ocio", action="store_true", help="also write the OCIO config")
    p.set_defaults(fn=_cmd_aces)

    p = sub.add_parser("bench", help="paired PU21-PSNR (+CVVDP) benchmark")
    p.add_argument("root", type=Path, help="directory containing ref/ and test/")
    p.add_argument("--nits-scale", type=float, default=1.0)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(fn=_cmd_bench)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
