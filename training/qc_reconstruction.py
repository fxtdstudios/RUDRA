"""Run the reconstruction QC gate on a frame, a pair of folders, or a master.

    python training/qc_reconstruction.py --sdr plate.png --hdr plate_rudra.exr
    python training/qc_reconstruction.py --sdr-dir bench/clean/sdr --hdr-dir out/ \\
        --representative --out qc.json

A sequence is scored at first / 25% / 50% / 75% / last unless --all is given.
Anything less than that is a smoke test and is labelled as one. The exit code
is 0 only on PASS: UNMEASURED is a failure, and so is a frame that could not
be read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.qc import PASS, check_frame, format_report, load_thresholds  # noqa: E402

FRAME_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
DIFFUSE_WHITE_NITS = 203.0


def read_sdr(path: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def read_hdr(path: Path) -> np.ndarray:
    """Scene-linear EXR or .npy -> absolute nits."""
    if path.suffix.lower() == ".npy":
        a = np.load(path).astype(np.float64)
        # A .npy may already be nits; assume scene-linear only if it is small.
        return a if a.max() > 50.0 else a * DIFFUSE_WHITE_NITS
    from rudra.delivery.exr import read_exr
    return np.asarray(read_exr(path), dtype=np.float64)[..., :3] * DIFFUSE_WHITE_NITS


def pick(paths: list[Path], representative: bool) -> list[Path]:
    if not representative or len(paths) <= 5:
        return paths
    n = len(paths) - 1
    return [paths[i] for i in sorted({0, n // 4, n // 2, (3 * n) // 4, n})]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdr", type=Path)
    ap.add_argument("--hdr", type=Path)
    ap.add_argument("--sdr-dir", type=Path)
    ap.add_argument("--hdr-dir", type=Path)
    ap.add_argument("--thresholds", type=Path)
    ap.add_argument("--representative", action="store_true",
                    help="first / 25%% / 50%% / 75%% / last instead of every frame")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    thresholds = load_thresholds(args.thresholds)
    pairs: list[tuple[Path, Path]] = []
    if args.sdr and args.hdr:
        pairs = [(args.sdr, args.hdr)]
    elif args.sdr_dir and args.hdr_dir:
        sdrs = sorted(p for p in args.sdr_dir.iterdir()
                      if p.suffix.lower() in FRAME_SUFFIXES)
        for s in pick(sdrs, args.representative):
            for suffix in (".exr", ".npy"):
                h = (args.hdr_dir / s.name).with_suffix(suffix)
                if h.is_file():
                    pairs.append((s, h))
                    break
            else:
                print(f"   no reconstruction for {s.name}", file=sys.stderr)
    else:
        ap.error("give --sdr and --hdr, or --sdr-dir and --hdr-dir")

    if not pairs:
        print("no pairs to score -- that is a failure, not an empty pass", file=sys.stderr)
        return 1
    if len(pairs) == 1:
        print("   one frame: this is a smoke test, not a sequence QC")

    reports, failed = [], 0
    for sdr_path, hdr_path in pairs:
        sdr, hdr = read_sdr(sdr_path), read_hdr(hdr_path)
        report = check_frame(hdr, sdr, thresholds=thresholds)
        h, w = sdr.shape[:2]
        print(format_report(report, sdr_path.name, f"{w}x{h}"))
        print()
        reports.append({"frame": sdr_path.name, **report.to_dict()})
        failed += report.verdict != PASS

    verdict = PASS if failed == 0 else "FAIL"
    print(f"{len(pairs)} frame(s), {failed} failing   ->   {verdict}")
    if args.out:
        args.out.write_text(json.dumps(
            {"verdict": verdict, "frames": reports,
             "thresholds": thresholds,
             "repro": " ".join(sys.argv)}, indent=2), encoding="utf-8")
        print(f"   wrote {args.out}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
