"""Master RUDRA float TIFF output to a Rec.2020/PQ HDR10 image or video.

Examples:
  python training/export_hdr10.py outputs/sdr2hdr_50k/input.tif --peak-nits 1000
  python training/export_hdr10.py outputs/clip --output outputs/clip_hdr10.mp4 --fps 24

The input must be a float TIFF or a directory containing a TIFF sequence using
RUDRA's contract: scene-linear RGB, with 1.0 equal to 10,000 nits.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.hdr10 import master_to_pq  # noqa: E402
from rudra.delivery import metadata as dm  # noqa: E402


def read_linear_tiff(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read {path}")
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected an RGB TIFF, got shape {image.shape} from {path}")
    if not np.issubdtype(image.dtype, np.floating):
        raise ValueError(f"Expected a float TIFF, got {image.dtype} from {path}")
    return cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32)


def write_pq_png(path: Path, pq_rgb: np.ndarray) -> None:
    encoded = (np.clip(pq_rgb, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    if not cv2.imwrite(str(path), cv2.cvtColor(encoded, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write {path}")


def x265_master_display(peak_nits: float, min_nits: float) -> str:
    # BT.2020 primaries and D65 white in the integer units required by x265.
    peak_units = int(round(peak_nits * 10000.0))
    min_units = int(round(min_nits * 10000.0))
    return (
        f"G(8500,39850)B(6550,2300)R(35400,14600)"
        f"WP(15635,16450)L({peak_units},{min_units})"
    )


def encode_video(pq_frames: list[np.ndarray], output: Path, fps: float,
                 peak_nits: float, min_nits: float, max_cll: int,
                 max_fall: int, crf: int, preset: str) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH; it is required for HDR10 video")
    height, width = pq_frames[0].shape[:2]
    params = ":".join((
        "hdr-opt=1", "repeat-headers=1", "colorprim=bt2020",
        "transfer=smpte2084", "colormatrix=bt2020nc",
        f"master-display={x265_master_display(peak_nits, min_nits)}",
        f"max-cll={max_cll},{max_fall}",
    ))
    command = [
        ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb48le",
        "-s:v", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-vf", (
            "zscale=matrixin=gbr:transferin=smpte2084:primariesin=2020:rangein=full:"
            "matrix=2020_ncl:transfer=smpte2084:primaries=2020:range=limited,"
            "format=yuv420p10le"
        ),
        "-an", "-c:v", "libx265", "-preset", preset, "-crf", str(crf),
        "-x265-params", params,
        "-color_primaries", "bt2020", "-color_trc", "smpte2084",
        "-colorspace", "bt2020nc", "-color_range", "tv", "-tag:v", "hvc1",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in pq_frames:
            packed = (np.clip(frame, 0, 1) * 65535.0 + 0.5).astype("<u2")
            process.stdin.write(packed.tobytes())
        process.stdin.close()
        return_code = process.wait()
    except BaseException:
        process.kill()
        raise
    if return_code:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Float TIFF or TIFF sequence directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--peak-nits", type=float, default=1000.0)
    parser.add_argument("--knee-nits", type=float)
    parser.add_argument("--min-nits", type=float, default=0.005)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--still-duration", type=float, default=5.0,
                        help="Seconds to hold one TIFF when exporting video")
    parser.add_argument("--crf", type=int, default=12)
    parser.add_argument("--preset", default="slow")
    parser.add_argument("--dynamic-metadata", action="store_true",
                        help="also write Dolby Vision L1 generate-JSON, HDR10+ "
                             "scene JSON and the RUDRA analysis sidecar")
    args = parser.parse_args()

    frames = sorted(args.input.glob("*.tif")) if args.input.is_dir() else [args.input]
    if not frames or not all(path.exists() for path in frames):
        raise FileNotFoundError(f"No TIFF input found at {args.input}")
    output = args.output or (
        args.input.with_name(args.input.stem + "_pq16.png") if args.input.is_file()
        else args.input.with_name(args.input.name + "_hdr10.mp4")
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    pq_frames: list[np.ndarray] = []
    frame_stats = []
    for index, path in enumerate(frames):
        pq, mastered = master_to_pq(read_linear_tiff(path), args.peak_nits, args.knee_nits)
        pq_frames.append(pq)
        # CTA-861.3 defines MaxCLL/MaxFALL on max(R,G,B) per pixel, NOT on
        # luminance — the previous luma-weighted computation understated both
        # (a saturated red highlight at 1000 nits read as ~263). The delivery
        # analyzer is the single implementation of that convention.
        frame_stats.append(dm.analyze_frame(mastered, index))

    max_cll, max_fall = dm.maxcll_maxfall(frame_stats)
    encoded_frames = pq_frames
    if output.suffix.lower() == ".png":
        if len(pq_frames) != 1:
            raise ValueError("PNG output requires exactly one input frame")
        write_pq_png(output, pq_frames[0])
    elif output.suffix.lower() in {".mp4", ".mov", ".mkv"}:
        if len(pq_frames) == 1:
            if args.still_duration <= 0:
                raise ValueError("still-duration must be greater than zero")
            encoded_frames = pq_frames * max(1, int(round(args.fps * args.still_duration)))
        encode_video(encoded_frames, output, args.fps, args.peak_nits, args.min_nits,
                     max_cll, max_fall, args.crf, args.preset)
    else:
        raise ValueError("Output extension must be .png, .mp4, .mov, or .mkv")

    metadata = {
        "source": str(args.input.resolve()),
        "output": str(output.resolve()),
        "input_encoding": "scene-linear Rec.2020 RGB; 1.0 = 10000 nits",
        "output_encoding": "Rec.2020 RGB with SMPTE ST 2084 (PQ)",
        "mastering_peak_nits": args.peak_nits,
        "mastering_min_nits": args.min_nits,
        "max_cll": max_cll,
        "max_fall": max_fall,
        "source_frames": len(frames),
        "encoded_frames": len(encoded_frames),
        "fps": args.fps if output.suffix.lower() != ".png" else None,
        "note": "PNG carries PQ sample values but not reliable HDR10 signalling; use the video container for HDR playback.",
    }
    sidecar = output.with_name(output.stem + "_metadata.json")
    sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.dynamic_metadata:
        dynamic_paths = dm.write_all_sidecars(
            frame_stats, output.with_suffix(""), mastering_peak_nits=args.peak_nits)
        metadata["dynamic_metadata"] = {k: str(v) for k, v in dynamic_paths.items()}
        sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
