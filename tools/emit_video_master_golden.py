"""Golden video mastering and spool frames for the native pipeline (Phase 4, step 4).

The oracle is rudra/video.py convert_video between the predictor and the
encoder: each HDR frame through profiles.encode_master for the delivery
format, the finite checks, the per-frame MaxCLL and frame average, the 16-bit
PNG spool frame (PQ or HLG code, BGR for OpenCV, the decoded alpha appended)
and the ceiled MaxCLL and MaxFALL handed to the encoder. convert_video runs
for real on the video goldens' clips with the predictor replaced by the step 3
goldens' HDR frames (native/tests/golden/video_predict), and is stopped at
encode_command, where the spool it wrote is copied out with the two ceilings.
The per-frame floats are computed here with the same numpy expressions the
loop uses, on the same arrays.

    python tools/emit_video_master_golden.py   # writes native/tests/golden/video_master/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CLIPS = REPO / "native" / "tests" / "golden" / "video" / "clips"
PREDICT = REPO / "native" / "tests" / "golden" / "video_predict" / "frames"
OUT = REPO / "native" / "tests" / "golden" / "video_master"
HDR = [f"tiled_cut_{i:02d}.hdr.npy" for i in range(6)]

# name: clip, output suffix, convert_video flags
RUNS = {
    "hdr10": ("h264_709.mp4", ".mp4", ["--format", "hdr10"]),
    "hdr10_4000_knee": ("h264_709.mp4", ".mp4", ["--format", "hdr10", "--peak-nits", "4000", "--knee-nits", "1000"]),
    "hlg": ("h264_709.mp4", ".mp4", ["--format", "hlg"]),
    "hlg_600": ("h264_709.mp4", ".mp4", ["--format", "hlg", "--peak-nits", "600"]),
    "prores4444_alpha": ("prores4444_alpha.mov", ".mov",
                         ["--format", "prores4444", "--alpha-mode", "straight", "--input-range", "limited"]),
}
MEAN_SIZES = [5, 8, 9, 127, 128, 129, 1000, 2304, 8191, 8193, 100003, 1920 * 1080]


class _Stop(Exception):
    pass


def run(name: str, clip: str, suffix: str, flags: list[str]) -> dict:
    import rudra.video as video
    from rudra.delivery.profiles import encode_master

    hdrs = [np.load(PREDICT / f) for f in HDR]
    calls = {"n": 0}
    captured: dict = {}

    class Predictor:
        def __init__(self, *a):
            pass

        def predict(self, rgb, contract, smoother):
            hdr = hdrs[calls["n"] % len(hdrs)]
            calls["n"] += 1
            return hdr, 1.0, calls["n"] == 1

    def encode_command(args, source, clock, spool, output, max_cll, max_fall):
        captured["max_cll"], captured["max_fall"] = max_cll, max_fall
        captured["spool"] = sorted(p for p in Path(spool).glob("*.png"))
        for p in captured["spool"]:
            shutil.copyfile(p, OUT / "spool" / f"{name}_{p.name}")
        raise _Stop

    parser = argparse.ArgumentParser()
    video.add_arguments(parser)
    real = video.Predictor, video.encode_command
    with tempfile.TemporaryDirectory() as tmp:
        args = parser.parse_args([str(CLIPS / clip), "--output", str(Path(tmp) / ("out" + suffix)),
                                  "--checkpoint", str(CLIPS / clip), *flags])
        video.Predictor, video.encode_command = Predictor, encode_command
        try:
            video.convert_video(args)
        except _Stop:
            pass
        finally:
            video.Predictor, video.encode_command = real
    frames = []
    for i in range(calls["n"]):
        pq, mastered = encode_master(hdrs[i % len(hdrs)], args.format, args.peak_nits, args.knee_nits)
        maxrgb = mastered.max(axis=2)
        png = f"{name}_{i:08d}.png"
        frames.append({"hdr": HDR[i % len(HDR)], "png": png,
                       "png_sha256": hashlib.sha256((OUT / "spool" / png).read_bytes()).hexdigest(),
                       "max_cll": float(maxrgb.max()), "frame_average": float(maxrgb.mean())})
    return {"name": name, "clip": clip, "format": args.format, "peak_nits": args.peak_nits,
            "knee_nits": args.knee_nits, "alpha": "prores4444" in name, "frames": frames,
            "max_cll": captured["max_cll"], "max_fall": captured["max_fall"]}


def main() -> int:
    import cv2
    shutil.rmtree(OUT / "spool", ignore_errors=True)
    (OUT / "spool").mkdir(parents=True)
    runs = [run(name, *spec) for name, spec in RUNS.items()]
    means = []
    for n in MEAN_SIZES:   # np.mean of float32, as maxrgb.mean() is taken; a hash both sides can make
        i = np.arange(n, dtype=np.uint64)
        a = (((i * np.uint64(2654435761)) % np.uint64(2 ** 32)).astype(np.float64) / 2 ** 32 * 1000).astype(np.float32)
        means.append({"n": n, "mean": float(a.mean()), "max": float(a.max())})
    with open(OUT / "index.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"oracle": "rudra/video.py convert_video frame loop (encode_master, spool, MaxCLL/MaxFALL)",
                   "opencv": cv2.__version__, "numpy": np.__version__, "runs": runs, "means": means}, f, indent=1)
        f.write("\n")
    print(f"video_master: {sum(len(r['frames']) for r in runs)} spool frames in {len(runs)} runs -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
