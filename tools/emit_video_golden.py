"""Golden video probes for the native pipeline (Phase 4, step 1).

The oracle is rudra/video.py: probe (ffprobe JSON), input_contract, timing,
has_alpha and decoder_filter, run in convert_video's order (one video stream,
contract, clock, alpha, constant frame size). The clips are small ones made
here with ffmpeg lavfi, each built to hit one rule: tagged and untagged,
already HDR, alpha, interlaced, rotated, anamorphic, odd sizes, variable
timing, RGB, two video streams. The clips are made once and kept (an encoder
upgrade must not churn them); --remake rebuilds them. ffprobe's JSON for each
clip is recorded too, so media/video_probe.cpp is checked on the same input
with or without ffprobe on the machine.

    python tools/emit_video_golden.py            # writes native/tests/golden/video/
    python tools/emit_video_golden.py --remake   # also rebuilds the clips

Step 2 (decode.json): convert_video itself is run on the clips the contract
accepts, with the predictor stubbed and stopped after the last frame, and the
decoder's command and every raw frame read_frame returns are recorded (the
command with the program and the clip's path made portable, each frame as its
SHA-256). The frame bytes depend on the ffmpeg build, so its version line is
recorded with them.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "native" / "tests" / "golden" / "video"
SRC = "testsrc2=s=64x36:r=24000/1001"
TAG709 = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv"]
X264 = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]

# name, ffmpeg arguments after the lavfi input (or a callable building from other clips)
CLIPS = {
    "h264_709.mp4": ["-f", "lavfi", "-i", SRC, "-frames:v", "6", *X264, *TAG709, "-output_ts_offset", "10"],
    "bars_709.mp4": ["-f", "lavfi", "-i", "smptebars=s=64x36:r=24000/1001", "-frames:v", "6", *X264, *TAG709],
    "untagged.mp4": ["-f", "lavfi", "-i", SRC, "-frames:v", "6", *X264],
    "srgb_full.mp4": ["-f", "lavfi", "-i", SRC, "-frames:v", "6", "-c:v", "libx264", "-preset", "ultrafast",
                      "-pix_fmt", "yuvj420p", "-color_primaries", "bt709", "-color_trc", "iec61966-2-1",
                      "-colorspace", "bt709", "-color_range", "pc"],
    "bt2020.mkv": ["-f", "lavfi", "-i", SRC, "-frames:v", "6", *X264, "-color_primaries", "bt2020",
                   "-color_trc", "bt709", "-colorspace", "bt2020nc", "-color_range", "tv"],
    "pq.mkv": ["-f", "lavfi", "-i", SRC, "-frames:v", "6", *X264, "-color_primaries", "bt2020",
               "-color_trc", "smpte2084", "-colorspace", "bt2020nc", "-color_range", "tv"],
    "hlg.mkv": ["-f", "lavfi", "-i", SRC, "-frames:v", "6", *X264, "-color_primaries", "bt2020",
                "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc", "-color_range", "tv"],
    "prores4444_alpha.mov": ["-f", "lavfi", "-i", "testsrc2=s=64x36:r=24,format=yuva444p10le", "-frames:v", "6",
                             "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", *TAG709],
    "interlaced.mp4": ["-f", "lavfi", "-i", "testsrc2=s=64x36:r=25", "-frames:v", "6", *X264,
                       "-flags", "+ildct+ilme", "-x264opts", "tff=1", *TAG709],
    "anamorphic.mp4": ["-f", "lavfi", "-i", SRC, "-frames:v", "6", "-vf", "setsar=4/3", *X264, *TAG709],
    "odd_width.mkv": ["-f", "lavfi", "-i", "testsrc2=s=64x36:r=24", "-frames:v", "6",
                      "-vf", "format=yuv444p,crop=63:36:0:0", "-c:v", "ffv1",
                      "-pix_fmt", "yuv444p", *TAG709],
    "vfr.mkv": ["-f", "lavfi", "-i", SRC, "-frames:v", "6",
                "-vf", "select=not(eq(n\\,3))", "-fps_mode", "passthrough",   # frame 3 dropped: a gap
                *X264, *TAG709],
    "rgb_png.mov": ["-f", "lavfi", "-i", "testsrc2=s=64x36:r=24", "-frames:v", "6", "-c:v", "png",
                    "-pix_fmt", "rgb24"],
    "two_video.mkv": ["-f", "lavfi", "-i", SRC, "-f", "lavfi", "-i", SRC, "-map", "0:v", "-map", "1:v",
                      "-frames:v", "6", *X264, *TAG709],
}
DERIVED = {   # made from another clip with stream copy
    "rotated.mp4": ("h264_709.mp4", ["-display_rotation:v:0", "90"]),
}

ARGS = {
    "default": {},
    "prores4444": {"format": "prores4444"},
    "prores4444_straight": {"format": "prores4444", "alpha_mode": "straight"},
    "prores4444_straight_limited": {"format": "prores4444", "alpha_mode": "straight", "input_range": "limited"},
    "srgb_709": {"input_transfer": "srgb", "input_primaries": "rec709"},
    "srgb_709_limited": {"input_transfer": "srgb", "input_primaries": "rec709", "input_range": "limited"},
    "explicit_709": {"input_transfer": "rec709", "input_primaries": "rec709", "input_matrix": "bt709",
                     "input_range": "limited"},
    "gamma24": {"input_transfer": "gamma24"},
}
CASES = [   # clip, argument variants
    ("h264_709.mp4", ["default", "gamma24"]),
    ("bars_709.mp4", ["default"]),
    ("untagged.mp4", ["default", "srgb_709", "explicit_709"]),
    ("srgb_full.mp4", ["default"]),
    ("bt2020.mkv", ["default"]),
    ("pq.mkv", ["default", "explicit_709"]),
    ("hlg.mkv", ["default"]),
    ("prores4444_alpha.mov", ["default", "prores4444", "prores4444_straight", "prores4444_straight_limited"]),
    ("interlaced.mp4", ["default"]),
    ("rotated.mp4", ["default"]),
    ("anamorphic.mp4", ["default"]),
    ("odd_width.mkv", ["default"]),
    ("vfr.mkv", ["default"]),
    ("rgb_png.mov", ["default", "srgb_709", "srgb_709_limited"]),
    ("two_video.mkv", ["default"]),
]


def make_clips(remake: bool) -> None:
    ffmpeg = shutil.which("ffmpeg")
    clips = OUT / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    for name, cmd in CLIPS.items():
        if remake or not (clips / name).exists():
            subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *cmd,
                            "-map_metadata", "-1", "-fflags", "+bitexact", str(clips / name)], check=True)
    for name, (base, opts) in DERIVED.items():
        if remake or not (clips / name).exists():
            subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *opts, "-i", str(clips / base),
                            "-c", "copy", "-fflags", "+bitexact", str(clips / name)], check=True)


def namespace(overrides: dict) -> SimpleNamespace:
    base = dict(format="hdr10", alpha_mode=None, input_transfer="auto", input_primaries="auto",
                input_matrix="auto", input_range="auto")
    base.update(overrides)
    return SimpleNamespace(**base)


def open_source(info: dict, frame_info: dict, args) -> dict:
    """convert_video's steps from probe to the frame-size check, in its order."""
    from rudra.video import decoder_filter, has_alpha, input_contract, timing
    videos = [s for s in info["streams"] if s["codec_type"] == "video"]
    if len(videos) != 1:
        raise ValueError("Select a source containing exactly one video stream")
    stream, frames = videos[0], frame_info["frames"]
    contract, clock = input_contract(stream, args), timing(stream, frames)
    alpha = has_alpha(stream)
    if any(f["width"] != stream["width"] or f["height"] != stream["height"] for f in frames):
        raise ValueError("Changing frame dimensions are unsupported")
    return {"contract": contract, "clock": clock, "alpha": alpha,
            "width": stream["width"], "height": stream["height"],
            "decoder_filter": decoder_filter(contract, alpha)}


DECODE_CASES = [("h264_709.mp4", "default"), ("bars_709.mp4", "default"), ("untagged.mp4", "explicit_709"), ("srgb_full.mp4", "default"),
                ("bt2020.mkv", "default"), ("rgb_png.mov", "srgb_709"),
                ("prores4444_alpha.mov", "prores4444_straight_limited")]


class _Stop(Exception):
    pass


def decode_record(clip: str, variant: str) -> dict:
    """Runs convert_video to the end of its decode loop and records what it read."""
    import hashlib
    import subprocess
    import tempfile
    import rudra.video as video

    path = OUT / "clips" / clip
    seen: dict = {"frames": []}
    real_popen, real_read = subprocess.Popen, video.read_frame

    def popen(cmd, *a, **k):
        seen["command"] = [Path(cmd[0]).stem, *[clip if c == str(path.resolve()) else c for c in cmd[1:]]]
        return real_popen(cmd, *a, **k)

    def read_frame(stream, count):
        data = real_read(stream, count)
        if data:
            seen["frames"].append(hashlib.sha256(data).hexdigest())
        return data

    class Predictor:
        def __init__(self, *a):
            pass

        def predict(self, rgb, contract, smoother):
            if len(seen["frames"]) == seen["total"]:
                raise _Stop
            return rgb, 0.0, False

    info = video.probe(path)
    seen["total"] = int(video.timing([s for s in info["streams"] if s["codec_type"] == "video"][0],
                                     video.probe(path, frames=True)["frames"])["frames"])
    parser = argparse.ArgumentParser()
    video.add_arguments(parser)
    flags = []
    for key, value in ARGS[variant].items():
        flags += ["--" + key.replace("_", "-"), value]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / ("out.mov" if "prores" in variant else "out.mp4")
        args = parser.parse_args([str(path), "--output", str(out), "--checkpoint", str(path), *flags])
        real_predictor = video.Predictor
        video.subprocess.Popen, video.read_frame, video.Predictor = popen, read_frame, Predictor
        try:
            video.convert_video(args)
        except _Stop:
            pass
        finally:
            video.subprocess.Popen, video.read_frame, video.Predictor = real_popen, real_read, real_predictor
    return {"clip": clip, "args": variant, "command": seen["command"], "frames": seen["frames"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remake", action="store_true")
    remake = parser.parse_args().remake
    from rudra.video import probe

    make_clips(remake)
    probes = OUT / "probes"
    probes.mkdir(exist_ok=True)
    cases = []
    for clip, variants in CASES:
        path = OUT / "clips" / clip
        info, frames = probe(path), probe(path, frames=True)
        for record in (info, frames):
            if "format" in record:
                record["format"]["filename"] = clip   # no machine paths in the goldens
        stem = clip.replace(".", "_")
        for suffix, record in (("streams", info), ("frames", frames)):
            with open(probes / f"{stem}.{suffix}.json", "w", encoding="utf-8", newline="\n") as f:
                json.dump(record, f, indent=1, sort_keys=True)
                f.write("\n")
        for variant in variants:
            try:
                result = {"ok": open_source(info, frames, namespace(ARGS[variant]))}
            except (ValueError, ZeroDivisionError) as error:
                result = {"error": str(error)}
            cases.append({"clip": clip, "args": variant, **result})
    with open(OUT / "index.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"oracle": "rudra/video.py probe, input_contract, timing, has_alpha, decoder_filter",
                   "args": ARGS, "cases": cases}, f, indent=1)
        f.write("\n")
    ffmpeg_version = subprocess.run([shutil.which("ffmpeg"), "-version"], capture_output=True,
                                    text=True).stdout.splitlines()[0]
    decodes = [decode_record(clip, variant) for clip, variant in DECODE_CASES]
    with open(OUT / "decode.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"oracle": "rudra/video.py convert_video decode loop (decoder command, read_frame)",
                   "ffmpeg": ffmpeg_version, "cases": decodes}, f, indent=1)
        f.write("\n")
    print(f"video: {len(cases)} cases on {len(CASES)} clips, {len(decodes)} decodes -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
