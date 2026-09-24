"""Golden export QC for the native pipeline (Phase 4, step 6).

The oracle is rudra/video.py quality_check and check_alpha. Good masters are
made by convert_video itself, end to end, with the predictor replaced by the
step 3 goldens' HDR frames: HDR10 with its audio copied and with AAC, HLG,
ProRes 422 and ProRes 4444 with alpha; the qc (and alpha) record each sidecar
carries is kept. Broken masters are derived from the good ones with ffmpeg
(a changed transfer tag, a dropped frame, the audio removed or shifted, the
HDR10 SEI stripped, a wrong ProRes tag, an HDR10 file checked as HLG, light
metadata that disagrees) and quality_check is run on each for its verdict.
ffprobe's JSON of every master is recorded in ffprobe's own key order, so the
native QC is checked on the same input with or without ffprobe.

    python tools/emit_video_qc_golden.py   # writes native/tests/golden/video_qc/
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CLIPS = REPO / "native" / "tests" / "golden" / "video" / "clips"
PREDICT = REPO / "native" / "tests" / "golden" / "video_predict" / "frames"
OUT = REPO / "native" / "tests" / "golden" / "video_qc"
HDR = [f"tiled_cut_{i:02d}.hdr.npy" for i in range(6)]

GOOD = {   # name: clip, output name, convert_video flags
    "hdr10_audio_copy": ("h264_709_audio.mp4", "hdr10_audio_copy.mp4", ["--format", "hdr10"]),
    "hdr10_audio_aac": ("h264_709_audio.mp4", "hdr10_audio_aac.mp4", ["--format", "hdr10", "--audio", "aac"]),
    "hlg": ("h264_709.mp4", "hlg.mp4", ["--format", "hlg"]),
    "prores422": ("bt2020.mkv", "prores422.mov", ["--format", "prores422"]),
    "prores4444_alpha": ("prores4444_alpha.mov", "prores4444_alpha.mov",
                         ["--format", "prores4444", "--alpha-mode", "straight", "--input-range", "limited"]),
}
# name: good master it comes from, ffmpeg arguments between input and output
BROKEN = {
    "wrong_transfer": ("hdr10_audio_copy", ["-c", "copy", "-bsf:v", "hevc_metadata=transfer_characteristics=1"]),
    "frame_dropped": ("hdr10_audio_copy", ["-c", "copy", "-frames:v", "5"]),
    "audio_removed": ("hdr10_audio_copy", ["-c", "copy", "-an"]),
    "sei_stripped": ("hdr10_audio_copy", ["-c:a", "copy", "-c:v", "libx265", "-preset", "ultrafast",
                                          "-x265-params", "repeat-headers=1:log-level=error", "-pix_fmt", "yuv420p10le",
                                          "-color_primaries", "bt2020", "-color_trc", "smpte2084",
                                          "-colorspace", "bt2020nc", "-color_range", "tv", "-tag:v", "hvc1"]),
    "prores_tag": ("prores422", ["-c", "copy", "-tag:v", "apch"]),
}
# name: master, the arguments quality_check is given (audio mode, format, alpha, expected-light change)
CHECKS = {
    "hdr10_audio_copy": ("hdr10_audio_copy", "copy", "hdr10", False, None),
    "hdr10_audio_aac": ("hdr10_audio_aac", "aac", "hdr10", False, None),
    "hlg": ("hlg", "copy", "hlg", False, None),
    "prores422": ("prores422", "copy", "prores422", False, None),
    "prores4444_alpha": ("prores4444_alpha", "copy", "prores4444", True, None),
    "wrong_transfer": ("wrong_transfer", "copy", "hdr10", False, None),
    "frame_dropped": ("frame_dropped", "copy", "hdr10", False, None),
    "audio_removed": ("audio_removed", "copy", "hdr10", False, None),
    "audio_none_requested": ("hdr10_audio_copy", "none", "hdr10", False, None),
    "sei_stripped": ("sei_stripped", "copy", "hdr10", False, None),
    "prores_tag": ("prores_tag", "copy", "prores422", False, None),
    "hdr10_checked_as_hlg": ("hdr10_audio_copy", "copy", "hlg", False, None),
    "light_differs": ("hdr10_audio_copy", "copy", "hdr10", False, {"max_cll": 1, "peak": 1.5}),
    "prores_as_4444": ("prores422", "copy", "prores4444", False, None),
}


def convert(name: str, clip: str, out_name: str, flags: list[str]) -> dict:
    import rudra.video as video
    hdrs = [np.load(PREDICT / f) for f in HDR]
    calls = {"n": 0}

    class Predictor:
        def __init__(self, *a):
            pass

        def predict(self, rgb, contract, smoother):
            hdr = hdrs[calls["n"] % len(hdrs)]
            calls["n"] += 1
            return hdr, 1.0, calls["n"] == 1

    parser = argparse.ArgumentParser()
    video.add_arguments(parser)
    output = OUT / "masters" / out_name
    for p in (output, output.with_suffix(output.suffix + ".json")):
        p.unlink(missing_ok=True)
    args = parser.parse_args([str(CLIPS / clip), "--output", str(output), "--checkpoint", str(CLIPS / clip), *flags])
    real = video.Predictor
    video.Predictor = Predictor
    try:
        video.convert_video(args)
    finally:
        video.Predictor = real
    report = json.loads(output.with_suffix(output.suffix + ".json").read_text(encoding="utf-8"))
    output.with_suffix(output.suffix + ".json").unlink()
    return {"clip": clip, "output": out_name, "report_qc": report["qc"],
            "report_alpha_json": json.dumps(report["qc"].get("alpha"), indent=2), "max_cll": report["max_cll"],
            "max_fall": report["max_fall"], "peak": args.peak_nits, "minimum": args.min_nits,
            "timing": report["timing"]}


def record_probes(path: Path) -> None:
    from rudra.video import probe
    stem = path.name.replace(".", "_")
    for suffix, rec in (("streams", probe(path)), ("frames", probe(path, frames=True))):
        rec.get("format", {})["filename"] = path.name
        with open(OUT / "probes" / f"{stem}.{suffix}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=1)   # ffprobe's own key order: the qc record keeps it
            f.write("\n")


def main() -> int:
    from rudra.video import probe, quality_check
    for d in ("masters", "probes"):
        shutil.rmtree(OUT / d, ignore_errors=True)
        (OUT / d).mkdir(parents=True)
    good = {name: convert(name, *spec) for name, spec in GOOD.items()}
    ffmpeg = shutil.which("ffmpeg")
    files = {name: g["output"] for name, g in good.items()}
    for name, (base, extra) in BROKEN.items():
        src = OUT / "masters" / files[base]
        dst = OUT / "masters" / (name + src.suffix)
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), *extra,
                        "-fflags", "+bitexact", str(dst)], check=True)
        files[name] = dst.name
        good[name] = {**good[base], "output": dst.name}
    for f in files.values():
        record_probes(OUT / "masters" / f)
    checks = []
    for name, (master, audio, fmt, alpha, light) in CHECKS.items():
        g = good[master]
        expected = dict(max_cll=g["max_cll"], max_fall=g["max_fall"], peak=g["peak"], minimum=g["minimum"])
        if light:
            expected = {k: expected[k] + light.get(k, 0) for k in expected}
        source_info = probe(CLIPS / g["clip"])
        try:
            ok = quality_check(source_info, g["timing"], OUT / "masters" / files[master], audio, expected, fmt, alpha)
            result = {"ok": ok, "ok_json": json.dumps(ok, indent=2)}
        except RuntimeError as error:
            result = {"error": str(error)}
        checks.append({"name": name, "master": files[master], "clip": g["clip"], "audio": audio, "format": fmt,
                       "alpha": alpha, "expected_hdr": expected, "clock": g["timing"], **result})
    ffmpeg_version = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    with open(OUT / "index.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"oracle": "rudra/video.py quality_check and check_alpha (via convert_video's sidecar)",
                   "ffmpeg": ffmpeg_version,
                   "converted": [{"name": n, **{k: v for k, v in g.items()}} for n, g in good.items() if n in GOOD],
                   "checks": checks}, f, indent=1)
        f.write("\n")
    print(f"video_qc: {len(GOOD)} masters converted, {len(BROKEN)} broken, {len(checks)} checks -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
