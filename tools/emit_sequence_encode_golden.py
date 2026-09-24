"""Golden sequence encodes for the native pipeline (Phase 4, step 8).

The oracle is rudra/delivery/video.py encode_sequence and the `rudra deliver`
command that drives it (rudra/delivery/cli.py _cmd_deliver). Three synthetic
frames in scene-linear units (a lit gradient with a clipped patch, brighter
each frame) are written as .npy into a folder; `rudra deliver` runs on it for
each of the four targets, with the ffmpeg command it built captured, its JSON
report kept and the file's colour tags read back. The pure part is recorded
on its own too: _encode_frame's 16-bit codes for every target from Rec.709,
Rec.2020 and P3 with and without the shoulder.

    python tools/emit_sequence_encode_golden.py   # writes native/tests/golden/sequence_encode/
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "native" / "tests" / "golden" / "sequence_encode"
H, W = 72, 128
TARGETS = ["hdr10", "hlg", "prores422hq", "prores4444"]
FRAME_CASES = [("rec709", True), ("rec2020", False), ("p3d65", True), ("rec2020", True)]


def frame(k: int) -> np.ndarray:
    """Scene-linear, 1.0 = diffuse white (203 nits as `deliver` reads it)."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    f = np.empty((H, W, 3))
    f[..., 0] = x / (W - 1) * 12.0 * (1 + 0.25 * k)
    f[..., 1] = y / (H - 1) * 6.0 + 0.02
    f[..., 2] = 0.3 + 0.2 * k + 0.1 * np.sin(x / 3.0)
    f[4:10, 30:40] = 49.0            # a clipped highlight, far above any peak
    f[20:24, 2:6] = 0.0005           # near black
    return f


def main() -> int:
    from rudra.delivery import cli as cli_mod
    from rudra.delivery import video as video_mod

    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "frames").mkdir(parents=True)
    (OUT / "codes").mkdir()
    for k in range(3):
        np.save(OUT / "frames" / f"shot_{k:04d}.npy", frame(k))

    codes = []
    nits = np.maximum(frame(1) * 203.0, 0.0)
    for target in TARGETS:
        for space, shoulder in FRAME_CASES:
            c = video_mod._encode_frame(nits, video_mod.TARGETS[target], 1000.0, space, shoulder)
            name = f"{target}_{space}_{'shoulder' if shoulder else 'raw'}.npy"
            np.save(OUT / "codes" / name, c)
            codes.append({"target": target, "source_space": space, "shoulder": shoulder, "peak_nits": 1000.0,
                          "input": "frames/shot_0001.npy", "nits_scale": 203.0, "codes": "codes/" + name})

    runs = []
    real_popen = video_mod.subprocess.Popen
    for target in TARGETS:
        seen = {}

        def popen(cmd, *a, **k):
            if isinstance(cmd, list) and "-x265-params" in cmd or "-f" in cmd and "rawvideo" in cmd:
                seen["command"] = list(cmd)
            return real_popen(cmd, *a, **k)
        work = OUT / "work"
        shutil.rmtree(work, ignore_errors=True)
        video_mod.subprocess.Popen = popen
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                cli_mod.main(["deliver", str(OUT / "frames"), "--output", str(work / "shot"), "--target", target,
                              "--fps", "23.976", "--peak-nits", "1000", "--source-space", "rec709"])
        finally:
            video_mod.subprocess.Popen = real_popen
        report = json.loads(out.getvalue())
        produced = Path(report["file"])
        command = [c if c != str(produced) else "OUTPUT" + produced.suffix for c in seen["command"]]
        report["file"] = "OUTPUT" + produced.suffix
        runs.append({"target": target, "fps": 23.976, "source_space": "rec709", "command": command,
                     "report": report, "report_json": json.dumps(report, indent=2),
                     "container_colr": video_mod.container_colr(produced),
                     "write_colr": video_mod._ffmpeg_supports("muxer", "mov", "write_colr"),
                     "prores_metadata": video_mod._ffmpeg_supports("bsf", "prores_metadata", "color_primaries")})
        shutil.rmtree(work)
    ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    with open(OUT / "index.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"oracle": "rudra/delivery/video.py encode_sequence via rudra deliver", "ffmpeg": ffmpeg,
                   "codes": codes, "runs": runs}, f, indent=1)
        f.write("\n")
    print(f"sequence_encode: {len(codes)} frame encodes, {len(runs)} deliveries -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
