"""Golden end-to-end video conversions for the native pipeline (Phase 4, step 7).

The oracle is rudra/video.py convert_video itself, on the shipped checkpoint on
CPU, with nothing replaced: HDR10 of a clip with audio (copied), HLG with
shadow smoothing and a colour-bars clip, and ProRes 4444 with alpha. The
published masters are kept with their sidecar reports; the native test runs
`convert_video` on the same clips with the exported package and holds the
report to these key for key, and the master to this one byte for byte when the
ffmpeg and the runtime are the recording ones, or to its decoded frames.

    python tools/emit_video_convert_golden.py   # writes native/tests/golden/video_convert/
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CLIPS = REPO / "native" / "tests" / "golden" / "video" / "clips"
OUT = REPO / "native" / "tests" / "golden" / "video_convert"
CHECKPOINT = REPO / "checkpoints" / "sdr2hdr_shadow_v1.pt"

CASES = {   # name: clip, output name, flags
    "hdr10_audio": ("h264_709_audio.mp4", "hdr10_audio.mp4", ["--format", "hdr10"]),
    "hlg_smoothed": ("bars_709.mp4", "hlg_smoothed.mp4", ["--format", "hlg", "--shadow-smoothing", "0.5",
                                                          "--peak-nits", "1200", "--tile-size", "32",
                                                          "--tile-overlap", "8"]),
    "prores4444_alpha": ("prores4444_alpha.mov", "prores4444_alpha.mov",
                         ["--format", "prores4444", "--alpha-mode", "straight", "--input-range", "limited",
                          "--audio", "none"]),
}


def main() -> int:
    import torch
    from rudra.video import main as video_main
    torch.set_num_threads(1)
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    cases = []
    for name, (clip, out_name, flags) in CASES.items():
        output = OUT / out_name
        video_main([str(CLIPS / clip), "--output", str(output), "--checkpoint", str(CHECKPOINT), *flags])
        sidecar = output.with_suffix(output.suffix + ".json")
        report = json.loads(sidecar.read_text(encoding="utf-8"))
        sidecar.rename(OUT / f"{name}.report.json")
        cases.append({"name": name, "clip": clip, "output": out_name, "flags": flags,
                      "report": f"{name}.report.json",
                      "master_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
    ffmpeg = subprocess.run([shutil.which("ffmpeg"), "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    with open(OUT / "index.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"oracle": "rudra/video.py convert_video, eager PyTorch %s on CPU" % torch.__version__,
                   "checkpoint": CHECKPOINT.name, "ffmpeg": ffmpeg, "cases": cases}, f, indent=1)
        f.write("\n")
    print(f"video_convert: {len(cases)} conversions -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
