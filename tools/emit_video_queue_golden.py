"""Golden video queues for the native pipeline (Phase 4, step 9).

The oracle is rudra/batch.py run_queue with rudra/video.py convert_video on
the shipped checkpoint: a three-clip queue (HDR10 with audio, HLG smoothed,
ProRes 4444 with alpha). Run twice in a scratch copy: once straight through,
keeping the final state; and once stopped as Ctrl+C stops it, at the second
clip's first frame, keeping the whole project as it was left (the state with
the second job interrupted, the first job's published master and report).
The native test resumes that project and must finish it to the Python's final
state; it also stops a native run the same way and must leave the Python's
interrupted state. Artifact digests depend on the reports' paths and times,
so the test holds them to the files on disk rather than to these.

    python tools/emit_video_queue_golden.py   # writes native/tests/golden/video_queue/
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CLIPS = REPO / "native" / "tests" / "golden" / "video" / "clips"
OUT = REPO / "native" / "tests" / "golden" / "video_queue"
CHECKPOINT = REPO / "checkpoints" / "sdr2hdr_shadow_v1.pt"

QUEUE = {
    "version": 1,
    "defaults": {"checkpoint": "sdr2hdr_shadow_v1.pt", "device": "cpu"},
    "jobs": [
        {"input": "h264_709_audio.mp4", "output": "out/a_hdr10.mp4", "options": {"format": "hdr10"}},
        {"input": "bars_709.mp4", "output": "out/b_hlg.mp4",
         "options": {"format": "hlg", "shadow_smoothing": 0.5}},
        {"input": "prores4444_alpha.mov", "output": "out/c_prores.mov",
         "options": {"format": "prores4444", "alpha_mode": "straight", "input_range": "limited", "audio": "none"}},
    ],
}


def project(where: Path) -> Path:
    where.mkdir(parents=True, exist_ok=True)
    for job in QUEUE["jobs"]:
        shutil.copyfile(CLIPS / job["input"], where / job["input"])
    shutil.copyfile(CHECKPOINT, where / CHECKPOINT.name)
    shutil.copyfile(OUT / "queue.json", where / "queue.json")
    return where / "queue.json"


def main() -> int:
    import rudra.video as video
    from rudra.batch import run_queue

    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    with open(OUT / "queue.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(QUEUE, f, indent=2)
        f.write("\n")

    with tempfile.TemporaryDirectory() as tmp:
        queue = project(Path(tmp) / "full")
        code = run_queue(queue)
        full = json.loads(Path(str(queue) + ".state.json").read_text(encoding="utf-8"))

    real = video.convert_video
    calls = {"n": 0}

    def convert(args, progress=None):
        calls["n"] += 1
        if calls["n"] == 2:
            def stop(value):
                if value.get("phase") == "inference":
                    raise KeyboardInterrupt
                progress(value)
            return real(args, progress=stop)
        return real(args, progress=progress)

    with tempfile.TemporaryDirectory() as tmp:
        queue = project(Path(tmp) / "stopped")
        video.convert_video = convert
        try:
            run_queue(queue)
            raise SystemExit("the queue was expected to stop")
        except KeyboardInterrupt:
            pass
        finally:
            video.convert_video = real
        left = OUT / "interrupted"
        left.mkdir()
        shutil.copyfile(Path(str(queue) + ".state.json"), left / "queue.json.state.json")
        shutil.copytree(queue.parent / "out", left / "out")
        interrupted = json.loads((left / "queue.json.state.json").read_text(encoding="utf-8"))

    with open(OUT / "index.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"oracle": "rudra/batch.py run_queue with rudra/video.py convert_video", "checkpoint": CHECKPOINT.name,
                   "full_exit": code, "full": full, "interrupted": interrupted}, f, indent=1)
        f.write("\n")
    print(f"video_queue: full run exit {code}, interrupted at job 2 -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
