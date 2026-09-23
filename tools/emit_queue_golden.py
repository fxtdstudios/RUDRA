"""Golden queue runs for the native queue (Phase 1 step 9, deliver/queue.cpp).

The oracle is rudra/batch.py run_queue and load_jobs, with the video export
replaced by a deterministic stand-in (the real one arrives natively with
libav in Phase 4). Recorded:

    state_first.json   a fresh run of a 3-job queue whose second job fails
    state_final.json   that state resumed with --retry-failed, all passing
    refusals           load_jobs's message for each malformed queue

The native test runs the same queue with the same stand-in and must write
byte-identical state files, and resumes the Python's state_first.json to the
Python's state_final.json. That is "a queue started by Python resumes in C++
and the other way round": both write the same bytes at every step.

    python tools/emit_queue_golden.py        # writes native/tests/golden/queue/
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra import batch  # noqa: E402

OUT = REPO / "native" / "tests" / "golden" / "queue"

QUEUE = {
    "version": 1,
    "defaults": {"checkpoint": "ckpt.pt", "format": "hdr10", "peak_nits": 1000},
    "jobs": [
        {"input": "a.mov", "output": "out/a.mp4"},
        {"input": "b.mov", "output": "out/b.mp4", "options": {"crf": 14}},
        {"input": "c.mov", "output": "out/c.mov", "options": {"format": "prores422hq"}},
    ],
}
FILES = {"a.mov": b"clip a", "b.mov": b"clip b", "c.mov": b"clip c", "ckpt.pt": b"weights"}

REFUSALS = {
    "bad_version": {"version": 2, "jobs": [{"input": "a.mov", "output": "o.mp4"}]},
    "no_jobs": {"version": 1, "jobs": []},
    "input_in_options": {"version": 1, "defaults": {"checkpoint": "ckpt.pt"},
                         "jobs": [{"input": "a.mov", "output": "o.mp4", "options": {"input": "x"}}]},
    "bad_choice": {"version": 1, "defaults": {"checkpoint": "ckpt.pt"},
                   "jobs": [{"input": "a.mov", "output": "o.mp4", "options": {"format": "dvd"}}]},
    "bad_int": {"version": 1, "defaults": {"checkpoint": "ckpt.pt"},
                "jobs": [{"input": "a.mov", "output": "o.mp4", "options": {"crf": 12.5}}]},
    "unknown_option": {"version": 1, "defaults": {"checkpoint": "ckpt.pt"},
                       "jobs": [{"input": "a.mov", "output": "o.mp4", "options": {"colour": "blue"}}]},
    "no_checkpoint": {"version": 1, "jobs": [{"input": "a.mov", "output": "o.mp4"}]},
    "output_is_source": {"version": 1, "defaults": {"checkpoint": "ckpt.pt"},
                         "jobs": [{"input": "a.mov", "output": "a.mov"}]},
    "two_jobs_one_output": {"version": 1, "defaults": {"checkpoint": "ckpt.pt"},
                            "jobs": [{"input": "a.mov", "output": "o.mp4"}, {"input": "b.mov", "output": "o.mp4"}]},
}


def stand_in(fail: set[str]):
    def convert_video(args, progress):
        if args.input.name in fail:
            raise RuntimeError("synthetic failure")
        progress({"phase": "encoding", "frame": 1, "of": 2})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(b"out:" + args.input.name.encode())
        args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps({"qc": {"passed": True}}))
    return convert_video


def project(root: Path) -> Path:
    for name, data in FILES.items():
        (root / name).write_bytes(data)
    q = root / "queue.json"
    q.write_text(json.dumps(QUEUE, indent=2))
    return q


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "project").mkdir(parents=True)
    project(OUT / "project")
    original = batch.video.convert_video
    try:
        with tempfile.TemporaryDirectory() as tmp:
            q = project(Path(tmp))
            batch.video.convert_video = stand_in({"b.mov"})
            first = batch.run_queue(q)
            shutil.copy(str(q) + ".state.json", OUT / "state_first.json")
            batch.video.convert_video = stand_in(set())
            final = batch.run_queue(q, retry_failed=True)
            shutil.copy(str(q) + ".state.json", OUT / "state_final.json")
    finally:
        batch.video.convert_video = original
    refusals = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        project(root)
        for name, spec in REFUSALS.items():
            q = root / f"{name}.json"
            q.write_text(json.dumps(spec))
            try:
                with contextlib.redirect_stderr(io.StringIO()):   # argparse's usage text
                    batch.load_jobs(q.resolve())
                refusals[name] = {"spec": spec, "message": None}
            except ValueError as e:
                msg = str(e).replace(str(root.resolve()), "<root>")
                refusals[name] = {"spec": spec, "message": msg}
    (OUT / "index.json").write_text(json.dumps({"oracle": "rudra/batch.py", "queue": QUEUE, "files": {k: v.decode() for k, v in FILES.items()},
                                                "first_exit": first, "final_exit": final,
                                                "stand_in": {"fails_first_run": ["b.mov"],
                                                             "progress": {"phase": "encoding", "frame": 1, "of": 2},
                                                             "output": "out:<input name>",
                                                             "sidecar": json.dumps({"qc": {"passed": True}})},
                                                "refusals": refusals}, indent=2))
    print(f"first run exit {first}, resumed exit {final}; {len(refusals)} refusals")
    for k, v in refusals.items():
        print(f"  {k}: {v['message']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
