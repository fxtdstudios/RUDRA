"""Golden results for the native sequence open (Phase 1 step 10, media/sequence.cpp).

The oracle is ui/sequence.py Sequence.open and describe() on fixture folders
built from the LAYOUTS below: natural order across powers of ten, case, leading
zeros, dot files, a suffix-less ".png", a folder named like a frame, text
files, an empty folder, a text file and a video. Paths are recorded with the
temporary root replaced by <root>, and the native test rebuilds the same
layouts and must give the same names and the same messages.

Video (Phase 4, step 11): the Python counts a video's packets with ffprobe
and extracts a frame by seeking; the empty "clips" are refused for having no
frames, and two real clips from the video goldens are opened, their frames 0
and 4 extracted, and the PNGs' digests recorded with the ffmpeg that made
them (the native test compares the bytes when it runs the same build).

    python tools/emit_sequence_golden.py        # writes native/tests/golden/sequence/
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ui.sequence import FRAME_SUFFIXES, VIDEO_SUFFIXES, Sequence, SequenceError  # noqa: E402

OUT = REPO / "native" / "tests" / "golden" / "sequence"

# name -> list of entries; a trailing "/" is a directory, anything else an empty file.
LAYOUTS = {
    "shot_A": ["frame_1.png", "frame_2.png", "frame_10.png", "Frame_3.PNG", "frame_007.png",
               "frame_100.jpg", "frame_20.JPEG", "a9.tif", "a10b2.tif", "a10b10.tiff", "10.bmp", "2.webp",
               "b.png", ".hidden.png", ".png", "trailing.", "frame_5.png.bak", "notes.txt", "sub.png/"],
    "empty": [],
    "no_frames": ["notes.txt", "readme.md", ".png", "dir.jpg/"],
    "single": ["only.tif"],
}
FILES = ["notes.txt", "clip.MOV", "clip.mp4"]
CLIPS = REPO / "native" / "tests" / "golden" / "video" / "clips"
REAL = {"real.mp4": "h264_709.mp4", "alpha clip.mov": "prores4444_alpha.mov"}   # a space, as real names have

CASES = [
    ("folder", "<root>/shot_A"),
    ("folder_quoted", '  "<root>/shot_A"  '),
    ("folder_single_quoted_slash", "'<root>/shot_A/'"),
    ("folder_messy", "<root>//shot_A/./"),
    ("single", "<root>/single"),
    ("empty", "<root>/empty"),
    ("no_frames", "<root>/no_frames"),
    ("text_file", "<root>/notes.txt"),
    ("missing", "<root>/missing"),
    ("missing_nested", "<root>/nope//deeper/"),
    ("blank", ""),
    ("whitespace", "  \t "),
    ("video_upper", "<root>/clip.MOV"),
    ("video", "<root>/clip.mp4"),
    ("video_real", "<root>/real.mp4"),
    ("video_real_quoted", '"<root>/alpha clip.mov"'),
]


def portable(text: str, root: Path) -> str:
    """<root> for the temporary folder, and '/' throughout, so every OS records the same golden."""
    return text.replace(str(root), "<root>").replace("\\", "/")


def build(root: Path) -> None:
    for folder, entries in LAYOUTS.items():
        (root / folder).mkdir()
        for e in entries:
            if e.endswith("/"):
                (root / folder / e[:-1]).mkdir()
            else:
                (root / folder / e).write_bytes(b"")
    for f in FILES:
        (root / f).write_bytes(b"")
    for name, clip in REAL.items():
        shutil.copyfile(CLIPS / clip, root / name)


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        build(root)
        for name, raw in CASES:
            text = raw.replace("<root>", str(root))
            entry = {"name": name, "raw": raw}
            try:
                seq = Sequence.open(text)
                d = seq.describe()
                entry.update({"kind": d["kind"], "count": d["count"], "name_field": d["name"],
                              "path": portable(d["path"], root), "names": d["names"]})
                if d["kind"] == "video":
                    entry["fps"] = d["fps"]
                    entry["frame_sha256"] = {str(i): hashlib.sha256(seq.frame_bytes(i)).hexdigest() for i in (0, 4)}
                    try:
                        seq.frame_bytes(d["count"])
                    except SequenceError as e:
                        entry["outside_error"] = str(e)
            except SequenceError as e:
                entry["error"] = portable(str(e), root)
            results.append(entry)
    # The real clips' entries depend on the ffmpeg that read them: kept apart, so
    # index.json is the same on every machine with ffmpeg on PATH.
    ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    videos = [r for r in results if r["name"].startswith("video_real")]
    results = [r for r in results if not r["name"].startswith("video_real")]
    (OUT / "video.json").write_text(json.dumps({"oracle": "ui/sequence.py", "ffmpeg": ffmpeg, "clips": REAL,
                                                "cases": videos}, indent=2), encoding="utf-8", newline="\n")
    (OUT / "index.json").write_text(json.dumps({
        "oracle": "ui/sequence.py", "frame_suffixes": sorted(FRAME_SUFFIXES),
        "video_suffixes": sorted(VIDEO_SUFFIXES), "layouts": LAYOUTS, "files": FILES,
        "cases": results}, indent=2), encoding="utf-8", newline="\n")
    for r in results + videos:
        print(r["name"], r.get("names", r.get("error"))[:3] if "names" in r else r.get("error"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
