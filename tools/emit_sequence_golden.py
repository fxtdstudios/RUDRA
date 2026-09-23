"""Golden results for the native sequence open (Phase 1 step 10, media/sequence.cpp).

The oracle is ui/sequence.py Sequence.open and describe() on fixture folders
built from the LAYOUTS below: natural order across powers of ten, case, leading
zeros, dot files, a suffix-less ".png", a folder named like a frame, text
files, an empty folder, a text file and a video. Paths are recorded with the
temporary root replaced by <root>, and the native test rebuilds the same
layouts and must give the same names and the same messages.

Video: the Python probes it with ffmpeg; natively it is refused as
Unsupported until libav arrives (Phase 4), so its entry records only that it
was taken for a video.

    python tools/emit_sequence_golden.py        # writes native/tests/golden/sequence/
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
]


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
            if Path(text.strip().strip('"').strip("'")).suffix.lower() in VIDEO_SUFFIXES and text.strip():
                entry["video"] = True
            else:
                try:
                    d = Sequence.open(text).describe()
                    entry.update({"kind": d["kind"], "count": d["count"], "name_field": d["name"],
                                  "path": d["path"].replace(str(root), "<root>"), "names": d["names"]})
                except SequenceError as e:
                    entry["error"] = str(e).replace(str(root), "<root>")
            results.append(entry)
    (OUT / "index.json").write_text(json.dumps({
        "oracle": "ui/sequence.py", "frame_suffixes": sorted(FRAME_SUFFIXES),
        "video_suffixes": sorted(VIDEO_SUFFIXES), "layouts": LAYOUTS, "files": FILES,
        "cases": results}, indent=2))
    for r in results:
        print(r["name"], r.get("names", r.get("error", "video")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
