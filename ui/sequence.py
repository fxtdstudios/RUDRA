"""Open a shot by path: a folder of frames, or a video file.

RUDRA Studio already plays a sequence -- the Frames rail, the scrubber and the
transport all work on a list. What it could not do was ACQUIRE one. Every
frame had to be dragged onto the window, which is fine for three stills and
absurd for a 240-frame plate, and a dropped .mov did nothing at all because
the page expects images.

The server runs on the same machine as the footage, so the fix is not to
upload anything. The page sends a PATH and the server reads it in place. A
1.4 GB ProRes file never crosses the socket; single frames do, already
decoded, in exactly the format /api/frame returns.

Video frames are extracted lazily with ffmpeg, one at a time, into a cache
beside the job. Extracting a whole plate up front would mean a minute of
nothing happening before the first picture appears, and most of the time an
artist looks at a handful of frames and moves on.

No state on disk beyond that cache, and no job survives a restart: this is a
local viewer, not a render farm.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# What the scanner will treat as a frame. Deliberately narrower than the
# training pipeline's list: these are the ones PIL opens without a plugin, and
# an EXR that silently failed to load would look like a black frame rather
# than an error.
FRAME_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mov", ".mp4", ".mxf", ".mkv", ".avi", ".m2ts", ".ts", ".webm", ".m4v"}

_NUMBER = re.compile(r"(\d+)")


def natural_key(name: str) -> list:
    """Sort frame_2.png before frame_10.png.

    Lexicographic order puts 10 before 2, which silently reverses parts of
    every sequence that crosses a power of ten and is the classic way to play
    a shot in the wrong order without anything reporting an error.
    """
    return [int(part) if part.isdigit() else part.lower()
            for part in _NUMBER.split(name)]


class SequenceError(Exception):
    """Something the user can fix, phrased for them rather than for a log."""


class Sequence:
    """One opened shot. Frames come out as encoded bytes, by index."""

    def __init__(self, path: Path):
        self.path = path
        self.kind = "frames"
        self.frames: list[Path] = []
        self.count = 0
        self._cache: Path | None = None
        self._fps: float | None = None

    # -- opening ----------------------------------------------------------
    @classmethod
    def open(cls, raw: str) -> "Sequence":
        if not raw or not raw.strip():
            raise SequenceError("No path given.")
        path = Path(raw.strip().strip('"').strip("'")).expanduser()
        if not path.exists():
            raise SequenceError(f"Nothing at {path}")
        sequence = cls(path)
        if path.is_dir():
            sequence._open_folder()
        else:
            sequence._open_video()
        return sequence

    def _open_folder(self) -> None:
        self.kind = "frames"
        self.frames = sorted(
            (p for p in self.path.iterdir()
             if p.is_file() and p.suffix.lower() in FRAME_SUFFIXES),
            key=lambda p: natural_key(p.name))
        self.count = len(self.frames)
        if not self.count:
            raise SequenceError(
                f"No frames in {self.path.name}. Looked for "
                + " ".join(sorted(FRAME_SUFFIXES)))

    def _open_video(self) -> None:
        if self.path.suffix.lower() not in VIDEO_SUFFIXES:
            raise SequenceError(
                f"{self.path.name} is not a folder or a video RUDRA can read. "
                "Point at a folder of frames, or a "
                + "/".join(sorted(s.lstrip('.') for s in VIDEO_SUFFIXES)) + " file.")
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise SequenceError(
                "ffmpeg is not on PATH, so video cannot be read. Install it "
                "(winget install Gyan.FFmpeg) and restart the server, or "
                "point at a folder of frames instead.")
        self.kind = "video"
        self.count, self._fps = _probe(self.path)
        if not self.count:
            raise SequenceError(f"ffprobe found no video frames in {self.path.name}")
        token = hashlib.sha1(str(self.path.resolve()).encode()).hexdigest()[:12]
        self._cache = Path(tempfile.gettempdir()) / "rudra_seq" / token
        self._cache.mkdir(parents=True, exist_ok=True)

    # -- reading ----------------------------------------------------------
    def name_of(self, index: int) -> str:
        if self.kind == "frames":
            return self.frames[index].name
        return f"{self.path.stem}_{index + 1:06d}"

    def frame_bytes(self, index: int) -> bytes:
        """Encoded image bytes for frame `index`, ready for run_frame()."""
        if not 0 <= index < self.count:
            raise SequenceError(f"frame {index} is outside 0..{self.count - 1}")
        if self.kind == "frames":
            return self.frames[index].read_bytes()

        cached = self._cache / f"{index:06d}.png"
        if not cached.is_file():
            _extract(self.path, index, self._fps or 24.0, cached)
        return cached.read_bytes()

    def describe(self) -> dict:
        # `names` is what the Frames rail shows and what a master is named
        # after, so it has to be the real frame identity, not "folder 12".
        # A few hundred short strings is nothing next to one decoded frame.
        return {"kind": self.kind, "count": self.count,
                "name": self.path.name, "path": str(self.path),
                "fps": self._fps,
                "names": [self.name_of(i) for i in range(self.count)]}


def _probe(video: Path) -> tuple[int, float]:
    """(frame count, fps). Counts packets rather than trusting the header.

    nb_frames is absent or wrong in plenty of professional containers, and a
    count that is too high shows the artist a scrubber with dead frames on the
    end. Packet counting is slower and right.
    """
    def ffprobe(*extra: str) -> str:
        done = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", *extra,
             "-of", "default=nokey=1:noprint_wrappers=1", str(video)],
            capture_output=True, text=True)
        return done.stdout.strip().splitlines()[0] if done.stdout.strip() else ""

    rate = ffprobe("-show_entries", "stream=avg_frame_rate")
    fps = 24.0
    if "/" in rate:
        num, _, den = rate.partition("/")
        try:
            fps = float(num) / float(den) if float(den) else 24.0
        except ValueError:
            fps = 24.0

    counted = ffprobe("-count_packets", "-show_entries", "stream=nb_read_packets")
    try:
        return int(counted), fps
    except ValueError:
        return 0, fps


def _extract(video: Path, index: int, fps: float, target: Path) -> None:
    """One frame, by seeking rather than decoding everything before it."""
    when = index / max(fps, 1e-6)
    partial = target.with_suffix(".tmp.png")
    done = subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         # -ss before -i seeks; -accurate_seek keeps it honest at the frame level.
         "-accurate_seek", "-ss", f"{when:.6f}", "-i", str(video),
         "-frames:v", "1", str(partial)],
        capture_output=True, text=True)
    if done.returncode != 0 or not partial.is_file():
        partial.unlink(missing_ok=True)
        raise SequenceError(
            f"ffmpeg could not read frame {index}: "
            + (done.stderr.strip().splitlines() or ["unknown error"])[-1])
    # Rename last, so a cache entry never exists half-written -- a partial PNG
    # would be served forever afterwards as a valid cached frame.
    partial.replace(target)
