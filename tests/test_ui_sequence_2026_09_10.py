"""Opening a shot by path: a folder of frames, or a video file.

RUDRA Studio could always PLAY a sequence -- rail, scrubber, transport and
wipe all worked on a list. What it could not do was ACQUIRE one: every frame
had to be dragged onto the window and a dropped .mov did nothing. ui/sequence.py
closes that, and the two ways it can be quietly wrong are worth a test each.

The first is ORDER. Lexicographic sort puts frame_10 before frame_2, which
reverses part of every sequence that crosses a power of ten and reports no
error at all -- the shot simply plays wrong.

The second is FRAME ACCURACY. Frames are extracted lazily by seeking rather
than by decoding everything before them, because a 900-frame plate must open
now and not in a minute. A seek that lands on the previous keyframe instead of
the requested frame gives you a picture -- just not that one -- and the only
place it shows is a stutter at GOP boundaries. So the video test encodes with a
known GOP and compares the seeked frames against a full decode, on both sides
of a boundary.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for path in (str(REPO), str(REPO / "ui")):
    if path not in sys.path:
        sys.path.insert(0, path)

sequence = pytest.importorskip("sequence", reason="ui/sequence.py")
Image = pytest.importorskip("PIL.Image", reason="Pillow")

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


@pytest.fixture()
def frames_dir(tmp_path):
    folder = tmp_path / "shot_010"
    folder.mkdir()
    for n in (1, 2, 10, 11):
        Image.new("RGB", (32, 24), (n * 8, 0, 0)).save(folder / f"frame_{n}.png")
    (folder / "notes.txt").write_text("not a frame", encoding="utf-8")
    return folder


def test_natural_order_survives_the_tens_boundary():
    names = ["f_10.png", "f_2.png", "f_1.png", "f_11.png"]
    assert sorted(names, key=sequence.natural_key) == [
        "f_1.png", "f_2.png", "f_10.png", "f_11.png"]


def test_folder_opens_in_shot_order_and_ignores_non_frames(frames_dir):
    shot = sequence.Sequence.open(str(frames_dir))
    assert shot.kind == "frames"
    assert shot.count == 4
    assert [shot.name_of(i) for i in range(4)] == [
        "frame_1.png", "frame_2.png", "frame_10.png", "frame_11.png"]


def test_folder_frame_bytes_are_the_file_on_disk(frames_dir):
    shot = sequence.Sequence.open(str(frames_dir))
    assert shot.frame_bytes(0) == (frames_dir / "frame_1.png").read_bytes()


def test_describe_carries_every_frame_name(frames_dir):
    described = sequence.Sequence.open(str(frames_dir)).describe()
    # The page names its masters from these. A made-up "folder 12" would put
    # twelve different frames into one EXR file name.
    assert described["names"][0] == "frame_1.png"
    assert len(described["names"]) == described["count"] == 4


def test_quoted_and_padded_paths_open(frames_dir):
    shot = sequence.Sequence.open(f'  "{frames_dir}"  ')
    assert shot.count == 4


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_path_is_a_user_error(bad):
    with pytest.raises(sequence.SequenceError):
        sequence.Sequence.open(bad)


def test_missing_path_names_itself(tmp_path):
    with pytest.raises(sequence.SequenceError, match="Nothing at"):
        sequence.Sequence.open(str(tmp_path / "nope"))


def test_empty_folder_says_what_it_looked_for(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(sequence.SequenceError, match="No frames"):
        sequence.Sequence.open(str(tmp_path / "empty"))


def test_a_stray_file_is_not_a_shot(frames_dir):
    with pytest.raises(sequence.SequenceError, match="not a folder or a video"):
        sequence.Sequence.open(str(frames_dir / "notes.txt"))


def test_index_outside_the_clip_is_refused(frames_dir):
    shot = sequence.Sequence.open(str(frames_dir))
    with pytest.raises(sequence.SequenceError, match="outside"):
        shot.frame_bytes(99)
    with pytest.raises(sequence.SequenceError):
        shot.frame_bytes(-1)


# -- video ----------------------------------------------------------------
GOP = 12


@pytest.fixture()
def encoded(tmp_path):
    video = tmp_path / "shot.mov"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc=size=64x48:rate=24:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", str(GOP), str(video)],
        check=True)
    return video


@pytest.mark.skipif(not HAS_FFMPEG, reason="needs ffmpeg and ffprobe")
def test_video_count_is_counted_not_believed(encoded):
    shot = sequence.Sequence.open(str(encoded))
    # 24 fps for 2 s. nb_frames is absent or wrong in plenty of professional
    # containers, and a count that is too high shows a scrubber with dead
    # frames on the end, so the probe counts packets.
    assert shot.kind == "video"
    assert shot.count == 48
    assert abs(shot.describe()["fps"] - 24.0) < 1e-6


@pytest.mark.skipif(not HAS_FFMPEG, reason="needs ffmpeg and ffprobe")
def test_video_frames_are_named_by_index(encoded):
    shot = sequence.Sequence.open(str(encoded))
    assert shot.name_of(0) == "shot_000001"
    assert shot.name_of(47) == "shot_000048"


@pytest.mark.skipif(not HAS_FFMPEG, reason="needs ffmpeg and ffprobe")
def test_seeked_frames_match_a_full_decode_across_gop_boundaries(encoded, tmp_path):
    reference = tmp_path / "ref"
    reference.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(encoded),
                    str(reference / "%06d.png")], check=True)
    shot = sequence.Sequence.open(str(encoded))
    # Around a boundary on both sides, plus the ends of the clip.
    for index in (0, 1, GOP - 1, GOP, GOP + 1, 2 * GOP, 47):
        got = np.asarray(
            Image.open(io.BytesIO(shot.frame_bytes(index))).convert("RGB"),
            dtype=np.int16)
        want = np.asarray(
            Image.open(reference / f"{index + 1:06d}.png").convert("RGB"),
            dtype=np.int16)
        assert np.abs(got - want).max() == 0, f"frame {index} is not that frame"


@pytest.mark.skipif(not HAS_FFMPEG, reason="needs ffmpeg and ffprobe")
def test_extraction_is_cached_and_never_half_written(encoded):
    shot = sequence.Sequence.open(str(encoded))
    first = shot.frame_bytes(7)
    assert shot.frame_bytes(7) == first
    cache = shot._cache
    assert (cache / "000007.png").is_file()
    # The temporary the extractor renames from must not survive: a partial PNG
    # left in the cache would be served forever afterwards as a valid frame.
    assert not list(cache.glob("*.tmp.png"))
