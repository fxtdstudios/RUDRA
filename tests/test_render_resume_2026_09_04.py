"""The HDRI renderer's resume behaviour.

Two bugs on 4 Sep 2026, both in the batched corpus build, both from the same
place -- deciding which panoramas are still to render:

  1. Nothing skipped what a previous batch had done, because nothing removes
     a rendered panorama from --hdri-dir unless --drop-source is on. Batch k
     re-rendered batches 1..k-1. 993 panoramas in 20 batches would have been
     10,500 renders.
  2. The fix made "everything here is already rendered" exit 1, sharing a code
     with "this directory has no panoramas at all". The corpus build aborted
     at batch 0 having done nothing wrong.

These run on empty files: select_sources decides from names and the done
file, and never opens a panorama, so the tests need no EXR decoder. CI has
none -- opencv-python-headless ships without OpenEXR -- which is exactly why
this is tested here and not through main().
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.render_hdri_moves import select_sources  # noqa: E402


def panoramas(root: Path, *names: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"")          # never opened


def test_an_empty_directory_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit):
        select_sources(tmp_path / "empty", None)


def test_a_directory_of_other_files_is_an_error(tmp_path):
    # The Poly Haven fetcher leaves its index beside the panoramas; a
    # directory holding only that is still empty for our purposes.
    root = tmp_path / "src"
    root.mkdir()
    (root / "_polyhaven_index.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        select_sources(root, None)


def test_everything_already_rendered_is_not_an_error(tmp_path):
    root = tmp_path / "src"
    panoramas(root, "a_4k.exr", "b_4k.exr")
    done = tmp_path / "rendered.txt"
    done.write_text("a_4k.exr\nb_4k.exr\n", encoding="utf-8")

    todo, already = select_sources(root, done)
    assert todo == [], "a finished batch must not look like an empty directory"
    assert already == {"a_4k.exr", "b_4k.exr"}


def test_only_the_unrendered_ones_come_back(tmp_path):
    root = tmp_path / "src"
    panoramas(root, "a_4k.exr", "b_4k.exr", "c_4k.hdr")
    done = tmp_path / "rendered.txt"
    done.write_text("b_4k.exr\n", encoding="utf-8")

    todo, already = select_sources(root, done)
    assert [p.name for p in todo] == ["a_4k.exr", "c_4k.hdr"]
    assert already == {"b_4k.exr"}


def test_no_done_file_means_render_everything(tmp_path):
    root = tmp_path / "src"
    panoramas(root, "a_4k.exr", "b_4k.exr")
    assert len(select_sources(root, None)[0]) == 2
    assert len(select_sources(root, tmp_path / "not_written_yet.txt")[0]) == 2


def test_a_blank_line_in_the_done_file_skips_nothing(tmp_path):
    # An empty entry must not match a panorama, and must not be counted.
    root = tmp_path / "src"
    panoramas(root, "a_4k.exr")
    done = tmp_path / "rendered.txt"
    done.write_text("\n\n  \n", encoding="utf-8")
    todo, already = select_sources(root, done)
    assert [p.name for p in todo] == ["a_4k.exr"]
    assert already == set()


def test_the_batched_loop_converges(tmp_path):
    # What the corpus build actually does: call this once per batch over a
    # directory that keeps growing. Every panorama must render exactly once,
    # and the pass after the last one must be quiet rather than fatal.
    root = tmp_path / "src"
    done = tmp_path / "rendered.txt"
    root.mkdir()
    rendered: list[str] = []

    for batch in range(4):
        panoramas(root, *[f"p{batch}_{i}_4k.exr" for i in range(3)])
        todo, _ = select_sources(root, done)
        rendered += [p.name for p in todo]
        done.write_text("\n".join(rendered) + "\n", encoding="utf-8")

    assert len(rendered) == 12
    assert len(set(rendered)) == 12, "a panorama was rendered twice"

    # One more pass with nothing new: empty, and no exception.
    todo, already = select_sources(root, done)
    assert todo == []
    assert len(already) == 12


# ── manifest hygiene ──────────────────────────────────────────────────────────

def test_the_manifest_ignores_interrupted_writes(tmp_path):
    """A killed render leaves ".<stem>.<random>.tmp.png" behind.

    Its suffix is .png and its stem still matches the sequence pattern, so
    the manifest builder used to take a half-written file for a real frame
    and hand it to the trainer as ground truth. Found on 4 Sep 2026 after a
    render was interrupted mid-frame.
    """
    from training.build_video_manifest import _index_frames

    d = tmp_path / "hdr"
    d.mkdir()
    for i in range(3):
        (d / f"tif_0000000_scene_a_0000{i}.png").write_bytes(b"")
    (d / ".tif_0000000_scene_a_00003.abcd1234.tmp.png").write_bytes(b"")

    grouped = _index_frames(d)[0]
    frames = [n for seq in grouped.values() for n in seq]
    assert sorted(frames) == [0, 1, 2], "an interrupted write entered the manifest"
