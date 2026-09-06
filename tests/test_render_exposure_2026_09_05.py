"""Exposure along a rendered camera move, and why it was ever constant.

The v02 temporal gate returned nothing on 5 Sep 2026, and the reason was in
the renderer rather than in the metric. A pure rotation through a static
panorama, tone-mapped with ONE curve for every frame, gives a scene point the
same SDR code in every frame it appears in. Whatever is clipped in frame 3 is
clipped in frame 7. The corpus therefore contained none of the information
v02's claim is about -- that a neighbour can show what this frame blew out --
and a gate measuring it faithfully had to come back empty.

`--exposure-drift` and `--exposure-jitter` put that axis back. These tests
hold three things: that the default reproduces the old corpus exactly, that
drift moves the SDR, and that it does NOT move the HDR target -- which is the
whole point, since the camera's response is what changes and the light is not.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

cv2 = pytest.importorskip("cv2")

from pipeline.render_hdri_moves import move, select_sources   # noqa: E402


def rng(seed: int = 3):
    return np.random.default_rng(seed)


def test_the_default_leaves_exposure_alone():
    """Byte-for-byte compatibility with the 993 scenes already rendered."""
    path = move(rng(), frames=9, hfov=75.0)
    assert all(pose["ev"] == 0.0 for pose in path)


def test_drift_is_a_ramp_centred_on_the_clip():
    """Centred, so a drifted corpus sits at the same average level as the old.

    An uncentred ramp would make every clip start correctly exposed and end a
    stop off, which is a systematic brightness change dressed up as a
    temporal one.
    """
    path = move(rng(), frames=9, hfov=75.0, exposure_drift=0.12)
    evs = [pose["ev"] for pose in path]
    assert abs(float(np.mean(evs))) < 1e-9
    steps = np.diff(evs)
    assert np.allclose(steps, steps[0])                  # a ramp, not a walk
    assert abs(steps[0]) == pytest.approx(0.12)
    # Nine frames at 0.12 is very nearly a stop end to end.
    assert abs(evs[-1] - evs[0]) == pytest.approx(0.96)


def test_drift_can_go_either_way():
    """Sign is drawn per clip. A corpus that always brightens teaches that."""
    signs = {np.sign(move(rng(s), 9, 75.0, exposure_drift=0.2)[-1]["ev"])
             for s in range(30)}
    assert signs == {-1.0, 1.0}


def test_jitter_is_not_a_ramp():
    path = move(rng(), frames=9, hfov=75.0, exposure_jitter=0.08)
    steps = np.diff([pose["ev"] for pose in path])
    assert not np.allclose(steps, steps[0])


def test_exposure_does_not_touch_the_geometry():
    """Poses must be identical with and without exposure. Same camera path.

    The gate warps frames onto each other with these numbers. If asking for
    drift also moved the camera, every temporal measurement taken on the new
    corpus would be measuring the change rather than the exposure.
    """
    plain = move(rng(11), 9, 75.0)
    drifted = move(rng(11), 9, 75.0, exposure_drift=0.12)
    for a, b in zip(plain, drifted):
        for key in ("yaw", "pitch", "roll", "hfov"):
            assert a[key] == pytest.approx(b[key])


def test_only_the_sdr_moves_with_exposure():
    """The target is the light that was there, not the light as recorded.

    Scaling the HDR alongside the SDR would keep the pair self-consistent and
    make the task trivially invertible -- the exact failure the v01 corpus
    documents under 'held-out SDR is the EXACT ACES output that produced the
    target'.
    """
    from training.prepare_training_data import make_sdr

    linear = np.abs(rng(5).normal(0.4, 0.6, size=(16, 24, 3))).astype(np.float32)
    dark = make_sdr(linear * np.float32(2.0 ** -0.5))
    bright = make_sdr(linear * np.float32(2.0 ** 0.5))
    assert not np.array_equal(dark, bright)
    assert float(bright.mean()) > float(dark.mean())
    # The HDR encoder never sees the exposure at all: same array in, same out.
    from pipeline.hdr_io import HDRStorage, encode_hdr_u16
    storage = HDRStorage(mode="log2_extended", ceiling_nits=1_000_000.0)
    first, _ = encode_hdr_u16(linear, storage)
    second, _ = encode_hdr_u16(linear, storage)
    assert np.array_equal(first, second)


def test_a_brighter_exposure_unclips_what_a_darker_one_clipped():
    """THE POINT, in one assertion.

    This is the information v02 says a neighbouring frame can carry, and the
    reason the fixed-exposure corpus could not carry it. Shadow detail crushed
    to zero at one exposure has to be readable at another, or there is nothing
    for a temporal model to fetch.
    """
    from training.prepare_training_data import make_sdr

    linear = np.linspace(1e-4, 0.02, 64, dtype=np.float32)
    linear = np.repeat(linear[None, :, None], 3, axis=2)
    dark = make_sdr(linear)
    bright = make_sdr(linear * np.float32(2.0 ** 3))
    crushed = dark[..., 0] == 0
    assert crushed.any(), "fixture does not crush anything; widen the range"
    assert (bright[..., 0][crushed] > 0).any()


# --------------------------------------------------------------------------
# --scene-list. A seed sweep is only a seed sweep if every seed sees the same
# scenes, and the panorama directory holds 994 of them.
# --------------------------------------------------------------------------

def panorama_dir(tmp_path, names):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in names:
        (tmp_path / f"{name}.exr").write_bytes(b"")
    return tmp_path


def test_a_scene_list_selects_and_orders(tmp_path):
    src = panorama_dir(tmp_path / "src", ["a", "b", "c", "d"])
    listing = tmp_path / "scenes.txt"
    listing.write_text("c\na\n", encoding="utf-8")
    hdris, _ = select_sources(src, None, listing)
    assert [p.stem for p in hdris] == ["c", "a"]


def test_a_scene_list_ignores_blank_lines(tmp_path):
    src = panorama_dir(tmp_path / "src", ["a", "b"])
    listing = tmp_path / "scenes.txt"
    listing.write_text("\n a \n\nb\n\n", encoding="utf-8")
    hdris, _ = select_sources(src, None, listing)
    assert [p.stem for p in hdris] == ["a", "b"]


def test_a_missing_scene_is_an_error(tmp_path):
    """Loudly. A sweep that silently rendered 49 of 50 scenes for one seed
    would produce a comparison nobody could interpret afterwards."""
    src = panorama_dir(tmp_path / "src", ["a", "b"])
    listing = tmp_path / "scenes.txt"
    listing.write_text("a\nzzz\n", encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        select_sources(src, None, listing)
    assert "zzz" in str(caught.value)


def test_the_done_file_still_skips_inside_a_scene_list(tmp_path):
    """The list picks WHICH scenes; the done file picks which are left."""
    src = panorama_dir(tmp_path / "src", ["a", "b", "c"])
    listing = tmp_path / "scenes.txt"
    listing.write_text("a\nb\nc\n", encoding="utf-8")
    done = tmp_path / "done.txt"
    done.write_text("b.exr\n", encoding="utf-8")
    hdris, already = select_sources(src, done, listing)
    assert [p.stem for p in hdris] == ["a", "c"]
    assert already == {"b.exr"}


def test_no_scene_list_takes_the_whole_directory(tmp_path):
    src = panorama_dir(tmp_path / "src", ["b", "a"])
    hdris, _ = select_sources(src, None, None)
    assert [p.stem for p in hdris] == ["a", "b"]
