"""Encoding a RUDRA sequence to something a timeline will accept.

RUDRA wrote EXRs and nothing else, while competing tools shipped ProRes and
HDR10. The pieces were already here -- PQ maths, MaxCLL/MaxFALL, ffmpeg -- and
the gap was that nothing joined them up.

Two failures these guard against, both silent. PQ is defined on ABSOLUTE nits,
so handing the encoder scene-linear values reports a 4,000-nit specular as
though it were 20: the file plays, it is simply dark, and this repo has made
that exact units slip before. And static HDR10 metadata that does not match the
pixels makes every display tone-map the file differently, with nothing
reporting a mismatch -- so the metadata is measured from the frames, never
supplied by hand.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.delivery.aces import write_acescg_exr, write_aces_exr      # noqa: E402
from rudra.delivery.colorspace import AP1_CHROMATICITIES, convert     # noqa: E402
from rudra.delivery.exr import read_exr                               # noqa: E402
from rudra.delivery.video import (TARGETS, EncodeError, colour_tags,  # noqa: E402
                                  container_colr, encode_sequence, hlg_oetf,
                                  prores_frame_tags)

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="needs ffmpeg and ffprobe")

SPECULAR_NITS = 3600.0


def frames(count: int = 6, height: int = 96, width: int = 160):
    for i in range(count):
        f = np.full((height, width, 3), 180.0)          # around diffuse white
        f[20:50, 20 + i * 6:60 + i * 6] = SPECULAR_NITS  # a moving specular
        yield f


def probe(path: Path, entries: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", entries,
         "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@needs_ffmpeg
@pytest.mark.parametrize("target", sorted(TARGETS))
def test_every_target_encodes_and_is_tagged(target, tmp_path):
    """The tags are the deliverable as much as the pixels are.

    A Rec.2020 PQ file with no colour tags plays as Rec.709 SDR in every
    player and NLE: washed out, wrong, and reported by nothing. The values
    below are written out in full rather than read back from TARGETS, because
    a test that asks the code under test what it expected cannot catch the
    code changing its mind.
    """
    out = encode_sequence(frames(), tmp_path / "shot", target=target, fps=24.0,
                          peak_nits=1000.0, maxcll=3600, maxfall=400)
    assert out.is_file() and out.stat().st_size > 0
    assert probe(out, "stream=pix_fmt")["streams"][0]["pix_fmt"] == TARGETS[target].pix_fmt

    want_trc = "smpte2084" if TARGETS[target].transfer == "pq" else "arib-std-b67"
    if "prores_ks" in TARGETS[target].codec:
        # ProRes keeps its colour description in every frame header, and that
        # is what a ProRes decoder reads. ffprobe does not report it, so this
        # reads the header directly rather than asking ffprobe a question it
        # cannot answer about this format.
        tags = prores_frame_tags(out)
        assert tags is not None, "no readable ProRes frame header"
    else:
        tags = colour_tags(out)              # every key present, "unknown" if absent
    assert tags["color_primaries"] == "bt2020", (target, tags)
    assert tags["color_transfer"] == want_trc, (target, tags)
    assert tags["color_space"] == "bt2020nc", (target, tags)


@needs_ffmpeg
@pytest.mark.parametrize("target", ["prores422hq", "prores4444"])
def test_prores_states_its_colour_in_both_places_when_the_build_allows(target, tmp_path):
    """Frame header AND container atom, checked separately.

    The frame header is the gate; the atom is a second statement of the same
    thing that some ffmpeg builds cannot express in a MOV. This test records
    what THIS build managed, so a regression in either half is visible rather
    than averaged away.
    """
    out = encode_sequence(frames(2), tmp_path / "shot", target=target)
    frame = prores_frame_tags(out)
    assert frame["color_primaries"] == "bt2020"
    atom = container_colr(out)
    if atom is None:
        pytest.skip("this ffmpeg wrote no colr atom for ProRes")
    assert atom["subtype"] in ("nclc", "nclx")
    if atom.get("color_primaries") != "bt2020":
        pytest.skip(f"this ffmpeg cannot express BT.2020 in a MOV colr atom: {atom}")
    assert atom["color_transfer"] == "smpte2084"
    assert atom["color_space"] == "bt2020nc"


@needs_ffmpeg
@pytest.mark.parametrize("target", ["prores422hq", "prores4444"])
def test_prores_with_unwritten_frame_headers_is_refused(target, tmp_path):
    """prores_ks leaves the frame headers unspecified unless they are written.

    This is the failure the frame-header check exists for: the file encodes,
    plays, and claims nothing about its colour.
    """
    from rudra.delivery import video as video_mod
    monkey = video_mod._ffmpeg_supports
    try:
        # Pretend this build has no prores_metadata filter, which is exactly
        # what an older ffmpeg looks like.
        video_mod._ffmpeg_supports = (
            lambda kind, name, flag: False if name == "prores_metadata" else monkey(kind, name, flag))
        with pytest.raises(EncodeError, match="frame header"):
            encode_sequence(frames(2), tmp_path / "bare", target=target)
    finally:
        video_mod._ffmpeg_supports = monkey


@needs_ffmpeg
def test_an_untagged_file_reads_back_as_unknown_not_as_a_missing_key(tmp_path):
    """The hole that let an untagged file pass review.

    ffprobe prints nothing at all for a colour field it cannot name, so a test
    that indexed the key straight out of the JSON raised KeyError on exactly
    the files that were broken, and the failure looked like a bad test rather
    than a bad file. colour_tags fills the gap in with "unknown" so the two
    cases are told apart.
    """
    raw = tmp_path / "raw.bin"
    raw.write_bytes((np.full((32, 32, 3), 20000, np.uint16)).tobytes() * 2)
    plain = tmp_path / "plain.mov"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
                    "-pix_fmt", "rgb48le", "-s", "32x32", "-r", "24", "-i", str(raw),
                    "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le",
                    str(plain)], check=True)
    tags = colour_tags(plain)
    assert set(tags) == {"color_primaries", "color_transfer", "color_space"}
    assert tags["color_primaries"] == "unknown", tags


@needs_ffmpeg
def test_delivery_refuses_a_file_whose_tags_did_not_land(tmp_path, monkeypatch):
    """Fail closed. An untagged master is worse than no master: it looks fine
    on the shelf and is wrong on every screen that opens it."""
    from rudra.delivery import video as video_mod
    monkeypatch.setattr(video_mod, "colour_tags",
                        lambda path: {"color_primaries": "unknown",
                                      "color_transfer": "unknown",
                                      "color_space": "unknown"})
    with pytest.raises(EncodeError, match="colour tags did not land"):
        encode_sequence(frames(2), tmp_path / "bad", target="hdr10")


@needs_ffmpeg
def test_verification_can_be_waived_but_is_on_by_default(tmp_path):
    from rudra.delivery import video as video_mod
    import inspect
    assert inspect.signature(video_mod.encode_sequence).parameters[
        "verify_tags"].default is True
    out = encode_sequence(frames(2), tmp_path / "ok", target="hdr10", verify_tags=False)
    assert out.is_file()


@needs_ffmpeg
def test_hdr10_carries_its_metadata_in_the_stream(tmp_path):
    """A PQ file without static metadata makes the display guess the mastering
    volume, and every display guesses differently."""
    out = encode_sequence(frames(), tmp_path / "m", target="hdr10",
                          peak_nits=1000.0, maxcll=3600, maxfall=400)
    got = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
         "-read_intervals", "%+#1", "-show_entries", "frame=side_data_list",
         "-of", "json", str(out)], capture_output=True, text=True, check=True).stdout
    assert "Mastering display metadata" in got
    assert "Content light level metadata" in got
    assert '"max_content": 3600' in got.replace(" ", " ")
    assert "10000000/10000" in got            # 1000 nits, in x265's units


def decoded_peak(path: Path) -> float:
    """Brightest code value in the first decoded frame, 0 to 1."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"],
        capture_output=True, check=True).stdout
    return float(np.frombuffer(raw, dtype=np.uint16).max()) / 65535.0


@needs_ffmpeg
def test_pq_is_fed_absolute_nits_not_scene_linear(tmp_path):
    """The units trap, measured on the decoded pixels rather than a proxy.

    A 3,600-nit specular put through PQ lands near the top of the code range.
    The same content handed over as scene-linear -- nits/203, which is how it
    is stored -- lands far lower, and the file still plays. It is just dark,
    which is why this slip survives review.

    File size does not catch it: both encode to within 1% of each other,
    because the picture has the same shape either way. The first version of
    this test compared sizes and passed on the broken case.
    """
    correct = decoded_peak(encode_sequence(frames(), tmp_path / "a", target="hdr10"))
    slipped = decoded_peak(
        encode_sequence((f / 203.0 for f in frames()), tmp_path / "b", target="hdr10"))
    assert correct > 0.7, correct
    assert slipped < 0.5, slipped
    assert correct > slipped + 0.25, (correct, slipped)


def test_hlg_and_pq_are_not_the_same_curve():
    """HLG is scene-referred and relative; PQ is display-referred and absolute.
    They are different deliverables, not one curve with two names."""
    from rudra.hdr10 import pq_oetf
    x = np.array([0.01, 0.1, 0.5, 1.0])
    assert not np.allclose(hlg_oetf(x), pq_oetf(x * 1000.0), atol=0.05)
    assert hlg_oetf(np.array([1.0]))[0] == pytest.approx(1.0, abs=1e-6)
    assert hlg_oetf(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-9)


def test_hlg_is_monotonic():
    x = np.linspace(0.0, 1.0, 512)
    assert np.all(np.diff(hlg_oetf(x)) >= -1e-12)


def test_an_unknown_target_is_refused(tmp_path):
    with pytest.raises(EncodeError, match="unknown target"):
        encode_sequence(frames(), tmp_path / "x", target="prores9999")


def test_an_empty_sequence_is_refused(tmp_path):
    with pytest.raises(EncodeError, match="no frames"):
        encode_sequence(iter([]), tmp_path / "x", target="hdr10")


@needs_ffmpeg
def test_a_size_change_mid_sequence_is_refused(tmp_path):
    """Silently letting it through gives a file whose second half is garbage."""
    def mixed():
        yield np.full((64, 64, 3), 100.0)
        yield np.full((32, 64, 3), 100.0)
    with pytest.raises(EncodeError, match="must not change size"):
        encode_sequence(mixed(), tmp_path / "x", target="hdr10")


def test_a_non_image_frame_is_refused(tmp_path):
    with pytest.raises(EncodeError, match=r"\(H, W, 3\)"):
        encode_sequence(iter([np.zeros((8, 8))]), tmp_path / "x", target="hdr10")


# -- ACEScg ---------------------------------------------------------------
def test_acescg_writes_ap1_chromaticities(tmp_path):
    rgb = np.full((8, 8, 3), 0.18, np.float32)
    _, attrs = read_exr(write_acescg_exr(rgb, tmp_path / "cg.exr"))
    assert attrs["chromaticities"][:2] == pytest.approx(AP1_CHROMATICITIES[:2], abs=1e-4)


def test_acescg_is_a_smaller_move_than_ap0(tmp_path):
    """AP1 is close to Rec.2020; AP0 is a much wider gamut. A pipeline handed
    AP0 usually converts straight out of it, which is the round trip this
    container exists to save."""
    rgb = np.array([[[0.5, 0.3, 0.1]]], np.float32)
    to_cg = convert(rgb, "rec2020", "ap1")
    to_ap0 = convert(rgb, "rec2020", "ap0")
    assert np.abs(to_cg - rgb).max() < np.abs(to_ap0 - rgb).max()


def test_the_two_containers_are_distinguishable_on_disk(tmp_path):
    rgb = np.full((4, 4, 3), 0.5, np.float32)
    _, cg = read_exr(write_acescg_exr(rgb, tmp_path / "a.exr"))
    _, ap0 = read_exr(write_aces_exr(rgb, tmp_path / "b.exr"))
    assert cg["chromaticities"][:2] != pytest.approx(ap0["chromaticities"][:2], abs=1e-4)
