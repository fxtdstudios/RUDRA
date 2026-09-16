"""Encode a RUDRA sequence to something a timeline will accept.

RUDRA wrote EXRs and nothing else. A colorist does not want a folder of 240
EXRs for a ten-second shot; they want a file they can drop on a timeline.
Competing tools ship ProRes and HDR10 out of the box, and the pieces to do it
were already here -- the HDR10 PQ maths in rudra/hdr10.py, MaxCLL and MaxFALL
in rudra/delivery/metadata.py, ffmpeg already required for reading footage.
This joins them up.

Four targets, because they answer four different questions:

  prores4444   12-bit, Rec.2020 linear-light or PQ. The finishing format. Big.
  prores422hq  10-bit. The review format: same picture, a third of the size.
  hdr10        HEVC Main10 PQ with static metadata, the mastering display and
               MaxCLL/MaxFALL written into the stream. The delivery format.
  hlg          HEVC Main10 with the BBC/NHK hybrid log-gamma curve, for
               broadcast paths that will not take PQ.

Everything is written frame by frame through a pipe: a 4K 16-bit sequence does
not fit in memory and should not have to.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..hdr10 import master_to_peak, pq_oetf
from .colorspace import convert

__all__ = ["TARGETS", "encode_sequence", "colour_tags", "expected_tags",
           "prores_frame_tags", "container_colr", "EncodeError",
           "shoulder_to_peak", "hlg_inverse_ootf"]

DIFFUSE_WHITE_NITS = 203.0


class EncodeError(RuntimeError):
    """Something the user can act on, phrased for them."""


@dataclass(frozen=True)
class Target:
    name: str
    codec: list[str]
    pix_fmt: str
    suffix: str
    transfer: str          # 'pq', 'hlg' or 'linear'
    note: str


TARGETS: dict[str, Target] = {
    "prores4444": Target(
        "prores4444",
        ["-c:v", "prores_ks", "-profile:v", "4444", "-qscale:v", "5"],
        "yuv444p12le", ".mov", "pq",
        "12-bit ProRes 4444, Rec.2020 PQ. The finishing format."),
    "prores422hq": Target(
        "prores422hq",
        ["-c:v", "prores_ks", "-profile:v", "3", "-qscale:v", "9"],
        "yuv422p10le", ".mov", "pq",
        "10-bit ProRes 422 HQ, Rec.2020 PQ. The review format."),
    "hdr10": Target(
        "hdr10",
        ["-c:v", "libx265", "-preset", "slow", "-crf", "12"],
        "yuv420p10le", ".mp4", "pq",
        "HEVC Main10 PQ with static HDR10 metadata. The delivery format."),
    "hlg": Target(
        "hlg",
        ["-c:v", "libx265", "-preset", "slow", "-crf", "12"],
        "yuv420p10le", ".mp4", "hlg",
        "HEVC Main10 hybrid log-gamma, for broadcast paths that refuse PQ."),
}


def hlg_oetf(scene_linear: np.ndarray) -> np.ndarray:
    """ARIB STD-B67 / ITU-R BT.2100 HLG, scene-referred, 1.0 = nominal white.

    Not PQ with a different name: HLG encodes a RELATIVE signal, so it carries
    no absolute nit anchor and a display decides the peak. That is exactly why
    broadcast likes it, and why a PQ master and an HLG master are not the same
    deliverable with a different curve.
    """
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    x = np.clip(np.asarray(scene_linear, dtype=np.float64), 0.0, None)
    return np.where(x <= 1.0 / 12.0,
                    np.sqrt(np.maximum(3.0 * x, 0.0)),
                    a * np.log(np.maximum(12.0 * x - b, 1e-12)) + c)


def hlg_inverse_ootf(display_normalized: np.ndarray, peak_nits: float = 1000.0) -> np.ndarray:
    """BT.2100 inverse OOTF: display-referred (1.0 = peak) -> scene-referred.

    HLG's OETF expects SCENE light. What RUDRA has is display light -- nits --
    and the display's OOTF (Y_d = Y_s ** gamma, gamma 1.2 at 1,000 nits and
    0.42 * log2(peak / 1000) more or less per BT.2100 note 5e) sits between the
    two. Feeding display light straight to the OETF, which this module did
    until 16 Sep 2026, put diffuse white 0.46 stop and 18% grey 0.96 stop dark
    on a 1,000-nit HLG display. The inverse is applied on luminance and the
    ratio carried to RGB, which is the form the standard gives it in.
    """
    d = np.maximum(np.asarray(display_normalized, dtype=np.float64), 0.0)
    gamma = 1.2 + 0.42 * np.log2(max(float(peak_nits), 1e-6) / 1000.0)
    y_d = np.sum(d * HLG_LUMA, axis=-1, keepdims=True)
    y_s = np.power(np.maximum(y_d, 1e-12), 1.0 / gamma)
    return d * (y_s / np.maximum(y_d, 1e-12))


HLG_LUMA = np.array([0.2627, 0.6780, 0.0593], dtype=np.float64)


def shoulder_to_peak(rgb_nits: np.ndarray, peak_nits: float) -> np.ndarray:
    """Hue-preserving roll-off of absolute nits into the mastering peak.

    A PQ stream declares its mastering display; pixels above that peak are
    not an artistic choice, they are a QC reject, and the analytic baseline
    puts SDR white at 2,552 nits. Until 16 Sep 2026 `deliver` clipped at
    10,000 and wrote MaxCLL from the unshouldered pixels, so a plain white
    shirt produced MaxCLL > MaxMDL. The knee is 75% of peak, as in
    rudra.hdr10.master_to_peak, whose curve this is.
    """
    return master_to_peak(np.asarray(rgb_nits, dtype=np.float64) / 10_000.0,
                          peak_nits=float(peak_nits)).astype(np.float64)


# -- colour tags ----------------------------------------------------------
#
# A Rec.2020 PQ file that carries no colour tags is not a cosmetic problem.
# Every player and NLE falls back to Rec.709 SDR, so the picture comes up
# desaturated and wrong and nothing anywhere reports an error. This repo has
# already shipped one untagged deliverable (an EXR with no chromaticities) and
# found out downstream, which is why the tags are asserted here rather than
# assumed.
#
# Two places the tags can be written, and ffmpeg does not treat them alike:
#   * the container (MOV writes a 'colr' atom, MP4 the same box), from the
#     -color_primaries / -colorspace / -color_trc output options;
#   * the bitstream itself (HEVC VUI), which for libx265 means x265's own
#     colorprim / transfer / colormatrix parameters.
# Relying on ffmpeg to forward the output options into x265's VUI works on
# some builds and not others, so both libx265 targets now state them outright.

PRIMARIES = "bt2020"
MATRIX = "bt2020nc"
TRANSFER_NAME = {"pq": "smpte2084", "hlg": "arib-std-b67"}


_CAPABILITY_CACHE: dict[str, bool] = {}


def _ffmpeg_supports(kind: str, name: str, flag: str) -> bool:
    """Does this ffmpeg build offer <flag> on <kind>=<name>?

    Asked rather than assumed, because the two options below are exactly the
    kind that get renamed between builds, and a flag ffmpeg does not recognise
    aborts the encode instead of degrading quietly.
    """
    key = f"{kind}={name}:{flag}"
    if key in _CAPABILITY_CACHE:
        return _CAPABILITY_CACHE[key]
    ok = False
    if shutil.which("ffmpeg"):
        done = subprocess.run(["ffmpeg", "-hide_banner", "-h", f"{kind}={name}"],
                              capture_output=True, text=True)
        ok = flag in (done.stdout + done.stderr)
    _CAPABILITY_CACHE[key] = ok
    return ok


def expected_tags(target: str) -> dict[str, str]:
    """What ffprobe should report for a correctly tagged file of this target."""
    if target not in TARGETS:
        raise EncodeError(f"unknown target {target!r}")
    spec = TARGETS[target]
    return {"color_primaries": PRIMARIES, "color_space": MATRIX,
            "color_transfer": TRANSFER_NAME.get(spec.transfer, "unknown")}


def colour_tags(path: Path) -> dict[str, str]:
    """Read the colour tags back off an encoded file.

    Returns every key, using ``"unknown"`` for a tag the file does not carry,
    so a caller can never confuse "absent" with "not asked for". ffprobe only
    prints an unknown-valued field when ``-show_optional_fields always`` is
    given, and that option does not exist on builds older than 5.0, hence the
    retry: without it a missing key means the same thing, and is filled in.
    """
    keys = ["color_primaries", "color_transfer", "color_space"]
    if not shutil.which("ffprobe"):
        raise EncodeError("ffprobe is not on PATH, so the colour tags of "
                          f"{path.name} cannot be checked. Install ffmpeg "
                          "(it ships ffprobe) or pass verify_tags=False.")
    base = ["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=" + ",".join(keys), "-of", "json", str(path)]
    for args in ([base[0], "-show_optional_fields", "always", *base[1:]], base):
        done = subprocess.run(args, capture_output=True, text=True)
        if done.returncode == 0:
            try:
                streams = json.loads(done.stdout).get("streams") or [{}]
            except json.JSONDecodeError:
                continue
            return {k: str(streams[0].get(k, "unknown")) for k in keys}
    raise EncodeError(f"ffprobe could not read {path.name}: "
                      + (done.stderr.strip().splitlines() or ["no output"])[-1])


# ProRes states its colour description twice over, and the two are written by
# different parts of ffmpeg:
#
#   frame header   Every ProRes frame carries primaries, transfer and matrix,
#                  per SMPTE RDD 36. This is intrinsic to the essence, and it
#                  is what Resolve, FCP and any ProRes decoder read. ffprobe
#                  does NOT surface it in stream= output.
#   colr atom      The QuickTime container's parallel statement, in the legacy
#                  'nclc' form for MOV. This is the only one ffprobe reports,
#                  and not every build can express BT.2020 primaries and PQ
#                  transfer in it.
#
# So a ProRes master can be right where it counts and still read as untagged
# through ffprobe. Both are parsed here so the two can be told apart instead
# of one standing in for the other.

_PRORES_PRIMARIES = {0: "unspecified", 1: "bt709", 2: "unspecified", 5: "bt470bg",
                     6: "smpte170m", 9: "bt2020", 11: "smpte431", 12: "smpte432"}
_PRORES_TRANSFER = {0: "unspecified", 1: "bt709", 2: "unspecified", 16: "smpte2084",
                    18: "arib-std-b67"}
_PRORES_MATRIX = {0: "unspecified", 1: "bt709", 2: "unspecified", 6: "smpte170m",
                  9: "bt2020nc"}


def prores_frame_tags(path: Path) -> dict[str, str] | None:
    """Colour description from the first ProRes frame header, or None.

    None means no ProRes frame was found, which for a .mov this module wrote
    is itself a failure, not an absence of opinion.
    """
    data = Path(path).read_bytes()
    marker = data.find(b"icpf")
    if marker < 0:
        return None
    head = marker + 4
    if len(data) < head + 20:
        return None
    size = int.from_bytes(data[head:head + 2], "big")
    width = int.from_bytes(data[head + 8:head + 10], "big")
    height = int.from_bytes(data[head + 10:head + 12], "big")
    if size < 20 or width == 0 or height == 0:
        return None                      # not a frame header, do not guess
    return {"color_primaries": _PRORES_PRIMARIES.get(data[head + 14], str(data[head + 14])),
            "color_transfer": _PRORES_TRANSFER.get(data[head + 15], str(data[head + 15])),
            "color_space": _PRORES_MATRIX.get(data[head + 16], str(data[head + 16]))}


def container_colr(path: Path) -> dict[str, str] | None:
    """The MOV/MP4 'colr' atom, decoded, or None when the file has none.

    A file with no atom is not necessarily untagged: HEVC states its colour
    description in the bitstream VUI and usually carries no atom at all.
    """
    data = Path(path).read_bytes()
    at = data.find(b"colr")
    if at < 0 or len(data) < at + 14:
        return None
    subtype = data[at + 4:at + 8].decode("ascii", "replace")
    if subtype not in ("nclc", "nclx", "prof", "rICC"):
        return None
    if subtype in ("prof", "rICC"):
        return {"subtype": subtype}
    pri = int.from_bytes(data[at + 8:at + 10], "big")
    trc = int.from_bytes(data[at + 10:at + 12], "big")
    mtx = int.from_bytes(data[at + 12:at + 14], "big")
    return {"subtype": subtype,
            "color_primaries": _PRORES_PRIMARIES.get(pri, str(pri)),
            "color_transfer": _PRORES_TRANSFER.get(trc, str(trc)),
            "color_space": _PRORES_MATRIX.get(mtx, str(mtx))}


def _ffmpeg_version() -> str:
    try:
        first = subprocess.run(["ffmpeg", "-version"], capture_output=True,
                               text=True).stdout.splitlines()[0]
    except (OSError, IndexError):
        return "unknown build"
    return first.strip()


def _mismatch(want: dict[str, str], got: dict[str, str]) -> str:
    wrong = {k: (v, got.get(k, "absent")) for k, v in want.items() if got.get(k) != v}
    return ", ".join(f"{k}: wanted {w}, file says {g}" for k, (w, g) in sorted(wrong.items()))


def _verify_tags(path: Path, target: str) -> None:
    """Check the colour description where the format actually keeps it.

    For ProRes that is the frame header, which is intrinsic to the essence and
    is what a ProRes decoder reads. For HEVC it is the bitstream VUI, which is
    what ffprobe reports. Checking each format in the wrong place is how this
    went round twice: ffprobe calling a ProRes file untagged says nothing about
    the frame headers, and the frame headers are the part that matters.
    """
    want = expected_tags(target)

    if "prores_ks" in TARGETS[target].codec:
        frame = prores_frame_tags(path)
        if frame is None:
            raise EncodeError(
                f"{path.name} has no readable ProRes frame header, so there is "
                f"no way to confirm what colour it claims to be. Not delivered. "
                f"Build: {_ffmpeg_version()}.")
        bad = _mismatch(want, frame)
        if bad:
            raise EncodeError(
                f"{path.name} encoded, but its ProRes frame headers do not say "
                f"what they should ({bad}). Resolve and FCP read those headers, "
                f"so this file would come up as the wrong colour with nothing "
                f"reporting it. The file is still at {path}. Build: "
                f"{_ffmpeg_version()}. Pass --no-verify-tags to accept it.")
        # The container atom is a second, weaker statement of the same thing.
        # Some builds cannot put BT.2020 primaries or PQ transfer in a MOV's
        # legacy 'nclc' atom at all. That is worth saying out loud, but it does
        # not make a correctly tagged master undeliverable.
        shown = colour_tags(path)
        if _mismatch(want, shown):
            print(f"note: {path.name} is correctly tagged in its ProRes frame "
                  f"headers, but this ffmpeg writes an incomplete colr atom, so "
                  f"ffprobe reports {shown}. Tools that read the frame headers "
                  f"(Resolve, FCP) are unaffected; anything reading only the "
                  f"container may assume Rec.709.", file=sys.stderr)
        return

    got = colour_tags(path)
    bad = _mismatch(want, got)
    if bad:
        raise EncodeError(
            f"{path.name} encoded, but its colour tags did not land ({bad}). "
            f"An untagged Rec.2020 file plays as Rec.709 SDR everywhere, with no "
            f"error shown, so it is not delivered. The file is still at {path} if "
            f"you want to look at it. This is an ffmpeg build difference, not "
            f"something in the pixels: {_ffmpeg_version()}. Pass --no-verify-tags "
            f"(or verify_tags=False) to accept the file as it is.")


def _encode_frame(rgb_nits: np.ndarray, target: Target, peak_nits: float,
                  source_space: str, shoulder: bool = True) -> np.ndarray:
    """One scene-linear frame in nits -> 16-bit code values for the pipe."""
    rgb = np.asarray(rgb_nits, dtype=np.float64)
    if source_space != "rec2020":
        rgb = convert((rgb / DIFFUSE_WHITE_NITS).astype(np.float32),
                      source_space, "rec2020").astype(np.float64) * DIFFUSE_WHITE_NITS
    if shoulder:
        rgb = shoulder_to_peak(rgb, peak_nits)
    if target.transfer == "pq":
        # pq_oetf takes ABSOLUTE nits. Handing it scene-linear reports a
        # 4,000-nit specular as though it were 20, which is the units slip
        # this repo has made before and written up.
        coded = pq_oetf(np.clip(rgb, 0.0, 10_000.0))
    elif target.transfer == "hlg":
        coded = hlg_oetf(hlg_inverse_ootf(rgb / max(peak_nits, 1e-6), peak_nits))
    else:
        coded = np.clip(rgb / max(peak_nits, 1e-6), 0.0, 1.0)
    return (np.clip(coded, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)


def encode_sequence(frames, output: Path, target: str = "hdr10", fps: float = 24.0,
                    peak_nits: float = 1000.0, maxcll: int | None = None,
                    maxfall: int | None = None, source_space: str = "rec2020",
                    min_nits: float = 0.005, verify_tags: bool = True,
                    shoulder: bool = True) -> Path:
    """Encode an iterable of scene-linear-nits frames to one file.

    ``frames`` yields (H, W, 3) float arrays in ABSOLUTE NITS. It is consumed
    lazily and written down a pipe, so a 900-frame 4K sequence never has to be
    held in memory.

    With ``shoulder`` (the default) every frame is rolled off into
    ``peak_nits`` before encoding, so no pixel exceeds the mastering display
    the stream declares. Pass ``shoulder=False`` only for frames that were
    already mastered to that peak; MaxCLL/MaxFALL must then describe those.

    With ``verify_tags`` (the default) the finished file is read back and its
    colour tags checked before it is handed over. An untagged Rec.2020 file
    plays as Rec.709 SDR in every player without reporting anything, so it
    fails here instead of two departments later.
    """
    if target not in TARGETS:
        raise EncodeError(f"unknown target {target!r}. Choose from: "
                          + ", ".join(sorted(TARGETS)))
    spec = TARGETS[target]

    # ARGUMENTS FIRST, THEN THE ENVIRONMENT, THEN ANYTHING WITH SIDE EFFECTS.
    #
    # This used to probe for ffmpeg before it looked at the frames, which meant
    # an empty sequence on a machine without ffmpeg reported "install ffmpeg"
    # -- true, and not the problem. The caller passed nothing to encode, and
    # that is worth saying whether or not a codec is installed. CI has no
    # ffmpeg and caught it; a user with a bad call and no ffmpeg would have
    # been sent to fix the wrong thing.
    #
    # Consuming the first frame here is also what validates it, so the shape
    # check comes for free before anything is created on disk.
    iterator = iter(frames)
    try:
        first = np.asarray(next(iterator), dtype=np.float64)
    except StopIteration:
        raise EncodeError("no frames to encode") from None
    if first.ndim != 3 or first.shape[2] != 3:
        raise EncodeError(f"expected (H, W, 3) frames, got {first.shape}")
    height, width = first.shape[:2]

    if not shutil.which("ffmpeg"):
        raise EncodeError("ffmpeg is not on PATH, so nothing can be encoded. "
                          "Install it (winget install Gyan.FFmpeg) and try again.")

    # Only now, once the call is known to be encodable, is a directory made.
    output = Path(output).with_suffix(spec.suffix)
    output.parent.mkdir(parents=True, exist_ok=True)

    args = ["ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb48le",
            "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
            # The RGB -> Y'CbCr conversion is swscale's, inserted automatically
            # for the pix_fmt change, and the -colorspace flag below only TAGS
            # the output: it does not tell swscale which coefficients to use.
            # Left alone, ffmpeg 6.1 encodes with BT.601 (pure red at PQ' 0.75
            # came out Y'/Cb/Cr 1044/1591/3397 where BT.2020 is 946/1673/3392)
            # under a bt2020nc tag, and the tag check cannot see it because
            # the tag is right. This filter makes the matrix what the tag says.
            "-vf", f"scale=out_color_matrix={MATRIX}:out_range=tv",
            *spec.codec, "-pix_fmt", spec.pix_fmt,
            "-color_primaries", PRIMARIES, "-colorspace", MATRIX,
            "-color_trc", TRANSFER_NAME[spec.transfer],
            "-color_range", "tv"]

    if spec.suffix == ".mov":
        # MOV carries the colour description in a 'colr' atom. Some builds
        # write it whenever the colour info is set and some only when asked,
        # and a ProRes master without it is read as Rec.709 SDR by everything.
        # The flag is documented as experimental, so its presence is checked
        # rather than assumed: an option ffmpeg does not know aborts the run.
        if _ffmpeg_supports("muxer", "mov", "write_colr"):
            args += ["-movflags", "+write_colr"]

    if "prores_ks" in spec.codec:
        # ProRes carries colour primaries, transfer and matrix in its own
        # FRAME headers, independently of the container, and that is what
        # Resolve and FCP read. prores_ks leaves them unspecified, so they are
        # written here. ffprobe reports the container's atom rather than this,
        # so a file can satisfy one and not the other: both are set.
        if _ffmpeg_supports("bsf", "prores_metadata", "color_primaries"):
            args += ["-bsf:v", f"prores_metadata=color_primaries={PRIMARIES}"
                               f":color_trc={TRANSFER_NAME[spec.transfer]}"
                               f":colorspace={MATRIX}"]

    if "libx265" in spec.codec:
        # State the colour description to x265 directly instead of trusting
        # ffmpeg to forward the output options above into the HEVC VUI. On the
        # builds where that forwarding does not happen the file encodes
        # perfectly and comes out untagged, which is the failure this whole
        # module is trying not to ship.
        params = [f"repeat-headers=1:colorprim={PRIMARIES}"
                  f":transfer={TRANSFER_NAME[spec.transfer]}:colormatrix={MATRIX}"
                  f":range=limited"]
        if spec.transfer == "pq":
            # Static HDR10 metadata belongs IN the stream. A PQ file without
            # it makes the display guess the mastering volume, and every
            # display guesses differently. HLG carries no absolute anchor, so
            # a mastering volume would be meaningless on it.
            display = (f"G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)"
                       f"L({int(round(peak_nits * 10_000))},{int(round(min_nits * 10_000))})")
            params.append(f":hdr10-opt=1:master-display={display}")
            if maxcll is not None and maxfall is not None:
                params.append(f":max-cll={int(maxcll)},{int(maxfall)}")
        args += ["-x265-params", "".join(params)]

    args.append(str(output))
    process = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    written = 0
    try:
        frame = first
        while True:
            process.stdin.write(_encode_frame(frame, spec, peak_nits, source_space,
                                              shoulder).tobytes())
            written += 1
            try:
                frame = np.asarray(next(iterator), dtype=np.float64)
            except StopIteration:
                break
            if frame.shape[:2] != (height, width):
                raise EncodeError(
                    f"frame {written} is {frame.shape[1]}x{frame.shape[0]}, "
                    f"the first was {width}x{height}. A sequence must not change size.")
        process.stdin.close()
    except BrokenPipeError:
        pass
    stderr = process.stderr.read().decode("utf-8", "replace").strip()
    if process.wait() != 0:
        raise EncodeError(f"ffmpeg failed after {written} frame(s): "
                          + (stderr.splitlines() or ["unknown error"])[-1])
    if verify_tags:
        _verify_tags(output, target)
    return output
