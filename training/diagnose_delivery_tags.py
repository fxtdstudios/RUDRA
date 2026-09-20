"""Report what this machine's ffmpeg actually writes into a delivery file.

Why this exists: RUDRA's delivery tests encode one file per target and read
the colour tags back. On one workstation every target came back tagged; on
another, ProRes and HLG came back with no colour tags at all while HDR10 was
fine. Same code, same flags, different ffmpeg build. Guessing which build does
what is how untagged masters get shipped, so this prints the evidence instead:
the two version strings, the exact ffmpeg command, and every colour field
ffprobe will admit to for each target.

    python training/diagnose_delivery_tags.py
    python training/diagnose_delivery_tags.py --keep out_dir

Nothing here is a test. It encodes four two-frame files into a temporary
directory and tells you what came out. Read the TAGS block: "unknown" means
the file does not carry that tag, which means a player will fall back to
Rec.709 SDR and show the wrong picture without reporting anything.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.delivery.video import (TARGETS, _ffmpeg_supports, colour_tags,  # noqa: E402
                                  container_colr, encode_sequence,
                                  prores_frame_tags)


def version(tool: str) -> str:
    if not shutil.which(tool):
        return f"{tool}: NOT ON PATH"
    out = subprocess.run([tool, "-version"], capture_output=True, text=True).stdout
    return (out.splitlines() or [f"{tool}: no output"])[0].strip()


def frames(count: int = 2):
    for i in range(count):
        f = np.full((96, 160, 3), 180.0)
        f[20:50, 20 + i * 6:60 + i * 6] = 3600.0
        yield f


def container_and_bitstream(path: Path) -> list[str]:
    """Where the colour description actually sits in this file.

    Two independent places, and a build can write one and not the other:

      colr atom      the MOV/MP4 container box. This is the only one ffprobe
                     reports. MOV uses the legacy 'nclc' form, and not every
                     build can express BT.2020 primaries or PQ transfer in it.
      frame header   ProRes carries primaries, transfer and matrix in every
                     frame, per SMPTE RDD 36. Resolve and FCP read these, and
                     ffprobe does not show them at all.

    A ProRes file can be correct in its frame headers and read as untagged
    through ffprobe. Knowing which half is short is the whole difference
    between a real problem and a reporting artefact.
    """
    lines = []
    atom = container_colr(path)
    if atom is None:
        lines.append("colr atom:              none "
                     "(normal for HEVC, which states colour in the bitstream)")
    else:
        rest = " ".join(f"{k.replace('color_', '')}={v}"
                        for k, v in atom.items() if k != "subtype")
        lines.append(f"colr atom:              {atom['subtype']} {rest}")
    frame = prores_frame_tags(path)
    if frame is not None:
        lines.append("ProRes frame header:    "
                     + " ".join(f"{k.replace('color_', '')}={v}" for k, v in frame.items()))
    return lines


def full_probe(path: Path) -> dict:
    """Everything ffprobe knows about the stream, not just the fields we ask
    for. If a build reports the tags somewhere unexpected, it shows up here."""
    args = ["ffprobe", "-v", "error", "-show_optional_fields", "always",
            "-select_streams", "v:0", "-show_streams", "-of", "json", str(path)]
    done = subprocess.run(args, capture_output=True, text=True)
    if done.returncode != 0:                      # build predates the option
        args = [a for a in args if a not in ("-show_optional_fields", "always")]
        done = subprocess.run(args, capture_output=True, text=True)
    if done.returncode != 0:
        return {"ffprobe_error": done.stderr.strip()}
    streams = json.loads(done.stdout).get("streams") or [{}]
    keep = ("codec_name", "profile", "pix_fmt", "color_range", "color_space",
            "color_transfer", "color_primaries", "chroma_location", "field_order")
    return {k: streams[0].get(k, "<field absent>") for k in keep}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", type=Path, default=None,
                    help="write the sample files here instead of a temp dir")
    args = ap.parse_args(argv)

    print(version("ffmpeg"))
    print(version("ffprobe"))
    print(f"python   {sys.version.split()[0]}")
    print(f"numpy    {np.__version__}")
    print()
    # The two build-dependent options the encoder reaches for. Either being
    # absent is a concrete explanation for an untagged file, not a mystery.
    print("build options this encoder wants:")
    print(f"  mov muxer write_colr      {_ffmpeg_supports('muxer', 'mov', 'write_colr')}")
    print(f"  prores_metadata bsf       {_ffmpeg_supports('bsf', 'prores_metadata', 'color_primaries')}")
    print()

    out_dir = args.keep or Path(tempfile.mkdtemp(prefix="rudra-delivery-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    verdicts = {}

    for name in sorted(TARGETS):
        print("=" * 72)
        print(name, "--", TARGETS[name].note)
        try:
            # verify_tags=False on purpose: the point is to SEE what came out,
            # not to stop at the first file that came out wrong.
            path = encode_sequence(frames(), out_dir / name, target=name,
                                   peak_nits=1000.0, maxcll=3600, maxfall=400,
                                   verify_tags=False)
        except Exception as exc:                   # noqa: BLE001 - report anything
            print(f"  ENCODE FAILED: {exc}")
            verdicts[name] = "encode failed"
            continue
        print(f"  file   {path.name}  ({path.stat().st_size:,} bytes)")
        want_trc = "smpte2084" if TARGETS[name].transfer == "pq" else "arib-std-b67"

        def correct(tags):
            return bool(tags) and (tags.get("color_primaries") == "bt2020"
                                   and tags.get("color_space") == "bt2020nc"
                                   and tags.get("color_transfer") == want_trc)

        # Judge each format where it actually keeps its colour description:
        # the frame header for ProRes, the bitstream (which is what ffprobe
        # reports) for HEVC. Asking ffprobe about a ProRes frame header is a
        # question it cannot answer, and reading its silence as "untagged" is
        # what sent this round twice.
        if "prores_ks" in TARGETS[name].codec:
            ok = correct(prores_frame_tags(path))
            second = correct(colour_tags(path))
            verdicts[name] = ("tagged" if ok and second
                              else "tagged in frame headers only" if ok
                              else "UNTAGGED")
        else:
            ok = correct(colour_tags(path))
            verdicts[name] = "tagged" if ok else "UNTAGGED"
        print(f"  TAGS   {'ok' if ok else 'WRONG OR MISSING'}")
        for key, value in full_probe(path).items():
            print(f"    {key:<18} {value}")
        for line in container_and_bitstream(path):
            print(f"    {line}")
    print("=" * 72)
    for name, verdict in verdicts.items():
        print(f"{name:<14} {verdict}")
    if args.keep:
        print(f"\nfiles kept in {out_dir}")
    partial = [n for n, v in verdicts.items() if v.endswith("frame headers only")]
    bad = [n for n, v in verdicts.items() if v == "UNTAGGED" or v == "encode failed"]
    if partial:
        print("\nA file tagged in its frame headers only is a correct ProRes "
              "master. Resolve and FCP read those headers. What is missing is "
              "the container's copy of the same statement, which this ffmpeg "
              "build cannot fully write into a MOV, so ffprobe and anything "
              "else reading only the container will assume Rec.709.")
    if bad:
        print("\nA file marked UNTAGGED encodes and plays, and is wrong: no "
              "player can tell it is Rec.2020 PQ, so all of them read it as "
              "Rec.709 SDR. Send this whole output along with the ffmpeg "
              "version line at the top.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
