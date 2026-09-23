"""Inventory an HDR source tree before ingesting anything from it.

Point it at any root -- a local drive, or a NAS share mapped to a drive letter --
and it answers the questions the August 2026 corpus could not:

  * what formats and colour encodings are actually present
  * how much real dynamic range each source carries (in stops, and peak nits)
  * how many INDEPENDENT scenes and video sequences exist

That last number is the one that matters. The post-fix run had 11,812 records
and looked healthy; 92% of them were frames of two clips, and the temporal
trainer silently ended up with a one-clip training set. This script reports
independent-sequence count up front so that can never be a surprise again.

    python pipeline/scan_sources.py R:\\08_Research --out hdrdata/source_inventory.jsonl

Reading is cheap by design: image headers plus a decimated pixel read, and only
``--video-probe-frames`` frames per movie. A large NAS tree scans in minutes.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.hdr_io import DIFFUSE_WHITE_NITS  # noqa: E402

IMAGE_EXT = {".exr", ".hdr", ".tif", ".tiff", ".dpx", ".png", ".jxl", ".avif", ".heic"}
VIDEO_EXT = {".mxf", ".mov", ".mp4", ".mkv", ".avi", ".m2ts", ".ts", ".webm"}

# Trailing frame numbers: name_00123.exr, name.00123.exr, name-0123.dpx
FRAME_RE = re.compile(r"^(?P<stem>.+?)[._-]?(?P<frame>\d{3,8})$")

ENCODING_KEYWORDS = (
    ("logc4", ("logc4", "alexa35", "alexa 35")),
    ("logc3", ("logc3", "logc", "alexa", "arri")),
    ("slog3", ("slog3", "s-log3", "slog", "venice", "fx9", "fx6", "sony")),
    ("hlg", ("hlg", "hybridlog")),
    # "stuttgart"/"hdm" cover the HdM-HDR-2014 and HdM-HFR-2017 sets. The 2014
    # graded TIFFs carry no transfer function in the path or the container, and
    # every frame in every sequence tops out at code 59150 -- which is exactly
    # the ST-2084 code for 4,000 nits, the grading ceiling the dataset states.
    # Without this they classify as UNKNOWN and 11,007 frames are skipped in
    # silence, or -- worse, with --include-unknown-encoding -- decoded as sRGB.
    ("pq", ("pq", "hdr10", "st2084", "st-2084", "2084", "dolby", "chimera",
            "netflix", "stuttgart", "hdm-hdr", "hdm_hdr", "hdm-hfr", "hdm_hfr")),
    # ACES 2065-1 (AP0) EXRs: the Sparks and Chimera ACES masters. Float, so
    # linear, but in AP0 primaries -- to_scene_linear converts to Rec.2020.
    # "acescg" is AP1 and stays under linear; it is tested first there.
    ("linear", ("acescg",)),
    ("aces", ("aces2065", "aces_2065", "_aces", "/aces/", "aces ")),
    ("linear", ("linear", "polyhaven", "poly haven", "hdri", "aces", "scene_linear")),
)


def guess_encoding(path: Path, declared: str | None = None) -> tuple[str, str]:
    """Return (encoding, why). Explicit about its own confidence.

    ``declared`` comes from --encoding and wins over every heuristic: the
    operator looked at the dataset's documentation, and a filename did not.
    """
    if declared:
        return declared, "declared:cli"
    blob = " ".join(p.lower() for p in path.parts)
    suffix = path.suffix.lower()
    # Two rules that outrank the keyword table, both found on G: on 23 Sep 2026.
    #
    # 1. The FILENAME can say PQ even inside a float container:
    #    SPARKS_P3_PQ_4000nit_*.exr holds PQ code values (max 0.9155), and the
    #    float-means-linear rule below would read them as ~200-nit radiance.
    #    Only whole tokens of the stem count, never a parent folder's name.
    tokens = set(re.split(r"[^a-z0-9]+", path.stem.lower()))
    if tokens & {"pq", "st2084", "hdr10"}:
        return "pq", "filename-token:pq"
    # 2. A linear-HDRI dataset outranks camera-log words that happen to be in
    #    a scene name: Poly Haven's venice_dawn_1 / venice_sunrise matched
    #    "venice" (a Sony camera) and were decoded as S-Log3 -- the two
    #    "non-finite source pixels" build_manifests has been dropping since
    #    23 Aug. They are ordinary linear EXRs.
    if suffix in (".exr", ".hdr"):
        for word in ("polyhaven", "poly haven", "poly_haven"):
            if word in blob:
                return "linear", f"keyword:{word}"
    for encoding, keywords in ENCODING_KEYWORDS:
        # A float container is scene-linear whatever the dataset is called: the
        # Sparks ACES EXRs live under netflix_sparks/, "netflix" is a PQ keyword,
        # and 1,799 of them were decoded as PQ codes and dropped for peaking
        # below 1 nit (corpus_v4b, 18 Sep 2026). Camera-log names still win --
        # a LogC EXR is a real thing -- but the display-referred keywords do not.
        if encoding in ("pq", "hlg") and suffix in (".exr", ".hdr"):
            continue
        if encoding == "aces" and suffix not in (".exr", ".hdr"):
            continue
        for word in keywords:
            if word in blob:
                return encoding, f"keyword:{word}"
    if suffix in (".exr", ".hdr"):
        return "linear", "extension:float-format"
    if suffix in (".tif", ".tiff", ".dpx"):
        return "UNKNOWN", "extension:ambiguous-needs-review"
    return "srgb", "extension:8bit-fallback"


def scene_key(path: Path) -> tuple[str, bool]:
    """(scene identity, is_sequence). Frame numbers collapse into one scene."""
    match = FRAME_RE.match(path.stem)
    if match and len(match.group("frame")) >= 3:
        return f"{path.parent.as_posix()}::{match.group('stem')}", True
    return f"{path.parent.as_posix()}::{path.stem}", False


def _read_decimated(path: Path, step: int = 4) -> np.ndarray | None:
    try:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if image is None:
            return None
        if image.ndim == 2:
            image = image[..., None]
        sub = image[::step, ::step, : min(3, image.shape[2])]
        if np.issubdtype(sub.dtype, np.integer):
            sub = sub.astype(np.float32) / float(np.iinfo(sub.dtype).max)
        return sub.astype(np.float32)
    except Exception:
        return None


def _ffprobe(path: Path) -> dict:
    try:
        raw = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
             "-show_format", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        payload = json.loads(raw.stdout or "{}")
        stream = (payload.get("streams") or [{}])[0]
        fmt = payload.get("format", {})
        return {
            "width": stream.get("width"),
            "height": stream.get("height"),
            "pix_fmt": stream.get("pix_fmt"),
            "codec": stream.get("codec_name"),
            "color_transfer": stream.get("color_transfer"),
            "color_primaries": stream.get("color_primaries"),
            "nb_frames": stream.get("nb_frames"),
            "duration_s": float(fmt.get("duration", 0.0) or 0.0),
            "frame_rate": stream.get("r_frame_rate"),
        }
    except Exception:
        return {}


def _radiometry(sample: np.ndarray, encoding: str) -> dict:
    """Dynamic range of a scene-linear-ish sample, in stops and nits."""
    flat = sample[np.isfinite(sample)]
    flat = flat[flat > 0]
    if flat.size < 64:
        return {}
    low, high = np.percentile(flat, [1.0, 99.99])
    peak = float(flat.max())
    stops = float(np.log2(max(high, 1e-9) / max(low, 1e-9))) if low > 0 else 0.0
    out = {
        "p1": float(low), "p50": float(np.median(flat)),
        "p99_99": float(high), "max": peak,
        "dynamic_range_stops": round(stops, 2),
    }
    if encoding in ("linear", "aces"):
        out["peak_nits_estimate"] = round(peak * DIFFUSE_WHITE_NITS, 1)
        out["above_10000_nits"] = bool(peak * DIFFUSE_WHITE_NITS > 10_000.0)
    return out


def inspect(path: Path, video_probe_frames: int,
            declared_encoding: str | None = None) -> dict:
    encoding, why = guess_encoding(path, declared_encoding)
    scene, is_sequence = scene_key(path)
    record = {
        "path": str(path),
        "suffix": path.suffix.lower(),
        "bytes": path.stat().st_size,
        "kind": "video" if path.suffix.lower() in VIDEO_EXT else "image",
        "encoding_guess": encoding,
        "encoding_reason": why,
        "scene_id": scene,
        "is_sequence_member": is_sequence,
    }
    if record["kind"] == "video":
        record.update(_ffprobe(path))
        transfer = (record.get("color_transfer") or "").lower()
        if transfer in ("smpte2084", "arib-std-b67"):
            record["encoding_guess"] = "pq" if transfer == "smpte2084" else "hlg"
            record["encoding_reason"] = f"container:{transfer}"
        record["probe_frames_requested"] = video_probe_frames
    else:
        sample = _read_decimated(path)
        if sample is not None:
            record["shape"] = list(sample.shape)
            record.update(_radiometry(sample, record["encoding_guess"]))
    return record


# Directory names that hold RUDRA's own OUTPUT, not source footage. Feeding
# derived pairs back in as sources is silent poison: 08_Research/data/hdr holds
# already-normalised August targets (linear * 203/10000), and re-ingesting them
# as scene-linear put 4,071 pairs -- 21% of the 23 Aug corpus -- roughly 5,000x
# too dark, each frame its own single-frame "scene".
DERIVED_DIRS = {"data", "pairs", "hdrdata", "checkpoints", "_trash", "outputs"}


def is_derived(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts[:-1]
    except ValueError:
        parts = path.parts[:-1]
    return any(part.lower() in DERIVED_DIRS for part in parts)


def walk(root: Path, follow_links: bool, include_derived: bool = False) -> list[Path]:
    found: list[Path] = []
    skipped = 0
    for path in root.rglob("*"):
        try:
            if not path.is_file():
                continue
        except OSError:
            continue
        if path.suffix.lower() not in IMAGE_EXT | VIDEO_EXT:
            continue
        if not include_derived and is_derived(path, root):
            skipped += 1
            continue
        found.append(path)
    if skipped:
        print(f"  skipped {skipped:,} file(s) under {sorted(DERIVED_DIRS)} "
              f"-- RUDRA output, not source (--include-derived overrides)", flush=True)
    return found


def summarise(records: list[dict]) -> dict:
    scenes: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        scenes[record["scene_id"]].append(record)

    image_scenes = {s for s, rs in scenes.items() if rs[0]["kind"] == "image"}
    video_files = [r for r in records if r["kind"] == "video"]
    sequence_scenes = {s for s, rs in scenes.items()
                       if rs[0]["kind"] == "image" and len(rs) > 1}
    still_scenes = image_scenes - sequence_scenes

    stops = [r["dynamic_range_stops"] for r in records if r.get("dynamic_range_stops")]
    above = [r for r in records if r.get("above_10000_nits")]

    return {
        "files": len(records),
        "by_suffix": dict(Counter(r["suffix"] for r in records).most_common()),
        "by_encoding_guess": dict(Counter(r["encoding_guess"] for r in records).most_common()),
        "unknown_encoding": sum(1 for r in records if r["encoding_guess"] == "UNKNOWN"),
        "total_scenes": len(scenes),
        "still_scenes": len(still_scenes),
        "image_sequence_scenes": len(sequence_scenes),
        "video_container_files": len(video_files),
        "independent_moving_sources": len(sequence_scenes) + len(video_files),
        "largest_scene_frames": max((len(v) for v in scenes.values()), default=0),
        "dynamic_range_stops": {
            "median": round(float(np.median(stops)), 2) if stops else None,
            "p90": round(float(np.percentile(stops, 90)), 2) if stops else None,
            "max": round(max(stops), 2) if stops else None,
        },
        "sources_above_10000_nits": len(above),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="Directory to scan (recursively)")
    parser.add_argument("--out", type=Path, default=Path("hdrdata/source_inventory.jsonl"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--video-probe-frames", type=int, default=8)
    parser.add_argument("--encoding",
                        choices=("pq", "hlg", "logc3", "logc4", "slog3", "linear", "srgb"),
                        default=None,
                        help="Declare the transfer function for everything under this root, "
                             "overriding the filename heuristic. Use it whenever the dataset documents its encoding and the path does not say so -- HdM-HDR-2014 is PQ graded to 4,000 nits and nothing in its filenames says PQ.")
    parser.add_argument("--follow-links", action="store_true")
    parser.add_argument("--include-derived", action="store_true",
                        help="Scan directories named data/pairs/hdrdata/... too. Almost always wrong: those hold RUDRA output, and re-ingesting it as source is what poisoned 21%% of the 23 Aug 2026 corpus.")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N files (smoke test)")
    args = parser.parse_args()

    if not args.root.exists():
        print(f"error: {args.root} does not exist or is not reachable", file=sys.stderr)
        return 2

    print(f"walking {args.root} ...", flush=True)
    paths = walk(args.root, args.follow_links, args.include_derived)
    if args.limit:
        paths = paths[: args.limit]
    print(f"  {len(paths):,} candidate files", flush=True)
    if not paths:
        print("nothing to inventory", file=sys.stderr)
        return 1

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(inspect, p, args.video_probe_frames, args.encoding): p for p in paths}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                records.append(future.result())
            except Exception as exc:  # a bad file must not kill the scan
                records.append({"path": str(futures[future]), "error": repr(exc)})
            if done % 500 == 0:
                print(f"  inspected {done:,}/{len(paths):,}", flush=True)

    records.sort(key=lambda r: r["path"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    good = [r for r in records if "error" not in r]
    summary = summarise(good)
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"INVENTORY  {args.root}")
    print("=" * 66)
    print(f"  files inspected            {summary['files']:>8,}")
    print(f"  by format                  {summary['by_suffix']}")
    print(f"  by encoding guess          {summary['by_encoding_guess']}")
    if summary["unknown_encoding"]:
        print(f"  !! UNKNOWN encoding        {summary['unknown_encoding']:>8,}  "
              f"(review before ingest -- a wrong EOTF silently corrupts every pair)")
    print(f"\n  total scenes               {summary['total_scenes']:>8,}")
    print(f"    stills                   {summary['still_scenes']:>8,}")
    print(f"    image sequences          {summary['image_sequence_scenes']:>8,}")
    print(f"    video containers         {summary['video_container_files']:>8,}")
    print(f"  INDEPENDENT MOVING SOURCES {summary['independent_moving_sources']:>8,}"
          f"   <- the temporal gate reads this")
    print(f"  largest single scene       {summary['largest_scene_frames']:>8,} frames")
    print(f"\n  dynamic range (stops)      {summary['dynamic_range_stops']}")
    print(f"  sources peaking >10k nits  {summary['sources_above_10000_nits']:>8,}"
          f"   <- these clip under pq_10000 storage")
    print(f"\nwrote {args.out}  and  {summary_path}")

    if summary["independent_moving_sources"] < 10:
        print("\n  WARNING: fewer than 10 independent moving sources. Temporal training "
              "on this alone produces a demo, not a model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
