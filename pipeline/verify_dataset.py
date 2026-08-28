"""The gate. Refuses a dataset that carries any defect the August 2026 audit found.

Run it after prepare_pairs.py + build_manifests.py and before any trainer. It
exits non-zero on FAIL so it can sit in front of training in a batch file:

    python pipeline/verify_dataset.py --pairs-dir hdrdata/pairs_v3 \\
        --manifest hdrdata/sdr_hdr_manifest.jsonl \\
        --video-manifest hdrdata/video_manifest_9f.jsonl || exit /b 1

Checks, each mapped to the defect it catches:

  1  sentinel present, hdr_io version matches      -- mixed-convention corpora
  2  encode/decode round-trip within tolerance     -- silent colour-pipeline drift
  3  clipped fraction under threshold              -- 78% of stills lost their sun
  4  shadow code precision                         -- 16-bit linear left ~8 bits
  5  no scene straddles a split                    -- leakage
  6  every split contains video                    -- val/test had zero frames
  7  corpus concentration                          -- 92% of records from 2 clips
  8  independent moving sources vs temporal floor  -- 1-clip temporal training
  9  loss ceiling vs model max_hdr                 -- supervision the net can't reach
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.hdr_io import (  # noqa: E402
    HDR_IO_VERSION, HDRStorage, UINT16_MAX, _code_for_nits, selftest, shadow_precision,
)

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str) -> None:
        self.rows.append((status, name, detail))

    @property
    def failed(self) -> bool:
        return any(status == FAIL for status, _, _ in self.rows)

    def render(self) -> str:
        width = max(len(name) for _, name, _ in self.rows)
        lines = []
        for status, name, detail in sorted(self.rows, key=lambda r: r[1]):
            lines.append(f"  [{status}] {name.ljust(width)}  {detail}")
        return "\n".join(lines)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def check_sentinel(pairs_dir: Path, report: Report) -> HDRStorage | None:
    sentinel = pairs_dir / "_ingest_config.json"
    if not sentinel.exists():
        report.add(FAIL, "1 sentinel", f"{sentinel} missing -- cannot know how targets are encoded")
        return None
    payload = json.loads(sentinel.read_text(encoding="utf-8"))
    storage = HDRStorage.from_dict(payload.get("storage", {}))
    if storage.version != HDR_IO_VERSION:
        report.add(FAIL, "1 sentinel", f"written by hdr_io v{storage.version}, this is v{HDR_IO_VERSION}")
        return storage
    report.add(PASS, "1 sentinel", storage.describe())
    return storage


def check_roundtrip(storage: HDRStorage, report: Report, tolerance: float) -> None:
    try:
        result = selftest(storage, tolerance)
    except AssertionError as exc:
        report.add(FAIL, "2 round-trip", str(exc))
        return
    report.add(PASS, "2 round-trip",
               f"worst {result['worst_relative_error']:.4%}, median "
               f"{result['median_relative_error']:.4%} over {result['probes']} probes")


def check_clipping(records: list[dict], report: Report, max_clipped: float) -> None:
    fractions = [r.get("clipped_fraction", 0.0) for r in records if "clipped_fraction" in r]
    if not fractions:
        report.add(WARN, "3 highlight clipping", "no clipped_fraction recorded -- re-run prepare_pairs")
        return
    arr = np.asarray(fractions, dtype=float)
    affected = float((arr > 0.0001).mean())
    detail = (f"{affected:.1%} of records clip any pixel; corpus mean "
              f"{arr.mean():.4%}, worst {arr.max():.3%}")
    status = FAIL if affected > max_clipped else PASS
    if status == FAIL:
        detail += f"  (threshold {max_clipped:.0%}; August corpus was 77.8%)"
    report.add(status, "3 highlight clipping", detail)


def check_shadow_precision(storage: HDRStorage, report: Report, minimum: int) -> None:
    codes_below_white = _code_for_nits(storage.diffuse_white_nits, storage) * UINT16_MAX
    detail = (f"{codes_below_white:,.0f} codes below diffuse white "
              f"({storage.diffuse_white_nits:g} nits); 16-bit linear gave 1,330")
    report.add(PASS if codes_below_white >= minimum else FAIL, "4 shadow precision", detail)


def check_leakage(rows: list[dict], report: Report) -> None:
    straddle = defaultdict(set)
    for row in rows:
        straddle[row["scene_id"]].add(row.get("split", "?"))
    leaked = [s for s, splits in straddle.items() if len(splits) > 1]
    if leaked:
        report.add(FAIL, "5 scene leakage", f"{len(leaked)} scenes span splits, e.g. {leaked[:3]}")
    else:
        report.add(PASS, "5 scene leakage", f"{len(straddle):,} scenes, none straddling")


def check_split_video(rows: list[dict], report: Report, min_share: float) -> None:
    has_video = any(r.get("is_video") for r in rows)
    if not has_video:
        report.add(WARN, "6 video in splits", "corpus contains no video at all")
        return
    details, worst = [], PASS
    for split in ("train", "val", "test"):
        subset = [r for r in rows if r.get("split") == split]
        if not subset:
            continue
        video = [r for r in subset if r.get("is_video")]
        share = len(video) / len(subset)
        scenes = len({r["scene_id"] for r in video})
        details.append(f"{split} {share:.0%}/{scenes}sc")
        if split in ("val", "test") and (scenes == 0 or share < min_share):
            worst = FAIL
    suffix = "" if worst == PASS else "  (August val/test held ZERO video frames)"
    report.add(worst, "6 video in splits", ", ".join(details) + suffix)


def check_concentration(rows: list[dict], report: Report, max_share: float) -> None:
    """How much of each split is one scene.

    Judged per split, and judged differently by split, because the consequence
    differs:

      * val/test are plain means over records, so a scene's share IS its weight
        in every number you will quote. One scene at 95.6% of val -- the Bar
        interior, 23 Aug 2026 -- means the held-out metric measures one room.
        That is a FAIL.
      * train draws through a scene-balanced weighted sampler, so record counts
        do not set a scene's influence there. A high share is worth saying out
        loud, but it is not the same defect. That is a WARN.
    """
    for split in ("train", "val", "test"):
        subset = [r for r in rows if r.get("split") == split]
        if not subset:
            continue
        counts = Counter(r["scene_id"] for r in subset)
        total = sum(counts.values())
        share = counts.most_common(1)[0][1] / total
        top_two = sum(c for _, c in counts.most_common(2)) / total
        detail = (f"{split}: largest scene {share:.1%} of records, top two "
                  f"{top_two:.1%} across {len(counts):,} scenes")
        if split == "train":
            status = WARN if share > max_share else PASS
            if status is WARN:
                detail += "  (scene-balanced sampling compensates; more scenes is the fix)"
        else:
            status = FAIL if share > max_share else (
                WARN if top_two > max_share * 1.5 else PASS)
            if status != PASS:
                detail += f"  (threshold {max_share:.0%}; August top two were 91.8%)"
        report.add(status, f"7 concentration/{split}", detail)


def check_temporal_floor(clips: list[dict], report: Report, minimum: int) -> None:
    if not clips:
        report.add(WARN, "8 temporal floor", "no video manifest -- temporal training unavailable")
        return
    per_split = defaultdict(set)
    for clip in clips:
        per_split[clip.get("split", "?")].add(clip["scene_id"])
    train = len(per_split.get("train", set()))
    held = len(per_split.get("val", set())) + len(per_split.get("test", set()))
    detail = f"{train} train scenes, {held} held out, {len(clips):,} clips"
    status = FAIL if train < minimum or held < 1 else PASS
    if status == FAIL:
        detail += (f"  (need >= {minimum} train scenes; the August run trained on 1 "
                   f"and every eval was worse than the baseline)")
    report.add(status, "8 temporal floor", detail)


def check_loss_ceiling(storage: HDRStorage, report: Report, max_hdr: float,
                       rows: list[dict] | None = None) -> None:
    """SDR2HDRNet(max_hdr) caps what the network can emit, in units of 10,000 nits."""
    net_ceiling = max_hdr * 10_000.0
    store_ceiling = 10_000.0 if storage.mode == "pq_10000" else storage.ceiling_nits
    if store_ceiling > net_ceiling:
        affected = ""
        if rows:
            peaks = [float(r.get("peak_nits", 0.0)) for r in rows]
            over = sum(1 for x in peaks if x == x and x > net_ceiling)
            affected = f" {over:,}/{len(peaks):,} records ({over / len(peaks):.1%}) reach past it;"
        report.add(WARN, "9 loss ceiling",
                   f"targets reach {store_ceiling:,.0f} nits, network caps at "
                   f"{net_ceiling:,.0f} (max_hdr={max_hdr}).{affected} the dataset clamps "
                   f"targets there so the residue is unreachable, not unlearnable -- "
                   f"raise max_hdr and DEFAULT_TARGET_CEILING together to recover it.")
    else:
        report.add(PASS, "9 loss ceiling",
                   f"targets <= {store_ceiling:,.0f} nits, network reaches {net_ceiling:,.0f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-manifest", type=Path)
    parser.add_argument("--max-clipped-records", type=float, default=0.02)
    parser.add_argument("--min-shadow-codes", type=int, default=8000)
    parser.add_argument("--min-video-share", type=float, default=0.15)
    parser.add_argument("--max-scene-share", type=float, default=0.25)
    parser.add_argument("--min-temporal-scenes", type=int, default=6)
    parser.add_argument("--max-hdr", type=float, default=4.0)
    parser.add_argument("--roundtrip-tolerance", type=float, default=0.01)
    args = parser.parse_args()

    report = Report()
    storage = check_sentinel(args.pairs_dir, report)
    if storage is not None:
        check_roundtrip(storage, report, args.roundtrip_tolerance)
        check_shadow_precision(storage, report, args.min_shadow_codes)

    rows = read_jsonl(args.manifest)
    check_clipping(rows, report, args.max_clipped_records)
    check_leakage(rows, report)
    check_split_video(rows, report, args.min_video_share)
    check_concentration(rows, report, args.max_scene_share)

    clips = read_jsonl(args.video_manifest) if args.video_manifest and args.video_manifest.exists() else []
    check_temporal_floor(clips, report, args.min_temporal_scenes)
    if storage is not None:
        check_loss_ceiling(storage, report, args.max_hdr, rows)

    print("=" * 78)
    print(f"DATASET VERIFICATION  {args.pairs_dir}")
    print("=" * 78)
    print(report.render())
    if storage is not None:
        print("\n  shadow code allocation under this encoding:")
        for name, info in shadow_precision(storage).items():
            print(f"    {name:<26} code {info['code']:>10,}")
    print()
    if report.failed:
        print("VERDICT: FAIL -- do not train on this dataset.")
        return 1
    print("VERDICT: PASS -- safe to train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
