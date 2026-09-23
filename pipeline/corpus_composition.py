"""What a manifest is made of, per source and per licence -- and what is left
when the non-commercial sources go.

    python pipeline/corpus_composition.py G:\\corpus_v4b\\sdr_hdr_manifest.jsonl

Per source: records, scenes, video share, median / p90 peak nits, share of
records whose SDR clips (verify_dataset check 3b, floor 20%), and share of
records at a grading ceiling. Then the commercial subset measured against the
gates it will have to pass: 3b floor, largest-scene share (check 7, 25% on
val/test), scene count, and whether the paper-bench scenes are in it.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.licences import classify  # noqa: E402


def summarise(rows: list[dict]) -> dict:
    peaks = sorted(float(r.get("peak_nits") or 0.0) for r in rows)
    scenes = Counter(r["scene_id"] for r in rows)
    clip = sum(1 for r in rows if float(r.get("sdr_clipped_fraction") or 0) > 1e-4)
    video = sum(1 for r in rows if r.get("is_video"))
    ceiling = sum(1 for r in rows if r.get("at_ceiling") or r.get("censored"))
    return {
        "records": len(rows), "scenes": len(scenes),
        "video_share": video / max(len(rows), 1),
        "video_scenes": len({r["scene_id"] for r in rows if r.get("is_video")}),
        "peak_median": statistics.median(peaks) if peaks else 0.0,
        "peak_p90": peaks[int(0.9 * (len(peaks) - 1))] if peaks else 0.0,
        "sdr_clip_share": clip / max(len(rows), 1),
        "largest_scene_share": (scenes.most_common(1)[0][1] / len(rows)) if rows else 0.0,
        "ceiling_share": ceiling / max(len(rows), 1),
    }


def line(name: str, s: dict) -> str:
    return (f"  {name:<18} {s['records']:>7,} rec {s['scenes']:>5,} sc  video {s['video_share']:>5.1%}"
            f" ({s['video_scenes']} sc)  peak med {s['peak_median']:>8,.0f} p90 {s['peak_p90']:>9,.0f} nits"
            f"  SDR clips {s['sdr_clip_share']:>5.1%}  top scene {s['largest_scene_share']:>5.1%}")


def main(path: Path) -> int:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_source = defaultdict(list)
    for r in rows:
        lic = classify(r)
        r["_src"], r["_ok"] = lic["source"], lic["commercial_ok"]
        by_source[r["_src"]].append(r)
    print(f"{path}  {len(rows):,} records\n")
    for src, rs in sorted(by_source.items(), key=lambda kv: -len(kv[1])):
        tag = "" if rs[0]["_ok"] else "  [non-commercial]"
        print(line(src, summarise(rs)) + tag)
    everything = summarise(rows)
    commercial = [r for r in rows if r["_ok"]]
    c = summarise(commercial)
    print("\n" + line("ALL", everything))
    print(line("COMMERCIAL", c))
    print(f"\n  commercial keeps {c['records'] / max(len(rows), 1):.1%} of records, "
          f"{c['scenes'] / max(everything['scenes'], 1):.1%} of scenes")
    verdicts = [
        ("check 3b: >= 20% of records clip an SDR pixel", c["sdr_clip_share"] >= 0.20),
        ("scene diversity: >= 300 scenes", c["scenes"] >= 300),
        ("highlight range: p90 peak >= 1,000 nits", c["peak_p90"] >= 1000),
        ("real video: >= 6 video scenes (temporal floor)", c["video_scenes"] >= 6),
    ]
    test = [r for r in commercial if r.get("split") == "test"]
    if test:
        t = Counter(r["scene_id"] for r in test)
        verdicts.append(("check 7: largest test scene <= 25%", t.most_common(1)[0][1] / len(test) <= 0.25))
    print()
    for label, ok in verdicts:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(Path(sys.argv[1])))
