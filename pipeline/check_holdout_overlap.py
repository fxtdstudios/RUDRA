"""Did the previous corpus's benchmark scenes land in this corpus's training set?

    python pipeline/check_holdout_overlap.py ^
        --old-manifest E:\\RUDRA_v3_20260822\\sdr_hdr_manifest.jsonl ^
        --manifest G:\\corpus_v4b\\sdr_hdr_manifest.jsonl

The paper's 429 frames are the old manifest's test split. When the same
sources are re-ingested under a fresh split, some of those scenes land in
train, and a model trained there cannot be scored on the old bench as
"held out". Scene ids embed the drive path, so matching uses scene_tail().

Exit 0 when none of the old test scenes are in train here, 1 when some are
(the count is printed), 2 when none of them exist here at all (wrong file).
Writes a JSON summary with --out.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.build_manifests import load_hold_out_scenes, scene_tail  # noqa: E402


def overlap(old_manifest: Path, manifest: Path) -> dict:
    wanted = load_hold_out_scenes(old_manifest)
    where: dict[str, set[str]] = {}
    records = Counter()
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tail = scene_tail(row.get("scene_id", ""))
            if tail in wanted:
                where.setdefault(tail, set()).add(row.get("split", "?"))
                records[row.get("split", "?")] += 1
    by_split = Counter()
    for splits in where.values():
        for split in splits:
            by_split[split] += 1
    return {
        "old_manifest": str(old_manifest), "manifest": str(manifest),
        "old_test_scenes": len(wanted), "found_here": len(where),
        "scenes_by_split": dict(by_split), "records_by_split": dict(records),
        "in_train": sorted(t for t, s in where.items() if "train" in s),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = overlap(args.old_manifest, args.manifest)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"old test scenes: {result['old_test_scenes']}   found here: {result['found_here']}   "
          f"by split: {result['scenes_by_split']}   records: {result['records_by_split']}")
    if result["found_here"] == 0:
        print("NONE of the old test scenes exist in this manifest -- wrong file, or ids do not line up")
        return 2
    if result["in_train"]:
        print(f"{len(result['in_train'])} old bench scenes are in TRAIN here; a model trained on this "
              f"manifest is not held out on the old bench. Rebuild with --hold-out-scenes.")
        return 1
    print("clean: no old bench scene is in train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
