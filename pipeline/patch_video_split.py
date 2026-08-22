"""Make the temporal dataset honour the manifest's split instead of inventing one.

``training/sdr2hdr_dataset.py`` currently computes its own held-out set:

    n_val = max(1, round(len(scenes) * val_fraction))

With two scenes in the manifest that is ``max(1, 0) == 1``: one clip to train on
and one to validate against. That is what produced the August 2026 temporal run
in which all 22 evaluations were worse than doing nothing.

This patch:
  * uses the per-clip ``split`` field when build_manifests.py wrote one;
  * falls back to the fractional split only when no field exists;
  * refuses outright below ``MIN_TEMPORAL_SCENES`` rather than silently
    producing a one-scene training set.

Idempotent. Writes a .bak the first time. Run with --check to see the diff only.

    python pipeline/patch_video_split.py --check
    python pipeline/patch_video_split.py --apply
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "training" / "sdr2hdr_dataset.py"

MARKER = "# --- pipeline/patch_video_split.py ---"

OLD = """        records = read_jsonl(manifest_path)
        if split in {"train", "val"} and val_fraction > 0:
            scenes = sorted({str(r["scene_id"]) for r in records})
            n_val = max(1, round(len(scenes) * val_fraction))
            val_scenes = set(scenes[-n_val:])
            records = [r for r in records if (str(r["scene_id"]) in val_scenes) == (split == "val")]"""

NEW = f"""        records = read_jsonl(manifest_path)
        {MARKER}
        MIN_TEMPORAL_SCENES = 6
        _scenes = sorted({{str(r["scene_id"]) for r in records}})
        if len(_scenes) < MIN_TEMPORAL_SCENES:
            raise ValueError(
                f"temporal training needs >= {{MIN_TEMPORAL_SCENES}} independent scenes, "
                f"the manifest has {{len(_scenes)}}: {{_scenes}}. Training on this many "
                f"produces a model of the clip, not of the task -- the August 2026 run "
                f"diverged from a one-scene training set. Add sources, or run image mode."
            )
        if split in {{"train", "val"}} and any("split" in r for r in records):
            # build_manifests.py assigned whole scenes; honour that, do not re-split.
            records = [r for r in records if str(r.get("split")) == split]
        elif split in {{"train", "val"}} and val_fraction > 0:
            n_val = max(1, round(len(_scenes) * val_fraction))
            val_scenes = set(_scenes[-n_val:])
            records = [r for r in records if (str(r["scene_id"]) in val_scenes) == (split == "val")]
        # --- end patch ---"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--apply", action="store_true")
    parser.add_argument("--target", type=Path, default=TARGET)
    args = parser.parse_args()

    if not args.target.exists():
        print(f"error: {args.target} not found", file=sys.stderr)
        return 2
    source = args.target.read_text(encoding="utf-8")

    if MARKER in source:
        print(f"already patched: {args.target}")
        return 0
    if OLD not in source:
        print("error: the expected block was not found -- the file has changed since this "
              "patch was written. Apply the change by hand:\n", file=sys.stderr)
        print(NEW, file=sys.stderr)
        return 3

    patched = source.replace(OLD, NEW)
    diff = difflib.unified_diff(source.splitlines(True), patched.splitlines(True),
                               fromfile=str(args.target), tofile=str(args.target) + " (patched)")
    print("".join(diff))

    if args.check:
        print("\n--check only; nothing written. Re-run with --apply.")
        return 0

    backup = args.target.with_suffix(args.target.suffix + ".bak")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
        print(f"backup -> {backup}")
    args.target.write_text(patched, encoding="utf-8")
    print(f"patched -> {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
