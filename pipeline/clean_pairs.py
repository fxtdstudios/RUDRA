"""Find and remove pair files that no manifest references any more.

Re-running prepare_pairs.py rewrites pairs_index.jsonl but does NOT delete the
PNGs an earlier run wrote: the stems encode the source's position in the
inventory, so a different source list produces different names and the old
files just accumulate. After the 23 Aug 2026 runs, E:\\RUDRA_v3_20260822\\pairs
held ~86k pairs' worth of files while the index referenced 8k.

Orphans are dead weight, not corruption -- the trainer reads the manifest, not
the directory. But they hide the real corpus size and fill the drive.

    python pipeline\\clean_pairs.py E:\\RUDRA_v3_20260822        # report only
    python pipeline\\clean_pairs.py E:\\RUDRA_v3_20260822 --apply   # move to _trash
    python pipeline\\clean_pairs.py E:\\RUDRA_v3_20260822 --apply --delete

Default --apply MOVES orphans to <work>\\_trash so a mistake is reversible;
add --delete to remove them outright. Checkpoints (.pt/.pth/.safetensors) are
never touched by this tool.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PAIR_SUBDIRS = ("sdr", "hdr", "meta")
PROTECTED_SUFFIXES = (".pt", ".pth", ".safetensors", ".ckpt")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} PB"


def referenced_files(pairs_dir: Path) -> set[str]:
    """Every file path the current index points at, lower-cased for Windows."""
    index = pairs_dir / "pairs_index.jsonl"
    if not index.exists():
        return set()
    keep: set[str] = set()
    for line in index.open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        for key in ("sdr_path", "hdr_path", "metadata_path"):
            value = record.get(key)
            if value:
                keep.add(Path(value).name.lower())
    return keep


def scan_pairs_dir(pairs_dir: Path) -> tuple[list[Path], int, int, int]:
    """Return (orphans, orphan_bytes, kept_count, kept_bytes)."""
    keep = referenced_files(pairs_dir)
    orphans: list[Path] = []
    orphan_bytes = kept = kept_bytes = 0
    for sub in PAIR_SUBDIRS:
        directory = pairs_dir / sub
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() in PROTECTED_SUFFIXES:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if path.name.lower() in keep:
                kept += 1
                kept_bytes += size
            else:
                orphans.append(path)
                orphan_bytes += size
    return orphans, orphan_bytes, kept, kept_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("work_dirs", nargs="+", type=Path,
                        help="work dir(s) containing a 'pairs' folder, or a pairs folder itself")
    parser.add_argument("--apply", action="store_true", help="actually act on the orphans")
    parser.add_argument("--delete", action="store_true",
                        help="with --apply: delete instead of moving to _trash")
    args = parser.parse_args()

    total_orphan_bytes = 0
    acted = 0
    for work in args.work_dirs:
        pairs_dir = work if (work / "pairs_index.jsonl").exists() else work / "pairs"
        if not (pairs_dir / "pairs_index.jsonl").exists():
            print(f"skip {work}: no pairs_index.jsonl (nothing to compare against)")
            continue

        print(f"\n=== {pairs_dir}")
        orphans, orphan_bytes, kept, kept_bytes = scan_pairs_dir(pairs_dir)
        print(f"  referenced by the index : {kept:>9,} files  {human(kept_bytes)}")
        print(f"  orphaned (no reference) : {len(orphans):>9,} files  {human(orphan_bytes)}")
        total_orphan_bytes += orphan_bytes

        if not orphans:
            print("  nothing to clean")
            continue
        for path in orphans[:3]:
            print(f"    e.g. {path.parent.name}/{path.name}")

        if not args.apply:
            print("  DRY RUN -- add --apply to move these to _trash "
                  "(or --apply --delete to remove them)")
            continue

        if args.delete:
            for path in orphans:
                try:
                    path.unlink()
                    acted += 1
                except OSError as exc:
                    print(f"    could not delete {path.name}: {exc}", file=sys.stderr)
            print(f"  deleted {acted:,} files")
        else:
            trash = pairs_dir.parent / "_trash" / pairs_dir.name
            for path in orphans:
                target = trash / path.parent.name
                target.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(path), str(target / path.name))
                    acted += 1
                except OSError as exc:
                    print(f"    could not move {path.name}: {exc}", file=sys.stderr)
            print(f"  moved {acted:,} files -> {trash}")
            print("  (delete that folder yourself once the next run looks right)")

    print(f"\ntotal orphaned: {human(total_orphan_bytes)}")
    if not args.apply and total_orphan_bytes:
        print("re-run with --apply to reclaim it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
