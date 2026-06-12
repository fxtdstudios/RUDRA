#!/usr/bin/env python3
"""
cleanup_checkpoints.py — reclaim disk in hdrdata/checkpoints.

Two jobs:
  1. DEAD runs — the rudra/ research-architecture decoders (NOT loadable in the
     node) and the OOM'd LoRA. Anything that isn't a node decoder run
     (turbo_* / full_*). Archived by default (moved aside), or deleted.
  2. PRUNE steps — inside the GOOD node runs, drop the intermediate per-step
     .pth / .safetensors and keep only `*_decoder_ema_best.safetensors` (the one
     you actually deploy) plus the newest checkpoint of each type.

Always dry-run first.

Usage
-----
  python training/cleanup_checkpoints.py --dry-run
  python training/cleanup_checkpoints.py --dead archive --prune-steps
  python training/cleanup_checkpoints.py --dead delete --prune-steps   # hard delete
"""

from __future__ import annotations

import argparse
import re
import shutil
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / "hdrdata" / "checkpoints"


def is_node_run(d: Path) -> bool:
    """Good runs are turbo_* / full_* dirs containing a RadianceDecoder best-EMA."""
    if not (d.name.startswith("turbo_") or d.name.startswith("full_")):
        return False
    return any(d.rglob("*_decoder_ema_best.safetensors"))


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def main():
    ap = argparse.ArgumentParser(description="Clean up checkpoint disk.")
    ap.add_argument("--dead", choices=["keep", "archive", "delete"], default="archive",
                    help="What to do with non-node (research/dead) runs.")
    ap.add_argument("--prune-steps", action="store_true",
                    help="In good runs, delete intermediate step files (keep best-EMA + newest).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not CKPT.is_dir():
        print(f"[error] {CKPT} not found"); return 1

    runs = [d for d in CKPT.iterdir() if d.is_dir() and not d.name.startswith("_archive")]
    good = [d for d in runs if is_node_run(d)]
    dead = [d for d in runs if d not in good]

    freed = 0
    print(f"=== node decoder runs (kept): {len(good)} ===")
    for d in sorted(good):
        print(f"  ✓ {d.name}")

    # 1. Dead runs
    print(f"\n=== dead / research runs: {len(dead)} ({args.dead}) ===")
    archive_root = CKPT / f"_archive_{time.strftime('%Y%m%d_%H%M%S')}"
    for d in sorted(dead):
        sz = dir_size(d)
        freed += sz
        print(f"  {d.name:<16} {human(sz):>9}  -> {args.dead}")
        if args.dry_run or args.dead == "keep":
            continue
        if args.dead == "archive":
            archive_root.mkdir(parents=True, exist_ok=True)
            shutil.move(str(d), str(archive_root / d.name))
        elif args.dead == "delete":
            shutil.rmtree(d, ignore_errors=True)

    # 2. Prune intermediate steps in good runs
    if args.prune_steps:
        print(f"\n=== pruning intermediate step files in good runs ===")
        for d in sorted(good):
            files = list(d.rglob("*decoder_step*")) + list(d.rglob("*decoder_ema_step*"))
            # keep the newest of each pattern; always keep *_ema_best.
            def stepnum(f):
                m = re.search(r"step0*(\d+)", f.name)
                return int(m.group(1)) if m else -1
            pth = sorted([f for f in files if f.suffix == ".pth"], key=stepnum)
            ema = sorted([f for f in files if f.suffix == ".safetensors"], key=stepnum)
            victims = pth[:-1] + ema[:-1]   # keep newest of each
            for f in victims:
                sz = f.stat().st_size
                freed += sz
                print(f"  {d.name}/{f.name}  {human(sz)}")
                if not args.dry_run:
                    f.unlink(missing_ok=True)

    print(f"\n=== {'WOULD free' if args.dry_run else 'freed'}: {human(freed)} ===")
    if args.dry_run:
        print("Re-run without --dry-run to apply (use --dead delete for hard delete).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
