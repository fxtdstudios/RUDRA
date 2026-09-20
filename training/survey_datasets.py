"""What is on the dataset drives, how big it is, and what still refers to it.

Proposes. Never deletes.

Disk gets reclaimed by deleting the wrong thing at least as often as the right
one, and on this project a deleted corpus is not recoverable: every checkpoint
pins its manifest by SHA-256, so a result whose corpus is gone stops being
reproducible the moment you need to defend it. So this walks the roots, walks
the manifests, and reports which directories are still spoken for. The decision
stays with a person.

    python training/survey_datasets.py ^
        --root G:\\datasets_rudra --root E:\\source_hdr ^
        --root E:\\RUDRA_v3_20260822 --root E:\\RUDRA_postfix_20260818 ^
        --manifest E:\\RUDRA_v3_20260822\\sdr_hdr_manifest.jsonl ^
        --manifest E:\\RUDRA_v3_20260822\\video_manifest_9f.jsonl ^
        --checkpoints checkpoints ^
        --output survey.json

Read the REFERENCED column before you read the size column.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HUMAN = [("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)]


def human(n: int) -> str:
    for unit, scale in HUMAN:
        if n >= scale:
            return f"{n / scale:,.1f} {unit}"
    return f"{n} B"


def walk(root: Path):
    """Total size, file count and newest mtime under each immediate child."""
    out = {}
    if not root.exists():
        return {"__missing__": {"bytes": 0, "files": 0, "mtime": 0.0}}
    children = [p for p in root.iterdir()] or [root]
    for child in children:
        total = files = 0
        newest = 0.0
        if child.is_file():
            try:
                st = child.stat()
                total, files, newest = st.st_size, 1, st.st_mtime
            except OSError:
                pass
        else:
            for dirpath, _, filenames in os.walk(child):
                for name in filenames:
                    try:
                        st = os.stat(os.path.join(dirpath, name))
                    except OSError:
                        continue
                    total += st.st_size
                    files += 1
                    newest = max(newest, st.st_mtime)
        out[child.name] = {"bytes": total, "files": files, "mtime": newest}
    return out


def manifest_paths(path: Path):
    """Every filesystem path a manifest mentions, lowercased.

    Manifests are JSON lines with a handful of path-ish keys. Rather than
    guessing which, take any string value that looks like a path: a missed key
    here marks live data as unreferenced, which is the direction that loses
    someone's corpus.
    """
    seen = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            stack = [row]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    stack.extend(item.values())
                elif isinstance(item, list):
                    stack.extend(item)
                elif isinstance(item, str) and ("/" in item or "\\" in item):
                    seen.add(item.replace("\\", "/").lower())
    return seen


def checkpoint_corpora(root: Path):
    """Manifest names and SHAs pinned inside checkpoint configs."""
    pins = []
    if not root.exists():
        return pins
    for cfg in root.rglob("config.json"):
        try:
            data = json.loads(cfg.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        keys = {k: v for k, v in data.items()
                if "manifest" in k.lower() or "corpus" in k.lower() or "sha" in k.lower()}
        if keys:
            pins.append({"checkpoint": str(cfg.parent.name), "pins": keys})
    return pins


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=[], type=Path, required=True)
    ap.add_argument("--manifest", action="append", default=[], type=Path)
    ap.add_argument("--checkpoints", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)

    referenced = set()
    for m in args.manifest:
        found = manifest_paths(m)
        print(f"manifest {m}: {len(found):,} distinct paths")
        referenced |= found
    if args.manifest and not referenced:
        print("\n  WARNING: the manifests named no paths at all. Every directory\n"
              "  below will look unreferenced, which is almost certainly wrong.\n"
              "  Check the manifest paths before deleting anything.\n")

    pins = checkpoint_corpora(args.checkpoints) if args.checkpoints else []
    if pins:
        print(f"\n{len(pins)} checkpoint(s) pin a corpus:")
        for p in pins:
            print(f"  {p['checkpoint']}: {json.dumps(p['pins'])[:110]}")

    report, grand = {}, 0
    print()
    for root in args.root:
        entries = walk(root)
        report[str(root)] = entries
        print("=" * 78)
        print(root)
        rows = sorted(entries.items(), key=lambda kv: -kv[1]["bytes"])
        for name, info in rows:
            key = (str(root / name)).replace("\\", "/").lower()
            hit = any(key in r or r.startswith(key) for r in referenced)
            info["referenced"] = hit
            grand += info["bytes"]
            mark = "referenced" if hit else "-"
            print(f"  {name[:38]:<38} {human(info['bytes']):>10}  "
                  f"{info['files']:>8,} files  {mark}")
    print("=" * 78)
    print(f"  {'total':<38} {human(grand):>10}")

    loose = [(root, name, info)
             for root, entries in report.items()
             for name, info in entries.items()
             if not info.get("referenced") and info["bytes"] > 0]
    if loose:
        loose.sort(key=lambda t: -t[2]["bytes"])
        freeable = sum(t[2]["bytes"] for t in loose)
        print(f"\nNot referenced by any manifest given here: {human(freeable)} across "
              f"{len(loose)} directories.")
        print("That is a CANDIDATE list, not a delete list. A directory can be")
        print("unreferenced because it is genuinely dead, or because the manifest")
        print("that referred to it was not passed to this script.\n")
        for root, name, info in loose[:40]:
            print(f"  {human(info['bytes']):>10}  {root}\\{name}")

    if args.output:
        args.output.write_text(json.dumps(
            {"roots": report, "checkpoint_pins": pins,
             "referenced_paths": len(referenced)}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
