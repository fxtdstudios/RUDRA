"""Re-run scan_sources.guess_encoding over an existing inventory, without a rescan.

    python pipeline/reclassify_inventory.py IN.jsonl OUT.jsonl

A full scan of the source drives reads every header and takes hours. When
only the classification rules change -- 23 Sep 2026: float containers are
linear whatever the dataset is called, which un-breaks the 1,799 Sparks ACES
EXRs that were decoded as PQ -- the paths and probes already on disk are
enough. Declared encodings (reason "declared:cli") are left alone. Prints what
changed, by old -> new encoding.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.scan_sources import guess_encoding  # noqa: E402


def main(src: Path, dst: Path) -> int:
    changed: Counter = Counter()
    out = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("encoding_reason") != "declared:cli" and row.get("path"):
            enc, why = guess_encoding(Path(row["path"]))
            if enc != row.get("encoding_guess"):
                changed[(row.get("encoding_guess"), enc)] += 1
                row["encoding_guess"], row["encoding_reason"] = enc, why
        out.append(json.dumps(row))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{len(out):,} sources -> {dst}")
    for (a, b), n in changed.most_common():
        print(f"  {n:>7,}  {a} -> {b}")
    if not changed:
        print("  nothing reclassified")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
