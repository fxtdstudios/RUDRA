"""Re-run scan_sources.guess_encoding over an existing inventory, without a rescan,
and point it at where the files live now.

    python pipeline/reclassify_inventory.py IN.jsonl OUT.jsonl [--remap OLD=NEW ...]

A full scan of the source drives reads every header and takes hours. When
only the classification rules change -- 23 Sep 2026: float containers are
linear whatever the dataset is called, which un-breaks the 1,799 Sparks ACES
EXRs that were decoded as PQ -- the paths and probes already on disk are
enough. Declared encodings (reason "declared:cli") are left alone. Prints what
changed, by old -> new encoding.

Paths: the v4b inventory was scanned before the sources moved to
G:\\datasets\\sources and three of them were renamed on the way
(pipeline/migrate_datasets_to_g.ps1). E:\\source_hdr is now a junction to
G:\\datasets\\sources, so the old prefix resolves but the renamed folders do
not -- on 23 Sep 2026 the v4c ingest skipped every HdM-HFR frame with
"No such file". Each missing path is tried against MIGRATIONS (plus any
--remap), longest old prefix first, case-insensitively, and the first that
exists wins. scene_id is left as scanned, so scene identities, the hold-out
match and the licence rules see the same text as v4b. Anything still missing
is listed and the script exits 1 unless --allow-missing.
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

# Old prefix -> new prefix, from pipeline/migrate_datasets_to_g.ps1.
MIGRATIONS = (
    (r"E:\source_hdr\HdM-HFR-2017_Color-Graded", r"G:\datasets\sources\hdm_hfr_2017"),
    (r"E:\source_hdr\Stuttgart_HDR_2014", r"G:\datasets\sources\stuttgart_hdr_2014"),
    (r"E:\source_hdr\Netflix\tif_DCI4k2398p", r"G:\datasets\sources\netflix_chimera"),
    (r"E:\source_hdr\Netflix", r"G:\datasets\sources\netflix_chimera"),
    (r"G:\datasets_rudra", r"G:\datasets\sources"),
    (r"E:\source_hdr", r"G:\datasets\sources"),
)


# Second renditions of footage already in the inventory. Ingesting both puts
# the same frames into the corpus twice under two scene ids -- one can land in
# train and the other in test. The kept rendition is named in the comment.
DUPLICATE_RENDITIONS = (
    # Sparks: SPARKS_ACES_* (scene-referred ACES 2065-1, to ~100,000 nits) is
    # kept; SPARKS_P3_PQ_4000nit_* is the same 13,776 frames as a 4,000-nit
    # P3 display grade.
    "sparks_p3_pq_4000nit",
)


def _norm(path: str) -> str:
    return path.replace("/", "\\")


def resolve(path: str, rules, exists=lambda p: Path(p).exists()) -> str | None:
    """The path where the file is now, or None."""
    if exists(path):
        return path
    flat = _norm(path)
    for old, new in sorted(rules, key=lambda r: -len(r[0])):
        if flat.lower().startswith(_norm(old).lower().rstrip("\\") + "\\"):
            candidate = _norm(new).rstrip("\\") + flat[len(_norm(old).rstrip("\\")):]
            if exists(candidate):
                return candidate
    return None


def main(src: Path, dst: Path, rules=MIGRATIONS, allow_missing: bool = False,
         commercial_only: bool = False) -> int:
    from pipeline.licences import classify
    dropped: Counter = Counter()
    changed: Counter = Counter()
    moved: Counter = Counter()
    missing: list[str] = []
    out = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if any(tag in str(row.get("path", "")).lower() for tag in DUPLICATE_RENDITIONS):
            dropped["duplicate rendition"] += 1
            continue
        if commercial_only:
            # Decided on the scanned text, before any file lookup: a source
            # that will not be ingested does not need to be found.
            lic = classify(row)
            if not lic["commercial_ok"]:
                dropped[lic["source"]] += 1
                continue
        if row.get("path") and row.get("kind", "image") == "image":
            now = resolve(row["path"], rules)
            if now is None:
                missing.append(row["path"])
            elif now != row["path"]:
                moved["\\".join(_norm(row["path"]).split("\\")[:3])] += 1
                row["scanned_path"], row["path"] = row["path"], now
        if row.get("encoding_reason") != "declared:cli" and row.get("path"):
            enc, why = guess_encoding(Path(row["path"]))
            if enc != row.get("encoding_guess"):
                changed[(row.get("encoding_guess"), enc)] += 1
                row["encoding_guess"], row["encoding_reason"] = enc, why
        out.append(json.dumps(row))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{len(out):,} sources -> {dst}")
    for sid, n in dropped.most_common():
        print(f"  {n:>7,}  dropped: {sid}")
    for (a, b), n in changed.most_common():
        print(f"  {n:>7,}  {a} -> {b}")
    if not changed:
        print("  nothing reclassified")
    print(f"  {sum(moved.values()):,} paths remapped to their current location")
    for root, n in moved.most_common(8):
        print(f"  {n:>7,}  from {root}")
    if missing:
        print(f"  {len(missing):,} sources exist nowhere known, e.g.:")
        for m in missing[:5]:
            print(f"           {m}")
        if not allow_missing:
            print("  add --remap OLD=NEW for their new location, or --allow-missing to drop them")
            return 1
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--remap", action="append", default=[], metavar="OLD=NEW")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--commercial-only", action="store_true",
                    help="drop every source pipeline/licences.py does not clear for sold "
                         "weights (HdM-HDR-2014, HdM-HFR-2017, unclassified) before ingest")
    a = ap.parse_args()
    extra = tuple(tuple(r.split("=", 1)) for r in a.remap)
    sys.exit(main(a.src, a.dst, extra + MIGRATIONS, a.allow_missing, a.commercial_only))
