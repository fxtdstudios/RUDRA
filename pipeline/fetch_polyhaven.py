#!/usr/bin/env python3
"""Fetch Poly Haven HDRIs, resumably, at a resolution the renderer will accept.

The video corpus has 935 clips and 13 SCENES, split 11 train / 1 val / 1 test,
so the temporal refiner cannot be evaluated. `render_hdri_moves.py` fixes that
by flying a virtual camera through scene-referred panoramas: one panorama is
one scene, and Poly Haven publishes 993 of them under CC0.

Resolution is not a preference here, it is a gate. The renderer refuses more
than `--max-upscale` (default 2.0x):

    2k  2048 px equirect ->  427 source px for a 1280-wide 75 deg frame -> 3.0x  REFUSED
    4k  4096 px          ->  853 px                                     -> 1.5x  ok
    8k  8192 px          -> 1707 px                                     -> 0.75x ok

So 4k is the smallest resolution that is usable, and it is what this defaults
to. 8k is ~90 MB an asset and better if you have the disk.

    python pipeline/fetch_polyhaven.py --list
    python pipeline/fetch_polyhaven.py --dest E:/source_hdr/PolyHaven_4k
    python pipeline/fetch_polyhaven.py --dest <dir> --skip 0 --limit 25   # one batch

Batching with --skip/--limit is how this pairs with `render_hdri_moves.py
--drop-source` when disk is short: fetch a batch, render it, let the renderer
delete it, fetch the next. A full 4k set is about 24 GB parked, and E: had
40 GB free on 4 Sep 2026.

Stdlib only. Safe to re-run: a file whose size already matches the API is
skipped, and a partial file resumes with an HTTP Range request.

LICENSE: CC0. https://polyhaven.com/license
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.polyhaven.com"
UA = {"User-Agent": "RUDRA-HDR/0.3 (+https://github.com/fxtdstudios/RUDRA)"}
RETRIES = 4


def get_json(url: str) -> dict:
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == RETRIES - 1:
                raise SystemExit(f"{url}: {type(exc).__name__}: {exc}")
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def listing() -> list[str]:
    """Every HDRI slug, in a stable order so --skip/--limit mean something."""
    return sorted(get_json(f"{API}/assets?t=hdris"))


def pick(slug: str, res: str, fmt: str) -> tuple[str, int] | None:
    """(url, size) for one asset at one resolution, or None if not published."""
    files = get_json(f"{API}/files/{slug}").get("hdri", {})
    entry = files.get(res, {}).get(fmt)
    if not entry or "url" not in entry:
        return None
    return entry["url"], int(entry.get("size", 0))


def download(url: str, dest: Path, expect: int) -> str:
    """Fetch to dest, resuming a partial file. Returns what happened."""
    if dest.is_file() and expect and dest.stat().st_size == expect:
        return "skipped"

    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.is_file() else 0
    headers = dict(UA)
    if have and expect and have < expect:
        headers["Range"] = f"bytes={have}-"
    else:
        have = 0

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as r, \
                open(part, "ab" if have else "wb") as f:
            # A server that ignores Range answers 200 with the whole file; if
            # we appended that to a partial we would silently write a corrupt
            # EXR that only fails much later, inside the renderer.
            if have and r.status != 206:
                f.close()
                part.unlink(missing_ok=True)
                return download(url, dest, expect)
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except (urllib.error.URLError, TimeoutError) as exc:
        return f"FAILED {type(exc).__name__}: {exc}"

    size = part.stat().st_size
    if expect and size != expect:
        part.unlink(missing_ok=True)
        return f"FAILED size {size} != {expect} from the API"
    part.replace(dest)
    return "resumed" if have else "fetched"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, help="where the panoramas go")
    ap.add_argument("--res", default="4k", help="1k 2k 4k 8k 16k (default 4k)")
    ap.add_argument("--format", default="exr", choices=("exr", "hdr"))
    ap.add_argument("--skip", type=int, default=0, help="skip the first N slugs")
    ap.add_argument("--limit", type=int, default=None, help="fetch at most N")
    ap.add_argument("--slugs", nargs="+", help="named assets instead of the whole set")
    ap.add_argument("--exclude-file", type=Path, default=None,
                    help="skip slugs whose file name appears in this list. Pair "
                         "with render_hdri_moves.py --done-file: a fetch / "
                         "render / drop loop otherwise re-fetches every "
                         "panorama the renderer has already deleted.")
    ap.add_argument("--list", action="store_true", help="print the slugs and exit")
    ap.add_argument("--count", action="store_true",
                    help="print how many HDRIs are published, and nothing else. "
                         "On stdout, because a caller that has to parse stderr "
                         "is a caller that breaks the first time Python warns "
                         "about anything.")
    ap.add_argument("--dry-run", action="store_true", help="resolve URLs, fetch nothing")
    args = ap.parse_args()

    slugs = args.slugs or listing()
    if args.count:
        print(len(slugs))
        return 0
    if args.list:
        print("\n".join(slugs))
        print(f"\n{len(slugs)} HDRIs published", file=sys.stderr)
        return 0
    if not args.slugs:
        slugs = slugs[args.skip:]
        if args.limit is not None:
            slugs = slugs[:args.limit]
    if not args.dest:
        ap.error("--dest is required unless --list is given")

    done: set[str] = set()
    if args.exclude_file and args.exclude_file.is_file():
        done = {line.strip() for line in
                args.exclude_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        before = len(slugs)
        slugs = [s for s in slugs
                 if f"{s}_{args.res}.{args.format}" not in done]
        print(f"   {before - len(slugs)} already rendered, skipping them")

    args.dest.mkdir(parents=True, exist_ok=True)
    index = args.dest / "_polyhaven_index.jsonl"

    print(f"   {len(slugs)} panoramas at {args.res} {args.format} -> {args.dest}")
    counts: dict[str, int] = {}
    total = 0
    started = time.time()
    with open(index, "a", encoding="utf-8") as log:
        for i, slug in enumerate(slugs, 1):
            found = pick(slug, args.res, args.format)
            if found is None:
                counts["unpublished"] = counts.get("unpublished", 0) + 1
                print(f"   {i:>4}/{len(slugs)} {slug}: no {args.res} {args.format}")
                continue
            url, size = found
            dest = args.dest / f"{slug}_{args.res}.{args.format}"
            if args.dry_run:
                counts["dry-run"] = counts.get("dry-run", 0) + 1
                total += size
                continue
            what = download(url, dest, size)
            counts[what.split()[0]] = counts.get(what.split()[0], 0) + 1
            if what.startswith("FAILED"):
                print(f"   {i:>4}/{len(slugs)} {slug}: {what}")
                continue
            total += size
            log.write(json.dumps({"slug": slug, "file": dest.name, "bytes": size,
                                  "res": args.res, "format": args.format,
                                  "url": url, "license": "CC0"}) + "\n")
            log.flush()
            if i % 10 == 0 or i == len(slugs):
                mb = total / 1048576
                rate = mb / max(time.time() - started, 1e-9)
                print(f"   {i:>4}/{len(slugs)}  {mb:8.0f} MB  {rate:5.1f} MB/s  {what}")

    print(f"\n   {dict(sorted(counts.items()))}   {total / 1073741824:.1f} GB")
    return 1 if any(k == "FAILED" for k in counts) else 0


if __name__ == "__main__":
    raise SystemExit(main())
