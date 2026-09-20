"""Find the sources whose HDR decode is wrong, before they are rendered again.

    python training\quarantine_broken.py --index E:\RUDRA_v3_20260822\pairs\pairs_index.jsonl
    python training\quarantine_broken.py --index ... --out quarantine.json

WHY THIS EXISTS

Measured on the shipped corpus (35 790 pairs in pairs_index.jsonl):

    clipped_fraction exactly 0 : 35 585   (99.43%)
    clipped_fraction > 0       :    205

and the 205 are not 205 frames with real blown highlights. Every one of them is
a pair whose peak_nits decoded to between 1e6 and 1.33e38 nits. The overlap is
exact, not approximate: 205 broken, 205 clipped, 205 in both, 0 clean pairs with
any clipping at all.

So of the correctly decoded training pairs, NOT ONE contains a clipped pixel.
The model's entire notion of what sits above a blown highlight was fitted to a
decode bug -- and that same bug is the 37.75-stop dynamic-range figure in the
NAS inventory that the paper lists as unresolved.

Deleting the pair cache does not fix this. The cache is regenerable; the SOURCE
files that decode wrong are still on disk and will be rendered again by the next
prepare_pairs run unless something excludes them. That is what this writes.

TWO DEFECTS, NOT ONE -- AND THE FIRST ONE IS BIGGER THAN THE 205

Chasing the 205 turned up something larger. Sweep the threshold and the tail is
continuous, and it is one set:

    > 10 000 nits   2 100 pairs   Netflix 1 639, PolyHaven 461
    > 20 000 nits     327 pairs   PolyHaven only
    > 50 000 nits     185 pairs   PolyHaven only
    >100 000 nits     126 pairs   PolyHaven only
    >500 000 nits      27 pairs   PolyHaven only
    >  1e6  nits      205 pairs   PolyHaven only   <- the ones a ceiling catches

PolyHaven's own pairs run median 601 nits, p90 26 796, max 962 784. A set whose
middle is below diffuse white and whose top decile is 26 000 nits does not have
a decode bug in 205 files. It has NO ABSOLUTE SCALE. Poly Haven panoramas are
relatively calibrated -- a MacBeth chart shot in the same bracket, per their own
technical standard -- and the ingest read them as though they carried absolute
luminance. Every nit figure for that set is meaningless, all 2 889 pairs of it,
and a ceiling only ever catches the top of the tail.

That is defect one: SCALE, confined to PolyHaven, 8.1% of the corpus.

Defect two is separate and is the one the paper is about. Netflix contributes
1 639 pairs genuinely above 10 000 nits, absolutely calibrated PQ, and NOT ONE
of them clips. Real range, correctly scaled, still no blown highlights --
because the render exposed down a stop before the ACES curve and truncated the
top code. Fixing the scale does nothing for that, and re-rendering at 0 EV does
nothing for the scale.

So: quarantine PolyHaven on scale, re-render everything else on exposure. Do not
conflate them; they are fixed in different files.

WHAT COUNTS AS BROKEN

  peak_nits is NaN, infinite, or above --ceiling (default 1e6 nits) -- and,
  reported separately, any SET whose distribution cannot be absolute.

This proposes. It writes a JSON list and changes nothing else.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def is_broken(peak, ceiling: float) -> str:
    if peak is None:
        return "missing"
    if isinstance(peak, str):
        return "not-a-number"
    if math.isnan(peak):
        return "nan"
    if math.isinf(peak):
        return "inf"
    if peak > ceiling:
        return "above-ceiling"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, required=True,
                    help="pairs_index.jsonl from the ingest run")
    ap.add_argument("--ceiling", type=float, default=1e6,
                    help="peak_nits above this is a decode failure (default 1e6)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the quarantine list here (JSON)")
    args = ap.parse_args()

    if not args.index.exists():
        print(f"no such index: {args.index}", file=sys.stderr)
        return 2

    rows = []
    for n, line in enumerate(args.index.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"  !! line {n} is not JSON, skipped", file=sys.stderr)
    if not rows:
        print("index is empty", file=sys.stderr)
        return 2

    broken, reasons = [], Counter()
    clipped = [r for r in rows if (r.get("clipped_fraction") or 0) > 0]
    for r in rows:
        why = is_broken(r.get("peak_nits"), args.ceiling)
        if why:
            reasons[why] += 1
            broken.append((r, why))

    print(f"index                  {args.index}")
    print(f"pairs                  {len(rows):>8,}")
    print(f"clipped_fraction == 0  {len(rows) - len(clipped):>8,}"
          f"   ({(len(rows)-len(clipped))*100.0/len(rows):.2f}%)")
    print(f"clipped_fraction  > 0  {len(clipped):>8,}")
    print(f"broken decode          {len(broken):>8,}   {dict(reasons)}")

    bset = {id(r) for r, _ in broken}
    both = sum(1 for r in clipped if id(r) in bset)
    clean_clipped = len(clipped) - both
    print(f"\n  clipped AND broken   {both:>8,}")
    print(f"  clipped and CLEAN    {clean_clipped:>8,}"
          f"   <-- the corpus's real clipped highlights")
    if clipped and clean_clipped == 0:
        print("\n  Every clipped pixel in this corpus comes from a bad decode.")
        print("  Not one correctly decoded pair contains a blown highlight.")

    good = sorted(r["peak_nits"] for r in rows
                  if isinstance(r.get("peak_nits"), (int, float))
                  and not math.isnan(r["peak_nits"])
                  and not math.isinf(r["peak_nits"])
                  and 0 < r["peak_nits"] <= args.ceiling)
    if good:
        def q(f): return good[min(int(f * len(good)), len(good) - 1)]
        print(f"\nplausible peak_nits ({len(good):,} pairs)")
        print(f"  median {statistics.median(good):>10,.0f}   p90 {q(.9):>10,.0f}"
              f"   p99 {q(.99):>10,.0f}   max {good[-1]:>12,.0f}")
        for t in (1000, 4000, 10000):
            n = sum(1 for p in good if p > t)
            print(f"  above {t:>6,} nits   {n:>7,}  ({n * 100.0 / len(good):5.1f}%)")

    # ---- per-set scale check --------------------------------------------
    def setof(sid: str) -> str:
        s = sid.replace("\\", "/")
        for k in ("PolyHaven", "Netflix", "Stuttgart", "HdM-HFR-2017"):
            if k in s:
                return k
        return "NAS/other"

    bysets: dict[str, list] = defaultdict(list)
    for r in rows:
        bysets[setof(r.get("scene_id", ""))].append(r)

    print(f"\n{'set':16} {'pairs':>7} {'scenes':>7} {'median':>9} {'p90':>10} "
          f"{'max':>11} {'p90/med':>8}  verdict")
    suspect, unmeasured = [], []
    for k, v in sorted(bysets.items(), key=lambda kv: -len(kv[1])):
        # Known-broken values are excluded here: including them would let 1e38
        # set the max and hide the shape of the rest, which is the whole point.
        pk = sorted(x["peak_nits"] for x in v
                    if isinstance(x.get("peak_nits"), (int, float))
                    and not math.isnan(x["peak_nits"])
                    and not math.isinf(x["peak_nits"])
                    and x["peak_nits"] <= args.ceiling)
        nsc = len({x["scene_id"] for x in v})
        if not pk:
            print(f"{k:16} {len(v):7,} {nsc:7,} {'-':>9} {'-':>10} {'-':>11} "
                  f"{'-':>8}  NO USABLE VALUES")
            continue
        med = statistics.median(pk)
        p90 = pk[min(int(.9 * len(pk)), len(pk) - 1)]
        if p90 < 1.0:
            # Nothing real peaks below one nit across a whole set. A field
            # holding 0.0002 to 0.32 is a NORMALIZED value that was never
            # scaled: the ingest skipped its multiply. Times 10 000 these land
            # at 2 to 3 245 nits, which is exactly what the source is.
            verdict, ratio = "NOT NITS -- NORMALIZED", (p90 / med if med > 0 else 0.0)
            unmeasured.append(k)
        else:
            ratio = p90 / med if med > 0 else float("inf")
            # THE TELL IS THE SPREAD, NOT ANY OUTLIER.
            # Absolutely-scaled HDR is bunched: a graded deliverable's p90 sits
            # within a couple of stops of its median (Stuttgart 1.2x, Netflix
            # 1.4x, HdM-HFR 1.4x). Relative data read as absolute fans out,
            # because each file's arbitrary scale lands somewhere different --
            # PolyHaven is 231x, about eight stops from middle to top decile.
            # No ceiling finds this; it has no outliers, it is all outlier.
            verdict = "NO ABSOLUTE SCALE" if ratio > 20.0 else "plausible"
            if ratio > 20.0:
                suspect.append(k)
        fmt = (lambda x: f"{x:,.4f}") if p90 < 1.0 else (lambda x: f"{x:,.0f}")
        print(f"{k:16} {len(v):7,} {nsc:7,} {fmt(med):>9} {fmt(p90):>10} "
              f"{fmt(pk[-1]):>11} {ratio:8.1f}  {verdict}")

    if suspect:
        print(f"\n  {', '.join(suspect)}: p90 is more than 20x the median.")
        print("  Relative data read as absolute. Quarantine the WHOLE SET --")
        print("  the 205 a ceiling catches are the tail, not the fault.")
    if unmeasured:
        print(f"\n  {', '.join(unmeasured)}: every pair peaks below 1 nit.")
        print("  That is not dark footage, it is the wrong unit -- a normalized")
        print("  0..1 value the ingest never multiplied. Times 10 000 these are")
        print("  ordinary HDR. Anything that filtered or weighted on dr_stops")
        print("  read these as black, and they are not.")

    scenes: dict[str, dict] = defaultdict(
        lambda: {"pairs": 0, "reasons": Counter(), "examples": []})
    for r, why in broken:
        s = scenes[r.get("scene_id", "<no scene_id>")]
        s["pairs"] += 1
        s["reasons"][why] += 1
        if len(s["examples"]) < 3:
            s["examples"].append({"asset_id": r.get("asset_id"),
                                  "peak_nits": r.get("peak_nits"),
                                  "clipped_fraction": r.get("clipped_fraction")})

    print(f"\n{len(scenes):,} source scenes to quarantine:")
    for sid, s in sorted(scenes.items(), key=lambda kv: -kv[1]["pairs"])[:25]:
        print(f"  {s['pairs']:>4}  {dict(s['reasons'])}  {sid}")
    if len(scenes) > 25:
        print(f"  ... and {len(scenes) - 25:,} more")

    payload = {
        "index": str(args.index),
        "ceiling_nits": args.ceiling,
        "pairs_total": len(rows),
        "pairs_broken": len(broken),
        "pairs_clipped": len(clipped),
        "pairs_clipped_and_clean": clean_clipped,
        "reasons": dict(reasons),
        "scenes": {sid: {"pairs": s["pairs"], "reasons": dict(s["reasons"]),
                         "examples": s["examples"]} for sid, s in scenes.items()},
        "scene_ids": sorted(scenes),
        "scale_suspect_sets": suspect,
        "wrong_unit_sets": unmeasured,
    }
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    print("\nNothing has been moved or deleted. Exclude these scene_ids from the")
    print("next prepare_pairs run, or fix pipeline/hdr_io.py and re-measure --")
    print("but do not render them again as they are.")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
