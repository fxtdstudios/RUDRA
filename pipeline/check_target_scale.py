"""Prove the trainer and the model agree about what a target pixel MEANS.

The 24 Aug 2026 image run burned 50,000 steps because nothing checked this.
``training/sdr2hdr_dataset.load_rgb`` divided the stored uint16 by 65535 and
handed the raw log2 transfer-function code to the loss, while SDR2HDRNet's
inverse-ACES baseline emitted linear radiance in nits/10,000.  Mid grey: target
0.465, baseline 0.020.  Every metric was garbage and no error was raised --
both numbers are finite, positive, and in [0,1].

This runs in under a minute and would have caught it:

    python pipeline/check_target_scale.py --manifest E:\\RUDRA_v3_20260822\\sdr_hdr_manifest.jsonl

For each sampled pair it compares the decoded target against the model's own
baseline for the same SDR.  The two are not equal -- the network exists because
they differ in clipped highlights and crushed shadows -- but through the
midtones, where inverse ACES is well posed, their medians must sit within a
stop or so.  A factor of 20 is a decode bug, not a modelling gap.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Ratio of median(target)/median(baseline) outside this band = wrong convention.
# Generous on purpose: real corpora skew bright, and the band still rejects the
# 23x error that the missing decode produced.
RATIO_LOW, RATIO_HIGH = 0.25, 4.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch  # noqa: E402  -- heavy; only needed here
    from rudra.sdr2hdr import sdr_to_baseline_hdr  # noqa: E402
    from training.sdr2hdr_dataset import (  # noqa: E402
        DEFAULT_TARGET_CEILING, NETWORK_PEAK_NITS, load_rgb, storage_for,
    )

    records = [json.loads(line) for line in args.manifest.open(encoding="utf-8") if line.strip()]
    records = [r for r in records if r.get("split") == args.split] or records
    sample = random.Random(args.seed).sample(records, min(args.samples, len(records)))

    storage = storage_for(sample[0]["hdr_path"])
    print(f"manifest : {args.manifest}")
    print(f"storage  : {storage.describe() if storage else 'NO SENTINEL -- legacy linear*203/10000'}")
    print(f"sampling : {len(sample)} {args.split!r} records\n")

    ratios, peaks, over_ceiling = [], [], 0
    for record in sample:
        sdr = load_rgb(record["sdr_path"], hdr=False)
        target = load_rgb(record["hdr_path"], hdr=True)
        baseline = sdr_to_baseline_hdr(
            torch.from_numpy(sdr).permute(2, 0, 1)[None]
        )[0].permute(1, 2, 0).numpy()
        # Midtones only: highlights and shadows are exactly where the two are
        # SUPPOSED to disagree, so comparing them there proves nothing.
        luma = sdr.mean(axis=2)
        mid = (luma > 0.15) & (luma < 0.80)
        if mid.sum() < 256:
            continue
        ratios.append(float(np.median(target[mid]) / max(np.median(baseline[mid]), 1e-9)))
        peak = float(target.max())
        peaks.append(peak)
        over_ceiling += peak >= DEFAULT_TARGET_CEILING - 1e-6

    ratio = float(np.median(ratios))
    print(f"  median target / median baseline (midtones) : {ratio:.3f}x")
    print(f"  target peak, median over samples           : {np.median(peaks):.4f}"
          f"  ({np.median(peaks) * NETWORK_PEAK_NITS:,.0f} nits)")
    print(f"  samples hitting the {DEFAULT_TARGET_CEILING:g} ceiling "
          f"({DEFAULT_TARGET_CEILING * NETWORK_PEAK_NITS:,.0f} nits) : "
          f"{over_ceiling}/{len(peaks)}")

    if RATIO_LOW <= ratio <= RATIO_HIGH:
        print("\n  PASS  targets and the model's baseline are on the same scale.")
        return 0
    print(f"\n  FAIL  {ratio:.2f}x apart. The targets are not in network units "
          f"(nits/{NETWORK_PEAK_NITS:,.0f}).")
    print("        Check that the pairs dir has _ingest_config.json and that "
          "load_rgb is decoding through pipeline/hdr_io.py. DO NOT TRAIN.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
