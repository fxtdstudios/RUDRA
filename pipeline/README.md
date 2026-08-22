# pipeline/ — corrected RUDRA training pipeline

Written 22 Aug 2026, after auditing the run at `E:\RUDRA_postfix_20260818`.
Everything here is **additive**: no existing file is modified except by
`patch_video_split.py`, which is explicit, reversible and idempotent.

## Why this exists

The post-fix run trained an image model that genuinely improved (log-PSNR
38.99 → 42.57 over the analytic baseline), inside a data pipeline that capped
what it could learn:

| Defect | Measured | Fixed by |
|---|---|---|
| HDR targets hard-clipped at 10,000 nits | 77.8% of stills peaked above it; one frame lost 6.4 stops | `hdr_io.py` — `log2_extended` mode, ceiling 1e6 nits |
| 16-bit **linear** target quantisation | 88.9% of pixels below code 256; 1,330 codes under diffuse white | `hdr_io.py` — 36,384 codes under diffuse white (**27×**) |
| val/test contained zero video frames | 96 val + 88 test, all stills | `build_manifests.py` — `--min-video-share` |
| temporal split computed as `max(1, round(2 × 0.1))` | one-clip training set; 22/22 evals worse than baseline | `patch_video_split.py` + explicit per-clip `split` |
| 92% of records from two clips | top-two scene share 91.8% | `verify_dataset.py` check 7 |
| nothing recorded peak nits or clipped fraction | invisible until audited | `prepare_pairs.py` per-record metadata |

## The scripts

| Script | Does | Gates on |
|---|---|---|
| `hdr_io.py` | The one implementation of how targets are encoded. Encode and decode live together so they cannot drift. `python hdr_io.py` prints round-trip error and shadow precision for both modes. | round-trip < 1% |
| `scan_sources.py` | Walks any root, reports formats, EOTF guesses, dynamic range in stops, and **independent moving sources**. Run this first on a new drive. | warns below 10 moving sources |
| `prepare_pairs.py` | Ingest. Reuses the repo's readers and EOTFs; replaces only the HDR write path. Multi-crop. Writes `_ingest_config.json` and refuses to mix conventions. | sentinel mismatch |
| `build_manifests.py` | Scene-safe splits with guaranteed video in val/test. Emits the temporal manifest with an explicit per-clip `split`. | video share, temporal scene floor |
| `verify_dataset.py` | **The gate.** Nine checks, one per audit finding. Exits non-zero on FAIL. | all of the above |
| `patch_video_split.py` | Makes `SDRHDRVideoDataset` honour the manifest split and refuse below 6 scenes. | — |
| `run_pipeline.bat` | Runs the sequence with the gate in the middle. | every step |

## Storage modes — the decision you have to make

`hdr_io.py` offers two, and the choice changes what every metric means. Make it
once, record it in the sentinel, do not mix.

**`log2_extended` (default, recommended for training targets)**
Scene-referred. 0.005 → 1,000,000 nits across 27.6 stops, ~2,380 codes per
stop, constant *relative* precision. An 822,000-nit sun survives intact. Round-
trip error 0.015% worst case. Use this so the supervision keeps everything the
source had, and let the *loss* decide what range to weight.

**`pq_10000`**
Display-referred. Matches HDR10 delivery and the ColorVideoVDP convention.
Still clips above 10,000 nits — but now that clipping is recorded per record as
`clipped_fraction` instead of being invisible. 38,055 codes below diffuse white.

Either way you get roughly 29× the shadow precision of the old linear scheme.

### The loss ceiling

`SDR2HDRNet(max_hdr=4.0)` can emit at most 40,000 nits (4.0 in units where
1.0 = 10,000). Under `log2_extended` your targets reach 1,000,000. That gap is
a **deliberate choice, not a bug**: store everything, then clamp the loss target
to what the network can represent. `verify_dataset.py` check 9 raises a WARN
naming the number. Either clamp the target at 40,000 nits in the loss, or raise
`max_hdr` and accept that reconstructing a 20-stop-clipped sun from 8-bit sRGB
is invention rather than recovery.

## Order of operations

```bat
python pipeline\scan_sources.py Z:\08_Research --out E:\RUDRA_v3\source_inventory.jsonl
python pipeline\prepare_pairs.py --inventory E:\RUDRA_v3\source_inventory.jsonl ^
    --dst E:\RUDRA_v3\pairs --mode log2_extended --crops 3
python pipeline\build_manifests.py --pairs-dir E:\RUDRA_v3\pairs --out-dir E:\RUDRA_v3
python pipeline\verify_dataset.py --pairs-dir E:\RUDRA_v3\pairs ^
    --manifest E:\RUDRA_v3\sdr_hdr_manifest.jsonl ^
    --video-manifest E:\RUDRA_v3\video_manifest_9f.jsonl
python pipeline\patch_video_split.py --apply
```

Or just `pipeline\run_pipeline.bat` after editing the paths at the top.

## Reading the temporal run

`train_sdr2hdr.py` seeds `best.pt` at step 0 with the baseline score and only
overwrites it on a strict improvement. That gate is why the August divergence
shipped nothing — keep it.

Use it as an alarm, not just a safety net: **if the first three evaluations do
not beat the step-0 baseline, stop the run.** In August all 22 failed to, over
two days, and the trend never turned. That is a data problem, not a warm-up.

## Verified

`hdr_io.py` round-trips both modes to better than 0.1% across their full range
(self-tested on import by `prepare_pairs.py`). `build_manifests.py` and
`verify_dataset.py` were run against a synthetic reconstruction of the August
corpus — the gate flags checks 3, 6, 7 and 8 and exits non-zero — and against a
corrected corpus, which passes. `scan_sources.py` and `prepare_pairs.py` are
**not yet exercised against real data**; they need the source drive mounted.
