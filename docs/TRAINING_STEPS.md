# Training steps — the next run

Phase 4 of `docs/RETRAIN_RUNBOOK.md`, expanded. Every flag below is real; they
come from `training/train_sdr2hdr.py` and `pipeline/verify_dataset.py`, not from
memory.

The run tests **one hypothesis**: that corpus content, not quantity, capacity or
objective, was the constraint. The paper already measured the other three — six
times the footage moved the bound 0.03 dB, capacity moved it 0.5 dB with the
sign varying, the objective moved it not at all. So nothing else changes this
run. Not the channel count, not the loss, not the sampler.

---

## Step 0 — the gate already has the floor it was missing. ✅ done

`pipeline/verify_dataset.py` now runs **two** clipping checks, not one:

- check **3** (`check_clipping`) — a *ceiling* on the HDR **target** clipping
  at the storage limit (`--max-clipped-records`, default 2%). Written to catch
  the August corpus at 77.8%.
- check **3b** (`check_sdr_clipping`) — a *floor* on the SDR **input** clipping
  at the top code (`--min-clipped-records`, default 20%). This is the defect the
  −1 EV corpus had: a median 0.000% clipped, 0.57% of records touching the top
  code at all, and every check passing. An inverse tone mapper trained on it
  never saw a blown highlight.

3b also FAILs a corpus that never recorded `sdr_clipped_fraction` at all, so a
stale `prepare_pairs.py` cannot slip through silently. Nothing to add here —
this step is closed (commit `f3f2749`).

---

## Step 1 — build the manifest. ✅ done (18 Sep 2026)

Built with the pipeline path, not `training\build_sdr_hdr_manifest.py` — that
one pairs files by name and carries none of the measurements the gate reads,
which is why the first `G:\corpus_v4` failed checks 1 and 3:

```
powershell -ExecutionPolicy Bypass -File BUILD_CORPUS_V4.ps1 -Scan -Ingest -Manifest -Verify
```

`scan_sources.py` → `prepare_pairs.py --tonemap-ev 0` → `build_manifests.py
--max-eval-scene-share 0.25` → `verify_dataset.py`, into **`G:\corpus_v4b`**.

What the builder reported:

| | records | scenes | video |
|---|---:|---:|---|
| train | 19,861 | 783 | 88.4% in 12 scenes |
| val | 383 | 97 | 24.8% in 1 scene |
| test | 384 | 97 | 25.0% in 1 scene |

Dropped on the way in, and worth knowing:

- **1,799 pairs from `netflix_sparks` peak below 1 nit** — no HDR content to
  learn. That is most of the Sparks download, and the same signature as the
  Chimera set in the corpus audit: normalised values never scaled back to
  nits. Sparks is one of the commercially clean sources, so this is worth
  chasing before a commercial build — it is probably an EOTF/scale guess in
  `scan_sources.py`, not bad footage.
- **7,231 records (35%) peak exactly at their grading ceiling** across 14
  delivery-graded scenes. Their highlights were already clipped by whoever
  graded them. The loss treats those as censored, not as ground truth
  (`tests/test_censored_highlights_2026_08_28.py`).

Two defaults that could not both be met were fixed on the way: the builder
thinned val/test to 35% while the gate failed anything over 25%, and the
builder's video floor (25%) sat on the same number as the scene cap, which is
unsatisfiable when a split's video is one scene. Now 0.25 and 0.15, matching
`verify_dataset.py`.

---

## Step 2 — run the gate, and let it stop you. ✅ passed (18 Sep 2026)

```
python pipeline\verify_dataset.py ^
    --pairs-dir G:\corpus_v4b ^
    --manifest  G:\corpus_v4b\sdr_hdr_manifest.jsonl ^
    --video-manifest G:\corpus_v4b\video_manifest_9f.jsonl
```

| check | result |
|---|---|
| **3b SDR clipping** | **PASS — 29.3% of records clip an SDR pixel**; mean 2.39%, p90 4.96%. The −1 EV corpus was 0.57%. This is the decisive experiment of the corpus programme and it came out positive: the corpus now contains what an inverse tone mapper exists to reconstruct |
| 3 target clipping | PASS — 0.3% of records |
| 7 concentration | PASS on val (24.8%) and test (25.0%); WARN on train, largest scene 31.2%, top two 53.0% of 783 scenes — the scene-balanced sampler compensates, more scenes is the fix |
| 8 temporal floor | **PASS — 12 train scenes, 2 held out, 641 clips.** Step 6 is back on the table |
| 9 loss ceiling | WARN — see below |

**Check 9 is a decision to make before Step 4, not after.** Targets reach
1,000,000 nits and the network caps at 40,000 (`max_hdr=4.0`): 417 of 20,628
records (2.0%) carry a target it cannot represent, and the dataset clamps
them. There is no CLI flag for it — raising it means editing
`DEFAULT_TARGET_CEILING` in `training/sdr2hdr_dataset.py` and passing
`max_hdr` through the model config together, which also changes the
architecture's encoding, so every result after it is a two-variable change.
**Leave it at 4.0 for this run.** 40,000 nits is ten times any mastering
display, 2% of records is small, and the run tests one hypothesis. Revisit
if Step 7 shows the loss is spending itself on specular suns.

---

## Step 3 — smoke run. Twenty minutes, not three days. ✅ done (22 Sep 2026)

> **Before this step, read `checkpoints/_invalid_corpus_v4/README.md`.** Between
> 18 and 22 Sep three runs were started on the *first* `G:\corpus_v4` (old
> ingest, no sidecars): `sdr2hdr_image_v4` (70k steps), `_v4_gate` (8k) and
> `sdr2hdr_temporal_v4`. Their configs record `corpus_ev: -1.0` over a 0 EV
> render, so every eval measured the baseline's one-stop error and called it
> gain (+13.9 dB "clean"). They are quarantined, not results. The
> `sdr2hdr_image_v4` name is free again for the run below.

```
python training\train_sdr2hdr.py ^
    --manifest G:\corpus_v4b\sdr_hdr_manifest.jsonl ^
    --output-dir checkpoints\_smoke ^
    --steps 200 --max-items 200 --max-val-items 40 ^
    --eval-every 50 --save-every 100 --log-every 10
```

Proves the manifest loads, the loop runs, the eval path works and the GPU fits,
before you commit days to it. Delete `checkpoints\_smoke` afterwards.

What it proved on corpus_v4b: `corpus_ev` reads **0.0** from the manifest
(`manifest_sha256 ffcd8bbb…`), the step-0 baseline sits at **48.99 dB clean /
28.76 dB hard** `psnr_log` — a baseline that inverts the render it was given,
against 25.3 dB on the -1 EV mismatch — and 200 steps, 4 evals and 2 saves
ran end to end. Composite gain at step 200 is −2.4 (clean −2.4, hard 0.0), which
is what 200 steps from scratch looks like and says nothing about the run.

---

## Step 4 — the image model, from scratch. ✅ ran 22–23 Sep 2026 — acceptance pending the hold-out check

Launched 22 Sep 21:51 on `G:\corpus_v4b\sdr_hdr_manifest.jsonl` (sha `ffcd8bbb…`,
the manifest **before** the hold-out rebuild), `corpus_ev 0.0`, seed 20260715,
finished 50,000 steps at 01:59 — **4 h 08 min on the 4080**, so a retrain is
cheap. 100 evals; `best.pt` = step **36,000** on the trailing-median-5
`composite_gain`:

| | psnr_log | vs baseline |
|---|---:|---:|
| clean | 27.09 dB | **+1.80 dB** (preserved +1.83) |
| hard | 20.78 dB | **+0.82 dB** (preserved +0.75) |
| composite | | **+0.82**, trailing median +0.69 |

Baseline at step 0: 25.29 dB clean / 19.96 dB hard on the full val split (the
smoke's 48.99 was 40 items). The model improves on its own baseline in **both**
conditions — v5 lost 3.0 dB on clean. That is the corpus hypothesis answered in
the direction it was asked, on the trainer's metric; Step 7 decides on the bench.

**Not yet accepted.** The run used the manifest from before `--hold-out-scenes`
existed, so the paper's 429 bench scenes may sit in its train split.
`pipeline/check_holdout_overlap.py` answers that; `scripts/next_steps_2026-09-22.ps1`
runs it, rebuilds the manifests with the hold-out, and either accepts this run
(no old bench scene in train) or moves it to `sdr2hdr_image_v4_prehold_*` and
relaunches on the rebuilt manifest. Four hours is not worth a footnote in §7.

```
python training\train_sdr2hdr.py ^
    --mode image ^
    --manifest G:\corpus_v4b\sdr_hdr_manifest.jsonl ^
    --output-dir checkpoints\sdr2hdr_image_v4 ^
    --steps 50000 --batch-size 4 --crop-size 256 --base-channels 32 ^
    --lr 2e-4 --weight-decay 1e-4 --workers 2 ^
    --best-eval hard --best-metric composite_gain --best-smoothing 5 ^
    --eval-every 500 --save-every 2000 --seed 20260715
```

**No `--init-checkpoint`, no `--resume`.** `--init-checkpoint` loads weights and
only resets the optimizer — it still inherits everything the old checkpoint
learned from HdM. Under the commercial scope that defeats the point of the
corpus work.

**`--base-channels 32` stays.** Capacity is the one axis the paper already swept,
and it moved the result 0.5 dB with the sign varying. Change it here and you
cannot attribute the outcome to the corpus.

**Leave scene-balanced sampling on** (it is the default). It is what keeps 9
Stuttgart scenes from dominating, and it is why check 7 only WARNs on train.

**Selection.** `composite_gain` is `hard_gain_db + min(0, clean_gain_db)` — the
improvement on degraded SDR, less any damage done to clean SDR. That second term
matters: v5 lost 3.0 dB to its own analytic baseline on clean input. Don't select
on `loss`.

On the 4080, if it OOMs: `--batch-size 2 --grad-accum 2`, not a smaller crop.
Halving the crop changes the receptive-field statistics; halving the batch does
not.

---

## Step 5 — the conditioning head, as its own run ← **next, once Step 4 is accepted**

Only after Step 4 has a `best.pt` you are happy with:

```
python training\train_sdr2hdr.py ^
    --manifest G:\corpus_v4b\sdr_hdr_manifest.jsonl ^
    --output-dir checkpoints\sdr2hdr_image_v4_gate ^
    --init-checkpoint checkpoints\sdr2hdr_image_v4\best.pt ^
    --gate-conditioning --freeze-except-gate --reset-best ^
    --steps 8000 --eval-every 250
```

Separate run, on purpose. The per-pixel priors decide *where* to reconstruct and
nothing decided *how much*; this head is the *how much*. Folding it into Step 4
means two changes at once and an unattributable result.

---

## Step 6 — temporal, only if check 8 passed

```
python training\train_sdr2hdr.py ^
    --mode temporal ^
    --manifest G:\corpus_v4b\video_manifest_9f.jsonl ^
    --image-checkpoint checkpoints\sdr2hdr_image_v4\best.pt ^
    --output-dir checkpoints\sdr2hdr_temporal_v4 ^
    --temporal-channels 24 --temporal-weight 0.5 ^
    --steps 20000 --eval-every 500
```

If check 8 failed, skip this entirely. Running it anyway reproduces August: one
train scene, every eval worse than the baseline, a week gone.

Since 22 Sep the trainer refuses to start temporal training when the image
checkpoint's `corpus_ev` differs from the video manifest's, and the video
manifest carries `tonemap_ev` per clip (`build_manifests.py`). A manifest built
before that date still works: `corpus_ev_of` reads the clip's first sidecar.
`sdr2hdr_temporal_v4` of 22 Sep was started with a -1 EV image model over the
0 EV clips and would have "gained" a stop; that is what the refusal is for.

---

## Step 7 — acceptance

Whole-frame PSNR is mostly pixels that were never clipped, so it cannot answer
the claim. Score the pixels the claim is about:

```
python training\measure_clipping.py ^
    --bench <the paper's bench> --score ^
    --checkpoint checkpoints\sdr2hdr_image_v4\best.pt ^
    --min-clip-pct 0.1 --out results_v4.json
```

Same bench, same metrics as the paper. A new bench makes the comparison
meaningless, and the comparison is the deliverable.

`--bench` must be laid out as `<bench>/clean/` and `<bench>/hard/`, each with a
`sdr/` and a `ref/` tree. `export_bench_pairs.py` produces that shape when run
twice with `--write-sdr` (`--out bench/clean --condition clean` and
`--out bench/hard --condition hard`); the `sdr/` tree is what `--score` reads,
and without it the census skips the condition. The baseline contender is scored
at the checkpoint's own `corpus_ev`, not the legacy -1 EV.

---

## What "complete" means

The run is finished when all five are true:

1. `verify_dataset.py` exits 0, **with the check 3b floor in place**.
2. `best.pt` selected on `composite_gain`, trailing median of 5 evals.
3. `measure_clipping.py --score` reports error in stops on clipped pixels, for
   all three contenders — leave-at-white, analytic inverse, RUDRA. On the one
   frame that had enough clipped pixels last time, the last two were identical
   to the digit. If they still are, the corpus was not the constraint either,
   and that is the finding.
4. Every scene in the manifest carries `licence`, `commercial_ok`, `sha256`,
   `absolute`.
5. `README.md` and `docs/` updated in the same pass, items ticked.
6. The new `best.pt` and its `config.json` copied up to `checkpoints/` under a
   release name, `SHA256SUMS` and `models.json` extended — `checkpoints/*/`
   is git-ignored, so a run directory is never a release by itself.

And the honest exit: if Step 2 check 3 fails, or Step 7 shows RUDRA still tied
with the analytic inverse on clipped pixels, **stop and write that down**. Two
negative results measured cheaply are worth more than a third retrain.

---

## After v4b: corpus_v4c and the curve head (23 Sep 2026)

The out-of-generator bench showed the v4b recipe cannot generalise past its
own ACES render. `scripts/critical_path_2026-09-23.ps1` runs the next corpus
and both models end to end; the commands it runs are:

```
python pipeline\reclassify_inventory.py G:\corpus_v4b\_inv\source_inventory.jsonl G:\corpus_v4c\_inv\source_inventory.jsonl
python pipeline\prepare_pairs.py --inventory G:\corpus_v4c\_inv\source_inventory.jsonl --dst G:\corpus_v4c ^
    --mode log2_extended --crops 3 --video-stride 8 --tonemap-ev 0 --sdr-render mix
python pipeline\build_manifests.py --pairs-dir G:\corpus_v4c --out-dir G:\corpus_v4c ^
    --max-eval-scene-share 0.25 --hold-out-scenes E:\RUDRA_v3_20260822\sdr_hdr_manifest.jsonl
python pipeline\build_manifests.py --pairs-dir G:\corpus_v4c --out-dir G:\corpus_v4c\studio ^
    --max-eval-scene-share 0.25 --hold-out-scenes E:\RUDRA_v3_20260822\sdr_hdr_manifest.jsonl --commercial-only
python training\train_sdr2hdr.py --mode image --manifest G:\corpus_v4c\sdr_hdr_manifest.jsonl ^
    --output-dir checkpoints\sdr2hdr_image_v4c --curve-head [Step 4 flags]
python training\train_sdr2hdr.py --mode image --manifest G:\corpus_v4c\studio\sdr_hdr_manifest.jsonl ^
    --output-dir checkpoints\sdr2hdr_image_v4c_studio --curve-head [Step 4 flags]
```

Then three benches (`bench\cp_oog`, `bench\cp_mix`, `bench\cp_aces`), every
model, and `training\paired_gate.py` into `reports\logs\cp_results.json`.

**N3 gate:** `oog/v4c vs baseline` both CIs above zero. **Studio gate (N7):**
v4c_studio within 0.3 dB / 0.03 JOD of v4c on `oog` and `mix`. **Clean
regression:** `aces/v4c` against `aces/v4b`, reported; see the proxy note in
STATUS before reading it.

