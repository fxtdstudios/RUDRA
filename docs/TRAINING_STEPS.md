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

## Step 0 — the gate cannot currently catch our defect. Fix that first.

`pipeline/verify_dataset.py` check 3 is one-sided:

```python
affected = float((arr > 0.0001).mean())
status = FAIL if affected > max_clipped else PASS      # max_clipped = 0.02
```

It FAILs a corpus where **too many** records clip — written to catch the August
corpus at 77.8%. Our corpus has 0.57% affected, so it **passes**. There is no
floor, and the defect we spent this week finding walks straight through the gate
that exists to stop it.

Add one:

```python
parser.add_argument("--min-clipped-records", type=float, default=0.10)
...
if affected < min_clipped:
    status = FAIL
    detail += (f"  (floor {min_clipped:.0%}; the v3 corpus had 0.57% and every "
               f"one of those was a broken PolyHaven decode)"
```

Without this, the gate blesses the new corpus whether or not the 0 EV render
worked, and you find out three days into training.

---

## Step 1 — build the manifest

```
python training\build_sdr_hdr_manifest.py ^
    --sdr-dir  G:\corpus_v4\pairs\sdr ^
    --hdr-dir  G:\corpus_v4\pairs\hdr ^
    --metadata-dir G:\corpus_v4\pairs\meta ^
    --output   G:\corpus_v4\sdr_hdr_manifest.jsonl ^
    --val-fraction 0.10 --test-fraction 0.10 --seed 20260715
```

Keep the seed. Changing it while also changing the corpus makes the comparison
against the paper's numbers meaningless.

---

## Step 2 — run the gate, and let it stop you

```
python pipeline\verify_dataset.py ^
    --pairs-dir G:\corpus_v4\pairs ^
    --manifest  G:\corpus_v4\sdr_hdr_manifest.jsonl ^
    --video-manifest G:\corpus_v4\video_manifest.jsonl
```

Exits non-zero on FAIL. Three checks decide whether the run is worth starting:

| check | what it means for this run |
|---|---|
| **3 highlight clipping** | with the floor from Step 0: did the 0 EV render actually produce blown highlights? If not, stop — that is the whole experiment, answered for the price of a render |
| **7 concentration** | val/test FAIL above 25% one scene. Train only WARNs, because the scene-balanced sampler compensates — Stuttgart at 46% of records is a WARN, not a blocker, and "more scenes is the fix" is the right reading |
| **8 temporal floor** | ≥ 6 train scenes and ≥ 1 held out, or temporal is off the table. The August run trained on 1 and every eval came out worse than the baseline |

---

## Step 3 — smoke run. Twenty minutes, not three days.

```
python training\train_sdr2hdr.py ^
    --manifest G:\corpus_v4\sdr_hdr_manifest.jsonl ^
    --output-dir checkpoints\_smoke ^
    --steps 200 --max-items 200 --max-val-items 40 ^
    --eval-every 50 --save-every 100 --log-every 10
```

Proves the manifest loads, the loop runs, the eval path works and the GPU fits,
before you commit days to it. Delete `checkpoints\_smoke` afterwards.

---

## Step 4 — the image model, from scratch

```
python training\train_sdr2hdr.py ^
    --mode image ^
    --manifest G:\corpus_v4\sdr_hdr_manifest.jsonl ^
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

## Step 5 — the conditioning head, as its own run

Only after Step 4 has a `best.pt` you are happy with:

```
python training\train_sdr2hdr.py ^
    --manifest G:\corpus_v4\sdr_hdr_manifest.jsonl ^
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
    --manifest G:\corpus_v4\video_manifest.jsonl ^
    --image-checkpoint checkpoints\sdr2hdr_image_v4\best.pt ^
    --output-dir checkpoints\sdr2hdr_temporal_v4 ^
    --temporal-channels 24 --temporal-weight 0.5 ^
    --steps 20000 --eval-every 500
```

If check 8 failed, skip this entirely. Running it anyway reproduces August: one
train scene, every eval worse than the baseline, a week gone.

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

---

## What "complete" means

The run is finished when all five are true:

1. `verify_dataset.py` exits 0, **with the Step 0 floor in place**.
2. `best.pt` selected on `composite_gain`, trailing median of 5 evals.
3. `measure_clipping.py --score` reports error in stops on clipped pixels, for
   all three contenders — leave-at-white, analytic inverse, RUDRA. On the one
   frame that had enough clipped pixels last time, the last two were identical
   to the digit. If they still are, the corpus was not the constraint either,
   and that is the finding.
4. Every scene in the manifest carries `licence`, `commercial_ok`, `sha256`,
   `absolute`.
5. `README.md` and `docs/` updated in the same pass, items ticked.

And the honest exit: if Step 2 check 3 fails, or Step 7 shows RUDRA still tied
with the analytic inverse on clipped pixels, **stop and write that down**. Two
negative results measured cheaply are worth more than a third retrain.
