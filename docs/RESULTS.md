# RUDRA results

Split out of the README on 10 Sep 2026, when the README became a page for the
people who use the tool. Nothing here changed; it moved.

Back to [the README](../README.md) . [STATUS.md](../STATUS.md) says what is
finished and what is open.

---

## Results

All numbers below come from the same 429 held-out frames at native 1280x720,
scene-linear with diffuse white at 1.0, scored at `--nits-scale 203`. Two
metrics against the same unclamped reference: PU21-PSNR and ColorVideoVDP JOD.
Whole scenes are held out, so no scene appears on both sides of a split.

`hard` is the deployment condition: unknown tone curve, 4:2:0 chroma, banding,
JPEG. `clean` is well-graded input, which the analytic inverse-ACES baseline
already handles well.

![Degraded input, four methods](compare/hard_deployment.png)

Three held-out frames under the degraded condition. The HDR columns are stopped
down by whole stops so the highlight range lands where a browser can show it,
and the exposure is picked from the reference alone, so no method is flattered
by its own output. The analytic inverse tone map has a hard ceiling and clips
into it; the sky is a flat plate. RUDRA keeps the roll-off the reference has.
Per-frame numbers under each panel come from the benchmark's own result files.

| Condition | Method | PU21-PSNR (dB) | CVVDP (JOD) |
|---|---|---:|---:|
| clean | analytic baseline | 45.99 | 9.448 |
| clean | v5, as shipped | 42.99 | 9.402 |
| clean | v6, 4x capacity | 43.24 | 9.452 |
| clean | v5, shadow arm off | 46.50 | 9.431 |
| clean | v5 + shadow gate | 46.06 | 9.561 |
| hard | analytic baseline | 25.92 | 7.362 |
| hard | v5, as shipped | 27.34 | 7.805 |
| hard | v6 | 26.88 | 7.706 |
| hard | v5, shadow arm off | 26.25 | 7.497 |
| hard | v5 + shadow gate | 27.15 | 7.751 |

The gate row is the one that matters. It is the only configuration we have
scored that beats the analytic baseline in both conditions on both metrics.

### The two metrics disagree

On degraded input v5 earns its keep: +1.43 dB and +0.44 JOD over the baseline,
winning 348 of 429 frames. That is the deployment condition and it is what the
project is for.

On clean input the same model loses 3.0 dB of PU21-PSNR and wins only 115
frames. CVVDP puts the same gap at -0.046 JOD, far below a just-noticeable
difference. The two instruments are two orders of magnitude apart on identical
frames. What RUDRA adds to well-graded input is highlight energy that PU21-PSNR
punishes and no viewer sees.

Do not report the clean PSNR row without the JOD beside it.

### The failure mode is in the shadows

Split the clean frames by how much headroom the ground truth has and the
regression stops looking diffuse.

| clean frames | mean delta vs baseline | median ground-truth peak |
|---|---:|---:|
| 60 worst | -10.76 dB | 238 nits |
| 60 best | +3.41 dB | 19,590 nits |

Correlation between `log2(peak_nits)` and gain is +0.46. The per-pixel luminance
gate never sees the frame as a whole, so it cannot tell a 238-nit studio
interior from a 20,000-nit sunset.

We guessed twice at the cause and were wrong twice. It is not invented
highlights: at the worst 1% of pixels the true luminance has a median of 4 nits
against 21 frame-wide, and only 46% of them are over-predicted. It is not a tail
either: discarding the worst 10% of pixels closes 2.3 dB of the 5.55 dB gap and
leaves 3.2 behind.

The damage comes from the shadow arm of the gate firing on low-dynamic-range
content that needs no reconstruction. Ablating that arm recovers the whole clean
deficit and gives up most of the hard gain, which is what pointed at the fix.

### One number per frame would fix it, and the input does not carry it

Give the residual a single global scale alpha and let an oracle pick it per
frame. The ceiling is large: +5.84 dB on clean and +0.29 dB on hard at the same
time. No constant reaches it. Clean wants alpha near 0.125, hard wants near 1.1,
and the constant that fixes clean throws away 85% of the hard gain.

So we tried to learn it. On 51 held-out frames, each clean and degraded,
cross-validated by frame:

| knowing | MAE on oracle alpha | R2 |
|---|---:|---:|
| nothing (predict the mean) | 0.501 | 0.000 |
| the features, via a linear readout | 0.487 | +0.031 |
| the condition, perfectly | 0.409 | +0.213 |
| the oracle itself | 0.000 | 1.000 |

The features explain 3% of the target's variance. 79% of that variance sits
within a condition rather than between conditions, so a perfect
clean-versus-degraded classifier caps out at R2 0.213. That is everything a
condition stem could ever buy. The remaining 79% asks whether this frame's
clipped region was a 200-nit lamp or a 20,000-nit sun, and an 8-bit frame does
not carry the evidence.

This is the information limit of single-image inverse tone mapping, measured
rather than asserted. Capacity (4x), corpus (6x) and objective were each varied
and none of them moved it.

### The gate that did work

The bound applies to the question, not to the problem. Asked for a continuous
scale, the input cannot answer. Asked "did this frame arrive clean or
degraded?", it can. That is the one axis that is detectable, and the
shadow-arm ablation showed it is the axis that matters. The target also needs no
oracle: at training time we know whether we degraded the frame.

`ShadowGate` is 21,121 parameters on the frozen v5 backbone. It predicts one
weight per frame on the shadow prior, supervised by binary cross-entropy against
the degradation label. Weight 1.0 reproduces `recovery_mode="all"` exactly and
0.0 reproduces `recovery_mode="highlights"` exactly, so it interpolates between
the two ablation endpoints and nothing else.

![Well-graded input, with and without the gate](compare/clean_gate.png)

Three low-headroom frames that need no reconstruction at all. The shipped v5
lifts them anyway, which is the shadow arm firing where it should not: a purple
cast in the black curtain, haze on the dark wall, milk in the foreground rocks.
The gate turns that arm down and lands on the reference.

Three seeds, gain over the analytic baseline:

| seed | clean dB | clean JOD | hard dB | hard JOD | clean CVVDP | clean frames won |
|---|---:|---:|---:|---:|---:|---:|
| 20260901 | +0.07 | +0.113 | +1.24 | +0.389 | 9.561 | 251 / 429 (59%) |
| 2 | +0.71 | +0.068 | +0.96 | +0.308 | 9.516 | 315 / 429 (73%) |
| 3 | +0.45 | +0.089 | +1.18 | +0.358 | 9.537 | 280 / 429 (65%) |
| mean +/- sd | +0.41 +/- 0.33 | +0.090 +/- 0.023 | +1.12 +/- 0.15 | +0.352 +/- 0.041 | | |

The shipped v5 wins 115 of 429 clean frames for comparison. All three runs are
positive on all four measures, and all three land a clean CVVDP above every
fixed alternative, including both ends of the ablation they interpolate between.
A hard binary switch could not do that. The gate emits intermediate weights and
finds per-frame settings that neither extreme reaches.

Against the shipped model that is +3.41 +/- 0.32 dB and +0.135 +/- 0.023 JOD on
clean, for 0.30 +/- 0.15 dB and 0.091 +/- 0.041 JOD on hard.

Seed 20260901 is the one committed here. It is also the worst of the three on
clean PU21 and the best on clean CVVDP, so reporting it alone understates the dB
result sixfold and overstates the JOD by a quarter. Three seeds support the
claim that the sign is stable. They do not support a confidence interval.

### One thing we tested and dropped

On 3 September two photographs were run through the shipped model, a stock
sunset and an Iceland landscape, neither in any split. Both suggested something
the luminance account cannot see: in the brightest 1% of pixels the red share of
R+G+B fell by about 0.07 while green rose by about 0.05. A hue rotation at
roughly constant luminance is nearly invisible to PU21-PSNR and to CVVDP as we
report them, so a real one would be a defect our own metrics would miss.

Measured on all 429 held-out frames against the reference, which the photographs
do not have, it does not survive:

| band | share of pixels | green shift | vs the reference | frames agreeing on sign |
|---|---:|---:|---|---:|
| < 20 nits | 41.2% | +0.005 | towards | 33% |
| 20-203 | 46.1% | +0.000 | - | 32% |
| 203-1000 | 11.4% | -0.001 | towards | 64% |
| > 1000 | 1.3% | +0.005 | away (0.0057 -> 0.0110) | 50% |

Only the top band moves away from the truth. The shift there is ten times
smaller than on the photographs, it covers 1.3% of pixels, and 50% sign
agreement across frames is a coin flip. Both photographs were saturated sunsets
whose highlights are nearly one hue, so a small per-channel error rotates them
coherently. The split's bright band averages near-neutral and per-frame
rotations cancel out. The effect is real on single-hue highlights and is not a
property of the model, so it is not in the paper.

It is recorded here because quietly dropping a hypothesis that did not work out
is how a repository ends up looking more certain than its evidence.
`training/analyze_chroma_shift.py` reproduces the table and
`docs/chroma_shift_shadow_v1.json` holds the raw sums.

### What we got wrong along the way

Five of the expensive mistakes this cycle were measurement defects rather than
model defects. Most are in the paper, because each one silently corrupted a
result we believed, and each is a mistake any comparable pipeline can make.

**Selection on the maximum of a noisy series.** `best.pt` was chosen by the
maximum of `composite_gain` over 102 evaluations, where `clean_gain_db` had mean
-1.43 dB and sd 1.29, and only 10 of 102 evals were ever positive. The shipped
checkpoint's +0.02 ranked 9th of 102. Selection now uses a trailing median over
five evals. Replayed, it picks step 72,000 instead of 81,000, which we then
scored. It is better on the criterion the selector optimises (composite -1.36
against -1.57) and worse on CVVDP in both conditions. Smoothing a selector does
not fix selecting on the wrong quantity.

**An evaluation that read the front of the split.** `DataLoader(val,
shuffle=False)` with `max_batches=N` reads the alphabetically first N records.
One run's eval was 32 records over 11 scenes, every name between
`abandoned_factory` and `blau_river`, leaving 403 records and 87 scenes
unmeasured. Since error correlates with headroom, that slice reported
`clean_gain +0.61` where the benchmark measured -3.0.

**A viewer that presented every frame upside down.** The default framebuffer
puts row 0 at the bottom and the display shader sampled `vUV` unchanged. No
numeric test caught it, because they all compare the float composite, which a
presentation flip leaves untouched. The one test that read the canvas did so
through `readPixels`, which returns rows bottom-first and cancelled the flip
against itself. Master EXR output was never affected. There is now a test that
looks at pixels.

**Two numbers transcribed and never checked back.** The paper said 346 frames
won on degraded input where the benchmark says 348, and gave v6 4,772,485
parameters where the checkpoint has 4,770,117. Both are small, and neither was
ever compared against its source. `training/audit_paper_numbers.py` now
recomputes every derived claim from `bench/results/*.json`, and
`tests/test_committed_checkpoint_2026_09_03.py` counts the tensors in every
committed model against what the registry and the paper claim. Both fail loudly.

**Reporting one seed as if it were the method.** The gate first shipped on seed
20260901 alone. It turned out to be the worst of three on clean PU21 and the
best on clean CVVDP, so the single-seed report understated one result sixfold
and overstated the other by a quarter. It had been selected on validation
accuracy, and nobody had checked what that accuracy tracked. The paper now
reports mean and spread everywhere.

---

## The paper

[**`paper/main.pdf`**](../paper/main.pdf) is the write-up of the SDR to HDR model:
15 pages, 11 sections, three figures, every number traceable to a command. The
PDF is committed, and so is the LaTeX it is built from, so a clone with no
LaTeX toolchain still has the document and a clone with one can rebuild it.

```bash
bash paper/build.sh      # -> paper/main.pdf
bash paper/mkarxiv.sh    # -> paper/rudra-arxiv.tar.gz, verified to build flat
```

`paper/main.tex` is the entry point. `paper/_body.tex` and `paper/_abstract.tex`
are generated, so do not edit them by hand. `paper/ABSTRACT_ARXIV.txt` is a
trimmed abstract for the submission form, which caps at 1,920 characters where
the PDF's abstract runs longer.

Every derived number in sections 5, 6 and 6.1 to 6.2 can be recomputed on
demand by two scripts that exit non-zero on any drift: 58 claims from the
benchmark files and five more from the headroom join. Commands are below.

Two limits the paper states and this README should repeat. One published method
has been scored on our split, so the numbers above are against our own analytic
baseline and against ExpandNet; three other public methods have runnable code
and have not been run. Section 6's trimmed-PSNR table and section 7's oracle
sweep have no script yet, because they need per-pixel statistics over the
reference frames rather than the per-frame results the benchmark writes.
Section 10 says so rather than leaving a reader to assume otherwise.

### Reproducing the numbers

```bash
powershell -ExecutionPolicy Bypass -File training\run_bench.ps1    # four exports, six scorings
python training/audit_paper_numbers.py --bench <bench dir>         # 58 claims, exits 1 on drift
python training/analyze_headroom.py --bench <dir> --manifest <m> --check   # section 6's split
python training/analyze_chroma_shift.py --bench <dir>/clean               # colour, per band
```

`run_bench.ps1` skips any stage whose output already exists, so an interrupted
run resumes. `score_checkpoint.ps1 -Checkpoint <ckpt> -Name <label>` adds one
model to an existing benchmark instead of redoing the reference, which is the
expensive half.

### Comparing against published work

ExpandNet, run from the authors' released weights on the same 429 frames:

| clean, 429 frames | PU21 dB | CVVDP JOD |
|---|---:|---:|
| RUDRA + gate, as deployed | 46.06 | 9.561 |
| analytic inverse-ACES baseline | 45.99 | 9.448 |
| ExpandNet | 27.50 | 7.532 |

That is -18.56 dB and -2.029 JOD, winning 1 of 429 frames on PU21 and 16 on
CVVDP.

Both metrics agree here, which is worth noting beside the disagreement above:
the two-order-of-magnitude gap is a property of small differences on well-graded
input, not a defect in either instrument.

![RUDRA against ExpandNet](compare/published_method.png)

ExpandNet loses the sun in the third frame outright and flattens the highlight
range in the other two. It is also the only column here that got a per-frame
exposure fit.

Three caveats live in section 5.1 of the paper and should travel with the
number. ExpandNet is run outside its training domain. Our reference is unclamped
to about a million nits. And it needs a per-frame exposure fit spanning 35.5x
where RUDRA needs 1.0x, because RUDRA predicts absolute nits and ExpandNet
predicts relative radiance. Given the best possible exposure it recovers only
+0.39 dB, so the fit is not what costs it.

Santos 2020, HDRCNN and SingleHDR have public code and no runner yet. The
harness that made this one work:

```bash
git clone https://github.com/dmarnerides/hdr-expandnet.git
python training/export_bench_pairs.py ... --write-sdr        # -> <out>/sdr/**.png
python training/run_expandnet.py --sdr <out>/sdr --repo hdr-expandnet \
    --out <out>/expandnet_raw
python training/import_method_output.py --out <dir> --from <out>/expandnet_raw \
    --name expandnet
rudra bench <dir> --nits-scale 203 --test-dir expandnet
```

Published single-image iTMO methods predict relative radiance with no nit
anchor, so the importer fits one global scalar per frame on the pixels the SDR
input did not clip. That is a free parameter RUDRA does not get, since it
predicts absolute nits and is scored as it stands. Run the importer on RUDRA's
own tree as well and report the symmetric row, or the table flatters us for
free. `import_<name>.json` records the fitted scales, any frames resized, and
any the method dropped.

---
