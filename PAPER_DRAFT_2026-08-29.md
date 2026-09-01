# Learned Inverse Tone Mapping Is Bounded by the Input, Not the Model

**Draft, 29 August 2026.** Supersedes the §6/§7.1 tables flagged as blocking in
`PAPER_ERRATA.md`. Every number below is measured, and the command that produced
it is named.

---

## Abstract

We train a compact network for single-image SDR-to-HDR reconstruction and
evaluate it against its own analytic inverse tone map on 429 held-out frames in
PU21-PSNR and ColorVideoVDP JOD. On degraded input — an unknown tone curve,
4:2:0 chroma, banding and JPEG, the condition real footage arrives in — the
model gains **+1.43 dB and +0.44 JOD**, winning **346 of 429 frames (81%)**. On
clean, well-graded input it *loses* **3.0 dB** of PU21-PSNR while CVVDP puts the
same gap at **−0.046 JOD**: a two-order-of-magnitude disagreement between the
metrics that is itself the result.

The clean regression is not uniform. It concentrates entirely in
low-dynamic-range frames: the 60 worst average **−10.76 dB** at a median
ground-truth peak of **238 nits**, the 60 best **+3.41 dB** at **19,590 nits**,
with correlation **+0.46** between gain and `log2(peak_nits)`. The model has no
representation of how much headroom a frame has, and so cannot decide to do
nothing. The penalty is carried by the tail of the error distribution rather
than its bulk (§6), which is itself a caution about the metric.

We then measure what fixing that would be worth, and whether it can be fixed. An
oracle per-frame scale on the residual is worth **+5.84 dB on clean and +0.29 dB
on hard simultaneously**, and no constant achieves it — clean wants 0.125, hard
wants 1.1, and the constant that fixes clean discards 85% of the hard gain. But
a linear readout of the features such a gate would use explains only **3%** of
that scale's variance, and **79% of the variance is within a condition rather
than between**, so even a perfect clean-versus-degraded classifier caps at
R² 0.213. The remaining 79% asks whether a clipped region was a 200-nit lamp or
a 20,000-nit sun — a question an 8-bit frame does not answer.

Our contribution is therefore not a better inverse tone mapper. It is a measured
account of where the ceiling sits and why: **the achievable gain is bounded by
the information in the SDR input, not by model capacity, corpus size, or
training objective**, each of which we vary and none of which moves the bound.

---

## 1. What is claimed, and what is not

This paper reports a **direct SDR-pixel to scene-linear-radiance image model**.
It does not report the DRE-transformer / cross-attention architecture described
in the earlier manuscript: that design has never been trained, and per
`PAPER_ERRATA.md` §2 the results attributed to it were aspirational. The
"HDR-VDP-3" column of the previous §6 corresponded to no metric this codebase
ever computed (`PAPER_ERRATA.md` §1). Both are withdrawn.

What follows is the system that exists, measured with the metrics that exist.

---

## 2. Related work

> **Draft note.** Citation slots are marked `[CITE]` and are deliberately
> unfilled. This section names the families of work this result sits among; it
> does not attribute numbers or claims to specific papers, because those
> attributions have not been verified against the sources. Filling them is a
> prerequisite for submission, not an optional polish. The failure mode this
> replaces — placeholder tables presented as measurements — is documented in
> `PAPER_ERRATA.md` and is the reason for the caution.

**Single-image inverse tone mapping.** The dominant framing learns a mapping
from an 8-bit frame to a higher-range one, typically with an encoder-decoder and
a loss in a perceptual or log domain, and typically evaluated on synthetically
clipped data. `[CITE]` Our architecture is deliberately smaller than this line
of work (1.2 M parameters) and is a *residual on an analytic inverse tone map*
rather than a direct predictor, which is what makes "does the network beat doing
nothing?" a question we can ask at every step of training.

**Highlight and clipped-region reconstruction.** A related family treats the
problem as inpainting the saturated regions specifically, rather than remapping
the whole frame. `[CITE]` Our highlight and shadow masks serve the same purpose,
but are predicted jointly with the residual and gated by a luminance prior
rather than by a detected saturation mask. §6 is a direct criticism of that
choice.

**Evaluation of HDR reconstruction.** PU21-PSNR `[CITE]` and ColorVideoVDP
`[CITE]` are the two public measuring sticks we use. §5 reports a case where
they disagree by two orders of magnitude in magnitude on the same frames, and §5
shows the PU21 penalty is tail-carried; we are not aware of that disagreement
being characterised for this task, and it is a contribution of this paper
independent of the model.

**Training data for HDR.** The scarcity of paired SDR/HDR footage shapes every
result here `[CITE]`; §4 documents a distribution shift between our own training
and held-out splits that we did not design and that a reader should weigh.

**What we could not compare against.** No published method is evaluated on our
split. `rudra bench --test-dir <method>` scores any third-party output against
the same reference; that comparison is the single most valuable addition to this
paper and its absence is stated again in §9.

---

## 3. Method

### 3.1 Storage and units

HDR targets are stored in `log2_extended`: 0.005 to 1,000,000 nits at 2,377
codes per stop. Network units are **nits / 10,000**, so 1.0 is 10,000 nits and
diffuse white (BT.2408, 203 nits) is 0.0203. The network clamps its output at
`max_hdr = 4.0` (40,000 nits).

### 3.2 Architecture

A compact U-Net (`rudra/sdr2hdr.py`, **1,196,197 parameters**, 4.6 MiB float32)
predicts three fields from the SDR frame and its analytic inverse-ACES baseline:
a log-domain residual, a highlight mask and a shadow mask. The composite is

```
pred_log = clamp(log1p(baseline · 16) + residual · gate,  0,  log1p(max_hdr · 16))
pred     = expm1(pred_log) / 16
```

where `gate = max(σ((y − 0.82)·24), σ((0.10 − y)·24))` is a per-pixel luminance
prior: the learned residual is applied where clipping or crushed shadows made
the SDR mapping non-invertible, and the physically-grounded analytic baseline is
preserved through ordinary midtones. Note for §5: this gate sees **one pixel's
brightness and nothing else**.

### 3.3 Censored observations

A pixel sitting on a source's delivery ceiling means "≥ ceiling", not
"= ceiling"; plain L1 against those pixels teaches the model to cap highlights.
The loss is one-sided there, with `CENSORED_HEADROOM_STOPS = 3.0` of free
reconstruction above the ceiling before the excess is charged, so a one-sided
loss cannot run every clipped highlight to `max_hdr`. Censoring is a small
effect on this corpus in practice: `censored_fraction` averages **0.0003**.

---

### 3.4 Training setup

| | v5 (shipped) | v6 (capacity ablation) |
|---|---|---|
| base channels | 32 | 64 |
| parameters | 1,196,197 | 4,772,485 |
| crop | 384 | 384 |
| batch × grad-accum | 2 × 2 | 2 × 2 |
| optimiser | AdamW, lr 2e-4, wd 1e-4, cosine to 5% | same |
| gradient clip | 1.0 | same |
| degradation probability | 0.65 | same |
| scene-balanced sampling | yes, 40% video mass | same |
| steps | 100,000 | 100,000 |
| eval | every 1,000 steps, 256 held-out crops, both conditions | same |
| selection | `composite_gain = hard_gain + min(0, clean_gain)` | same |
| hardware | one NVIDIA RTX 4080 SUPER | same |
| seed | 20260715 | 20260715 |

The corpus manifest is pinned by SHA-256 in every checkpoint's `config.json`, so
a run can be tied to the exact record set it saw.


## 4. Corpus, splits, and a distribution shift we did not intend

27,678 training / 435 validation / 429 test records; val and test are 97 scenes
each at 1280×720. Sampling is scene-balanced with a 40% video mass.

The held-out splits are **not** drawn from the training distribution:

| | median peak | < 1000 nits | < 400 nits | video share |
|---|---:|---:|---:|---:|
| train, as sampled | 1,713 nits | 42.2% | **27.2%** | 40.0% |
| val | 548 nits | 69.0% | 38.2% | 33.8% |
| test | 546 nits | 58.0% | **45.2%** | 32.9% |

The band in which the model fails (§6) is **45.2% of the test split and 27.2% of
what the sampler draws**. We report this because it is a confound in our own
results, and because §6 shows it is not the binding constraint.

---

## 5. Results

`training/export_bench_pairs.py` renders each held-out record at native
resolution into scene-linear EXR trees; `rudra bench --nits-scale 203` scores
them. Both conditions use the same unclamped reference — clamping the reference
at the network's own ceiling would score the model against a ground truth
cropped to its own limits. `hard` applies a seeded camera/codec degradation.

**429 frames, PU21-PSNR and ColorVideoVDP JOD, against the analytic baseline:**

| Condition | Method | PU21-PSNR (dB) | CVVDP (JOD) |
|---|---|---:|---:|
| clean | analytic baseline | **45.99** | 9.448 |
| clean | v5 (32 ch, step 81,000) | 42.99 | 9.402 |
| clean | v6 (64 ch) | 43.24 | **9.452** |
| hard | analytic baseline | 25.92 | 7.362 |
| hard | **v5** | **27.34** | **7.805** |
| hard | v6 | 26.88 | 7.706 |

Per-frame, v5 against the baseline:

| | mean Δ PU21 | median Δ | frames won | mean Δ JOD | median Δ JOD |
|---|---:|---:|---:|---:|---:|
| hard | **+1.427 dB** | +0.609 | **346 / 429 (81%)** | **+0.443** | +0.134 |
| clean | −2.999 dB | −2.131 | 115 / 429 (27%) | **−0.046** | −0.040 |

![Per-frame gain distributions under both metrics](docs/figures/fig3_metric_disagreement.png)

**Figure 1.** Per-frame gain over the analytic baseline, 429 held-out frames.
Vertical rules mark the means. The clean distribution is broad and left-shifted
in PU21-PSNR and collapses onto zero in CVVDP: the same frames, the same models,
two metrics that disagree in magnitude by two orders of magnitude.

**The metrics disagree by two orders of magnitude on clean input.** A 3.0 dB
PU21-PSNR loss corresponds to −0.046 JOD, far below a just-noticeable
difference. What the model adds to well-graded input is highlight energy that
PU21-PSNR punishes and no viewer sees. *The clean PU21 row should never be
reported without the JOD beside it.*

### 5.1 Capacity is not the constraint

v6 quadruples width to 4.77 M parameters. It is better on clean (+0.25 dB,
+0.05 JOD) and worse on hard (−0.46 dB, −0.10 JOD) than v5 — movement in both
directions smaller than the spread between conditions.

---

## 6. The failure mode has a name

The clean regression is concentrated, not diffuse. Splitting clean frames by the
headroom the ground truth actually has:

| clean frames | mean Δ vs baseline | median ground-truth peak |
|---|---:|---:|
| 60 worst | **−10.76 dB** | 238 nits |
| 60 best | **+3.41 dB** | 19,590 nits |

![Gain against ground-truth peak luminance](docs/figures/fig1_gain_vs_headroom.png)

**Figure 2.** Per-frame gain against the headroom the ground truth actually has,
log x-axis, with binned medians. On clean input the trend rises through zero at
roughly 10,000 nits; on degraded input it is flat and positive at every
headroom. The model's error is a function of the scene, and only when the input
arrives clean.

Correlation between `log2(peak_nits)` and gain: **+0.46**. The per-pixel
luminance gate of §3.2 cannot distinguish a 238-nit studio interior from a
20,000-nit sunset, because it never sees the frame.

**What the error actually is: shadows, and only partly a tail.** On 51 clean
held-out frames we computed the per-pixel PU21 squared error for both the model
and the baseline, then recomputed each PSNR with the worst pixels progressively
discarded. If the penalty were carried by a few bad pixels the gap would close.

| discard worst | baseline | RUDRA | gap |
|---|---:|---:|---:|
| — | 54.40 | 48.84 | **−5.55 dB** |
| 0.1% | 54.70 | 49.40 | −5.30 |
| 1% | 55.36 | 51.14 | −4.23 |
| 10% | 57.58 | 54.34 | **−3.23 dB** |

*Frames below 400 nits, n = 25.* The gap **narrows but does not close**:
discarding a tenth of every frame removes 2.3 dB of the 5.55 and leaves 3.2. So
the penalty is **partly** tail-carried and mostly broad — the worst 1% of pixels
do carry **32%** of the model's total squared error, but the bulk of the
distribution is genuinely worse too.

**And the damage is in the shadows, not the highlights.** At the worst 1% of
pixels the *true* luminance has median **4 nits**, against **21 nits**
frame-wide — those pixels are darker than typical, not brighter. Only **46%** of
them are over-predicted; the rest are under. On frames at or above 400 nits the
model and the baseline are level (−0.03 dB) and the worst pixels move to the
bright end instead (median 560 nits against 57 frame-wide).

This relocates the failure. §3.2's gate has two arms, a highlight prior and a
shadow prior, and on a low-dynamic-range scene the shadow arm fires on content
that needs no reconstruction at all. The headroom correlation of +0.46 is real,
but the mechanism behind it is the **shadow** path, not invented highlights —
which is a specific, testable target that the oracle experiments of §7, being a
single global scale, could not have isolated.

We record that this passage has been corrected twice against fresh measurement.
The first draft claimed the model invents highlights; the second claimed the
penalty was tail-carried. Both were wrong, and both were plausible.

## 7. What an adaptive gate is worth, and why it cannot be had

### 7.1 The opportunity

Give the composite a single global scale `α` on the residual and let an oracle
choose it per frame (27 held-out scenes, PU21 gain over the analytic baseline):

| α | clean | hard |
|---|---:|---:|
| 1.0 — as shipped | **−4.15** | +1.08 |
| 0.125 — best single constant | +0.83 | **+0.17** |
| per-frame oracle | **+1.69** | **+1.37** |

![The alpha sweep, clean and degraded](docs/figures/fig2_alpha_sweep.png)

**Figure 3.** Mean gain as a function of a global scale α on the learned
residual. The curves cross: every α that helps one condition hurts the other,
and the shipped α = 1 is near-optimal for degraded input and near-worst for
clean.

The conditions want opposite settings: clean improves monotonically as α falls
to ≈0.125, hard as it rises to ≈1.1. **No constant serves both** — the constant
that fixes clean discards 85% of the hard gain. Per-frame adaptation beats the
shipped configuration by **5.84 dB on clean and 0.29 dB on hard at the same
time**, which is not a trade-off at all.

And the oracle's α is legible. Median oracle α, split by reference peak:

| | peak < 1000 nits | ≥ 1000 nits |
|---|---:|---:|
| clean | **0.125** | 0.969 |
| hard | 1.250 | 1.094 |

On clean input it is near-binary in headroom. On degraded input it wants full
strength regardless, because degradation destroys information the baseline
cannot recover whatever the scene's range. A gate would need to see **headroom
and condition together**.

### 7.2 The ceiling

We built that gate — a pooled-feature head predicting α (`ConditionGate`,
21,121 parameters) — and trained it three times: twice through the
reconstruction loss (16,000 steps total) and once by direct supervision on the
oracle. All three produced a near-constant output. The trained head emits α in
**[1.0125, 1.0802]** with correlation **−0.037** against `log2(peak_nits)`,
where the oracle wanted 0.125.

Rather than a fourth attempt we measured the ceiling. 51 held-out frames, each
clean and degraded (102 samples), exactly the features the head receives, a
linear readout, cross-validated **by frame** so a clean/degraded pair can never
straddle the split:

| knowing | MAE on oracle α | R² |
|---|---:|---:|
| nothing (predict the mean) | 0.501 | 0.000 |
| the features, via a linear readout | 0.487 | **+0.031** |
| the condition, **perfectly** | 0.409 | **+0.213** |
| the oracle itself | 0.000 | 1.000 |

Two numbers close the question. The features explain **3%** of the target's
variance. And **79% of that variance is within a condition, only 21% between**
(clean α: mean 0.436, sd 0.471; degraded: 0.925, sd 0.469) — so a *perfect*
clean-versus-degraded classifier, the most any condition-detection architecture
could buy, caps at R² 0.213. 55% of frames want α at one end of the grid, so it
is close to a binary decision per frame.

The remaining 79% asks whether *this* frame's clipped region was a 200-nit lamp
or a 20,000-nit sun. **That is the information limit of single-image inverse
tone mapping, not a feature-engineering gap.** The +5.84 dB an oracle gate is
worth is mostly unreachable from the input.

### 7.3 Three levers, one bound

| lever varied | change | effect on the bound |
|---|---|---|
| capacity | 1.20 M → 4.77 M parameters | ±0.5 dB, sign varies (§5.1) |
| corpus | 6× footage (v4) | +0.03 dB |
| objective | censored loss; oracle supervision | no movement in α |

---

## 8. Evaluation methodology: three defects we found in our own harness

We report these because each silently corrupted a result we believed, and each
is a mistake any comparable pipeline can make.

**Selection on a maximum over noise.** `best.pt` was chosen by the maximum of
`composite_gain = hard_gain + min(0, clean_gain)` over 102 evaluations. Across
that run `clean_gain_db` had mean **−1.43 dB** and standard deviation **1.29
dB**, with only **10 of 102** evaluations positive; the shipped checkpoint's
`+0.02` ranked **9th of 102**. Its promised hard gain of +1.80 dB measured
**+1.43 dB** on held-out frames — between the eval mean (+1.19) and its selected
maximum, exactly where an inflated in-training figure should land. Selection now
uses a trailing median over five evaluations; replayed on the same series it
selects step 72,000 instead of 81,000, for a five-eval neighbourhood averaging
−0.19 dB on clean against 81,000's −0.91.

**An evaluation that read the front of the split.** `DataLoader(val,
shuffle=False)` with `max_batches=N` evaluates the alphabetically *first* N
records — 32 records across 11 scenes, every name between `abandoned_factory`
and `blau_river`, with 403 records and 87 scenes never measured. Since error
correlates with headroom, that slice reported `clean_gain +0.61` where the
benchmark measured −3.0. Fixed with a seeded permutation applied identically to
both conditions.

**A viewer that presented every frame upside down.** The default framebuffer
places row 0 at the bottom; the display shader sampled `vUV` unchanged. No
numeric test caught it because they all compare the float composite, which a
presentation flip leaves untouched — and the one test that read the canvas did
so through `readPixels`, which returns rows bottom-first, cancelling the flip
against itself. Master EXR output was never affected.

---

## 9. Limitations

- **Held-out size.** 429 test frames over 97 scenes. §6's oracle and ceiling
  analyses use 27 and 51 frames respectively, one per scene; that sample runs
  ≈1.15 dB pessimistic on clean against the full set (−4.15 dB at α=1 where the
  full 429 read −3.00). Ordering is reliable, absolute values will move.
- **Train/test distribution shift** (§4) is a confound in the clean result. §7.2
  argues it is not the binding constraint, but we have not retrained under a
  matched sampler to prove it.
- **The oracle is an upper bound.** It reads the ground truth.
- **Single-image only.** The temporal refiner is not evaluated here; its
  held-out set is 4 validation and 5 test clips of one scene each, too small to
  report.
- **One degradation model.** `hard` is our own seeded camera/codec pipeline, not
  a corpus of real degraded footage.
- **No comparison to published inverse tone mapping methods.** Every number here
  is against our own analytic baseline. `rudra bench --test-dir <method>` scores
  any third-party output against the same reference, and that comparison is the
  obvious next step. This is the paper's largest gap and we do not minimise it.
- **The related-work section carries unfilled `[CITE]` markers** and must not be
  submitted in that state (§2).
- **The shadow-versus-highlight result in §6 rests on 51 frames** at one
  degradation setting, and the shadow-arm hypothesis it suggests is untested:
  we have not ablated the shadow prior to confirm it.

---

## 10. Reproducibility

```bash
# the full benchmark: four exports, six scorings, resumable
powershell -ExecutionPolicy Bypass -File training\run_bench.ps1

# one more checkpoint against the same reference
powershell -ExecutionPolicy Bypass -File training\score_checkpoint.ps1 `
    -Checkpoint <ckpt> -Name <label>
```

Results land as `bench/RESULTS.md`, per-clip JSON and per-frame CSV. The
composite exists in three languages — torch (`rudra/sdr2hdr.py`), GLSL
(`ui/compositor.js`) and numpy (`tests/compose_reference.py`) — pinned against
each other to 8.4e-06 relative error by `tests/webgl_parity/parity.py`. 156
tests pass, including canvas orientation, PU21 torch-versus-numpy parity, and a
smoke test that presses all 54 controls of the viewer.

Weights, pairs and HDR sources are not committed; `training/` regenerates them.

---

## 11. Conclusion

A 1.2 M-parameter residual on an analytic inverse tone map is worth **+1.43 dB
and +0.44 JOD on degraded input, on 81% of held-out frames** — and on clean
input it is a 3 dB PSNR regression that is perceptually invisible at −0.046 JOD.
Its one clear defect is that it does not know when to do nothing, and we measure
that an oracle fixing it would be worth 5.84 dB, that no constant can, and that
97% of the signal needed to predict it is absent from the input.

Capacity, corpus and objective each moved the result by less than the noise. The
binding constraint on single-image inverse tone mapping, at least here, is what
an 8-bit frame can tell you about the scene that produced it.
