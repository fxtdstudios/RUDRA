# RUDRA

**Radiometric Dynamic-Range Conditioning for HDR-Aware Diffusion Models**
[FXTD Studios](https://fxtdstudios.com) / Radiance Research

Diffusion models are trained on tone-mapped images, so they learn a world where
nothing is brighter than white. RUDRA gives them the rest of the range back. It
decodes latents straight to scene-linear HDR, and conditions the backbone on a
radiometric descriptor so luminance, exposure and wide-gamut colour survive
generation.

A second path needs no diffusion model at all: hand it an ordinary 8-bit frame
and it reconstructs what the tone map threw away. That path is what most of this
README is about, because it is the part we finished measuring.

📦 Weights — [huggingface.co/fxtdstudios/RUDRA](https://huggingface.co/fxtdstudios/RUDRA/tree/main)
📄 Paper — [`PAPER_DRAFT_2026-08-29.md`](PAPER_DRAFT_2026-08-29.md), *What an
8-Bit Frame Can and Cannot Say About the Scene Behind It*. Run
`bash paper/build.sh` for the PDF.
🔎 Status — [`STATUS.md`](STATUS.md) says which of the three things called RUDRA
are actually done. Short version: the production decoders are; the research
pipeline's Stage 3 is not; the SDR→HDR model is measured and written up.

---

## RUDRA Studio

![RUDRA Studio](docs/rudra_studio.png)

Drop a frame in, or a whole sequence. The network runs once per frame on the
GPU and hands the page its raw fields. Everything after that is composed on your
own GPU, so the controls move at frame rate instead of at one round trip each.

Two ways to compare against the analytic inverse tone map. Hold **B**, or press
and hold the image, and it swaps without the frame moving — the same pixels,
no seam to track. Press **W** for a wipe instead: baseline on the left,
reconstruction on the right, drag anywhere on the image to move the split.
Arrow keys nudge it, Shift for fine, W or Escape to leave.

The flip is better for judging whether a change is real. The wipe is better for
showing someone where it is.

Region EV is a grade, not a preview. Drag a value to scrub it, double-click to
zero it. The bands are soft luminance qualifiers from
`rudra/delivery/controls.py`, and **Master EXR** applies the identical qualifier
and gain to the file: full-resolution ACES 2065-1, AP0 primaries, ST 2065-4
chromaticities, and a sidecar recording exactly how it was graded.

Waveform, histogram and every measured number follow the composite on screen.
MaxCLL and MaxFALL come from an exact GPU reduction, not a downsample. Press `?`
for the keyboard.

```bash
python ui/server.py                          # finds a checkpoint, opens a browser
python ui/server.py --checkpoint <path> --port 8422 --device cuda
```

Without `--checkpoint` it looks for the newest model under your data directory,
then falls back to the one committed in `checkpoints/`, then to demo mode.
`?frame=<url>` loads a frame from the server instead of dropping one, and
`?demo=1` loads the bundled sample — which is how the screenshot above is taken,
so it can be regenerated rather than staged.

Needs WebGL2 and float render targets. Any current Chrome, Edge, Firefox or
Safari will do.

---

## Install

Python 3.10 to 3.13.

```bash
git clone https://github.com/fxtdstudios/RUDRA.git && cd RUDRA
pip install -e .                          # core + the `rudra` CLI
pip install -e ".[metrics]"               # cvvdp and lpips, for the real metrics
pip install -e ".[test]" && pytest tests/ # 180-odd tests, a few minutes
```

CUDA is only needed for training and inference. The delivery layer
(`rudra/delivery/`) is torch-free, so render and mastering machines need nothing
but Python and numpy.

Every model from the SDR→HDR path ships with the repo — 38 MB, six files — so a
fresh clone can reconstruct a frame and reproduce the tables below without
downloading anything:

```bash
sha256sum -c checkpoints/SHA256SUMS
python ui/server.py                                  # loads the shipped model
python ui/server.py --checkpoint sdr2hdr_image_v6.pt # or any other by name
```

`checkpoints/models.json` is the registry the server reads, and
[`checkpoints/README.md`](checkpoints/README.md) says what each file is, what
it scores, and how to load one in your own code. The training pairs and the HDR
sources stay out of git.

### ComfyUI decoders

Download the decoder for your backbone into `ComfyUI/models/radiance/`, enable
`rudra_decoder` in the *Radiance HDR VAE Decode* node, pick a `decoder_size`.

```bash
huggingface-cli download fxtdstudios/RUDRA --include "rudra_*_decoder_*.safetensors" \
    --local-dir "ComfyUI/models/radiance"
```

| Backbone | VAE latent | Decoder | PSNR_log |
|---|---|---|---|
| Flux.1 | 16ch / 8× | full | 29.77 |
| Wan | 16ch / 8× | full | 32.45 |
| LTX | 128ch / 8× | full | 25.47 |
| SDXL | 4ch / 8× | turbo | 33.86 |
| Z-Image | = Flux VAE | use the Flux decoder | — |
| Qwen-Image | 16ch / 8× | turbo | 26.67 |
| Flux.2 Klein | 128ch / **16×** | turbo | 28.57 |

---

## Use it

Reconstruct a frame or a sequence:

```bash
python training/infer_sdr2hdr.py input/ --output-dir out/ \
    --checkpoint checkpoints/sdr2hdr_shadow_v1.pt
```

Master and deliver. No GPU required:

```bash
rudra info out/ --nits-scale 203                                           # nits, PQ codes, percentiles
rudra metadata out/ --nits-scale 203 --output out/master --peak-nits 1000  # DoVi L1 + HDR10+
rudra aces out/ --output delivery/aces --ocio                              # ACES EXR + OCIO
rudra grade in.exr --output graded/ --region 400:2000:1.0:0.5              # +1 EV, 400–2000 nits
rudra bench bench_root/ --output results.json                              # PU21-PSNR / CVVDP
```

Train the SDR→HDR model. Four stages, and each one refuses to hand work to the
next if it can't vouch for it:

```bash
python pipeline/scan_sources.py /path/to/source_hdr --out work/inventory.jsonl
python pipeline/prepare_pairs.py --inventory work/inventory.jsonl --dst work/pairs \
    --mode log2_extended --crops 3 --video-stride 2
python pipeline/build_manifests.py --pairs-dir work/pairs --out-dir work
python pipeline/verify_dataset.py --pairs-dir work/pairs \
    --manifest work/sdr_hdr_manifest.jsonl --video-manifest work/video_manifest_9f.jsonl
python training/train_sdr2hdr.py --mode image --manifest work/sdr_hdr_manifest.jsonl \
    --output-dir work/checkpoints/image --steps 100000 --best-metric composite_gain --device cuda
```

Then train the shadow gate on top of the frozen result. Fifteen minutes on one
4080, and it is the difference between a model that helps on degraded input and
one that helps on both:

```bash
python training/train_shadow_gate.py --manifest work/sdr_hdr_manifest.jsonl \
    --init-checkpoint work/checkpoints/image/best.pt \
    --output-dir work/checkpoints/shadow --seed 20260901
```

---

## Results

Everything here is 429 held-out frames at native 1280×720, scene-linear with
diffuse white at 1.0, scored at `--nits-scale 203`. Two metrics, both against
the same unclamped reference: PU21-PSNR and ColorVideoVDP JOD. Whole scenes are
held out, so no scene straddles a split.

`hard` is the deployment condition — unknown tone curve, 4:2:0 chroma, banding,
JPEG. `clean` is well-graded input that the analytic inverse-ACES baseline
already handles.

| Condition | Method | PU21-PSNR (dB) | CVVDP (JOD) |
|---|---|---:|---:|
| clean | analytic baseline | 45.99 | 9.448 |
| clean | v5, as shipped | 42.99 | 9.402 |
| clean | v6, 4× capacity | 43.24 | 9.452 |
| clean | v5, shadow arm off | **46.50** | 9.431 |
| clean | **v5 + shadow gate** | 46.06 | **9.561** |
| hard | analytic baseline | 25.92 | 7.362 |
| hard | **v5, as shipped** | **27.34** | **7.805** |
| hard | v6 | 26.88 | 7.706 |
| hard | v5, shadow arm off | 26.25 | 7.497 |
| hard | v5 + shadow gate | 27.15 | 7.751 |

The gate row is the one to read. It is the only configuration we have scored
that beats the analytic baseline in both conditions on both metrics.

### The two metrics disagree, and that is a result

On degraded input v5 earns its keep: +1.43 dB and +0.44 JOD over the baseline,
winning 348 of 429 frames. That is the deployment condition and it is what the
project is for.

On clean input the same model loses 3.0 dB of PU21-PSNR and wins only 115
frames. But CVVDP puts that gap at **−0.046 JOD**, far below a just-noticeable
difference. Two orders of magnitude apart on the same frames. What RUDRA adds to
well-graded input is highlight energy that PU21-PSNR punishes and no viewer
sees.

Never report the clean PSNR row without the JOD beside it.

### The failure mode is in the shadows

Split the clean frames by how much headroom the ground truth actually has and
the regression stops looking diffuse:

| clean frames | mean Δ vs baseline | median ground-truth peak |
|---|---:|---:|
| 60 worst | **−10.76 dB** | 238 nits |
| 60 best | **+3.41 dB** | 19,590 nits |

Correlation between `log2(peak_nits)` and gain is **+0.46**. The per-pixel
luminance gate never sees the frame, so it cannot tell a 238-nit studio interior
from a 20,000-nit sunset.

We guessed twice about what the error was and were wrong both times. It is not
invented highlights: at the worst 1% of pixels the *true* luminance has a median
of 4 nits against 21 frame-wide, and only 46% of them are over-predicted. It is
not a tail either: discarding the worst 10% of pixels closes 2.3 dB of the 5.55
dB gap and leaves 3.2.

The damage is the **shadow arm** of the gate firing on low-dynamic-range content
that needs no reconstruction at all. Ablating that arm recovers the whole clean
deficit and gives up most of the hard gain, which is what made the fix obvious.

### One number per frame would fix it, and the input does not carry it

Give the residual a single global scale α and let an oracle pick it per frame,
and the ceiling is large: **+5.84 dB on clean and +0.29 dB on hard at the same
time**. No constant gets there. Clean wants α ≈ 0.125, hard wants ≈ 1.1, and the
constant that fixes clean throws away 85% of the hard gain.

So we tried to learn it, and measured why it does not work. On 51 held-out
frames, each clean and degraded, cross-validated by frame:

| knowing | MAE on oracle α | R² |
|---|---:|---:|
| nothing (predict the mean) | 0.501 | 0.000 |
| the features, via a linear readout | 0.487 | **+0.031** |
| the condition, **perfectly** | 0.409 | **+0.213** |
| the oracle itself | 0.000 | 1.000 |

The features explain 3% of the target's variance. And 79% of that variance sits
*within* a condition rather than between, so a perfect clean-versus-degraded
classifier — everything a condition stem could ever buy — caps out at R² 0.213.
The remaining 79% asks whether this frame's clipped region was a 200-nit lamp or
a 20,000-nit sun, and an 8-bit frame does not carry the evidence.

That is not a feature-engineering gap. It is the information limit of
single-image inverse tone mapping, measured rather than asserted. Capacity (4×),
corpus (6×) and objective were each varied and none of them moved it.

### The gate that did work

The bound is on the question, not the problem. Asked for a continuous scale the
input cannot answer. Asked **"did this frame arrive clean or degraded?"** — the
one axis that is detectable, and the axis the shadow-arm ablation showed
matters — it can. That target needs no oracle either: at training time we know
whether we degraded the frame.

`ShadowGate` is **21,121 parameters** on the frozen v5 backbone, predicting one
weight per frame on the shadow prior, supervised by binary cross-entropy against
the degradation label. Weight 1.0 reproduces `recovery_mode="all"` exactly and
0.0 reproduces `recovery_mode="highlights"` exactly, so it interpolates between
the two ablation endpoints and nothing else.

Three seeds, gain over the analytic baseline:

| seed | clean dB | clean JOD | hard dB | hard JOD | clean CVVDP | clean frames won |
|---|---:|---:|---:|---:|---:|---:|
| 20260901 | +0.07 | +0.113 | +1.24 | +0.389 | 9.561 | 251 / 429 (59%) |
| 2 | +0.71 | +0.068 | +0.96 | +0.308 | 9.516 | 315 / 429 (73%) |
| 3 | +0.45 | +0.089 | +1.18 | +0.358 | 9.537 | 280 / 429 (65%) |
| **mean ± sd** | **+0.41 ± 0.33** | **+0.090 ± 0.023** | **+1.12 ± 0.15** | **+0.352 ± 0.041** | | |

The shipped v5 wins 115 of 429 clean frames for comparison. All three runs are
positive on all four measures, and all three land a clean CVVDP above every
fixed alternative including both ends of the ablation they interpolate. A hard
binary switch could not do that; the gate emits intermediate weights and finds
per-frame settings neither extreme reaches.

Against the shipped model that is **+3.41 ± 0.32 dB and +0.135 ± 0.023 JOD on
clean, for 0.30 ± 0.15 dB and 0.091 ± 0.041 JOD on hard.**

Seed 20260901 is the one committed to this repo. It is also the *worst* of the
three on clean PU21 and the *best* on clean CVVDP, so reporting it alone
understates the dB result sixfold and overstates the JOD by a quarter. Three
seeds support "the sign is stable". They do not support a confidence interval.

### One thing we tested and dropped

Two photographs run through the shipped model on 3 Sep — a stock sunset and an
Iceland landscape, neither in any split — both suggested something §6's
luminance account cannot see: in the brightest 1% of pixels the red share of
R+G+B fell by about 0.07 while green rose by about 0.05. A hue rotation at
roughly constant luminance is nearly invisible to PU21-PSNR and to CVVDP as we
report them, so if it were real it would be a defect our own metrics would miss.

Measured on all 429 held-out frames against the reference — which the
photographs do not have — it does not survive:

| band | share of pixels | green shift | vs the reference | frames agreeing on sign |
|---|---:|---:|---|---:|
| < 20 nits | 41.2% | +0.005 | towards | 33% |
| 20–203 | 46.1% | +0.000 | — | 32% |
| 203–1000 | 11.4% | −0.001 | towards | 64% |
| > 1000 | 1.3% | +0.005 | **away** (0.0057 → 0.0110) | 50% |

Only the top band moves away from the truth. The shift there is ten times
smaller than on the photographs, it covers 1.3% of pixels, and 50% sign
agreement across frames is a coin flip. Both photographs were saturated sunsets
whose highlights are nearly one hue, so a small per-channel error rotates them
coherently; the split's bright band averages near-neutral and per-frame
rotations cancel. Real on single-hue highlights, not a property of the model,
and **not in the paper**.

It is here because the alternative — quietly dropping a hypothesis that did not
work out — is how a repository ends up looking more certain than its evidence.
`training/analyze_chroma_shift.py` reproduces the table;
`docs/chroma_shift_shadow_v1.json` has the raw sums.

### What we got wrong along the way

Five of the expensive mistakes this cycle were measurement defects, not model
defects. Most are in the paper, because each one silently corrupted a result we
believed, and each is a mistake any comparable pipeline can make.

**Selection on the maximum of a noisy series.** `best.pt` was chosen by the
maximum of `composite_gain` over 102 evaluations, where `clean_gain_db` had mean
−1.43 dB and sd 1.29, and only 10 of 102 evals were ever positive. The shipped
checkpoint's +0.02 ranked 9th of 102. Selection now uses a trailing median over
five evals. Replayed, it picks step 72,000 instead of 81,000 — which we then
scored, and it is better on the criterion the selector optimises (composite
−1.36 against −1.57) and *worse on CVVDP in both conditions*. Smoothing a
selector does not fix selecting on the wrong quantity.

**An evaluation that read the front of the split.** `DataLoader(val,
shuffle=False)` with `max_batches=N` reads the alphabetically first N records.
One run's eval was 32 records over 11 scenes, every name between
`abandoned_factory` and `blau_river`, with 403 records and 87 scenes never
measured. Since error correlates with headroom, that slice reported
`clean_gain +0.61` where the benchmark measured −3.0.

**A viewer that presented every frame upside down.** The default framebuffer
puts row 0 at the bottom and the display shader sampled `vUV` unchanged. No
numeric test caught it, because they all compare the float composite, which a
presentation flip leaves untouched — and the one test that read the canvas did
so through `readPixels`, which returns rows bottom-first and cancelled the flip
against itself. Master EXR output was never affected. There is now a test that
looks at pixels.

**Two numbers transcribed and never checked back.** The paper said 346 frames
won on degraded input where the benchmark says 348, and gave v6 4,772,485
parameters where the checkpoint has 4,770,117. Both small, and neither was ever
compared to its source — the same shape of hole `PAPER_ERRATA.md` documents.
`training/audit_paper_numbers.py` now recomputes all 58 derived claims from
`bench/results/*.json`, and `tests/test_committed_checkpoint_2026_09_03.py`
counts the tensors in every committed model against what the registry and the
paper claim. Both fail loudly.

**Reporting one seed as if it were the method.** The gate first shipped on seed
20260901 alone. It turned out to be the *worst* of three on clean PU21 and the
*best* on clean CVVDP, so the single-seed report understated one result sixfold
and overstated the other by a quarter. It had been selected on validation
accuracy, and nobody had checked what that tracked. The paper now reports mean
and spread everywhere.

---

## The paper

[`PAPER_DRAFT_2026-08-29.md`](PAPER_DRAFT_2026-08-29.md) is the write-up of the
SDR→HDR model: 14 pages, 11 sections, three figures, and every number in it
traceable to a command. The markdown is the source of truth. The LaTeX under
`paper/` is generated.

```bash
bash paper/build.sh      # -> paper/main.pdf
bash paper/mkarxiv.sh    # -> paper/rudra-arxiv.tar.gz, verified to build flat
```

Never edit `paper/_body.tex` or `paper/_abstract.tex`; both are regenerated on
every build. `paper/ABSTRACT_ARXIV.txt` is a trimmed abstract for the
submission form, which caps at 1,920 characters where the PDF's runs longer.

Every derived number in §5, §6 and §6.1–6.2 is recomputed on demand by two
scripts that exit non-zero on any drift — 58 claims from the benchmark files,
five more from the headroom join. Both are below.

Two things the paper is honest about and this README should be too. No published
method has been scored on our split, so every number above is against our own
analytic baseline; that is the largest gap — `training/run_expandnet.py` closes
the mechanical half of it, and running it is the next job. And §6's
trimmed-PSNR table and §7's oracle sweep still have no script, because they need
per-pixel statistics over the reference frames rather than the per-frame results
the benchmark writes. §10 says so rather than leaving a reader to assume
otherwise.

### Reproducing the numbers

```bash
powershell -ExecutionPolicy Bypass -File training\run_bench.ps1    # four exports, six scorings
python training/audit_paper_numbers.py --bench <bench dir>         # 58 claims, exits 1 on drift
python training/analyze_headroom.py --bench <dir> --manifest <m> --check   # §6's split
python training/analyze_chroma_shift.py --bench <dir>/clean               # colour, per band
```

`run_bench.ps1` skips any stage whose output already exists, so an interrupted
run resumes. `score_checkpoint.ps1 -Checkpoint <ckpt> -Name <label>` adds one
model to an existing benchmark instead of redoing the reference, which is the
expensive half.

### Comparing against published work

The harness is built, and ExpandNet has a runner that imports the authors' own
model and weights from a clone of their repository rather than reimplementing
the method:

```bash
git clone https://github.com/dmarnerides/hdr-expandnet.git
python training/export_bench_pairs.py ... --write-sdr        # -> <out>/sdr/**.png
python training/run_expandnet.py --sdr <out>/sdr --repo hdr-expandnet \
    --out <out>/expandnet_raw
python training/import_method_output.py --out <dir> --from <out>/expandnet_raw \
    --name expandnet
rudra bench <dir> --nits-scale 203 --test-dir expandnet
```

Santos 2020, HDRCNN and SingleHDR have no runner yet.

Published single-image iTMO methods predict *relative* radiance with no nit
anchor, so the importer fits one global scalar per frame on the pixels the SDR
input did not clip. That is a free parameter RUDRA does not get — it predicts
absolute nits and is scored as it stands. Run the importer on RUDRA's own tree
as well and report the symmetric row, or the table flatters us for free.
`import_<name>.json` records the fitted scales, any frames resized, and any the
method dropped.

---

## Training data

The released model was trained on public HDR footage plus proprietary FXTD
material. None of it is committed here; `pipeline/scan_sources.py` reads
whatever you point it at.

| Source | What it gives | Grade ceiling | Licence |
|---|---|---|---|
| [Poly Haven HDRIs](https://polyhaven.com/hdris) | 963 scene-referred panoramas, real suns above 100,000 nits | none — scene-referred | [CC0](https://polyhaven.com/license) |
| [HdM-HDR-2014](https://hdm-stuttgart.de/vmlab/hdm-hdr-2014/) | 9 cinematic scenes: fireworks, forge sparks, fire, stage lights | 4,000 nits, Rec.2020 | free for academic use; commercial needs a licence from HdM |
| HdM-HFR-2017 | Bar and Fire scenes, 192 fps, same FTP host | 1,000 nits (PQ-1K) | as above |
| [Netflix Chimera](https://opencontent.netflix.com/) | 4K live action, P3-PQ | 10,000 nits | [CC BY 4.0](http://download.opencontent.netflix.com/) |

Both HdM sets come off the same plain-FTP host, `hdr-2014.hdm-stuttgart.de`
(user `HdM-HDR-2014`). `pipeline/fetch_stuttgart.py` handles the download and
resumes if it drops:

```bash
python pipeline/fetch_stuttgart.py --list-root
python pipeline/fetch_stuttgart.py --root HdM-HDR-2014_Color-Graded-for-HDR \
    --dest /path/to/source_hdr/Stuttgart_HDR_2014 --only carousel_fireworks fireplace
```

The grade ceilings in that table are not decoration. Three of the four sources
stop at a hard delivery ceiling, which is why the loss treats those pixels as
censored.

Three things that make a training log readable:

- Every eval scores two conditions. `clean_*` is the held-out frame as prepared,
  `hard_*` is the same frame under a seeded camera and codec degradation. Only
  the hard numbers describe deployment, so only they choose `best.pt`.
- Both report `gain_db` against the analytic inverse tone map the network sits
  on top of, so "better than doing nothing" is a number rather than a guess.
- Graded sources are censored. A pixel at exactly 4,000 nits in a 4,000-nit
  grade means *at least* 4,000, so the loss goes one-sided there with three
  stops of free headroom above the ceiling. Otherwise the model learns to cap.
  The temporal refiner did not do this until 28 Aug 2026, on a video corpus
  where 84% of clips carry a hard ceiling — it would have undone the image
  model's highlights frame by frame.

The distribution shift is worth knowing before you read any clean number. The
band where the model fails, below 400 nits, is 45.2% of the test split and 27.2%
of what the sampler actually draws. Weighted median peak: 1,713 nits in train
against 546 in test. We did not design that and a reader should weigh it.

---

## Layout

```
rudra/            Descriptor, DRE transformer, cross-attention, FiLM decoder,
                  DR-gated LoRA, losses, CVVDP metric, and sdr2hdr.py
rudra/delivery/   Torch-free: DoVi L1 / HDR10+, ACES / EXR / OCIO, grade
                  controls, benchmarks, the `rudra` CLI
pipeline/         Corpus construction and its gates: scanner, pair preparation,
                  HDR storage (hdr_io), manifests, verify_dataset
training/         Trainers, evaluation, inference, the exporter that turns a
                  checkpoint into benchmark pairs, run_bench.ps1, the ExpandNet
                  runner and third-party importer, and the scripts that
                  recompute and probe the paper's numbers
ui/               RUDRA Studio: the page, its GPU compositor, the inference
                  server behind them
paper/            LaTeX build of the markdown paper; build.sh and mkarxiv.sh
checkpoints/      Every SDR→HDR model, plus models.json, the registry the
                  viewer reads (see its README)
config/, configs/ VAE registry and training recipes
tests/            Curve round-trips, corpus guards, delivery, target decode,
                  censored highlights, GPU/torch composite parity, canvas
                  orientation, and a smoke test that presses every control
```

## How it is checked

The composite lives in three languages — torch in `rudra/sdr2hdr.py`, GLSL in
`ui/compositor.js`, numpy in `tests/compose_reference.py` — and the numpy one is
the reference the other two are measured against, so the picture on screen and
the EXR that Master writes cannot drift apart.

| Check | Result |
|---|---|
| Shader vs reference, 11 cases incl. shadow weight 0.0–1.0 | 8.4 × 10⁻⁶ relative |
| Wipe: which side is which, and does the seam move | baseline 49, RUDRA 255 |
| Torch vs reference, untiled | 3 × 10⁻⁶ relative |
| Torch vs reference, tiled (overlap bands only) | up to a few % |
| MaxCLL, page vs the EXR it writes | 0.023% apart |
| Every control on the page, pressed | 54 checks, 0 failed |

```bash
pytest tests/                                  # everything
python tests/webgl_parity/parity.py            # shader vs the reference
python tests/webgl_parity/orientation.py       # the canvas itself, not a buffer
python tests/webgl_parity/wipe.py              # the wipe, also from pixels
python ui/server.py --port 8100 --no-browser --preload --device cpu
python tests/ui_smoke/press_everything.py --url http://127.0.0.1:8100/
```

The shader checks run headless on software GL, so they need no GPU and no torch.
The page gets pressed rather than read, because a menu item that runs nothing
looks exactly like one that works.

Tiling is the one place the two compositions genuinely disagree, because
feathering fields and feathering composed predictions are not the same operation
either side of `expm1`. The viewer and Master both ask for an untiled pass, fall
back to tiles only on OOM, and say which they used.

The screenshot at the top of this file is generated against a running Studio and
refuses to overwrite itself with anything under 200 KB, which is the size of an
error page:

```bash
python ui/capture_shot.py
```

Every script has `--help` and a docstring saying what it does and why.
`pipeline/verify_dataset.py` lists the nine corpus invariants.
`pipeline/hdr_io.py` is the single source of truth for how targets are stored.

---

## Notes

Six checkpoints are committed, 38 MB in total: the shipped model, its two other
seeds, the v5 backbone, the v6 capacity ablation, and the temporal refiner. The
first five are what the Results tables are made of. Training pairs and HDR
sources stay out; generate them with the scripts above, or pull the diffusion
decoders from
[HuggingFace](https://huggingface.co/fxtdstudios/RUDRA/tree/main).

The production decoders need no text conditioning. Stages 2 and 3 follow the
original paper and target SDXL's U-Net cross-attention; porting them to Flux is
the §10 generalization experiment. Stage 3 has never been trained — see
[`PAPER_ERRATA.md`](PAPER_ERRATA.md) §2 and [`STATUS.md`](STATUS.md) for which
lines of work are finished and which are not.

The temporal refiner exists and is not evaluated here. Its held-out set is 4
validation and 5 test clips of one scene each, which is too small to report.
