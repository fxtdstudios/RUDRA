# RUDRA

**Radiometric Dynamic-Range Conditioning for HDR-Aware Diffusion Models**
[FXTD Studios](https://fxtdstudios.com) / Radiance Research

Diffusion models are trained on tone-mapped images, so they learn a world where
nothing is brighter than white. RUDRA gives them the rest of the range back: it
decodes latents straight to scene-linear HDR, and conditions the backbone on a
radiometric descriptor so luminance, exposure and wide-gamut colour survive
generation. A separate path needs no diffusion model at all — hand it an
ordinary 8-bit frame and it reconstructs the highlights the tone map threw away.

📦 Weights: [huggingface.co/fxtdstudios/RUDRA](https://huggingface.co/fxtdstudios/RUDRA/tree/main)
📄 Paper: [research/RUDRA_V01.pdf](research/RUDRA_V01.pdf)

---

## RUDRA Studio

![RUDRA Studio](docs/rudra_studio.png)

Drop a frame in — or a whole sequence. The network runs once per frame on the
GPU and hands the page its raw fields; everything after that is composed on
your own GPU, so the controls move at frame rate instead of at one round trip
each.

**Compare by flipping.** Hold **B**, or press and hold the image, and it swaps
to the analytic inverse tone map without the frame moving. You are comparing
the same pixels rather than tracking a seam.

**Region EV is a grade, not a preview.** Drag a value to scrub it, double-click
to zero it. The bands are soft luminance qualifiers from
`rudra/delivery/controls.py`, and **Master EXR** applies the identical
qualifier and gain to the file — full-resolution ACES 2065-1, AP0 primaries,
ST 2065-4 chromaticities, and a sidecar recording exactly how it was graded.

Waveform, histogram and every measured number follow the composite on screen;
MaxCLL and MaxFALL come from an exact GPU reduction rather than a downsample.
The frames rail, the transport and the menus all do what they say — press `?`
for the keyboard, and see `tests/ui_smoke/press_everything.py`, which presses
every control and fails if one turns out to be decoration.

`?frame=<url>` loads a frame straight from the server instead of dropping one,
and `?demo=1` loads the bundled sample — which is how the screenshot above is
taken, so it can be regenerated rather than staged.

Needs a browser with WebGL2 and float render targets — any current Chrome,
Edge, Firefox or Safari.

```bash
python ui/server.py --preload
```

It serves on `localhost:8422` and loads the newest `shipped_*.pt` it can find
under the checkpoint roots — at present v5 step 81 000. Pin a different one
with `--checkpoint <path>`, or compare two without restarting: the page can
score any checkpoint against the same frame in place.

Asset URLs are stamped with each file's mtime and size, and the port is
deliberately not 8080 — both so a stale entry another project left in your
browser cache can never answer for RUDRA's.

---

## Setup

```bash
git clone https://github.com/fxtdstudios/RUDRA.git && cd RUDRA
pip install -e .                             # the torch-free `rudra` CLI
pip install -r requirements-metrics.txt      # cvvdp + lpips, for the real metrics
pytest tests/
```

CUDA is only needed for training and inference. The delivery layer is pure
numpy, so render and mastering machines need nothing but Python.

### ComfyUI decoders

Download the decoder for your backbone into `ComfyUI/models/radiance/`, then
enable `rudra_decoder` in the *Radiance HDR VAE Decode* node and pick a
`decoder_size`.

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

**Reconstruct a frame or a sequence.**

```bash
python training/infer_sdr2hdr.py input/ --output-dir out/ \
    --checkpoint work/checkpoints/image/best.pt
```

**Master and deliver.** No GPU required.

```bash
rudra metadata out/ --nits-scale 203 --output out/master --peak-nits 1000  # DoVi L1 + HDR10+
rudra aces out/ --output delivery/aces --ocio                              # ACES EXR + OCIO
rudra grade in.exr --output graded/ --region 400:2000:1.0:0.5              # +1 EV, 400–2000 nits
rudra bench bench_root/ --output results.json                              # PU21-PSNR / CVVDP
```

**Train the SDR→HDR model.** Four stages; each refuses to hand work to the next
if it can't vouch for it.

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

### Training data

The released model was trained on public HDR footage. None of it is committed
here; `pipeline/scan_sources.py` reads whatever you point it at.

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
censored — see the third bullet below.

Three things that will make a training log make sense:

- Every eval scores two conditions. `clean_*` is the held-out frame as prepared;
  `hard_*` is the same frame under a seeded camera and codec degradation. Only
  the hard numbers describe deployment, so only they choose `best.pt`.
- Both report `gain_db` against the analytic inverse tone map the network sits
  on top of, so "better than doing nothing" is a number, not a guess.
- Graded sources are treated as censored. A pixel at exactly 4,000 nits in a
  4,000-nit grade means *at least* 4,000, so the loss goes one-sided there —
  otherwise the model learns to cap. Both trainers do this. The temporal
  refiner did not until 28 Aug 2026, on a video corpus where 84% of clips
  carry a hard ceiling, which would have had it undo the image model's
  highlights frame by frame.

---

## Results

Every figure below comes from the same held-out split and the same fixed
evaluation set. `clean_baseline_psnr_log` is 52.757246777 and
`hard_baseline_psnr_log` is 30.238924973 in all three runs, identical to nine
decimal places, so the runs are directly comparable and the differences are not
sampling noise.

`hard` is the deployment condition — unknown tone curve, 4:2:0 chroma, banding,
JPEG. `clean` is well-graded input that the analytic inverse-ACES baseline
already handles. Both columns are gains in log-radiance PSNR over that baseline.

**Direct SDR → HDR, image model**

| Checkpoint | hard | clean | composite |
|---|---|---|---|
| v3b, step 44 000 | +1.71 dB | −0.90 dB | — |
| v3b, step 48 000 | +1.81 dB | −4.22 dB | — |
| v4, step 58 000 (6× data) | +1.83 dB | −2.45 dB | −0.62 |
| v4, step 78 000 | +1.74 dB | −0.18 dB | +1.56 |
| **v5, step 81 000 — shipped** | **+1.80 dB** | **+0.02 dB** | **+1.80** |

Peak highlight recovery has not moved in three runs. Around +1.8 dB is where
this architecture sits on this corpus, and more data did not change that: v4 saw
six times the footage for 0.03 dB.

What moved is the price. v3b and v4 bought their highlight recovery by damaging
well-graded input, by as much as 4.2 dB. v5 gets the same recovery for nothing.
That is the censored-highlight loss working: 78% of public HDR footage is
delivery-graded, and plain L1 against pixels sitting on a grading ceiling teaches
the model to cap. Treating those pixels as *at least* the ceiling rather than
*exactly* the ceiling removes the pressure.

**Both long runs peak near 80 000 steps and decay after.**

| | step 78–81 k | step 100 000 |
|---|---|---|
| v4, clean | −0.18 dB | −3.44 dB |
| v5, clean | +0.02 dB | −3.96 dB |

Every one of v5's last fifteen evaluations is negative on clean input, and
`clean_log_l1` climbs from 0.00496 — better than the 0.00513 baseline — to
0.00722, while hard-condition highlight error barely moves. v4 does the same
thing without the censored loss, so this is the cosine schedule annealing the
model onto the delivery-graded majority, not an artefact of the new term. Treat
80 000 steps as the useful budget for this recipe; best-checkpoint selection is
what rescues a longer run.

**preserve_outside**, v5 step 81 000. Blending back to the baseline where the
learned masks are cold:

| Mode | clean PSNR_log | gain | hard PSNR_log | gain |
|---|---|---|---|---|
| inverse-ACES baseline | 52.76 | — | 30.24 | — |
| plain | 52.78 | +0.02 | 32.04 | +1.80 |
| **preserve_outside** | **53.42** | **+0.67** | 31.91 | +1.67 |

It costs 0.13 dB on degraded input and adds 0.65 dB on clean input, so it stays
the shipped default.

**The model.** 1 196 197 parameters, 4.6 MiB of float32 — small because it
never synthesises an image. It predicts a residual on top of the analytic
inverse-ACES baseline, gated to learned highlight and shadow masks, so its head
is five channels: three of residual and two of mask logits. Half the weights
sit in the middle block at quarter resolution. A checkpoint fits in a browser
download and runs a 1600×900 frame in one untiled pass.

Every run so far — v3, v3b, v4, v5 — used `--base-channels 32`. **Capacity has
never been ablated.** The plateau above is read as a corpus ceiling, and v4 is
real evidence for that (six times the footage, +0.03 dB), but "more data did not
help" and "more capacity would not help" are different claims and only the first
has been tested. `--base-channels 64` is 4.77 M parameters and one overnight
run; until it exists the ceiling is an inference, not a measurement.

**Corpus.** 28 542 pairs across 976 physical scenes — 963 Poly Haven stills and
13 video scenes. Whole scenes are held out; no scene straddles a split.

Reproduce with `python training/sweep_inference.py --checkpoint <ckpt> --manifest
<manifest>`.

> These are gains over RUDRA's own analytic baseline, not a comparison against
> published inverse tone mapping work. CVVDP JOD numbers are not in yet,
> highlight-mask headroom has not been re-measured on v5, and no model wider
> than 32 base channels has been trained. Read them as internal progress, not
> as a benchmark result.

**Getting numbers that are comparable to published work.** `rudra bench`
scores paired directories in PU21-PSNR and CVVDP JOD — the only public
measuring sticks in this field — but nothing produced its input, so between a
trained checkpoint and a JOD there was no step at all. There is now:

```bash
python training/export_bench_pairs.py --checkpoint <ckpt> \
    --manifest <manifest> --split test --condition clean --out <dir>
python -m rudra.delivery.cli bench <dir> --nits-scale 203
python -m rudra.delivery.cli bench <dir> --nits-scale 203 --test-dir baseline
```

That writes `ref/`, `test/` and `baseline/` trees of scene-linear EXRs over the
429 held-out frames, and scores the model and the analytic baseline against one
reference. `--condition hard` reuses the eval's seeded degradation, so the
exported frames are the ones behind the `hard_*` numbers rather than a fresh
draw. The reference is decoded without the network's `max_hdr` clamp: clamping
it would score the model against a ground truth cropped to the model's own
ceiling.

CVVDP needs `pip install cvvdp`. It had never run once before 28 Aug 2026 — the
metric was handed a device string where pycvvdp wanted a `torch.device`, three
layers of `except Exception` turned the resulting error into "install torch +
pycvvdp", and that advice was given to a machine which had them installed. The
summary now reports the real reason whenever the backend is missing.

**The reconstruction is assembled in three languages** — torch in the model,
GLSL in the viewer, numpy as the reference both are checked against — so the
picture on screen and the EXR that Master writes cannot drift apart:

| Check | Result |
|---|---|
| Shader vs reference, 8 control combinations | 8.4 × 10⁻⁶ relative |
| Torch vs reference, untiled | 3 × 10⁻⁶ relative |
| Torch vs reference, tiled (overlap bands only) | up to a few % |
| MaxCLL, page vs the EXR it writes | 0.023% apart |
| Every control on the page, pressed | 52 checks, 0 failed |

Tiling is the one place the two compositions genuinely disagree, because
feathering fields and feathering composed predictions are not the same
operation either side of `expm1`. The viewer and Master both ask for an
untiled pass and fall back to tiles only on OOM, and say which they used.

---

## Layout

```
rudra/            Descriptor, DRE transformer, cross-attention, FiLM decoder,
                  DR-gated LoRA, losses, CVVDP metric
rudra/delivery/   Torch-free: DoVi L1 / HDR10+, ACES / EXR / OCIO, grade
                  controls, benchmarks, the `rudra` CLI
pipeline/         Corpus construction and its gates: scanner, pair preparation,
                  HDR storage (hdr_io), manifests, verify_dataset
training/         Trainers, evaluation, inference, and the exporter that
                  turns a checkpoint into benchmark pairs
ui/               RUDRA Studio: the page, its GPU compositor and the
                  inference server behind them
config/, configs/ VAE registry and training recipes
tests/            Curve round-trips, corpus guards, delivery, target decode,
                  censored highlights, GPU/torch composite parity, and a
                  smoke test that presses every control on the page
```

The composite lives in three languages — torch in `rudra/sdr2hdr.py`, GLSL in
`ui/compositor.js`, numpy in `tests/compose_reference.py` — and the last one is
the reference the other two are checked against, so the picture on screen and
the EXR that Master writes cannot drift apart:

```bash
pytest tests/test_frame_fields_2026_08_28.py   # torch  vs the reference
python tests/webgl_parity/parity.py            # shader vs the reference
```

The shader check runs headless on software GL, so it needs no GPU and no torch.
The page itself gets pressed rather than read, because a menu item that runs
nothing looks exactly like one that works:

```bash
python ui/server.py --port 8100 --no-browser --preload --device cpu
python tests/ui_smoke/press_everything.py --url http://127.0.0.1:8100/
```

The screenshot at the top of this file is generated the same way, against a
running Studio, and refuses to overwrite itself with anything under 200 KB —
the size of an error page:

```bash
python ui/capture_shot.py
```

Measurements on the page come from fields that crossed the wire as float16, so
they land within about 0.03% of the master written from float32 — close enough
that MaxCLL agrees to a handful of nits out of twenty-five thousand, and near
enough documented that nobody has to wonder.

Every script has `--help` and a docstring saying what it does and why.
`pipeline/verify_dataset.py` lists the nine corpus invariants;
`pipeline/hdr_io.py` is the single source of truth for how targets are stored.

## Notes

Weights, pairs and HDR sources are deliberately not committed. Generate them
with the scripts above.

The production decoders need no text conditioning. Stages 2 and 3 follow the
paper and target SDXL's U-Net cross-attention; porting them to Flux is the §10
generalization experiment.
