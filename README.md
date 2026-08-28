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

Drop a frame in. The network runs once, on the GPU, and hands the page its raw
fields — everything after that (residual strength, recovery mode, preserve,
display peak) is composed on your own GPU, so the controls move at frame rate
rather than at one round trip each.

Compare by flipping: hold **B**, or press and hold the image, and it swaps to
the analytic inverse tone map with the frame staying exactly where it is. That
is a much easier thing for the eye to read than a seam travelling across the
picture, because you are comparing the same pixels rather than tracking a
moving edge. Waveform, histogram and every measured number follow the composite
you are actually looking at; MaxCLL and MaxFALL come from an exact GPU
reduction, not from a downsample. **Master EXR** writes a full-resolution ACES
2065-1 container with AP0 primaries, ST 2065-4 chromaticities and a JSON
sidecar of delivery metadata.

Needs a browser with WebGL2 and float render targets — any current Chrome,
Edge, Firefox or Safari.

```bash
python ui/server.py --preload
```

It loads the newest `shipped_*.pt` it can find under the checkpoint roots — at
present v5 step 81 000. Pin a different one with `--checkpoint <path>`, or
compare two without restarting: the page can score any checkpoint against the
same frame in place.

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
  otherwise the model learns to cap.

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

**Corpus.** 28 542 pairs across 976 physical scenes — 963 Poly Haven stills and
13 video scenes. Whole scenes are held out; no scene straddles a split.

Reproduce with `python training/sweep_inference.py --checkpoint <ckpt> --manifest
<manifest>`.

> These are gains over RUDRA's own analytic baseline, not a comparison against
> published inverse tone mapping work. CVVDP JOD numbers are not in yet, and
> highlight-mask headroom has not been re-measured on v5. Read them as internal
> progress, not as a benchmark result.

---

## Layout

```
rudra/            Descriptor, DRE transformer, cross-attention, FiLM decoder,
                  DR-gated LoRA, losses, CVVDP metric
rudra/delivery/   Torch-free: DoVi L1 / HDR10+, ACES / EXR / OCIO, grade
                  controls, benchmarks, the `rudra` CLI
pipeline/         Corpus construction and its gates: scanner, pair preparation,
                  HDR storage (hdr_io), manifests, verify_dataset
training/         Trainers, evaluation, inference, benchmarks
ui/               RUDRA Studio: the page, its GPU compositor and the
                  inference server behind them
config/, configs/ VAE registry and training recipes
tests/            Curve round-trips, corpus guards, delivery, target decode
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

Every script has `--help` and a docstring saying what it does and why.
`pipeline/verify_dataset.py` lists the nine corpus invariants;
`pipeline/hdr_io.py` is the single source of truth for how targets are stored.

## Notes

Weights, pairs and HDR sources are deliberately not committed. Generate them
with the scripts above.

The production decoders need no text conditioning. Stages 2 and 3 follow the
paper and target SDXL's U-Net cross-attention; porting them to Flux is the §10
generalization experiment.
