# RUDRA

**Radiometric Dynamic-Range Conditioning for HDR-Aware Diffusion Models**
FXTD Studios / Radiance Research

📦 **Pretrained weights:** [huggingface.co/fxtdstudios/RUDRA](https://huggingface.co/fxtdstudios/RUDRA/tree/main)

RUDRA makes latent-diffusion backbones HDR-aware. Instead of tone-mapping scene-linear
imagery down to SDR before the model ever sees it, RUDRA (a) decodes diffusion latents
directly into scene-linear HDR / OpenEXR, and (b) conditions the backbone on a compact
radiometric descriptor `R(x) = [L, E, H, x_CIE, y_CIE]` so generation preserves luminance,
exposure, highlight structure and wide-gamut color.

---

## What's here

The repo has **four cooperating systems**:

1. **Production HDR decoders** (`fast_vae.py` in the ComfyUI `radiance` node) — fast,
   distilled `latent → log image → scene-linear` decoders. Trained by
   `training/train_turbo_decoder.py`. These are what the *Radiance HDR VAE Decode* node
   loads. Trained, validated, and deployed for **7 backbones**.
2. **The RUDRA research pipeline** (`rudra/`) — the paper's contribution: a 5-channel
   spatial descriptor → 12-layer Dynamic Range Encoder (DRE) → cross-attention token
   injection, plus a FiLM-conditioned decoder and DR-gated LoRA. Trained by
   `training/train_rudra.py` (Stages 1–3).
3. **Direct SDR recovery** (`rudra/sdr2hdr.py`) — a compact mask-aware image network
   accepting ordinary 8-bit sRGB pixels, plus an optional lightweight real-clip temporal
   refiner. Trained by `training/train_sdr2hdr.py`. Unlike systems 1–2, this path does not
   require diffusion/VAE latents as input.
4. **The delivery layer** (`rudra/delivery/`, **torch-free**) — everything after radiance
   exists: Dolby Vision L1 / HDR10+ dynamic-metadata analysis, ACES 2065-1 container EXR
   export + OCIO config, artist grade controls (EV, per-region EV, luminance qualifiers,
   hue-preserving knee/peak), a paired PU21-PSNR/CVVDP benchmark harness, and the headless
   `rudra` CLI. Runs on numpy alone — no CUDA/torch needed on render or delivery machines.

## Trained decoders (production)

| Backbone | VAE latent | Best decoder | PSNR_log |
|---|---|---|---|
| Flux.1   | 16ch / 8×  | full  | 29.77 |
| Wan      | 16ch / 8×  | full  | 32.45 |
| LTX      | 128ch / 8× | full  | 25.47 |
| SDXL     | 4ch / 8×   | turbo | 33.86 |
| Z-Image  | = Flux VAE | (use the Flux decoder) | — |
| Qwen-Image | 16ch / 8× | turbo | 26.67 |
| Flux.2 Klein | 128ch / **16×** | turbo | 28.57 |

### Pretrained weights (Hugging Face)

Trained decoders are hosted at
**[huggingface.co/fxtdstudios/RUDRA](https://huggingface.co/fxtdstudios/RUDRA/tree/main)**
(they are intentionally not committed to git). Download the ones you need into the ComfyUI
models folder:

```bash
huggingface-cli download fxtdstudios/RUDRA --include "rudra_*_decoder_*.safetensors" \
    --local-dir "ComfyUI/models/radiance"
```
or in Python:
```python
from huggingface_hub import hf_hub_download
hf_hub_download("fxtdstudios/RUDRA", "rudra_full_decoder_flux_ema.safetensors",
               local_dir="ComfyUI/models/radiance")
```
File names follow `rudra_{turbo|full}_decoder_{backbone}_ema.safetensors` — drop them in
`ComfyUI/models/radiance/`, enable `rudra_decoder` in the node, and pick `decoder_size`.

---

## Quick start

```bash
pip install -r requirements-metrics.txt      # cvvdp (real HDR-VDP metric) + lpips
pytest tests/                                 # curve round-trips, freeze, conditioning
```

### Build a decoder for any backbone
Downloads the VAE, generates pairs from `hdrdata/Source_HDR`, trains turbo + full, and
deploys into the ComfyUI models folder:
```bash
python training/build_all_decoders.py --only flux,wan,ltx,sdxl,qwen,flux2-klein
```

### Inspect / evaluate a trained decoder
```bash
python training/scan_model.py hdrdata/checkpoints/turbo_flux --model-type flux \
    --pair_dir hdrdata/hdr_pairs --samples 200
```

### Train direct 8-bit SDR to HDR recovery

Four stages: inventory the sources, build pairs, build scene-safe manifests, train.
Each stage refuses to proceed on a corpus it cannot vouch for.

```bash
# 1. inventory. --encoding declares the transfer function when the dataset
#    documents it and the filenames do not (HdM-HDR-2014 is PQ graded to 4,000
#    nits and says so nowhere in its paths).
python pipeline/scan_sources.py /path/to/source_hdr --out work/inventory.jsonl

# 2. pairs. Targets are stored through pipeline/hdr_io.py -- log2_extended keeps
#    0.005..1,000,000 nits at ~2,380 codes per stop, so nothing clips and the
#    shadows get 36,384 codes below diffuse white instead of 1,330.
python pipeline/prepare_pairs.py --inventory work/inventory.jsonl --dst work/pairs \
  --mode log2_extended --crops 3 --video-stride 2

# 3. manifests. Whole scenes are held out; framings of one setup collapse to one
#    scene; delivery-graded sources are tagged with the ceiling they were graded to.
python pipeline/build_manifests.py --pairs-dir work/pairs --out-dir work

# 4. gate, then train. verify_dataset checks nine invariants and refuses on any.
python pipeline/verify_dataset.py --pairs-dir work/pairs \
  --manifest work/sdr_hdr_manifest.jsonl --video-manifest work/video_manifest_9f.jsonl
python pipeline/check_target_scale.py --manifest work/sdr_hdr_manifest.jsonl
python training/train_sdr2hdr.py --mode image --manifest work/sdr_hdr_manifest.jsonl \
  --output-dir work/checkpoints/image --steps 100000 --best-metric composite_gain --device cuda
```

Three properties worth knowing about, because each exists to stop a specific
class of silent failure:

**Every eval scores two conditions.** `clean_*` is the held-out SDR as prepared;
`hard_*` is the same frames under a seeded camera/codec degradation — unknown
tone curve, 4:2:0 chroma, banding, JPEG. Only `hard_*` reflects deployment, so
only `hard_*` selects `best.pt`. Both report `gain_db` against the analytic
inverse-ACES baseline the network sits on top of, so "is this better than doing
nothing" is a logged number, never an inference.

**`--best-metric composite_gain`** scores `hard_gain_db + min(0, clean_gain_db)`:
the improvement on degraded input, less any harm done to clean input. Selecting
on raw loss instead always picks the last checkpoint, which is rarely the best
trade.

**Delivery-graded targets are treated as censored.** A pixel at exactly 4,000
nits in a 4,000-nit graded source means "≥ 4,000", not "= 4,000". Plain L1
against those pixels teaches the model to cap. `build_manifests.py` detects a
grading ceiling (a peak value many frames land on bit-for-bit — natural scene
peaks never repeat exactly) and the loss goes one-sided there: predicting above
the ceiling is free up to `CENSORED_HEADROOM_STOPS`, predicting below still costs.

### RUDRA Studio (local UI)

![RUDRA Studio](docs/rudra_studio.png)

```bash
python ui/server.py --checkpoint work/checkpoints/image/best.pt --preload
```

<sub>The screenshot is captured from the running UI, not mocked up — regenerate
it with `python ui/capture_shot.py` while the server is running. `?demo=1` runs
the bundled frame through the loaded checkpoint on page load, so the numbers in
it are that checkpoint's real output. The script refuses to write unless a model
is actually loaded and the capture is a plausible size: Chrome exits 0 and
writes a valid PNG when a page fails to load, so an unguarded capture taken with
the server down silently commits a picture of "This site can't be reached".</sub>

A local page on `http://localhost:8080` for looking at the model on your own
footage, which no metric substitutes for. Drop an SDR image and it reports what
is actually measurable without a reference: **MaxCLL / MaxFALL** from the same
`rudra.delivery.metadata` code that writes the HDR10 sidecar, peak and P99 nits,
the share of pixels above diffuse white, and how far the model moved from its
analytic baseline — globally, inside each learned mask, and as an RMS departure
in stops.

The compare slider wipes the model against the **inverse-ACES baseline**, not
against the SDR, so it shows what the network added rather than what the tone
map already gave you. The exposure control moves the display peak
(`203 × 2^EV`), not the prediction: raising it lifts the clip point so
reconstructed highlights become visible on an SDR monitor.

There is deliberately no LPIPS or JOD here. Both need the ground-truth HDR,
which a file you just dropped in does not have; reference metrics live in
`training/sweep_inference.py` on the held-out split.

### Master, measure, and export (delivery layer — no GPU required)

```bash
pip install -e .          # installs the torch-free `rudra` console script

# per-shot Dolby Vision L1 + HDR10+ + analysis sidecar from linear frames
rudra metadata outputs/clip/ --nits-scale 203 --output outputs/clip/master --peak-nits 1000
dovi_tool generate --json outputs/clip/master_dovi_generate.json --rpu-out clip.rpu

# ACES 2065-1 container EXR master + OCIO v2 config for Resolve/Nuke
rudra aces outputs/clip/ --output delivery/aces --ocio

# graded HDR10: +1 EV on the 400–2000-nit band, mastered to 1000 nits
rudra grade in.exr --output graded/ --region 400:2000:1.0:0.5 --peak-nits 1000

# paired benchmark (root/ref/**, root/test/**) → PU21-PSNR, +CVVDP JOD when torch present
rudra bench bench_root/ --output results/stuttgart.json
```

HDR10 video export (`training/export_hdr10.py`) computes MaxCLL/MaxFALL on max(R,G,B)
per CTA-861.3 and gains `--dynamic-metadata` to emit all three sidecars alongside the mux.

### The research program (RUDRA-Full, on SDXL — the paper's intended backbone)
```bash
python training/research_sdxl.py --phase stage3     # DRE + cross-attention (core thesis)
python training/research_sdxl.py --phase sweep      # §7.3 λ conditioning-strength sweep
python training/research_sdxl.py --phase ablate     # §7.2 descriptor channel ablation
```

### Benchmark vs. a baseline (e.g. LTX IC-LoRA-HDR), real perceptual metric
```bash
python training/benchmark_hdr.py --gt gt --a rudra_preds --b iclora_preds \
    --name-a RUDRA --name-b IC-LoRA-HDR
```

---

## Repo layout

```
rudra/            Research package: descriptor, DRE transformer, cross-attention,
                  FiLM decoder, DR-gated LoRA, losses, ColorVideoVDP metric, pipeline
pipeline/         Corpus construction and its gates: source inventory, pair
                  preparation, HDR storage (hdr_io), scene-safe manifests,
                  verify_dataset, check_target_scale
ui/               RUDRA Studio: static page + torch-backed inference server
rudra/delivery/   Torch-free delivery layer: DoVi L1/HDR10+ metadata, ACES/EXR/OCIO,
                  grade controls, PU21/CVVDP bench, the `rudra` CLI
pipeline/         Corrected data pipeline v3: hdr_io storage modes, source scanner,
                  scene-safe manifests, verify_dataset gate (run before every training run)
training/         Trainers + orchestration:
                    train_turbo_decoder.py   production decoder (node)
                    train_rudra.py           Stages 1–3 (decoder / LoRA / DRE)
                    train_sdr2hdr.py         direct 8-bit SDR image / temporal recovery
                    evaluate_sdr2hdr.py      frozen-test comparison vs physical baseline
                    infer_sdr2hdr.py         image/video to float HDR frame sequence
                    build_all_decoders.py    download → pairs → train → deploy
                    research_sdxl.py         full SDXL research program
                    scan_model.py / scan_highlights.py / benchmark_hdr.py
                    cleanup_checkpoints.py / download_models.py
config/, configs/ Per-model VAE config registry + training recipes
tests/            Curve round-trip, freeze-count, conditioning, hdrvdp tests
research/         RUDRA_V01.pdf
```

## Metrics

HDR quality is reported with **ColorVideoVDP** (the pip-installable successor to HDR-VDP
from Mantiuk et al.) in JOD units, alongside ΔE2000, EV-error, highlight-reconstruction
accuracy, and tone-mapped PSNR/SSIM. Install `cvvdp` for the real metric; without it the
code falls back to a clearly-labeled proxy.

## Reference

- **[research/RUDRA_V01.pdf](research/RUDRA_V01.pdf)** — the paper.
- Every script carries its own `--help` and a module docstring stating what it
  does and why it exists; `pipeline/verify_dataset.py` documents the nine corpus
  invariants, and `pipeline/hdr_io.py` is the single source of truth for how HDR
  targets are stored on disk.

## Notes

- Weights, pairs, and HDR sources are intentionally **not** committed (see `.gitignore`).
  Generate decoders/pairs locally with the scripts above.
- The production decoders need no text conditioning; the RUDRA-Full conditioning stages
  (2/3) follow the paper and are intended for SDXL (U-Net cross-attention). Porting the
  conditioning to Flux is the paper's §10 generalization experiment.
