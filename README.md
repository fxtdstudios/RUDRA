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

See **[DECODER_CHEATSHEET.md](DECODER_CHEATSHEET.md)** for which `decoder_size` to select
per backbone and the node settings.

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

```bash
python training/build_sdr_hdr_manifest.py --sdr-dir G:/data/sdr --hdr-dir G:/data/hdr \
  --metadata-dir G:/data/meta --output hdrdata/sdr_hdr_manifest.jsonl
python training/train_sdr2hdr.py --mode image --manifest hdrdata/sdr_hdr_manifest.jsonl \
  --output-dir hdrdata/checkpoints/sdr2hdr_image_50k --steps 50000 --device cuda
python training/evaluate_sdr2hdr.py --manifest hdrdata/sdr_hdr_manifest.jsonl \
  --checkpoint hdrdata/checkpoints/sdr2hdr_image_50k/best.pt --split test
```

`best.pt` is protected by a held-out baseline gate: training starts with the analytic
inverse-tone-map checkpoint and only replaces it when validation log-radiance error improves.

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

## Documentation

- **[DELIVERY_2026-08-22.md](DELIVERY_2026-08-22.md)** — delivery layer: usage, conventions, verification.
- **[MOAT_REVIEW_2026-08-22.md](MOAT_REVIEW_2026-08-22.md)** — system-design review vs Runway Ruby / Topaz Hyperion / Beeble.
- **[RUDRA_TECHNICAL_REVIEW.md](RUDRA_TECHNICAL_REVIEW.md)** — full code review + every fix applied.
- **[RUDRA_RETRAIN_RUNBOOK.md](RUDRA_RETRAIN_RUNBOOK.md)** — step-by-step (re)training guide.
- **[DECODER_CHEATSHEET.md](DECODER_CHEATSHEET.md)** — decoder selection + lessons learned.
- **[research/RUDRA_V01.pdf](research/RUDRA_V01.pdf)** — the paper.

## Notes

- Weights, pairs, and HDR sources are intentionally **not** committed (see `.gitignore`).
  Generate decoders/pairs locally with the scripts above.
- The production decoders need no text conditioning; the RUDRA-Full conditioning stages
  (2/3) follow the paper and are intended for SDXL (U-Net cross-attention). Porting the
  conditioning to Flux is the paper's §10 generalization experiment.
