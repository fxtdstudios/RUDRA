# RUDRA

**Radiometric Dynamic-Range Conditioning for HDR-Aware Diffusion Models**
FXTD Studios / Radiance Research

RUDRA makes latent-diffusion backbones HDR-aware. Instead of tone-mapping scene-linear
imagery down to SDR before the model ever sees it, RUDRA (a) decodes diffusion latents
directly into scene-linear HDR / OpenEXR, and (b) conditions the backbone on a compact
radiometric descriptor `R(x) = [L, E, H, x_CIE, y_CIE]` so generation preserves luminance,
exposure, highlight structure and wide-gamut color.

---

## What's here

The repo has **two cooperating systems**:

1. **Production HDR decoders** (`fast_vae.py` in the ComfyUI `radiance` node) — fast,
   distilled `latent → log image → scene-linear` decoders. Trained by
   `training/train_turbo_decoder.py`. These are what the *Radiance HDR VAE Decode* node
   loads. Trained, validated, and deployed for **7 backbones**.
2. **The RUDRA research pipeline** (`rudra/`) — the paper's contribution: a 5-channel
   spatial descriptor → 12-layer Dynamic Range Encoder (DRE) → cross-attention token
   injection, plus a FiLM-conditioned decoder and DR-gated LoRA. Trained by
   `training/train_rudra.py` (Stages 1–3).

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
training/         Trainers + orchestration:
                    train_turbo_decoder.py   production decoder (node)
                    train_rudra.py           Stages 1–3 (decoder / LoRA / DRE)
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
