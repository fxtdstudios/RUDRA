---
license: other
license_name: rudra-weights-noncommercial
license_link: https://github.com/fxtdstudios/RUDRA/blob/main/checkpoints/LICENSE
library_name: comfyui
pipeline_tag: image-to-image
tags:
  - hdr
  - diffusion
  - comfyui
  - vae-decoder
  - openexr
  - radiance
  - rudra
---

# RUDRA — HDR Decoders for Diffusion Models

**Radiometric Dynamic-Range Conditioning for HDR-Aware Diffusion Models**
FXTD Studios / Radiance Research

Distilled decoders that turn diffusion **latents into scene-linear HDR / OpenEXR** instead
of tone-mapped SDR. They replace the standard VAE decode inside the *Radiance HDR VAE
Decode* ComfyUI node, preserving highlights, exposure, and wide-gamut color.

➡️ **Code, training, and docs:** https://github.com/fxtdstudios/RUDRA

## Files

Each file is a trained `RadianceTurboDecoder` / `RadianceFullDecoder` for one backbone:

```
rudra_{turbo|full}_decoder_{backbone}_ema.safetensors
```

| Backbone | Recommended file | Quality (PSNR_log) |
|---|---|---|
| Flux.1 | `rudra_full_decoder_flux_ema.safetensors` | 29.77 |
| Wan | `rudra_full_decoder_wan_ema.safetensors` | 32.45 |
| SDXL | `rudra_turbo_decoder_sdxl_ema.safetensors` | 33.86 |
| Qwen-Image | `rudra_turbo_decoder_qwen_ema.safetensors` | 26.67 |
| Flux.2 Klein | `rudra_turbo_decoder_flux2-klein_ema.safetensors` | 28.57 |
| LTX (2.3) | `rudra_full_decoder_ltx-video_ema.safetensors` | 25.47 |
| Z-Image | use the Flux decoder (shares the FLUX.1 VAE) | — |

`turbo` (~0.5 M params) is fast and strong on simple latents (SDXL); `full` (~5.6 M) wins
on Flux/Wan/LTX. Both are provided where trained.

## Usage (ComfyUI)

1. Download into `ComfyUI/models/radiance/`:
   ```bash
   huggingface-cli download fxtdstudios/RUDRA --include "rudra_*.safetensors" \
       --local-dir "ComfyUI/models/radiance"
   ```
2. In the **Radiance HDR VAE Decode** node: set `rudra_decoder = Enabled`, pick
   `decoder_size` (`rudra_turbo` or `rudra_full`) per the table above, and set
   `target_space` to your output color space (Linear / ACEScg / Rec.2020 / LogC4…).

## Notes

- **Backbone-specific:** a decoder is tied to its VAE latent space — use the matching file
  for the model feeding the node (Flux decoder for a Flux workflow, etc.).
- **Flux.2 Klein** uses a 128-channel / 16× VAE, so its decoder has an extra upsample stage
  (requires the updated `fast_vae.py` from the GitHub repo).
- Quality is reported as held-out **log-space PSNR**; perceptual HDR evaluation uses
  **ColorVideoVDP** (JOD). SDXL/Qwen/Klein were trained on a smaller pair set and can be
  improved with more data.

## Citation

> RUDRA: Radiometric Dynamic-Range Conditioning for HDR-Aware Diffusion Models.
> FXTD Studios / Radiance Research.

## Licence

**Non-commercial.** The weights are licensed for research, teaching, evaluation,
benchmarking and personal use only
([`checkpoints/LICENSE`](https://github.com/fxtdstudios/RUDRA/blob/main/checkpoints/LICENSE));
the code is licensed under the
[PolyForm Noncommercial License 1.0.0](https://github.com/fxtdstudios/RUDRA/blob/main/LICENSE).
Commercial licensing: [FXTD Studios](https://fxtdstudios.com).
