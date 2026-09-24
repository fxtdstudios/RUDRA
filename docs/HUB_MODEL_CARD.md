---
license: other
license_name: rudra-weights-noncommercial
license_link: https://github.com/fxtdstudios/RUDRA/blob/main/checkpoints/LICENSE
library_name: comfyui
pipeline_tag: image-to-image
tags:
  - hdr
  - inverse-tone-mapping
  - sdr-to-hdr
  - vae-decoder
  - diffusion
  - comfyui
  - openexr
  - radiance
  - rudra
---

# RUDRA

**Radiometric Dynamic-Range Conditioning for HDR-Aware Diffusion Models**
[FXTD Studios](https://fxtdstudios.com) · [github.com/fxtdstudios/RUDRA](https://github.com/fxtdstudios/RUDRA)

Two independent families live in this repository.

| family | what it does | files |
|---|---|---|
| **`sdr2hdr/`** | takes an ordinary 8-bit frame and reconstructs scene-linear HDR | 6 |
| **decoders (root)** | decode diffusion latents straight to scene-linear HDR, skipping the tone-mapped VAE | 14 |

They share a name and a radiometric convention and nothing else. Pick the one
that matches your problem.

---

## `sdr2hdr/` — single-image inverse tone mapping

A compact U-Net that predicts a **residual on an analytic inverse-ACES tone
map**, not the image itself. Six channels in (SDR RGB plus the analytic
baseline), five out (a log-domain residual and two masks). The curve does the
mapping; the network supplies only what a curve cannot know.

**Start with `sdr2hdr_shadow_v1.safetensors`.**

| file | params | what it is |
|---|---:|---|
| **`sdr2hdr_shadow_v1`** | 1,217,318 | **the shipped model.** v5 backbone plus a trained shadow gate, seed 20260901 |
| `sdr2hdr_shadow_s2` | 1,217,318 | same recipe, seed 2 |
| `sdr2hdr_shadow_s3` | 1,217,318 | same recipe, seed 3 |
| `sdr2hdr_image_v5` | 1,196,197 | the backbone alone, no gate |
| `sdr2hdr_image_v6` | 4,770,117 | 4× capacity ablation, 64 base channels |
| `sdr2hdr_temporal_v1` | 37,011 | temporal refiner. **Not an `SDR2HDRNet`** and will not load as one |

Input is 8-bit normalised sRGB RGB. Output is scene-linear RGB normalised to
10,000 nits, with diffuse white at 203 (ITU-R BT.2408).

### Loading

Each file carries its own architecture config in its safetensors metadata, so
nothing else is needed to build the right network:

```python
import json, torch
from safetensors import safe_open
from safetensors.torch import load_file
from rudra.sdr2hdr import SDR2HDRNet          # pip install -e . from the GitHub repo

path = "sdr2hdr/sdr2hdr_shadow_v1.safetensors"
with safe_open(path, framework="pt") as f:
    config = json.loads(f.metadata()["config"])
model = SDR2HDRNet.from_config(config)
model.load_state_dict(load_file(path), strict=True)
model.eval()
```

`SHA256SUMS` and `index.json` sit beside the weights.

### What it scores

429 held-out frames at native 1280×720, PU21-PSNR and ColorVideoVDP JOD,
against the analytic inverse-ACES baseline the model sits on top of. `hard` is
the deployment condition (unknown tone curve, 4:2:0 chroma, banding, JPEG);
`clean` is well-graded input the baseline already handles.

| model | clean dB | clean JOD | hard dB | hard JOD |
|---|---:|---:|---:|---:|
| shadow_v1 | +0.07 | +0.113 | +1.24 | +0.389 |
| shadow_s2 | +0.71 | +0.068 | +0.96 | +0.308 |
| shadow_s3 | +0.45 | +0.089 | +1.18 | +0.358 |
| **mean ± sd** | **+0.41 ± 0.33** | **+0.090 ± 0.023** | **+1.12 ± 0.15** | **+0.352 ± 0.041** |
| image_v5 | −3.00 | −0.046 | +1.43 | +0.443 |
| image_v6 | −2.75 | +0.004 | +0.96 | +0.343 |

The three gate seeds are the only configurations that come out positive on all
four measures.

### Limitations, stated plainly

- **No published method has been scored on our split.** Every number above is
  against our own analytic baseline. Read them as internal progress, not as a
  comparison, until that gap is closed.
- **The backbone alone regresses on clean input.** `image_v5` loses 3.0 dB of
  PU21-PSNR on well-graded frames, and the failure concentrates in
  low-dynamic-range scenes: the 60 worst average −10.76 dB at a median
  ground-truth peak of 238 nits, the 60 best +3.41 dB at 19,590 nits. The gate
  is what fixes it.
- **The metrics disagree, and it matters.** That same 3.0 dB loss is −0.046 JOD
  in CVVDP, far below a just-noticeable difference. Do not quote the PU21 row
  without the JOD beside it.
- **The achievable gain is bounded by the input, not the model.** A linear
  readout of the features explains 3% of the variance in the per-frame scale an
  oracle would use, and 79% of that variance sits within a condition rather than
  between. Capacity (4×), corpus (6×) and objective were each varied and none
  moved it.
- **Three seeds is a spread, not a distribution.**
- **The temporal refiner is unevaluated.** Its held-out set is 4 validation and
  5 test clips of one scene each, too small to report.

Training data is documented in the GitHub README: Poly Haven HDRIs (CC0),
HdM-HDR-2014 and HdM-HFR-2017 (academic licence), Netflix Chimera (CC BY 4.0),
plus proprietary FXTD footage that is not redistributed.

---

## Decoders — diffusion latents to scene-linear HDR

Diffusion models are trained on tone-mapped images, so their VAEs decode to a
world where nothing is brighter than white. These replace that decoder and
return scene-linear radiance instead.

Two sizes per backbone: `full` reconstructs more faithfully, `turbo` is roughly
a tenth the size and fast enough to sit in a live graph.

```bash
huggingface-cli download fxtdstudios/RUDRA \
    --include "rudra_*_decoder_*.safetensors" \
    --local-dir "ComfyUI/models/radiance"
```

Then enable `rudra_decoder` in the **Radiance HDR VAE Decode** node and pick a
`decoder_size`.

| file | size |
|---|---:|
| `rudra_full_decoder_flux_ema.safetensors` | 21.5 MB |
| `rudra_full_decoder_ltx-video_ema.safetensors` | 22.0 MB |
| `rudra_full_decoder_sdxl_ema.safetensors` | 21.4 MB |
| `rudra_full_decoder_wan_ema.safetensors` | 21.5 MB |
| `rudra_full_decoder_zimage_ema.safetensors` | 21.5 MB |
| `rudra_turbo_decoder_flux_ema.safetensors` | 2.2 MB |
| `rudra_turbo_decoder_flux2_ema.safetensors` | 2.0 MB |
| `rudra_turbo_decoder_flux2-klein_ema.safetensors` | 3.0 MB |
| `rudra_turbo_decoder_ltx_ema.safetensors` | 2.4 MB |
| `rudra_turbo_decoder_ltx-video_ema.safetensors` | 3.5 MB |
| `rudra_turbo_decoder_qwen_ema.safetensors` | 2.2 MB |
| `rudra_turbo_decoder_sdxl_ema.safetensors` | 2.1 MB |
| `rudra_turbo_decoder_wan_ema.safetensors` | 2.2 MB |
| `rudra_turbo_decoder_zimage_ema.safetensors` | 2.2 MB |

Measured reconstruction quality, for the size we recommend per backbone. The
table covers the recommended decoder per backbone, not every file above:

| backbone | VAE latent | recommended | PSNR_log |
|---|---|---|---:|
| Flux.1 | 16ch / 8× | full | 29.77 |
| Wan | 16ch / 8× | full | 32.45 |
| LTX | 128ch / 8× | full | 25.47 |
| SDXL | 4ch / 8× | turbo | 33.86 |
| Qwen-Image | 16ch / 8× | turbo | 26.67 |
| Flux.2 Klein | 128ch / **16×** | turbo | 28.57 |

---

## Licence and citation

**Non-commercial.** The weights here are licensed for research, teaching,
evaluation, benchmarking and personal use only
([`checkpoints/LICENSE`](https://github.com/fxtdstudios/RUDRA/blob/main/checkpoints/LICENSE)),
because one of their training sources (HdM-HDR-2014 / HdM-HFR-2017) is free
for academic use only. The code that runs them is licensed under the
[PolyForm Noncommercial License 1.0.0](https://github.com/fxtdstudios/RUDRA/blob/main/LICENSE).
Commercial licensing: [FXTD Studios](https://fxtdstudios.com).

If the SDR→HDR work is useful in yours:

```bibtex
@misc{rudra2026,
  title  = {What an 8-Bit Frame Can and Cannot Say About the Scene Behind It:
            bounds for inverse tone mapping, and a gate that reaches one of them},
  author = {{FXTD Studios}},
  year   = {2026},
  note   = {https://github.com/fxtdstudios/RUDRA}
}
```
