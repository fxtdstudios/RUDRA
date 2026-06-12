# RUDRA / Radiance HDR Decoder — Cheatsheet

The node ("Radiance HDR VAE Decode") loads **`RadianceTurboDecoder` / `RadianceFullDecoder`**
(from `fast_vae.py`, trained by `train_turbo_decoder.py`). The `rudra/` package decoders
(`train_rudra.py`) are a *separate research architecture* and are NOT loadable in the node.

## Which decoder to use, per backbone

| Backbone | Use `decoder_size` | PSNR_log | Deployed file (`ComfyUI/models/radiance/`) |
|---|---|---|---|
| **Flux** | **rudra_full** | 29.77 | `rudra_full_decoder_flux_ema.safetensors` |
| **Wan** | **rudra_full** | 32.45 | `rudra_full_decoder_wan_ema.safetensors` |
| **LTX** | **rudra_full** | 25.47 | `rudra_full_decoder_ltx-video_ema.safetensors` |
| **SDXL** | **rudra_turbo** | 33.86 | `rudra_turbo_decoder_sdxl_ema.safetensors` |
| **Z-Image** | same as Flux | — | copy of the flux decoder (Z-Image uses the FLUX.1 VAE) |
| **Qwen** | **rudra_turbo** | 26.67 | `rudra_turbo_decoder_qwen_ema.safetensors` |
| **Flux.2 Klein** | **rudra_turbo** | 28.57 | `rudra_turbo_decoder_flux2-klein_ema.safetensors` (128ch, 16x — needs the dynamic-decoder fast_vae.py) |

- **Flux/Wan/LTX → full** (it genuinely beats turbo on these).
- **SDXL → turbo** (turbo beats every SDXL full; SDXL full is slow + data-starved — ignore it).

## Node settings

- `rudra_decoder` = **Enabled**
- `decoder_size` = per table above
- `source_space` = the curve the decoder was trained on (**ARRI LogC4** for flux/wan; **Sony S-Log3** for ltx)
- `hdr_mode` = **Compress (Log)** (forced on when the decoder is enabled)
- `target_space` = your output space (Linear, ACEScg, Rec.2020, etc. — ACEScg gamut fix applied)
- `decode_noise_scale` = 0

## What was learned (so future training is right)

- **Always use the `*_ema_best.safetensors`** checkpoint, never the final step — these decoders
  peak early (10k–18k) then oscillate; best-EMA captures the peak.
- **Full decoder LR = 3e-4** (not 1e-4). Lower LR gave a *smoother but lower* peak; best-EMA
  already handles the oscillation. Keep warmup + EMA 0.9995.
- **`--knee 0.6`**, not the default 0.96 — the data's log codes top out ~0.79, so a 0.96 knee
  makes the highlight loss never fire.
- **Full decoder is ~16× slower on 64×64-latent backbones** (Flux/Wan/SDXL → decodes at full
  512×512). It's only cheap on LTX (16×16 latent → 128×128). Turbo is fast everywhere.
- **Full ≈ 5.6M params** (not the ~32M the docstring claims). Widening to 256ch is *not* worth
  it — gains are data/latent-limited, not capacity-limited.
- **SDXL had only 963 pairs vs ~13k** for the others → data-starved. Regenerate multi-crop pairs
  if you ever want SDXL full to be competitive (but turbo already wins, so usually skip).

## Retrain recipe (if you add a backbone)

```
$env:PYTHONPATH = "D:\A.I\ComfyUI\custom_nodes;D:\A.I\ComfyUI\custom_nodes\radiance"
python training/build_all_decoders.py --only <name> --sizes turbo,full
```
Then it auto-deploys to `models/radiance/rudra_{turbo,full}_decoder_<type>_ema.safetensors`.
Fill the VAE URL in the script's REGISTRY first for brand-new backbones.
