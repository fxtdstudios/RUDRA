# RUDRA — Training & Research Status

> **Updated 1 Sep 2026.** The snapshot below the line dates from 22 Aug and is
> still accurate for what it covers. Read this section first: the repository
> holds **three separate lines of work** that share a name, and "is RUDRA
> finished?" has a different answer for each.
>
> | line | what it is | state |
> |---|---|---|
> | **A. Production decoders** | distilled log-space VAE decoders, 7 backbones, ComfyUI node | **complete** — measured, deployed |
> | **B. Research pipeline (Stages 1-3)** | descriptor + FiLM + DR-gated LoRA + DRE cross-attention, the *original paper's core thesis* | **incomplete** — Stage 3 never trained |
> | **C. Direct SDR-to-HDR image model** | `rudra/sdr2hdr.py`, v5, RUDRA Studio, the delivery path | **measured and written up** |
>
> **The paper drafted on 29 Aug (`PAPER_DRAFT_2026-08-29.md`) is about line C.**
> It is not the manuscript `PAPER_ERRATA.md` refers to, which is line B. Line B's
> completion path is unchanged and is listed below; nothing in this session
> advanced it.
>
> **Line C, as of 1 Sep 2026:** v5 (1,196,197 parameters) benchmarked on 429
> held-out frames — **+1.43 dB PU21-PSNR and +0.44 JOD on degraded input, 348 of
> 429 frames**; **-3.0 dB / -0.046 JOD on clean input**. Failure mode located
> (error correlates +0.46 with scene headroom), oracle bound measured (+5.84 dB
> clean), and that bound shown to be mostly unreachable from an 8-bit input
> (features explain 3% of the oracle scale's variance; 79% of its variance is
> within-condition). Capacity (4x), corpus (6x) and objective were each varied;
> none moved the bound. Three evaluation-harness defects found and fixed:
> selection on a noisy maximum, an eval reading the alphabetical front of the
> split, and a viewer presenting every frame upside down.
>
> **Line C remaining:** a comparison against any published method on this split
> (the largest gap), a seed sweep on the shadow gate to put an error bar on its
> +0.07 dB clean margin, scoring `step_0072000.pt` to confirm the selection fix,
> characterising the error tail on more than 26 frames, LaTeX/arXiv formatting,
> and the temporal refiner — whose held-out set is 4 validation and 5 test clips
> of one scene each, too small to report. The paper's related work is cited
> (9 references + 4 standards, 2026-09-03); no `[CITE]` markers remain.

---

Snapshot of what is trained, with measured values, and what remains for the paper.

## Production decoders (ComfyUI node) — ✅ complete

Distilled `RadianceTurbo/FullDecoder` (`fast_vae.py`), trained by
`training/train_turbo_decoder.py`. Metric: log-space PSNR on held-out pairs.

| Backbone | Decoder used | PSNR_log | peak step | pairs |
|---|---|---|---|---|
| Flux.1 | **full** | 29.77 | 26k | 12,281 |
| Wan | **full** | 32.45 | 16k | 13,227 |
| LTX (2.3, **32× fixed**) | **full** | 25.47 | 40k | 13,227 |
| SDXL | **turbo** | 33.86 | 18k | 963 |
| Qwen-Image | **turbo** | 26.67 | 20k | 963 |
| Flux.2 Klein (128ch/16×) | **turbo** | 28.57 | 18k | 963 |
| Z-Image | use Flux decoder | — | — | (Flux VAE) |

Turbo also exists for flux/wan/ltx/sdxl; the table lists the *best* per backbone.
Notes: SDXL/Qwen/Klein are on 963 pairs (data-starved) — regenerate multi-crop pairs to
lift them. Full is slow on 64×64-latent backbones (decodes at 512×512). Always deploy the
`*_decoder_ema_best.safetensors`.

## RUDRA research pipeline (the paper) — partial

`rudra/` package + `training/train_rudra.py`, Stages 1–3.

| Paper stage | Status | Measured value | Blocker / next |
|---|---|---|---|
| **Stage 1** — RUDRA-Lite descriptor + FiLM decoder (§3.6) | ✅ flux/wan/ltx @ 50k | ColorVideoVDP **JOD 9.0–9.2**, PSNR_tm ~25–26, EV-error ~0.05 stops, HRA ~0.18 | done |
| **Stage 2** — DR-gated LoRA (RUDRA-Lite conditioning) | ⚠️ partial (400 steps) | — | Flux-16GB OOM; runs on SDXL or 256px pairs |
| **Stage 3** — DRE transformer + cross-attention injection (§3.4–3.5, *core thesis*) | ❌ not trained (dryrun only) | — | run on **SDXL** (`research_sdxl.py --phase stage3`) |
| §7.1 component ablation | ❌ TBD | — | derived from Stage 1→3 runs |
| §7.2 descriptor channel ablation (L / L+E / L+H / L+xy / full R⁵) | ❌ TBD | — | `research_sdxl.py --phase ablate` |
| §7.3 λ conditioning-strength sweep (0.0 … 1.25) | ❌ TBD | — | `research_sdxl.py --phase sweep` |
| §6 results vs baselines (incl. LTX IC-LoRA-HDR) | ❌ TBD | — | `benchmark_hdr.py` + ColorVideoVDP |
| §8 qualitative grids / EV-over-time | ❌ TBD | — | after the above |

## Completion path

1. **Stage 3 on SDXL** — the paper's intended backbone (U-Net cross-attention; ~5× lighter
   than Flux, fits 16 GB). This is the core contribution. `research_sdxl.py --phase stage3`.
2. **λ sweep + descriptor ablation** — `--phase sweep` and `--phase ablate` (short runs).
3. **Benchmark** RUDRA vs LTX IC-LoRA-HDR with ColorVideoVDP (JOD), ΔE2000, EV-error, HRA.
4. **Fill the paper tables** (§6/§7) with the measured values; drop the placeholder numbers.
5. *(Optional)* Flux port of the conditioning — the paper's §10 generalization experiment.

## Code health

All P0/P1/P2 review items closed (see `RUDRA_TECHNICAL_REVIEW.md`). Real HDR metric is
ColorVideoVDP (`rudra/hdrvdp.py`); the old hand-rolled "HDR-VDP-3 ≈ 80" numbers were a
placeholder and must not be reported. `pytest tests/` covers curve round-trips, the freeze
guarantee, conditioning, and the metric backend.
