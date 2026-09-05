# RUDRA — Training & Research Status

> **Updated 5 Sep 2026.** The snapshot below the line dates from 22 Aug and is
> still accurate for what it covers. Read this section first: the repository
> holds **four separate lines of work** that share a name, and "is RUDRA
> finished?" has a different answer for each.
>
> | line | what it is | state |
> |---|---|---|
> | **A. Production decoders** | distilled log-space VAE decoders, 7 backbones, ComfyUI node | **complete** — measured, deployed |
> | **B. Research pipeline (Stages 1-3)** | descriptor + FiLM + DR-gated LoRA + DRE cross-attention, the *original paper's core thesis* | **incomplete** — Stage 3 never trained |
> | **C. Direct SDR-to-HDR image model** | `rudra/sdr2hdr.py`, v5, RUDRA Studio, the delivery path | **measured and written up** |
> | **D. Temporal (v02)** | rendered camera-move corpus, clip metric, the oracle gate | **gate passed, nothing trained yet** |
>
> **The paper ([`paper/main.pdf`](paper/main.pdf)) is about line C.** It is not
> the manuscript `PAPER_ERRATA.md` refers to, which is line B. Line B's
> completion path is unchanged and is listed below; nothing since 22 Aug has
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
> **Line C, done since:** ExpandNet run from the authors' released weights on
> the same 429 frames (−18.56 dB, −2.03 JOD, 1 frame won on PU21 and 16 on
> CVVDP); the shadow gate scored on three seeds (+0.41 ± 0.33 dB clean,
> +1.12 ± 0.15 dB hard); `step_0072000.pt` scored, confirming the selection fix
> picks it and that it is better on the criterion the selector optimises and
> worse on CVVDP in both conditions; the LaTeX build, the arXiv package and the
> committed PDF. Related work cited (9 references + 4 standards); no `[CITE]`
> markers remain.
>
> **Line C remaining:** nothing measurable. The blockers are the arXiv
> endorsement (a person has to say yes), the HuggingFace upload, and the
> weights licence below. Section 10 still declares two gaps honestly: three
> published methods have runnable code and have not been run, and §6's
> trimmed-PSNR table and §7's oracle sweep have no script because they need
> per-pixel statistics the benchmark does not write.
>
> ---
>
> **Line D — v02, the temporal track (opened 4 Sep 2026).** The v01 video corpus
> is 935 clips across **13 scenes**, which is why the temporal refiner is
> reported as unevaluated. `pipeline/render_hdri_moves.py` flies a virtual
> camera through the CC0 Poly Haven panoramas, one panorama being one scene:
> the corpus is now **993 scenes, 17,874 frames, 128 GB**.
>
> Three things were measured before any model was trained, in the shape of §7:
>
> - `TemporalHDRRefiner`'s receptive field is **±4 frames, ±4 pixels**, measured
>   by gradient, against **4.3–27.3 px/frame** of camera motion. It cannot fetch
>   a value from where the content was; it can only smooth. Retraining it is the
>   control, not a candidate.
> - Scoring per frame is **blind to flicker by construction**. Two clips with an
>   identical 0.0600 relative error on every frame score 10.000 and 10.000
>   per-frame, and 10.000 and 5.111 as clips. `hdr_vdp3_clip_jod` runs
>   ColorVideoVDP in video mode; `rudra/pose_warp.py` adds a temporal
>   consistency measure with no flow estimator in it, ground truth against
>   itself sitting at 0.0145 stops.
> - **The gate.** `training/gate_temporal_oracle.py`, 2 clips: on clean frames
>   the per-frame model already scores 9.972 JOD of 10, so nothing can be won
>   and the run says nothing. Under degradation, per-frame is −1.962 JOD and an
>   **aligned mean of the warped neighbours — no ground truth, the simplest
>   thing a model could learn — reaches 7.346 JOD**, matching the per-pixel
>   oracle. Achievable **+9.31 JOD** against a +0.5 threshold.
>
> Two caveats travel with that: the −1.962 floor is far below the 7.805 the
> benchmark reports for its hard condition, and the degradation is seeded per
> frame, which is the best possible case for temporal averaging. **Line D
> remaining:** calibrate the floor, repeat the aligned-mean measurement with
> *estimated* optical flow rather than the analytic camera poses (real footage
> has no poses — this decides the architecture and must precede any training),
> then train rung 1 as the control and rung 2 as flow-warp plus multi-scale
> fusion, testing on the 13 real scenes the v01 corpus provides.
>
> ---
>
> **Licensing (5 Sep 2026).** The code is Apache 2.0. The **weights are not**:
> HdM-HDR-2014 and HdM-HFR-2017 are **75.6% of the training corpus** and are
> free for academic use only, with commercial use requiring a separate
> agreement with HdM Stuttgart. `checkpoints/LICENSE` now licenses the weights
> non-commercially and `NOTICE` carries the Netflix Chimera CC BY 4.0
> attribution. Academic use is permitted, so **the paper is unaffected**.
>
> The intended resolution is two model families: `rudra-research` as it stands,
> and a `rudra-studio` retrained without HdM — Netflix Chimera (14.3%, CC BY,
> median peak 7,094 nits) plus Poly Haven (10.1%, CC0) plus the 17,874 rendered
> frames plus FXTD's own footage, which is comparable corpus volume with the
> high-nit end covered. That retrain depends on nobody's permission.

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
