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
> | **C. Direct SDR-to-HDR image model** | `rudra/sdr2hdr.py`, v5, RUDRA Studio (opens a folder of frames or a video by path), the delivery path | **measured and written up** |
> | **D. Temporal (v02)** | rendered camera-move corpus, clip metric, the oracle gate | **CLOSED.** Exact poses +0.60 JOD, RAFT +0.34, DIS −0.07, against a +0.5 threshold fixed in advance. Nothing a plate can supply clears it; no temporal model trained, and that is the result |
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
> - **The gate.** `training/gate_temporal_oracle.py`. On clean frames the
>   per-frame model already scores 9.972 JOD of 10, so nothing can be won and
>   the run says nothing; the gate has to be read on `--condition hard`.
>
> **The gate failed, and then said why (5 Sep 2026).** The 4 Sep reading of
> **+9.31 JOD achievable** does not survive. Two faults produced it, and both
> are fixed:
>
> - It scored a **bare `model(x)` forward pass**, not RUDRA as deployed. The
>   benchmark runs `predict_image(preserve_outside=True)`, where outside the
>   learned masks the output *is* the analytic baseline. That is why the floor
>   sat at −1.96 JOD against the benchmark's 7.805 on the same checkpoint and
>   the same 1280×720 frames — it was never a crop-size or resolution
>   mismatch.
> - `degrade_like_eval` **reseeds from the frame index**, so every frame of a
>   nine-frame clip got its own exposure, tone curve, white balance,
>   saturation, chroma subsampling, bit depth and JPEG quality. Averaging nine
>   independent draws of a corruption is worth √9 whether or not the frames
>   carry information. `--degradation {per-frame,per-clip,realistic,codec}`
>   makes that assumption a flag; `codec` puts the clip through a real H.264
>   round trip and is the one with a GOP in it.
>
> | degradation | clips | per-frame | achievable | ceiling |
> |---|---|---|---|---|
> | `per-frame` (4 Sep) | 2 | −2.396 JOD | **+9.397** | +9.408 |
> | `realistic` | 40 | 6.961 | **+0.033** | +0.191 |
> | `codec` | 40 | 6.864 | **+0.034** | +0.165 |
>
> Against a +0.5 JOD threshold, **the gate fails** — and even the unreachable
> bound (omniscient per-pixel selection over exactly aligned neighbours)
> reaches only +0.19.
>
> **That is a fact about the corpus, not about video.** `make_sdr` tone-maps
> every frame with one fixed curve and one fixed EV offset, and the renderer's
> camera only rotates through a static panorama — so a scene point carries the
> **same SDR code in every frame it appears in**, and what is blown in frame 3
> is blown in frame 7. The corpus contains none of the information v02's claim
> is about. `--exposure-drift` and `--exposure-jitter` put that axis back
> (defaults 0.0; the `_ingest_config` sentinel refuses to mix a drifted render
> into the existing 993 scenes).
>
> **The gate passes once exposure moves (6 Sep 2026).** 50 panoramas rendered
> twice with `PYTHONHASHSEED=0` — identical camera paths (checked to 1e-9),
> identical H.264 degradation, the SDR exposure the only difference. Scene
> list in `_gate50_scenes.txt`.
>
> | exposure | per-frame | achievable | ceiling |
> |---|---|---|---|
> | constant | 6.209 JOD | +0.110 | +0.264 JOD / +0.632 dB |
> | ±0.48 stops | 5.786 | **+0.511** | +0.851 JOD / **+4.554 dB** |
>
> **+0.511 against a +0.5 threshold, with the fixed-exposure control at +0.110
> on the very same scenes and camera paths.** The PU21 ceiling is the clearer
> signal: +0.63 dB without exposure movement, +4.55 dB with it.
>
> **Repeated on three seeds (6 Sep 2026, `training/run_drift_gate.ps1`, on the
> 4080):**
>
> | seed | flat | drift | drift ceiling | clips |
> |---|---|---|---|---|
> | 20260903 | +0.110 | +0.511 | +0.851 | 50 |
> | 20260906 | +0.035 | **+0.603** | +1.090 | 40 |
> | 20260907 | +0.046 | **+0.661** | +1.142 | 40 |
> | **mean** | **+0.064** | **+0.592** | +1.028 | |
>
> **The result holds.** Every drift arm clears the threshold; every flat arm
> is an order of magnitude below it. The ~+0.53 JOD separation between arms is
> far larger than the spread across seeds, which is what makes this a result
> rather than a run. The hypothesis behaves exactly as stated — a neighbour is
> worth something when it saw the scene at a different exposure and close to
> nothing when it did not.
>
> Two things travel with it. The 40 in the later rows is a defect:
> `stage_oracle_clips` defaulted `--count` to 40 and the sweep passed none, so
> those seeds used the first 40 of the 50 scenes — the *same* 40 in both arms,
> so each pairing is intact, and the default is now 0. And drift makes the
> per-frame job harder (6.209 → 5.786 on seed 20260903), so part of the gain
> is repairing damage the drift itself did; the oracle ceiling separates the
> two, and +0.26 JOD flat against +0.85 drift is information that exists to be
> fetched, not merely damage to undo.
>
> What it licenses is narrow: a proposition about footage whose exposure
> breathes — handheld, documentary, anything riding auto-exposure — not about
> video in general.
>
> **And then estimated alignment took it away (6 Sep 2026).** Every number
> above used the camera angles the renderer wrote down. A plate has none, so
> the same 50 drifted clips were re-scored with the correspondence estimated
> by DIS optical flow from the degraded SDR — what a deployed model holds:
>
> | arm | per-frame | control | aligned | oracle | **achievable** | ceiling |
> |---|---|---|---|---|---|---|
> | pose (exact) | 5.786 | 5.835 | 6.296 | 6.687 | **+0.511** | +0.851 |
> | flow (DIS) | 5.786 | 5.844 | 5.802 | 5.839 | **+0.016** | −0.005 |
> | flow + texture gate | 5.786 | 5.828 | 5.658 | 5.514 | **−0.128** | −0.314 |
>
> **Read the ceiling column.** Under estimated alignment the *oracle* — ground
> truth consulted per pixel, better than any architecture or weighting could
> ever do — beats the per-frame model by +0.053 JOD, while the control, which
> carries no information at all, beats it by +0.058. The oracle's whole margin
> is selection on resampling noise. **There is no headroom left to build for.**
>
> The alignment is not visibly bad, which is what makes this worth stating
> carefully: DIS matched the analytic poses to a median 0.04 px at one frame
> of separation and 0.09 px at eight, coverage within 1%, warped frames
> agreeing to under half an 8-bit code value. PU21-PSNR keeps 75% of the gain
> (+0.326 of +0.435 dB). The JOD keeps 3%. What survives a per-frame PSNR and
> not a video metric is a small, spatially coherent error that changes every
> frame — the exact artefact a temporal model exists to remove.
>
> Where it fails is legible: the six worst clips are open sky, the four best
> interiors. On the sky scene flow error in low-texture regions is **1.35 px
> against 0.15 px** in textured regions of the same frame, and
> forward-backward consistency passed **57%** of those bad pixels, because in
> a flat region any displacement round-trips perfectly. Flow fails exactly
> where the highlights are. Gating on texture energy made it worse (−0.128):
> a hard mask leaves nine-frame averaging beside one-frame averaging with a
> seam between them. A softer weighting is not worth trying — the oracle row
> bounds every weighting there is.
>
> **The learned estimator, and the close (10 Sep 2026).** DIS is a fast
> classical estimator, so the flow arm was re-run with RAFT-large — markedly
> better in exactly the low-texture regions where DIS failed, 0.35 px against
> 2.61 px on an open-sky clip. Seed 20260906, 40 drifted clips, same clips
> every row:
>
> | arm | per-frame | aligned | oracle | **achievable** | ceiling |
> |---|---|---|---|---|---|
> | pose (exact) | 5.925 | 6.528 | 7.063 | **+0.603** | +1.090 |
> | flow (DIS) | 5.925 | 5.856 | 6.115 | **−0.069** | +0.122 |
> | flow (RAFT) | 5.925 | 6.266 | 6.659 | **+0.341** | +0.671 |
>
> RAFT recovers 57% of what exact poses give and **misses the threshold** —
> +0.341 against the +0.5 set before any of this was measured. Not the flat
> zero DIS gave: the ceiling moved from +0.122 to +0.671, so there was room a
> better combiner might have reached.
>
> One combiner was declared and tried, *before* it was run — weight each
> neighbour by its forward-backward residual instead of averaging equally.
> On the same clips: **mean +0.351, confidence +0.350.** Nothing.
>
> The reason is the same wall from a third angle. RAFT's forward-backward
> drift is 0.04–0.09 px on nearly every pixel that passes, so the weight is ~1
> everywhere and the signal has no dynamic range. **The failures are not
> low-confidence matches — they are confident wrong ones**, in flat regions
> where any displacement round-trips perfectly. That is precisely what
> forward-backward consistency cannot see, and therefore what weighting by it
> cannot fix.
>
> **Line D is closed.** Three alignment arms and two combiners against a
> threshold fixed in advance: exact camera poses clear it, nothing a plate can
> supply does. The +0.603 belongs to the renderer's angles. No temporal model
> was trained because there is nothing measurable for one to learn — **that is
> the result, not the absence of one**, and it is worth more than the month it
> would have taken to find out the other way.
>
> The corpus re-render, the architecture ladder and the video split are off
> the board. What would reopen it is a corpus whose neighbouring frames carry
> information these do not — real parallax, moving subjects, genuine
> multi-exposure capture — not a better estimator and not a better
> architecture. The gate would have to be re-run from scratch on it.
>
> **The question the gate did not answer** was whether the per-frame model is
> temporally *stable* enough to deliver — the gate settled only whether a
> temporal model could fetch more information (it cannot).
> `training/measure_temporal_stability.py` measures the flicker gap,
> `per_frame_jod - clip_jod`, which is what a per-frame metric is blind to.
> First two clips, 9 frames each, 10 Sep 2026:
>
> | clip | RUDRA | analytic baseline |
> |---|---|---|
> | abandoned_workshop_02_4k_c0 | **1.109** | 1.175 |
> | abandoned_workshop_4k_c0 | **1.151** | 1.188 |
>
> RUDRA is *below* the baseline on both, so **the model adds no flicker of its
> own** — the ~1.15 JOD gap is in the input, which is why the baseline shows it
> too. Indicative, not settled: two synthetic Poly Haven clips, not the 13 real
> scenes. `shadow_weight` travel differs sharply between them (std 0.028 vs
> 0.107, max frame jump 0.037 vs 0.129), and `gate_travel()` sampled no
> `residual_scale` at all — that half measured nothing.
>
> **Still open, neither a training run:** `sdr2hdr_temporal_v1.pt` ships in
> `models.json` marked *"Unevaluated"* and belongs to the closed line — pull
> it or measure it. And v6 (4× capacity, moved neither condition) is a keep
> or drop.
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
