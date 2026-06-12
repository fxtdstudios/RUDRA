# RUDRA — Technical Review

**Reviewer perspective:** AI researcher / ML engineer
**Scope:** Full repository (`rudra/` core package, `training/` scripts, `config/`, `configs/`, research paper `RUDRA_V01.pdf`)
**Codebase size:** ~9,800 lines of Python across 35 files
**Date:** June 2026

---

## 1. Verdict

RUDRA is a well-conceived and unusually clean research codebase. The central idea — treating radiometric dynamic range as a first-class conditioning signal for a frozen diffusion backbone, analogous to ControlNet/T2I-Adapter but targeting luminance/exposure/highlight/chromaticity instead of spatial layout — is genuinely novel and well-motivated for VFX/OpenEXR workflows. The module decomposition mirrors the paper faithfully, adapter-engineering hygiene (zero/near-zero init, closed-gate init, EMA, staged losses, scene-aware splitting) is above the norm for research code, and several numerical components (ΔE2000, SSIM) are implemented carefully.

However, **the project is not yet in a trustworthy training/evaluation state.** There are several correctness bugs that silently corrupt the supervision signal and the "frozen backbone" guarantee, the headline paper metrics do not correspond to any metric the code actually computes, and the reproducibility package the paper itself promises is absent. The good news: most issues are localized and fixable without architectural change.

I'd classify this as a strong **v0.3 research prototype** that needs a correctness-hardening pass before any reported number should be believed.

---

## 2. Architecture & research soundness

The implementation matches the paper's two-path design:

- **RUDRA-Lite** (`descriptor.py` → `encoder.py` → `decoder.py` + `adapter.py`): a 26-dim global descriptor → MLP projection → FiLM-conditioned HDR decoder, plus a dynamic-range–gated LoRA on the backbone.
- **RUDRA-Full** (`spatial_descriptor.py` → `dre_transformer.py` → `cross_attention.py`): the paper's 5-channel per-pixel descriptor `R(x) = [L, E, H, x_CIE, y_CIE]` → 12-layer transformer DRE → cross-attention token injection `C_total = concat(C_text, λ·C_R)`.

This is faithful and sensible. The descriptor math is thoughtful — log-normalized luminance, EV-from-median exposure, smooth sigmoid highlight masks, true CIE xy chromaticity through proper RGB→XYZ matrices. The adapter design choices are correct: near-zero final-layer init so the adapter starts as identity, zero-init LoRA up-projection, gate bias `-2.0` so DR gating starts mostly closed. These are the right instincts.

**Conceptual caveat worth surfacing in the paper:** in Stage 1 the decoder is conditioned on the descriptor of the *target* image it is reconstructing (`pipeline(latent, target_bchw, ...)` in `train_rudra.py:313`). At inference the descriptor would come from a *source* of different dynamic range. So Stage 1 as written is closer to a DR-conditioned autoencoder decode than to the inference-time task. This may be intentional (decoder pre-training), but the train/inference conditioning distribution gap should be acknowledged and ideally bridged with source-vs-target descriptor augmentation.

---

## 3. Critical correctness bugs

These change results and should be fixed before any further training runs.

**3.1 Log-curve round-trip is inconsistent (silently corrupts the supervision domain).**
Targets are encoded to log with the *true* vendor curves (`dataset_hdr.linear_to_log_coded` → `color_utils.tensor_linear_to_logc4`, etc.). But the inverse used everywhere downstream — `normalization._generic_log_decode` — is a crude `(x−0.5)·stops` exponential approximation, **not** the true inverse of LogC4/S-Log3/etc. So `generic_decode(true_logc4_encode(linear)) ≠ linear`. The "scene-linear ground truth" fed to every auxiliary loss, and the dynamic-range descriptor computed for any log-format input, are therefore physically wrong. This affects the descriptor (`descriptor.py`, `spatial_descriptor.py`), the losses (`losses.py`), and the training linearization (`train_rudra.py:320-321`). *Fix: route all log decodes through the same accurate vendor inverses used for encoding; the generic decode should only be a last-resort fallback for unknown formats.*

**3.2 Decoder output domain ≠ loss domain.**
`RUDRADecoder` emits `softplus`-activated `scene_linear_positive` values (can exceed 1.0). Training then passes that prediction through `normalize_to_scene_linear(pred, format_id=logc4)` (`train_rudra.py:320`), which treats it as a `[0,1]` LogC4 *code* (clamping >1 to 1) and exponentiates it. The prediction and the target therefore enter the loss in different domains, and HDR highlights in the prediction are clamped away before the loss sees them — directly undermining the highlight-preservation objective. *Fix: define one canonical comparison space (e.g. scene-linear) and ensure the decoder output and the target are mapped into it consistently; do not re-linearize an already-linear prediction.*

**3.3 Cross-attention injection unfreezes the entire backbone.**
In `cross_attention.inject_rudra_cross_attention` (lines ~225-229):
```python
for p in injector.parameters():
    if p is not injector.original_attn.parameters():   # Parameter `is not` generator → always True
        p.requires_grad = True
```
`injector.original_attn.parameters()` returns a fresh generator, so `p is not <generator>` is **always True**. Every parameter — including the frozen backbone cross-attention weights wrapped inside the injector — gets `requires_grad=True`. For SD/SDXL (which have `attn2` modules) this trains the base model, violating the paper's "frozen backbone" claim and making the reported trainable-parameter count meaningless. *Fix: collect injector-owned parameters explicitly (the `lambda_param` and any new projections), and freeze `original_attn` by id-set membership.*

**3.4 DR conditioning collapses to zero in the cached training path.**
In Stages 2 and 3, when a `cache_dir` of pre-encoded latents is used, there is no paired image, so the code sets `target_bchw = torch.zeros(...)` (`train_rudra.py:610`, `:830`). The descriptor / spatial-descriptor is then computed on a **black frame**, producing a constant, content-free `dr_proj` / `c_r`, and the gate-regularization target `bright_ratio` is identically 0. The dynamic-range signal — the entire point of RUDRA — vanishes precisely on the fast training path most likely to be used at scale. *Fix: cache the descriptor (or the source image stats) alongside the latent so conditioning survives offline caching.*

**3.5 Resume loses training state.**
The `.safetensors` resume branch hardcodes `start_step = 0` (`train_rudra.py:236`) and does not restore optimizer/scheduler, so a resumed cosine schedule restarts and step-based logic (early stopping, LR decay) is wrong. The `.pth` branch is correct.

---

## 4. ML soundness concerns

**4.1 Reported metrics do not match implemented metrics.** The paper's results table reports `HDR-VDP-3 = 80.4` (and baselines 65–72). The code does not implement HDR-VDP-3 — it has a hand-rolled `hdr_vdp_proxy` that returns a JOD score in `[0, 10]`. No code path produces values near 80. The headline numbers therefore correspond to no computation in this repo and should be treated as aspirational placeholders, not measured results. Either integrate the real HDR-VDP-3 (it is freely available for research) or relabel the table and rescale to the proxy's JOD range. This is the single most important credibility issue for an arXiv release.

**4.2 Highlight loss is numerically unstable.** `highlight_preservation_loss` is an unbounded L1 in scene-linear RGB (`|pred − target|`, values up to `1e4`) weighted by the highlight mask. Gradients are dominated by the few brightest pixels, exactly where the softplus/decode interactions above are worst. The paper specifies scene-linear *luminance*; the code uses RGB. Prefer a log-domain or relative ( `|p−t| / (t+ε)` ) highlight error to bound the dynamic range of the gradient.

**4.3 Adapter trains only on unconditional (null) text.** Stages 2/3 call `get_null_embed(...)` for every step, so the LoRA/DRE adapter never sees real captions during HDR adaptation. If the production model is text-conditioned, the adapter may interact poorly with real prompts at inference. At minimum, mix real and null embeddings.

**4.4 Backbone-specific scaling is wrong for the stated primary target (Flux).** `RUDRA_MODEL_CONFIGS["flux"]` uses `scale_factor = 0.18215` — the SD1.5 VAE constant. Flux's VAE uses a different scale **and** shift (≈0.3611 / 0.1159). Using SD1.5 scaling on Flux latents corrupts the latent statistics the decoder learns from. Verify per-model VAE scale/shift against each backbone.

**4.5 DRE positional-embedding interpolation is 1-D.** `_get_pos_embed` interpolates the flattened token axis linearly, which scrambles 2-D spatial neighborhoods when input resolution exceeds `max_grid_size`. Interpolate on the `(H/8, W/8)` grid in 2-D instead.

**4.6 LoRA target set is very broad.** `_DEFAULT_TARGET_MODULES` includes `fc1/fc2/proj/dense/linear1/linear2/up_proj/down_proj`, so LoRA is injected into MLP and miscellaneous linears, not just attention. With only an `in_features < 32` guard this may wrap layers where DR-gating is meaningless and inflate trainable params. Consider scoping to attention projections first and ablating MLP injection.

---

## 5. Code quality & engineering

Strengths: clear module boundaries, good docstrings, dataclass config, EMA + grad-clip + cosine schedule + early stopping, scene-aware splitting that genuinely addresses the paper's leakage requirement, and careful float64 ΔE2000 / Gaussian-window SSIM.

Issues:

- **Two disconnected config systems.** `configs/*.yaml` (ACEScg working space, `loss_schedule`, `sampler`, grad-accum) are **not read by `train_rudra.py`**, which is pure argparse. The YAMLs are effectively dead documentation, and they disagree with code defaults (YAML `color_space: ACEScg` vs `RUDRAConfig` default `rec2020`). The paper explicitly requires fixing *one* working color space — the repo currently ships three different answers (code default, YAML, and per-model `log_curve`). Wire one config path and pick one space.
- **Broad `try/except` that swallows failures.** Eval (`train_rudra.py:449`), backbone forward (multiple nested fallbacks), and LPIPS init all catch-and-continue. This hides exactly the kind of shape/dtype bug that §3 describes. Prefer narrow exceptions and fail loud during development.
- **Eval OOM risk.** The eval loop concatenates the *entire* validation set into one batch (`torch.cat(eval_l)`), which won't scale to the paper's 3,420-pair val set.
- **No reproducibility scaffolding.** No `requirements.txt`/`pyproject.toml`, no `README`, no `LICENSE`, no git history, no tests, no global seeding (only the split is seeded). Format/noise randomness is unseeded.
- Minor: deprecated `merge_andunload` alias retained; `gc`/`math` imported unused in places.

---

## 6. Reproducibility vs. the paper's own checklist

The paper's §9 checklist is a good standard to hold the repo to. Current status:

- Preprocessing scripts for SDR/PQ/HLG/OpenEXR — *present but inconsistent* (§3.1).
- Train/eval working color space stated — *contradicted across code/YAML* (§5).
- Scene/shot split files — *mechanism present (`split_by_scene`), no published split files*.
- DRE config (depth/dim/heads/proj) — *present in `config.py`*. ✅
- Optimizer/LR/schedule/batch/steps/precision/GPU count — *partially in argparse + JSON dump; precision (bf16) only in unused YAML*.
- Baseline conversion rules for SDR→HDR comparison — *not in code*.
- Reproducible eval notebook / qualitative grids — *absent*.

So the headline claims are not currently reproducible from this repository.

---

## 7. Prioritized recommendations

**P0 — correctness (do before any new training run):**
1. Unify log encode/decode on accurate vendor curves (§3.1).
2. Fix decoder-output vs loss-domain mismatch; compare in one space (§3.2).
3. Fix the always-true freeze check in cross-attention injection (§3.3).
4. Cache descriptor/source stats with latents so Stage 2/3 conditioning isn't zeroed (§3.4).
5. Fix `.safetensors` resume step/optimizer restoration (§3.5).

**P1 — credibility & stability:**
6. Integrate real HDR-VDP-3 or relabel/rescale the results table to the proxy's JOD range (§4.1).
7. Make the highlight loss bounded (log or relative error) (§4.2).
8. Verify per-backbone VAE scale/shift, especially Flux (§4.4).
9. Mix real + null text embeddings in adapter training (§4.3).

**P2 — engineering & reproducibility:**
10. Wire `configs/*.yaml` into the trainer (or delete them); pick and enforce one working color space.
11. Add `requirements.txt`/`pyproject.toml`, `README`, `LICENSE`, global seeding, and a small `tests/` suite (descriptor shape/range, round-trip identity, freeze-count assertions, loss finiteness).
12. Replace silent `try/except` with narrow handling; stream eval batches.
13. Publish split files and a minimal evaluation notebook to satisfy the paper's §9.

---

## 8. Quick wins (low effort, high value)

- A 10-line unit test asserting `trainable_params == injector-only count` would have caught §3.3 immediately.
- A round-trip test `assert allclose(decode(encode(x)), x, atol)` per format would have caught §3.1.
- A guard `assert target_bchw.abs().sum() > 0` in Stage 2/3 would have caught §3.4.

These three tests cover the three most damaging bugs in the codebase.

---

## 9. Changelog — P0 fixes applied (June 2026)

The five P0 correctness fixes have been implemented. Only Stage 1 (decoder) had been
trained (~5,200 steps, Flux + Wan); Stages 2–3 never ran, so no checkpoint depended on
the freeze or cached-conditioning bugs. The prior Stage 1 run is not worth resuming —
its log shows `highlight = 0.0` every step and `exposure = 15–19 EV`, i.e. the HDR
objectives never engaged — so retrain Stage 1 from scratch on the fixed code.

- **§3.1 — self-contained accurate curves.** New `rudra/color_curves.py` defines exact
  forward+inverse pairs for LogC3/LogC4/S-Log3/V-Log/Log3G10/DaVinci (round-trip error
  ~1e-13, verified). `normalization.py` now decodes log formats with these exact inverses
  instead of `_generic_log_decode`; `dataset_hdr.linear_to_log_coded` encodes through the
  same module, so encode and decode are guaranteed inverses (no longer dependent on an
  external `color_utils` that wasn't shipped in this repo).
- **§3.2 — decoder/loss domain unified.** `train_rudra.py` Stage 1 now decodes the target
  to scene-linear once with its true curve and compares the (already scene-linear) decoder
  output directly — no more re-running the prediction through `normalize_to_scene_linear`,
  which had clamped every highlight. `multi_curve` augmentation now re-encodes the linear
  target into the simulated source format via `encode_scene_linear_to_format` so the
  descriptor sees a consistently-coded input while the reconstruction target stays linear.
- **§3.3 — freeze fixed.** `inject_rudra_cross_attention` now skips backbone params by an
  explicit `id()` set, so the wrapped attention stays frozen and the trainable count is
  accurate.
- **§3.4 — no silent zero conditioning.** Stage 2/3 cached-latent paths now raise a clear
  error (pointing to `--pair_dir`) instead of substituting a black frame; Stage 2 also
  supports a cached `dr_raw` descriptor if present.
- **§3.5 + eval — resume & visibility.** `.safetensors` resume recovers the step from the
  filename; eval streams a bounded number of val batches (no full-set `cat` OOM) and now
  records failures to the run log instead of failing silently.
- **Tests.** `tests/test_color_curves.py` (round-trip identity, monotonicity, dispatch)
  and `tests/test_rudra_fixes.py` (freeze count, scene-linear HDR output, non-zero
  highlight loss, format round-trip) pin all of the above. Run with `pytest tests/`.

**Validation note:** curve round-trips were verified numerically (max err ~1e-13) and all
new/edited files were syntax-checked. The `pytest` suite requires `torch`, so run it in
your training environment before kicking off the retrain — confirm `highlight` and
`exposure` read sane non-zero/small values within the first ~200 steps.

**Not yet addressed after P0 (now mostly closed by P1 below):** real HDR-VDP-3 vs the JOD
proxy (§4.1) remains open.

---

## 10. Changelog — P1 fixes applied (June 2026)

- **§4.2 — HDR losses bounded.** `losses.py`: `highlight_preservation_loss` now measures
  the H-weighted error in `log1p` radiance instead of raw scene-linear RGB (where a few
  ~1e4 highlight pixels dominated the gradient). `rudra_reconstruction_loss` gained a
  `recon_domain` argument defaulting to `"log"`, so the base reconstruction is also in
  log-radiance; pass `"linear"` to restore the old behavior.
- **§4.4 — VAE scale/shift corrected.** Added a `shift_factor` field and fixed the
  registries in `config.py` and `config/model_map.py`: Flux `scale 0.3611 / shift 0.1159`
  (was the SD1.5 0.18215 with no shift), SD3 `shift 0.0609`, SDXL `0.13025`, plus
  Hunyuan/Cog approximations. Matters most for Stage 2/3, which feed latents back into the
  real backbone. `verify_unified_config.py` expectations updated to match.
- **§4.3 — text conditioning plumbing.** New `_select_text_embed` uses a real caption
  embedding when the dataset provides one (`text_embed` key), with CFG-style dropout to
  null (`--text_dropout`, default 0.1), falling back to null-only otherwise. Removes the
  hard-coded null-only assumption in Stages 2/3 (full benefit requires a caption-bearing
  dataset, which is a data task).
- **§5 — YAML wired + single working color space.** `train_rudra.py` now reads
  `configs/*.yaml` via `--config` (mapped onto CLI defaults with `parser.set_defaults`, so
  explicit flags still win), so the previously-dead configs drive runs. The working color
  space is now a single knob (`--color_space`, or `RUDRAConfig.color_space`) that flows to
  the descriptor, the auxiliary losses (`highlight`/`chromaticity` now build their
  descriptor in that space), and the metrics — `delta_e_2000` / `validation_metrics` take a
  `color_space` argument instead of hardcoding Rec.2020, with matrices for
  rec709/rec2020/acescg. The trainer logs the resolved space once at startup.

**Still open (data / validation tasks):** a caption-bearing dataset to make §4.3 effective,
per-model VAE scale/shift validation against each real checkpoint, and rerunning the paper's
results + ablation/λ-sweep tables on the retrained models with ColorVideoVDP.

**Validation note for P1:** all P1 logic files were edited via the authoritative file
tools; `losses.py`, `metrics.py`, `normalization.py`, and `train_rudra.py` were
byte-verified by compilation. A workspace mount-cache glitch blocked in-sandbox
compilation of `config.py` / `model_map.py` / `cross_attention.py` / `dataset_hdr.py`, but
their full contents were verified correct by direct read. Run `pytest tests/` and a quick
`python -c "import rudra"` in your training environment before the retrain.

---

## 11. Changelog — real HDR-VDP metric (§4.1, June 2026)

The fake `hdr_vdp_proxy` (a 0–10 number matching no published metric, and not the scale of
the paper's reported "HDR-VDP-3" column) is replaced by a real perceptual metric:

- **New `rudra/hdrvdp.py`.** `hdr_vdp3_jod(pred, target, color_space, diffuse_white_nits)`
  returns `(jod, backend)`. Reference HDR-VDP-3 proper is MATLAB-only, so the backend is
  **ColorVideoVDP** (`pycvvdp`) — the pip-installable PyTorch successor from the same lab
  (Mantiuk et al., SIGGRAPH 2024), which handles calibrated HDR and reports JOD units
  (10 = identical). The function maps scene-linear → absolute cd/m² (scene-linear 1.0 →
  `diffuse_white_nits`, default 200) and converts the working space to Rec.2020 linear for
  cvvdp's `standard_hdr_linear` display model.
- **Honest fallback.** If `cvvdp` isn't installed (or its API version differs), it falls
  back to the internal proxy, now explicitly documented as **"NOT HDR-VDP-3"**, and reports
  `hdr_vdp_backend="proxy"` so logged numbers are never mistaken for the real metric.
- **Metrics wiring.** `validation_metrics` now returns `hdr_vdp3` and `hdr_vdp_backend`
  (plus the legacy `hdr_vdp_proxy` key for back-compat). The training eval log records both,
  so every reported HDR quality number is tagged with the backend that produced it.
- **Install.** `pip install -r requirements-metrics.txt` (adds `cvvdp` and `lpips`).
  Tests in `tests/test_hdrvdp.py` cover the proxy-fallback path, backend reporting, and the
  color-space conversion branches.

**Reporting guidance for the paper:** only cite the `hdr_vdp3` numbers produced with
`hdr_vdp_backend == "colorvideovdp"`, and report them as **ColorVideoVDP (JOD)** — not
"HDR-VDP-3" — unless you separately run the MATLAB HDR-VDP-3 reference. The previously
tabulated ~80 values should be dropped; ColorVideoVDP JODs live on a ~0–10 scale.

---

## 12. Changelog — P2 code polish (June 2026)

- **§4.5 — 2-D DRE positional embedding.** `dre_transformer.py` now treats the learned
  `pos_embed` as a `(grid_side × grid_side)` 2-D grid: within bounds it slices the top-left
  `(H/8 × W/8)` sub-grid (exact, preserves learned positions), and beyond `max_grid_size` it
  interpolates the grid to the target shape with **bicubic** in 2-D. The old code did 1-D
  interpolation over the flattened sequence, mixing spatially distant positions. `_get_pos_embed`
  now takes `(h, w)` and the forward pass passes the real token-grid dimensions.
- **§4.6 — LoRA target scoping.** `adapter.py` splits targets into
  `_ATTENTION_TARGET_MODULES` (the new default) and `_MLP_TARGET_MODULES` (opt-in via
  `inject_rudra_lora(..., include_mlp=True)`). The old default also wrapped MLP and generic
  `proj/dense/linear*` names, injecting LoRA almost everywhere. Stage 2 exposes this as
  `--lora_mlp` so attention-only vs +MLP is an explicit ablation, not a silent default.
- **Tests.** `tests/test_rudra_fixes.py` adds a DRE forward-pass check within and beyond the
  grid (shape + finiteness) and an injection-scope check (attention-only by default, MLP only
  with `include_mlp=True`). `dre_transformer.py`, `adapter.py`, and the tests compile clean.
