# RUDRA Retrain Runbook (post-fix)

All prior checkpoints (turbo + full decoders for flux1/flux2/wan21/wan22/ltx, and the
Stage 2 LoRA) were trained on buggy code — their logs show the HDR objectives were inert
(`highlight=0`, `exposure` stuck at ~18 EV) and the LoRA loss was numerically broken
(`base_mse` ~1e8). This runbook rebuilds them on the fixed code.

**Golden rule: do not launch a multi-day run until the Step 0 sanity gate passes.** It
takes ~2 minutes and tells you the §3.1/§3.2 fixes actually took.

---

## Step 0 — Pre-flight (once)

```bash
# From the repo root (D:\A.I\Devlopments\rudra)
pip install -r requirements-metrics.txt        # cvvdp (real HDR-VDP) + lpips
pytest tests/ -q                                # curves, freeze, conditioning, hdrvdp
python -c "import rudra; print('import OK', rudra.__version__, 'cvvdp:', rudra.colorvideovdp_available())"
```

Everything in `tests/` must be green. If `cvvdp` won't install, validation still runs but
falls back to the labeled proxy (`hdr_vdp_backend="proxy"`) — fine for now, but install it
before generating numbers for the paper.

**Data check — will the HDR objective even fire?** RUDRA's whole advantage is highlight
preservation; if the training crops have no near-clipping highlights, the highlight loss
stays 0 (which is why `hl=0.0000` shows up) and there is no HDR signal to learn. Scan first:

```bash
python training/scan_highlights.py hdrdata/hdr_pairs                 # flux/wan (LogC4)
python training/scan_highlights.py hdrdata/ltx_pairs --curve slog3   # ltx (S-Log3)
```

If it reports ❌/⚠️ (few images with highlights), curate higher-dynamic-range sources
(skies, sun, neon, specular, fire, windows) before spending GPU time — otherwise RUDRA
cannot demonstrate an HDR edge over a baseline.

---

## Step 1 — The sanity gate (≈300 steps, ~2 min)

Run a tiny decoder job and read the first log lines. This is the whole point of the fix.

```bash
python training/train_rudra.py --stage decoder \
  --pair_dir hdrdata/hdr_pairs \
  --output_dir hdrdata/checkpoints/_sanity \
  --model_type flux --model_size turbo \
  --steps 300 --batch_size 1 --multi_curve --color_space rec2020
```

Open `hdrdata/checkpoints/_sanity/flux_rudra_stage1/rudra_decoder_log.jsonl` and check the
first few lines against the OLD broken run:

| field | OLD (broken) | NEW (fixed) — what you want |
|---|---|---|
| `highlight` | `0.0` every step | **> 0**, and trending down |
| `exposure` | 13–19 (never moves) | **< ~2** and decreasing |
| `chromaticity` | frozen ~0.10 | moving (down) |
| `loss` | L1 only | finite, smooth |

If `highlight` is still 0 or `exposure` is still ~18, **stop** — something didn't pick up
the fix (wrong branch/paths). Delete `_sanity` once it passes.

---

## Step 2 — Stage 1 decoders (turbo, then full)

Re-train per backbone. Turbo first (cheap, 50k); full is the 100k high-capacity head.

```bash
# Turbo (per backbone) — reuse existing pairs; targets are now decoded correctly
python training/train_rudra.py --stage decoder --model_type flux  --model_size turbo \
  --pair_dir hdrdata/hdr_pairs      --output_dir hdrdata/checkpoints/flux1_turbo  \
  --steps 50000 --batch_size 8 --lr 3e-4 --multi_curve --color_space rec2020 \
  --val_split 0.05 --patience 8

python training/train_rudra.py --stage decoder --model_type wan   --model_size turbo \
  --pair_dir hdrdata/wan_hdr_pairs  --output_dir hdrdata/checkpoints/wan21_turbo \
  --steps 50000 --batch_size 8 --lr 3e-4 --multi_curve --color_space rec2020 --val_split 0.05

python training/train_rudra.py --stage decoder --model_type ltx-video --model_size turbo \
  --pair_dir hdrdata/ltx_pairs      --output_dir hdrdata/checkpoints/ltx_turbo \
  --steps 50000 --batch_size 4 --lr 3e-4 --multi_curve --color_space rec2020 --val_split 0.05

# Full (per backbone), 100k, lower LR
python training/train_rudra.py --stage decoder --model_type flux --model_size full \
  --pair_dir hdrdata/hdr_pairs --output_dir hdrdata/checkpoints/flux1_full \
  --steps 100000 --batch_size 4 --lr 1e-4 --multi_curve --color_space rec2020 --val_split 0.05
# …repeat full for wan / wan22 as before.
```

Notes:
- **Pick one `--color_space` and keep it everywhere** (rec2020 is the safe, fully-tested
  default; pass `--config configs/rudra_decoder_only.yaml` if you want the ACEScg recipe —
  it now flows through descriptor, losses, and ΔE2000 consistently).
- `--val_split 0.05` re-enables eval; it now streams (no OOM) and logs ColorVideoVDP JOD as
  `hdr_vdp3` + `hdr_vdp_backend`. Watch `ev_error` actually fall this time.
- The decoder pairs are reusable as-is (targets are LogC4, now inverted correctly).

---

## Step 3 — Stage 2 LoRA (DR-gated)

The old LoRA was broken by the Flux scale/shift bug. Use `--pair_dir` (NOT a latent-only
`--cache_dir`, which the fix now rejects, because it can't carry DR conditioning).

```bash
python training/train_rudra.py --stage lora --model_type flux \
  --model_path <path-to>/flux1-dev-fp8.safetensors \
  --pair_dir hdrdata/hdr_pairs \
  --output_dir hdrdata/checkpoints/flux1_lora \
  --steps 10000 --batch_size 2 --lr 1e-4 --text_dropout 0.1
```

Sanity: `base_mse` in `rudra_stage2_log.jsonl` should now start in a **sane range
(roughly 0.1–10), not 1e8**. If it's still huge, the latents feeding the backbone are
mis-scaled — verify the VAE encode used the corrected Flux scale 0.3611 / shift 0.1159.

---

## Step 4 — Stage 3 DRE (only had a dryrun before)

```bash
python training/train_rudra.py --stage dre --model_type flux \
  --model_path <path-to>/flux1-dev-fp8.safetensors \
  --pair_dir hdrdata/hdr_pairs \
  --output_dir hdrdata/checkpoints/flux1_dre \
  --steps 10000 --batch_size 2 --lr 1e-4 --text_dropout 0.1
```

The cross-attention freeze fix means the backbone now stays frozen; the logged trainable
count should be small (DRE transformer + projection + λ), not the whole U-Net.

---

## Step 5 — Validate & report

With `cvvdp` installed, `validation_metrics` reports a real `hdr_vdp3` (JOD) tagged
`hdr_vdp_backend="colorvideovdp"`. For the paper, report these as **"ColorVideoVDP (JOD,
0–10)"** — not "HDR-VDP-3" — and drop the old ~80 numbers. Only cite rows where the backend
is `colorvideovdp`.

---

## Step 6 — Benchmark vs LTX IC-LoRA-HDR (the real "is it better?" test)

Lightricks ships an official `LTX-2.3-22b-IC-LoRA-HDR`, so "better" can be measured, not
argued. Produce HDR predictions from each method on the SAME held-out inputs, then score
both against ground truth identically:

```bash
# 1) generate predictions (your pipelines): rudra_preds/ and iclora_preds/, GT in gt/
# 2) score them apples-to-apples
python training/benchmark_hdr.py --gt gt --a rudra_preds --b iclora_preds \
    --name-a RUDRA --name-b IC-LoRA-HDR --color-space rec2020 --out bench.csv
```

It reports per-method means and a per-metric winner for ColorVideoVDP (JOD), ΔE2000,
EV-error, highlight reconstruction, and tone-mapped PSNR/SSIM, and writes a per-clip CSV.
Expect RUDRA's edge (if any) to be on the HDR-specific metrics (HRA, EV-error, HDR-VDP in
highlights); a small from-scratch decoder may trail on detail/LPIPS. The harness does not
run inference — it scores whatever EXR/HDR predictions you drop in the folders.

## Data regeneration (only if needed)

You generally do **not** need to regenerate the decoder pairs. Regenerate only if:
- you switch the working color space (e.g. to ACEScg), or
- you want the Stage 2/3 latents re-encoded with the corrected VAE scale/shift.

```python
from training.dataset_hdr import HDRPairDataset
HDRPairDataset.generate_pairs(
    exr_dirs=["hdrdata/Source_HDR/PolyHaven", "hdrdata/Source_HDR/Alexa", ...],
    output_dir="hdrdata/hdr_pairs_v2",
    vae=<your ComfyUI VAE>,           # must apply the corrected per-model scale/shift
    log_curve="ARRI LogC4",
)
```

---

## Suggested order

1. Step 0 pre-flight → Step 1 sanity gate (flux turbo).
2. flux1 turbo full run; eyeball the eval `ev_error` falling before committing the rest.
3. Roll out remaining turbo + full decoders.
4. Stage 2 LoRA → confirm `base_mse` sane → Stage 3 DRE.
5. Validate with ColorVideoVDP; regenerate the results table.
