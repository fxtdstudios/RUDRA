"""Settle the training-vs-inference latent convention empirically (2026-07-03 review).

THE QUESTION
------------
Pair generation (dataset_hdr.py) stores latents exactly as the VAE wrapper
returned them — the diffusers backend returns ``latent_dist.sample()`` with NO
scale/shift, and train_turbo_decoder.py feeds those latents to the decoder
unchanged. But the inference path (fast_vae.decode_to_linear_realtime) divides
the incoming latent by the per-model ``scale_factor`` (flux 0.3611, sdxl
0.13025, default 0.18215) before the decoder.

If the latent handed to the node at inference lives in the same space as the
stored training pairs, that division is a systematic ~2.8–7.7× amplitude
mismatch between training and inference. The symptom would be a global
gain/contrast error in the decoded HDR — which nodes/hdr/uplift_universal.py
currently *masks* with a median-luma gain-match (0.1–10×) against the naive
base reconstruction.

WHAT THIS SCRIPT DOES
---------------------
Loads N stored .npz pairs (latent + log_coded target) and the deployed decoder
for the family, then decodes the SAME latent under three conventions:

    A. raw            : model(latent)                      (training convention)
    B. div-scale      : model(latent / scale)              (current inference)
    C. mul-scale      : model(latent * scale)              (in case pairs were pre-divided)

and reports PSNR of each decode against the stored log_coded target. The
correct convention is the one that reproduces (approximately) the training
PSNR from STATUS.md; the wrong ones will be dramatically worse.

RUN (Windows, training venv):
    python training/verify_latent_convention.py --pair_dir G:\\data\\flux_pairs ^
        --model_type flux --model_size full ^
        --ckpt D:\\A.I\\ComfyUI\\models\\radiance\\rudra_full_decoder_flux_ema.safetensors

ACTION ON RESULT
----------------
* If A wins  -> remove the ``x = x / _scale`` division in
  fast_vae.decode_to_linear_realtime (or pass scale_factor=1.0 everywhere),
  and delete the compensating gain-match in uplift_universal once re-tested.
* If B wins  -> training pairs were scaled after all; document it in
  DECODER_CHEATSHEET.md and pin it with a metadata key on the next retrain.
* Either way -> Radiance Studio integration must adopt the winning convention.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for p in (_REPO,):
    if p not in sys.path:
        sys.path.insert(0, p)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair_dir", required=True, help="Directory of training .npz pairs")
    ap.add_argument("--ckpt", required=True, help="Deployed decoder .safetensors")
    ap.add_argument("--model_type", default="flux")
    ap.add_argument("--model_size", default="full", choices=["full", "turbo"])
    ap.add_argument("--n", type=int, default=8, help="Number of pairs to test")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    import safetensors.torch

    # Architecture comes from the ComfyUI node's fast_vae.py (the deployed
    # loader). Prefer it on PYTHONPATH; fail with a clear message otherwise.
    try:
        from fast_vae import RadianceFullDecoder, RadianceTurboDecoder
    except ImportError:
        sys.exit("fast_vae.py not importable — add the ComfyUI 'radiance' node dir "
                 "to PYTHONPATH (see DECODER_CHEATSHEET.md retrain recipe).")

    try:
        from config.model_map import MODEL_VAE_CONFIG
        cfg = MODEL_VAE_CONFIG.get(args.model_type, {})
    except ImportError:
        cfg = {}
    scale = float(cfg.get("scale_factor", 0.18215))
    import math
    n_up = max(1, int(round(math.log2(float(cfg.get("vae_spatial_factor", 8))))))

    state = safetensors.torch.load_file(args.ckpt)
    latent_ch = state["layers.0.weight"].shape[1]
    cls = RadianceFullDecoder if args.model_size == "full" else RadianceTurboDecoder
    model = cls(latent_channels=latent_ch, output_channels=3, n_upsample=n_up)
    model.load_state_dict(state, strict=True)
    model.eval().to(args.device)

    pairs = sorted(glob.glob(os.path.join(args.pair_dir, "*.npz")))[: args.n]
    if not pairs:
        sys.exit(f"no .npz pairs found in {args.pair_dir}")

    scores = {"A raw (training conv.)": [], "B latent/scale (current inference)": [],
              "C latent*scale": []}
    with torch.no_grad():
        for p in pairs:
            d = np.load(p)
            z = torch.from_numpy(d["latent"].astype(np.float32)).unsqueeze(0).to(args.device)
            if z.ndim == 5:  # (1, C, T, H, W) stored by video VAEs
                z = z[:, :, 0]
            target = d["log_coded"].astype(np.float32)          # (H, W, 3)
            for name, zz in (("A raw (training conv.)", z),
                             ("B latent/scale (current inference)", z / scale),
                             ("C latent*scale", z * scale)):
                out = model(zz)[0].permute(1, 2, 0).float().cpu().numpy()
                h = min(out.shape[0], target.shape[0]); w = min(out.shape[1], target.shape[1])
                scores[name].append(psnr(np.clip(out[:h, :w], 0, 1), np.clip(target[:h, :w], 0, 1)))

    print(f"\nmodel={args.model_type}/{args.model_size}  scale={scale}  pairs={len(pairs)}")
    print(f"{'convention':40s}  mean PSNR (log-coded)")
    best = max(scores, key=lambda k: np.mean(scores[k]))
    for k, v in scores.items():
        mark = "  <-- WINNER" if k == best else ""
        print(f"{k:40s}  {np.mean(v):6.2f} dB{mark}")
    print("\nCompare the winner against STATUS.md training PSNR_log "
          "(flux full 29.77 / sdxl turbo 33.86 / wan full 32.45).")
    if best.startswith("A"):
        print("=> inference's '/ scale' division is a train/inference mismatch: "
              "fix decode_to_linear_realtime to feed raw latents.")


if __name__ == "__main__":
    main()
