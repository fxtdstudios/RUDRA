# RUDRA

**Radiometric Dynamic-Range Conditioning for HDR-Aware Diffusion Models**
FXTD Studios / Radiance Research

Diffusion models are trained on tone-mapped images, so they learn a world where
nothing is brighter than white. RUDRA gives them the rest of the range back: it
decodes latents straight to scene-linear HDR, and conditions the backbone on a
radiometric descriptor so luminance, exposure and wide-gamut colour survive
generation. A separate path needs no diffusion model at all — hand it an
ordinary 8-bit frame and it reconstructs the highlights the tone map threw away.

📦 Weights: [huggingface.co/fxtdstudios/RUDRA](https://huggingface.co/fxtdstudios/RUDRA/tree/main)
📄 Paper: [research/RUDRA_V01.pdf](research/RUDRA_V01.pdf)

---

## RUDRA Studio

![RUDRA Studio](docs/rudra_studio.png)

Drop a frame in. The wipe compares the reconstruction against the analytic
inverse tone map, so you see what the network added rather than what the tone
map already gave you. Waveform and histogram are measured from the prediction.
**Master EXR** writes a full-resolution ACES 2065-1 container with AP0
primaries, ST 2065-4 chromaticities and a JSON sidecar of delivery metadata.

```bash
python ui/server.py --checkpoint work/checkpoints/image/best.pt --preload
```

---

## Setup

```bash
git clone https://github.com/fxtdstudios/RUDRA.git && cd RUDRA
pip install -e .                             # the torch-free `rudra` CLI
pip install -r requirements-metrics.txt      # cvvdp + lpips, for the real metrics
pytest tests/
```

CUDA is only needed for training and inference. The delivery layer is pure
numpy, so render and mastering machines need nothing but Python.

### ComfyUI decoders

Download the decoder for your backbone into `ComfyUI/models/radiance/`, then
enable `rudra_decoder` in the *Radiance HDR VAE Decode* node and pick a
`decoder_size`.

```bash
huggingface-cli download fxtdstudios/RUDRA --include "rudra_*_decoder_*.safetensors" \
    --local-dir "ComfyUI/models/radiance"
```

| Backbone | VAE latent | Decoder | PSNR_log |
|---|---|---|---|
| Flux.1 | 16ch / 8× | full | 29.77 |
| Wan | 16ch / 8× | full | 32.45 |
| LTX | 128ch / 8× | full | 25.47 |
| SDXL | 4ch / 8× | turbo | 33.86 |
| Z-Image | = Flux VAE | use the Flux decoder | — |
| Qwen-Image | 16ch / 8× | turbo | 26.67 |
| Flux.2 Klein | 128ch / **16×** | turbo | 28.57 |

---

## Use it

**Reconstruct a frame or a sequence.**

```bash
python training/infer_sdr2hdr.py input/ --output-dir out/ \
    --checkpoint work/checkpoints/image/best.pt
```

**Master and deliver.** No GPU required.

```bash
rudra metadata out/ --nits-scale 203 --output out/master --peak-nits 1000  # DoVi L1 + HDR10+
rudra aces out/ --output delivery/aces --ocio                              # ACES EXR + OCIO
rudra grade in.exr --output graded/ --region 400:2000:1.0:0.5              # +1 EV, 400–2000 nits
rudra bench bench_root/ --output results.json                              # PU21-PSNR / CVVDP
```

**Train the SDR→HDR model.** Four stages; each refuses to hand work to the next
if it can't vouch for it.

```bash
python pipeline/scan_sources.py /path/to/source_hdr --out work/inventory.jsonl
python pipeline/prepare_pairs.py --inventory work/inventory.jsonl --dst work/pairs \
    --mode log2_extended --crops 3 --video-stride 2
python pipeline/build_manifests.py --pairs-dir work/pairs --out-dir work
python pipeline/verify_dataset.py --pairs-dir work/pairs \
    --manifest work/sdr_hdr_manifest.jsonl --video-manifest work/video_manifest_9f.jsonl
python training/train_sdr2hdr.py --mode image --manifest work/sdr_hdr_manifest.jsonl \
    --output-dir work/checkpoints/image --steps 100000 --best-metric composite_gain --device cuda
```

Three things that will make a training log make sense:

- Every eval scores two conditions. `clean_*` is the held-out frame as prepared;
  `hard_*` is the same frame under a seeded camera and codec degradation. Only
  the hard numbers describe deployment, so only they choose `best.pt`.
- Both report `gain_db` against the analytic inverse tone map the network sits
  on top of, so "better than doing nothing" is a number, not a guess.
- Graded sources are treated as censored. A pixel at exactly 4,000 nits in a
  4,000-nit grade means *at least* 4,000, so the loss goes one-sided there —
  otherwise the model learns to cap.

---

## Layout

```
rudra/            Descriptor, DRE transformer, cross-attention, FiLM decoder,
                  DR-gated LoRA, losses, CVVDP metric
rudra/delivery/   Torch-free: DoVi L1 / HDR10+, ACES / EXR / OCIO, grade
                  controls, benchmarks, the `rudra` CLI
pipeline/         Corpus construction and its gates: scanner, pair preparation,
                  HDR storage (hdr_io), manifests, verify_dataset
training/         Trainers, evaluation, inference, benchmarks
ui/               RUDRA Studio: the page and its inference server
config/, configs/ VAE registry and training recipes
tests/            Curve round-trips, corpus guards, delivery, target decode
```

Every script has `--help` and a docstring saying what it does and why.
`pipeline/verify_dataset.py` lists the nine corpus invariants;
`pipeline/hdr_io.py` is the single source of truth for how targets are stored.

## Notes

Weights, pairs and HDR sources are deliberately not committed. Generate them
with the scripts above.

The production decoders need no text conditioning. Stages 2 and 3 follow the
paper and target SDXL's U-Net cross-attention; porting them to Flux is the §10
generalization experiment.
