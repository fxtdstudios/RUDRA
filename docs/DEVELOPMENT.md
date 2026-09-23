# Development

Split out of the README on 16 Sep 2026, when the README became a page for the
people who use RUDRA rather than the people who work on it. Nothing here
changed; it moved.

Back to [the README](../README.md). [`STATUS.md`](../STATUS.md) says what is
finished and what is open.

---

## Layout

```
rudra/            Descriptor, DRE transformer, cross-attention, FiLM decoder,
                  DR-gated LoRA, losses, CVVDP metric, and sdr2hdr.py
rudra/delivery/   Torch-free: DoVi L1 / HDR10+, ACES / EXR / OCIO, grade
                  controls, benchmarks, the `rudra` CLI
pipeline/         Corpus construction and its gates: scanner, pair preparation,
                  HDR storage (hdr_io), manifests, verify_dataset, and the
                  HDRI camera-move renderer that makes a video split possible
training/         Trainers, evaluation, inference, the exporter that turns a
                  checkpoint into benchmark pairs, run_bench.ps1, the ExpandNet
                  runner and third-party importer, and the scripts that
                  recompute and probe the paper's numbers
ui/               RUDRA Studio: the page, its GPU compositor, the inference
                  server behind them, and sequence.py, which opens a folder
                  of frames or a video by path
run_studio.bat    One-file launchers: set up on the first run, check and start
run_studio.sh     on every run after that
paper/            LaTeX source and the built PDF; build.sh and mkarxiv.sh
docs/             RESULTS.md, TRAINING.md, TRAINING_STEPS.md (the next run,
                  step by step), RETRAIN_RUNBOOK.md, INTERNALS.md and CORPUS.md;
                  paper figures, the Studio screenshot, the comparison strips
                  and make_compare.py, which rebuilds them from a scored bench
checkpoints/      Every released SDR to HDR model as checkpoints/*.pt, plus
                  models.json, the registry the viewer reads (see its README).
                  A training run writes checkpoints/<run>/ (best.pt, step_*.pt,
                  train.jsonl); those directories are git-ignored and a run is
                  promoted by copying its best.pt up a level under a release
                  name. checkpoints/_invalid_corpus_v4/ is quarantine, not a
                  release
BUILD_CORPUS_V4.ps1  The corpus build: scan → ingest at 0 EV → manifests (with
                  the previous manifest's test scenes held out) → the gate
scripts/          AUDIT_REPO.ps1 (secrets and provenance over the full
                  history) and finalize_*.ps1 batches; scripts/archive/ holds
                  the one-shot commit/push scripts that did earlier pushes,
                  git-ignored, kept because their headers say why each push
                  happened
reports/          Dated audits and reviews (HTML and Markdown), git-ignored
config/, configs/ VAE registry and training recipes
tests/            Curve round-trips, corpus guards, delivery, target decode,
                  censored highlights, GPU/torch composite parity, canvas
                  orientation, and a smoke test that presses every control
```

---

## How it is checked

The composite lives in three languages: torch in `rudra/sdr2hdr.py`, GLSL in
`ui/compositor.js`, numpy in `tests/compose_reference.py`. The numpy one is the
reference the other two are measured against, so the picture on screen and the
EXR that Master writes cannot drift apart.

| Check | Result |
|---|---|
| Shader vs reference, 11 cases incl. shadow weight 0.0-1.0 | 8.4e-6 relative |
| Wipe: which side is which, and does the seam move | baseline 49, RUDRA 255 |
| Torch vs reference, untiled | 3e-6 relative |
| Torch vs reference, tiled (overlap bands only) | up to a few % |
| MaxCLL, page vs the EXR it writes | 0.023% apart |
| Every control on the page, pressed | 54 checks, 0 failed |

```bash
pytest tests/                                  # everything
python tests/webgl_parity/parity.py            # shader vs the reference
python tests/webgl_parity/orientation.py       # the canvas itself, not a buffer
python tests/webgl_parity/wipe.py              # the wipe, also from pixels
python ui/server.py --port 8100 --no-browser --preload --device cpu
python tests/ui_smoke/press_everything.py --url http://127.0.0.1:8100/
```

The shader checks run headless on software GL, so they need no GPU and no torch.
The page is pressed rather than read, because a menu item that runs nothing
looks exactly like one that works.

All of this runs on every push. `.github/workflows/tests.yml` has two jobs: the
suite on Python 3.10 and 3.13 against CPU torch, with a flake8 pass for syntax
errors and undefined names, and the three shader checks against a real WebGL2
context on SwiftShader. Neither needs a GPU. The whole thing is about a minute.

Tiling is the one place the two compositions genuinely disagree, because
feathering fields and feathering composed predictions are not the same operation
either side of `expm1`. The viewer and Master both ask for an untiled pass, fall
back to tiles only on OOM, and report which they used.

The screenshot at the top of this file is the problem, measured. It is the
Studio with no checkpoint loaded, so what it shows is the analytic inverse-ACES
baseline -- and on that frame the SDR clips on **8.43% of pixels** while the
baseline saturates at **1 466 nits** against its own ceiling of 1 470, which is
`7.24 x 203`, the point where the Narkowicz inverse runs out. The probe is open
on the sun: SDR `255,255,255`, baseline `+2.85 stops`, at saturation. Everything
above that line is what an 8-bit frame does not carry and what the model exists
to put back.

The scopes in it are computed from that frame's own pixels, not drawn. The
baseline-against-reconstruction wipe needs a checkpoint, so it comes from a
running Studio rather than from a static render. The capture refuses to
overwrite itself with anything under 200 KB, which is the size of an error
page:

```bash
python ui/capture_shot.py
```

The comparison strips under `docs/compare/` are generated the same way, from a
scored benchmark rather than from a screenshot. Every caption in them is read
back out of the benchmark's own result JSON, so a strip cannot claim a gain the
benchmark does not have, and the display exposure is picked from the reference
alone rather than from any method's output:

```bash
python docs/make_compare.py --bench <bench dir>
python docs/make_compare.py --bench <bench dir> \
    --list-candidates hard baseline shadow_v1 20    # how the frames were picked
```

Every script has `--help` and a docstring saying what it does and why.
`pipeline/verify_dataset.py` lists the nine corpus invariants.
`pipeline/hdr_io.py` is the single source of truth for how targets are stored.

---

---

## Notes

Six checkpoints are committed, 38 MB in total: the shipped model, its two other
seeds, the v5 backbone, the v6 capacity ablation, and the temporal refiner. The
first five are what the Results tables are made of. Training pairs and HDR
sources stay out. Generate them with the scripts above, or pull the diffusion
decoders from
[HuggingFace](https://huggingface.co/fxtdstudios/RUDRA/tree/main).

The production decoders need no text conditioning. Stages 2 and 3 follow the
original paper and target SDXL's U-Net cross-attention; porting them to Flux is
the section 10 generalization experiment. Stage 3 has never been trained.
[`STATUS.md`](STATUS.md) says which lines of work are finished and which are
not.

The temporal refiner exists and is not evaluated here. Its held-out set is 4
validation and 5 test clips of one scene each, which is too small to report.

No temporal model was trained beyond it, and that is a measurement rather than
a plan that ran out of time. `training/gate_temporal_oracle.py` asks what a
*perfectly* aligned temporal model could win before one is built. Exact
correspondence from the renderer's camera poses, omniscient per-pixel
selection among the aligned neighbours. On 40 rendered camera-move clips under
a real H.264 round trip it comes back at **+0.03 JOD achievable against a +0.5
threshold**, with the unreachable bound itself at +0.17.

The reason is in the corpus rather than in video. Those clips are pure camera
rotations through a static panorama, tone mapped with one fixed curve, so a
scene point carries the same 8-bit code in every frame it appears in and a
neighbour has nothing to add. Rendering fifty panoramas **twice**, identical
camera paths, identical compression, the SDR exposure the only difference,
separates the two claims:

| exposure | achievable (3 seeds) | ceiling |
| --- | --- | --- |
| constant | +0.110, +0.035, +0.046 JOD | +0.26 JOD / +0.63 dB |
| ±0.48 stops | **+0.511, +0.603, +0.661 JOD** | +1.03 JOD / **+4.55 dB** |

Every drifted arm clears the +0.5 threshold; every fixed-exposure arm is an
order of magnitude below it. So the information a temporal model would fetch
is *exposure variation*, present when the exposure moved between neighbours,
close to absent when it did not.

That is measured with the camera angles the renderer wrote down, and a plate
has none. Re-scoring the same clips with the correspondence **estimated** from
the pixels, which is what a deployed model actually holds, is what closed the line:

| alignment | achievable | oracle ceiling |
| --- | --- | --- |
| exact poses | +0.603 JOD | +1.090 JOD |
| RAFT optical flow | **+0.341 JOD** | +0.671 JOD |
| DIS optical flow | −0.069 JOD | +0.122 JOD |

The threshold was +0.5, fixed before any of it was measured. A learned
estimator recovers 57% of what exact poses give and still misses. One
combiner was declared and tried, weighting each neighbour by its
forward-backward residual instead of averaging equally, and it moved the number
by 0.001.

The reason is worth carrying away from this project. The flow is *accurate*:
0.04 to 0.09 px median against the analytic poses. It fails in flat regions, and
forward-backward consistency cannot detect that failure, because any
displacement round-trips perfectly through a constant area. **The bad matches
are not low-confidence, they are confident and wrong**, in exactly the blown
sky a temporal model would be asked to reconstruct.

So RUDRA is a per-frame model by measurement, not for want of trying. No
temporal model was trained because there is nothing measurable for one to
learn. [`STATUS.md`](STATUS.md) carries the full tables and what would have to
be true of a corpus to reopen the question.

The viewer looks for a model in `RUDRA_CHECKPOINT_ROOTS` first, then in this
repo's `checkpoints/`, so a clone with nothing else still starts on a real
model. Point it at your own training output to have the newest run there win:

```bash
export RUDRA_CHECKPOINT_ROOTS=/path/to/your/checkpoints     # or set on Windows
```
