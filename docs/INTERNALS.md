# RUDRA internals

Split out of the README on 10 Sep 2026, when the README became a page for the
people who use the tool. Nothing here changed; it moved.

Back to [the README](../README.md) . [STATUS.md](../STATUS.md) says what is
finished and what is open.

---

## Training data

The released model was trained on public HDR footage plus proprietary FXTD
material. None of it is committed here. `pipeline/scan_sources.py` reads
whatever you point it at.

| Source | What it gives | Grade ceiling | Licence |
|---|---|---|---|
| [Poly Haven HDRIs](https://polyhaven.com/hdris) | 963 scene-referred panoramas, real suns above 100,000 nits | none, scene-referred | [CC0](https://polyhaven.com/license) |
| [HdM-HDR-2014](https://hdm-stuttgart.de/vmlab/hdm-hdr-2014/) | 9 cinematic scenes: fireworks, forge sparks, fire, stage lights | 4,000 nits, Rec.2020 | free for academic use; commercial needs a licence from HdM |
| HdM-HFR-2017 | Bar and Fire scenes, 192 fps, same FTP host | 1,000 nits (PQ-1K) | as above |
| [Netflix Chimera](https://opencontent.netflix.com/) | 4K live action, P3-PQ | 10,000 nits | [CC BY 4.0](http://download.opencontent.netflix.com/) |

Both HdM sets come off the same plain-FTP host, `hdr-2014.hdm-stuttgart.de`
(user `HdM-HDR-2014`). `pipeline/fetch_stuttgart.py` handles the download and
resumes if it drops:

```bash
python pipeline/fetch_stuttgart.py --list-root
python pipeline/fetch_stuttgart.py --root HdM-HDR-2014_Color-Graded-for-HDR \
    --dest /path/to/source_hdr/Stuttgart_HDR_2014 --only carousel_fireworks fireplace
```

The grade ceilings in that table matter. Three of the four sources stop at a
hard delivery ceiling, which is why the loss treats those pixels as censored.

Three things that make a training log readable:

- Every eval scores two conditions. `clean_*` is the held-out frame as prepared.
  `hard_*` is the same frame under a seeded camera and codec degradation. Only
  the hard numbers describe deployment, so only they choose `best.pt`.
- Both report `gain_db` against the analytic inverse tone map the network sits
  on top of, so "better than doing nothing" is a number rather than a guess.
- Graded sources are censored. A pixel at exactly 4,000 nits in a 4,000-nit
  grade means at least 4,000, so the loss goes one-sided there with three stops
  of free headroom above the ceiling. Otherwise the model learns to cap. The
  temporal refiner did not do this until 28 August 2026, on a video corpus where
  84% of clips carry a hard ceiling. It would have undone the image model's
  highlights frame by frame.

### Video is a different problem

There are 935 clips but only 13 scenes, split 11 train / 1 val / 1 test. A
number measured on one held-out scene describes that scene, so the temporal
refiner is reported as unevaluated rather than given a figure. More clips would
not help: 926 training clips from 11 scenes is 11 scenes sliced 84 ways.

`pipeline/render_hdri_moves.py` is the way out. It flies a virtual camera
through the 963 scene-referred Poly Haven panoramas (pans, tilts, rolls and slow
zooms) and writes SDR/HDR clip pairs with exact ground truth, because both
halves come from the same radiance. Frames are named for
`build_video_manifest.py`, so the output drops into the existing pipeline:

```bash
python pipeline/render_hdri_moves.py --hdri-dir <polyhaven> --dst work/pairs_moves
python training/build_video_manifest.py --hdr-dir work/pairs_moves/hdr \
    --sdr-dir work/pairs_moves/sdr --output work/video_manifest_moves.jsonl \
    --clip-length 9 --frame-step 1
```

Two limits worth knowing. A panorama has no parallax, so nothing occludes
anything as the camera turns and nothing in the scene moves. That covers a large
share of real plates and none of the hardest ones. And the 2k panoramas are
2048x1024, so a 75 degree field of view samples 427 source pixels across a
1280-wide frame. `--max-upscale` refuses that by default instead of quietly
training on softened sources. Fetch the 4k or 8k versions, same licence, for
anything you intend to report.

One distribution shift is worth knowing before reading any clean number. The
band where the model fails, below 400 nits, is 45.2% of the test split and 27.2%
of what the sampler actually draws. Weighted median peak is 1,713 nits in train
against 546 in test. We did not design that and a reader should weigh it.

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
docs/             Paper figures, the Studio screenshot, the comparison strips
                  and make_compare.py, which rebuilds them from a scored bench
checkpoints/      Every SDR to HDR model, plus models.json, the registry the
                  viewer reads (see its README)
config/, configs/ VAE registry and training recipes
tests/            Curve round-trips, corpus guards, delivery, target decode,
                  censored highlights, GPU/torch composite parity, canvas
                  orientation, and a smoke test that presses every control
```

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

The screenshot at the top of this file is generated against a running Studio and
refuses to overwrite itself with anything under 200 KB, which is the size of an
error page:

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
[`STATUS.md`](../STATUS.md) says which lines of work are finished and which are
not.

The temporal refiner exists and is not evaluated here. Its held-out set is 4
validation and 5 test clips of one scene each, which is too small to report.

No temporal model was trained beyond it, and that is a measurement rather than
a plan that ran out of time. `training/gate_temporal_oracle.py` asks what a
*perfectly* aligned temporal model could win before one is built: exact
correspondence from the renderer's camera poses, omniscient per-pixel
selection among the aligned neighbours. On 40 rendered camera-move clips under
a real H.264 round trip it comes back at **+0.03 JOD achievable against a +0.5
threshold**, with the unreachable bound itself at +0.17.

The reason is in the corpus rather than in video. Those clips are pure camera
rotations through a static panorama, tone mapped with one fixed curve, so a
scene point carries the same 8-bit code in every frame it appears in and a
neighbour has nothing to add. Rendering fifty panoramas **twice**, with
identical camera paths, identical compression and the SDR exposure the only
difference, separates the two claims:

| exposure | achievable (3 seeds) | ceiling |
| --- | --- | --- |
| constant | +0.110, +0.035, +0.046 JOD | +0.26 JOD / +0.63 dB |
| ±0.48 stops | **+0.511, +0.603, +0.661 JOD** | +1.03 JOD / **+4.55 dB** |

Every drifted arm clears the +0.5 threshold; every fixed-exposure arm is an
order of magnitude below it. So the information a temporal model would fetch
is *exposure variation*: present when the exposure moved between neighbours,
close to absent when it did not.

That is measured with the camera angles the renderer wrote down, and a plate
has none. Re-scoring the same clips with the correspondence **estimated** from
the pixels, which is what a deployed model actually holds, is what closed the
line:

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
0.04–0.09 px median against the analytic poses. It fails in flat regions, and
forward-backward consistency cannot detect that failure, because any
displacement round-trips perfectly through a constant area. **The bad matches
are not low-confidence, they are confident and wrong**, in exactly the blown
sky a temporal model would be asked to reconstruct.

So RUDRA is a per-frame model by measurement, not for want of trying. No
temporal model was trained because there is nothing measurable for one to
learn. [`STATUS.md`](../STATUS.md) carries the full tables and what would have to
be true of a corpus to reopen the question.

The viewer looks for a model in `RUDRA_CHECKPOINT_ROOTS` first, then in this
repo's `checkpoints/`, so a clone with nothing else still starts on a real
model. Point it at your own training output to have the newest run there win:

```bash
export RUDRA_CHECKPOINT_ROOTS=/path/to/your/checkpoints     # or set on Windows
```
