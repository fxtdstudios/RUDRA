# RUDRA

**SDR in, HDR out.** Point it at a frame, a folder or a movie and it expands
ordinary 8-bit footage into scene-linear HDR you can grade, master and deliver.

[FXTD Studios](https://fxtdstudios.com) / Radiance Research

![RUDRA Studio](docs/rudra_studio.png)

*Frames rail, viewer, scopes and the reconstruction controls. Every number on
screen is measured from the composite you are looking at, not from a preview.*

---

## Start here

Double-click **`run_studio.bat`** on Windows, or run **`./run_studio.sh`**
anywhere else. The first run builds its own environment and installs what it
needs; after that it goes straight to the viewer. A browser window opens on its
own.

Nothing else to set up. Six models ship inside the repo, so a fresh clone can
reconstruct a frame without downloading anything.

### Your first shot, in three moves

1. **Open it.** Type a path into **Open shot** at the foot of the Frames rail:
   a folder of frames, or a movie (`.mov`, `.mp4`, `.mxf`, `.mkv`, `.avi`,
   `.m2ts`, `.webm`). You can also drag stills onto the window. Nothing is
   uploaded: RUDRA runs on the same machine as the footage and reads it where
   it sits, so a 1.4 GB ProRes never moves. Frames decode as you scrub.
2. **Look at it.** Hold **B** to flip to the plain inverse tone map, or press
   **W** to wipe. That comparison is the honest one, because it shows what
   RUDRA added over doing the obvious thing.
3. **Deliver it.** **Master EXR** writes a full-resolution scene-linear file
   with an HDR10 sidecar. `M` is the shortcut.

Press **?** for the rest of the keyboard.

---

## Does it actually help?

![Degraded input, four methods](docs/compare/hard_deployment.png)

Three held-out frames, all four methods, on the condition that looks like real
work: unknown tone curve, 4:2:0 chroma, banding, JPEG. The HDR columns are
stopped down by whole stops so the highlight range lands where a browser can
show it, and the exposure is picked from the **reference** alone, so no method
is flattered by its own output. The analytic inverse tone map has a hard ceiling
and clips into it, so its sky is a flat plate. RUDRA keeps the roll-off the
reference has.

429 held-out frames, whole scenes held out so nothing appears on both sides:

| condition | metric | plain inverse tone map | **RUDRA, as shipped** |
|---|---|---:|---:|
| clean, well-graded input | PU21-PSNR | 45.99 dB | **46.06 dB** |
| clean | ColorVideoVDP | 9.448 JOD | **9.561 JOD** |
| hard, real-world input | PU21-PSNR | 25.92 dB | **27.15 dB** |
| hard | ColorVideoVDP | 7.362 JOD | **7.751 JOD** |

The shipped model is the only configuration measured that beats the plain
inverse tone map in **both** conditions on **both** metrics. On clean,
well-graded input the gap is small, because that input is a job the analytic
curve already does well. On degraded input, which is most footage, it is +1.2 dB and
+0.39 JOD.

Against published work, ExpandNet on the same frames and the same reference,
see [the comparison strip](docs/RESULTS.md#comparing-against-published-work) further down.

Everything above is reproducible: [**Reproducing the numbers**](docs/RESULTS.md#reproducing-the-numbers).

### The three controls that matter

| control | what it does | when to move it |
|---|---|---|
| **Mode**: All / Highlights / Shadows / Off | which regions the learned part is allowed to touch | Highlights if you only want blown areas opened up and the rest left alone |
| **Residual strength** | how far the learned correction goes, 0 to 2× | back it off if recovered highlights look invented |
| **Display peak** | what your monitor is pretending to be, 100 to 10 000 nits | set it to your actual mastering display; it changes the view, never the file |

**Preserve outside masks** is on by default and should stay on. With it off you
see the network's raw prediction everywhere, including regions it was never
meant to touch.

### Two delivery switches, and why they exist

Both live in the Deliver rail and both are on by default. They apply to
**Master EXR**, not to the live view.

**Anchor to source exposure.** RUDRA's analytic baseline carries a fixed +1 stop
that comes from how the training corpus was built. On a frame from that corpus
it is correct. On your plate it is not. Measured on a real image, mid-grey came
back **+1.43 stops**, and 89% of the picture that was never clipped had been
re-exposed. Anchoring puts unclipped picture back exactly where it was and lets
only the highlights expand. Turn it **off** only if your source came out of
`prepare_training_data.py`.

**Carry source chroma.** The expansion runs per channel and is steep near white,
so two 8-bit codes one step apart in red come out far apart, which shows up as
coloured flecks in a smooth sky. This takes hue from the source below the clip
and keeps the model's own above it. Measured chroma noise dropped from 16.6 to
7.0 against the source's own 3.1.

---

## Feed it good pixels

This is the single biggest thing you control, and it is upstream of RUDRA.

**Save 16-bit, not 8-bit.** The expansion multiplies whatever fine detail it is
given by roughly 30×, quantisation included. Same sky, same curve, only the
source's depth changing:

| source | grain in the output | colour flecking |
|---|---:|---:|
| 8-bit | 2.36% | 1.53 |
| 10-bit | 1.04% | 0.67 |
| 12-bit | 0.85% | 0.50 |
| float | 0.83% | 0.49 |

Roughly two thirds of the sparkle in an 8-bit sky is the eight bits, and it is
gone by ten. In a ComfyUI graph the VAE decodes to float and the frame is
quantised on the way to a PNG, one step before RUDRA is asked to put back what
the quantiser removed. Save 16-bit PNG or EXR from that decode instead.

RUDRA reads 8- and 16-bit PNG and TIFF and display-encoded float, and every
response reports `source_bits` so you can see what it actually received. It
refuses a float file that peaks above 1.0: that is already HDR.

---

## Check the result

```bash
python training/qc_reconstruction.py --sdr plate.png --hdr plate_rudra.exr
```

PASS, FAIL, or UNMEASURED per check, with the number and the threshold beside
it. A metric that could not be measured counts as a failure, never a pass. The
thresholds live in [`configs/qc_reconstruction.json`](configs/qc_reconstruction.json),
each with the reason it holds that value; change them deliberately and say why.

For a sequence, `--sdr-dir`/`--hdr-dir` with `--representative` scores first,
25%, 50%, 75% and last. Exit code is 0 only on PASS, so it can gate a build.

---

## What RUDRA actually does, and what it does not

It **expands** an SDR frame into an HDR container and **estimates** what belongs
above white. Doing nothing at all leaves clipped highlights about 3.2 stops too
dark, so the expansion is unambiguously worth doing.

It does **not** recover data that was destroyed. A pixel clipped at 255 has lost
its value; what comes back is a plausible estimate, not the original. On the
frames measured so far the learned model lands about 0.07 stops from the
ground truth on clipped pixels against the analytic curve's 0.14. A real edge,
on a small sample. [`training/measure_clipping.py`](training/measure_clipping.py)
settles it across the whole test split, and
[`STATUS.md`](STATUS.md) records what has and has not been measured.

Say "expands SDR into HDR". Do not say "reconstructs the original data".

---

## Command line

The viewer is one way in. Everything it does is available without it:

```bash
# reconstruct a frame or a folder
python training/infer_sdr2hdr.py input/ --output-dir out/ \
    --checkpoint checkpoints/sdr2hdr_shadow_v1.pt

# inspect, master and deliver -- no GPU needed
rudra info out/ --nits-scale 203                                           # nits, PQ codes, percentiles
rudra metadata out/ --nits-scale 203 --output out/master --peak-nits 1000  # Dolby Vision L1 + HDR10+
rudra aces out/ --output delivery/aces --ocio                              # ACES EXR + OCIO config
rudra grade in.exr --output graded/ --region 400:2000:1.0:0.5              # +1 EV over 400-2000 nits
rudra bench bench_root/ --output results.json                              # PU21-PSNR / CVVDP
```

Train it on your footage: [**Train on your own footage**](docs/TRAINING.md).

---

## Installing by hand

Only needed if you are not using the launcher.

Python 3.10 to 3.13.

```bash
git clone https://github.com/fxtdstudios/RUDRA.git && cd RUDRA
pip install -e .                          # core plus the `rudra` CLI
pip install -e ".[metrics]"               # cvvdp and lpips, for the real metrics
pip install -e ".[test]" && pytest tests/
```

CUDA is only needed for training and inference. `rudra/delivery/` is torch-free,
so render and mastering machines need Python and numpy and nothing else.

Video needs `ffmpeg` on `PATH` (`winget install Gyan.FFmpeg`); a folder of
frames needs nothing.

```bash
sha256sum -c checkpoints/SHA256SUMS     # the six shipped models, 38 MB
```

`checkpoints/models.json` is the registry the server reads, and
[checkpoints/README.md](checkpoints/README.md) says what each file is. Training
pairs and HDR sources stay out of git.

Needs WebGL2 and float render targets in the browser. Any current one will do.

---

---

## When something goes wrong

**"no model" in the title bar.** The viewer could not find a checkpoint. The six
that ship with the repo live in `checkpoints/`; if you moved them, point at one
directly:

```bash
python ui/server.py --checkpoint checkpoints/sdr2hdr_shadow_v1.pt
```

**A movie will not open.** Video needs `ffmpeg` on `PATH`. On Windows,
`winget install Gyan.FFmpeg`, then restart the viewer. A folder of frames needs
nothing, so exporting frames is always a way round it.

**The picture is black, or the page never draws.** The compositor needs WebGL2
with float render targets. Any current browser has it; a remote desktop session
or a very old GPU driver may not.

**It is slow.** The first frame loads the model, which takes a few seconds.
After that, expect well under a second a frame on a recent GPU. If every frame
is slow, check the title bar: it says `CPU` when CUDA was not available.

**The master looks brighter or flatter than the viewer.** The viewer composes on
your GPU from the raw fields and does not apply the two delivery switches. They
are applied when you press Master EXR. That difference is the source's own
exposure, and the toggles' hints say so.

**Colours look wrong in Nuke or Resolve.** The EXR is scene-linear with BT.2020
primaries and 1.0 = 203 nits. If your host is reading it as Rec.709 or as ACES
AP0, that is a primaries mismatch, not a RUDRA bug. `rudra aces` writes an
AP0-tagged file if that is what your pipeline wants.

---

## Train it on your own footage

If you have your own HDR material, you can train a model on it. One command:

```bash
train.bat            # Windows
./train.sh           # anywhere else
```

It asks for the folder, then runs the whole thing: inventory, pairs, manifests,
verification, backbone, shadow gate, benchmark. Each stage checks its own output
and refuses to hand work to the next if the check fails, and the whole thing is
resumable.

You need **HDR** sources. The pipeline makes the SDR side itself, and there is
nothing to learn from SDR alone. Scene variety matters far more than frame
count.

The full guide, including the five commands underneath and six ways to waste a
week, is in [**docs/TRAINING.md**](docs/TRAINING.md).

---

## Going deeper

- [**STATUS.md**](STATUS.md) what is finished, what is measured, what is open.
  Three separate things share the name RUDRA and this keeps them apart.
- [`paper/main.pdf`](paper/main.pdf) *What an 8-Bit Frame Can and Cannot Say
  About the Scene Behind It*, 15 pages.
- Weights: [huggingface.co/fxtdstudios/RUDRA](https://huggingface.co/fxtdstudios/RUDRA/tree/main).


## Licence

The code is Apache 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE).

The weights are **non-commercial**, licensed in
[`checkpoints/LICENSE`](checkpoints/LICENSE). Research, teaching, evaluation,
benchmarking and publishing results with them are all permitted. Selling them,
or anything built with them, is not. HdM-HDR-2014 and HdM-HFR-2017 are free
for academic use and need a separate agreement with HdM Stuttgart for
commercial use, and that is not a term FXTD Studios can waive for you. Credit
Netflix Open Content (CC BY 4.0) wherever you credit sources. Commercial
licensing: ask.

`pipeline/render_hdri_moves.py` is the way out of that condition. Weights
retrained on the CC0 panoramas plus FXTD's own footage carry no upstream
restriction, and are intended for release under Apache 2.0.

---

[FXTD Studios](https://fxtdstudios.com) · Cairo
