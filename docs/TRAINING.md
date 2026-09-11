# RUDRA training

Split out of the README on 10 Sep 2026, when the README became a page for the
people who use the tool. Nothing here changed; it moved.

Back to [the README](../README.md) . [STATUS.md](../STATUS.md) says what is
finished and what is open.

---

## Train on your own footage

RUDRA is meant to be retrained. A model fitted to your cameras, your grades and
your delivery ceilings will beat a general one on your material, and the data
never leaves your machine. Nothing here phones home.

Training on footage you own also sidesteps the licence on the released weights
entirely. See [Licence](#licence). Your model, your data, your terms.

### What you need

**HDR ground truth. This is not optional.** RUDRA learns to invert a tone map,
so it needs to see the answer. The SDR half of every pair is *generated* from
your HDR by the pipeline. You do not supply it, and you cannot train from SDR
alone. If all you have is SDR, there is nothing here to learn from.

Anything with real range works:

| You have | Works | Note |
|---|---|---|
| Scene-referred EXR / Radiance HDR | best | no grade ceiling, the model sees true radiance |
| Graded HDR masters (PQ / HLG) | yes | declare the ceiling, see below |
| HDR video (MXF, MOV, ProRes, MP4…) | yes | frames are extracted; `--video-stride` thins them |
| Log footage (S-Log, V-Log, LogC) | yes, via EXR | convert to scene-linear first, in Resolve or Nuke |
| 8-bit SDR only | **no** | no target to learn |

Formats the scanner accepts: `.exr .hdr .tif .tiff .dpx .png .jxl .avif .heic`
and `.mxf .mov .mp4 .mkv .avi .m2ts .ts .webm`.

**How much.** The shipped model saw ~28,500 records. Scene *diversity* matters
far more than frame count. 926 clips drawn from 11 scenes is 11 scenes, and
the model will overfit to them no matter how many crops you cut. As a rough
floor: 20+ distinct scenes and a few thousand records before the numbers mean
anything. Below that you are measuring your test split.

**Your grade ceiling matters.** If your masters are graded to 1,000 or 4,000
nits, every pixel sitting exactly at the ceiling means *at least* that bright,
not *exactly* that bright. The loss goes one-sided there with three stops of
free headroom. Without it the model learns to cap, and your highlights die.
`scan_sources.py` probes for this, but check its output against what you know
about your deliverables.

### One command

```bash
./train.sh /path/to/your_hdr_footage         # macOS / Linux
train.bat  D:\path\to\your_hdr_footage       # Windows, or double-click it
```

That is the whole thing. It checks your environment, inventories the footage,
builds the pairs, splits them by scene, verifies the corpus, trains the
backbone, trains the shadow gate on top, and scores the result against the
analytic baseline on your own held-out frames. It installs the requirements the
first time if they are missing.

It is **resumable**: a stage whose output already exists is skipped with a
note, and interrupting the backbone and re-running picks up from its last
checkpoint. Nothing is lost by stopping it.

```bash
python training/train_from_footage.py FOOTAGE --dry-run   # print the plan, run nothing
python training/train_from_footage.py FOOTAGE --from pairs # redo from a stage
python training/train_from_footage.py FOOTAGE --steps 40000 --device cpu
```

It stops early rather than late. Pointed at 8-bit SDR it says there is nothing
to learn from, instead of training a model that has learned the identity
function. Pointed at 200 files it warns that you will be measuring your test
split. If the corpus fails verification it stops there and tells you the checks
exist because something got past them once.

### The five commands underneath

Run these directly when you want something the driver's defaults do not give.
Each stage checks its own output and refuses to hand work forward if the check
fails.

```bash
# 1. Inventory. What is on disk, what range it carries, how it is encoded.
python pipeline/scan_sources.py /path/to/your_hdr --out work/inventory.jsonl

# 2. Pairs. Generates the SDR side, encodes the HDR side, writes the sentinel.
python pipeline/prepare_pairs.py --inventory work/inventory.jsonl --dst work/pairs \
    --mode log2_extended --crops 3 --crop-size 512 --video-stride 2

# 3. Manifests. Scene-held-out splits, not frame-held-out.
python pipeline/build_manifests.py --pairs-dir work/pairs --out-dir work \
    --val-frac 0.10 --test-frac 0.10

# 4. Verify. Run this. It is the cheapest hour you will spend.
python pipeline/verify_dataset.py --pairs-dir work/pairs \
    --manifest work/sdr_hdr_manifest.jsonl \
    --video-manifest work/video_manifest_9f.jsonl

# 5. Train the backbone.
python training/train_sdr2hdr.py --mode image --manifest work/sdr_hdr_manifest.jsonl \
    --output-dir work/checkpoints/image --steps 100000 \
    --best-metric composite_gain --device cuda
```

Then the shadow gate, on the frozen backbone. Fifteen minutes on one 4080, and
it is what takes the model from helping only on degraded input to helping on
both conditions:

```bash
python training/train_shadow_gate.py --manifest work/sdr_hdr_manifest.jsonl \
    --init-checkpoint work/checkpoints/image/best.pt \
    --output-dir work/checkpoints/shadow --seed 20260901
```

The backbone is the long stage, and the shipped model is step 81,000. Everything
after it is minutes.

### Reading the log

Every eval scores two conditions, and the distinction is the whole point:

- `clean_*`: the held-out frame as prepared. A well-graded plate.
- `hard_*`: the same frame under a seeded camera and codec degradation.

**Only the hard numbers describe deployment, and only they choose `best.pt`.**
A model that wins on clean and loses on hard is a model that will disappoint
the first time an artist points it at real archive material.

Both report `gain_db` against the analytic inverse tone map the network sits on
top of, so "better than doing nothing" is a number rather than an impression.
A gain near zero on hard means the network is not earning its inference cost.

### Check it against the shipped model

Same held-out frames, same metrics, both models:

```bash
python training/export_bench_pairs.py --checkpoint work/checkpoints/shadow/best.pt \
    --manifest work/sdr_hdr_manifest.jsonl --out bench/hard \
    --condition hard --test-name mine
rudra bench bench/hard --output results.json
```

PU21-PSNR and ColorVideoVDP JOD, against the same unclamped reference. Read the
JOD. [The two metrics disagree](#the-two-metrics-disagree) by two orders of
magnitude on small differences, and the JOD is the one calibrated against human
observers.

### Deploy it

Point the viewer and the CLI at your checkpoints:

```bash
export RUDRA_CHECKPOINT_ROOTS=/path/to/work/checkpoints    # or set on Windows
```

That root is searched before this repo's `checkpoints/`, so your model wins
without touching the repo. To have it appear by name in the Studio picker, add
an entry to `models.json` beside the checkpoint: `file`, `kind: sdr2hdr`,
`title`, and a `note` saying what it is and what it measured. The registry
exists because discovery-by-newest-file once picked a temporal refiner and tried
to load it as an image model.

### Six ways to waste a week

Every one of these cost us real time. They are in the checks now, but the
checks only help if you read what they say.

1. **Mixing storage conventions.** `_ingest_config.json` beside your pairs is
   the sentinel that says how the 16-bit HDR PNGs decode. Append pairs written
   under a different `--mode` or `--ceiling-nits` and the corpus becomes
   unreadable in a way that still trains. The pipeline refuses to mix them;
   use a fresh `--dst` rather than arguing with it.
2. **Two different meanings of 1.0.** The network's output is `1.0 = 10,000
   nits`. Storage and EXR are `1.0 = diffuse white = 203 nits`. Conflating them
   is a 5.6-stop error that produces entirely plausible-looking numbers. Ours
   read 11.53 dB where the truth was 46.
3. **Not declaring the grade ceiling.** See above. The model learns to cap.
4. **Frame-held-out splits.** Two crops of one frame on both sides of a split
   is leakage, and it makes a model look far better than it is. Scenes are held
   out here, and `verify_dataset.py` enforces a cap on how much of the eval set
   any one scene may be.
5. **Reporting one seed.** The shipped gate first looked like a large win on
   seed 1 and a small one on seeds 2 and 3. Train three, report the spread.
6. **Judging on clean.** RUDRA is near-neutral on well-graded input by design,
   because there is nothing to recover. Optimising the clean number optimises for the
   case that did not need you.

---
