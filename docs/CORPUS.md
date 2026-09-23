# The corpus

What RUDRA is trained on, what is wrong with what it was trained on, and what a
production-quality set has to contain.

Read this before proposing to collect anything. The headline result of the paper
is that **corpus size is not the constraint**: six times the footage moved the
measured bound by 0.03 dB. What the pilot below adds is that corpus *content*
may well be, and that the two are not the same claim. Capacity moved it 0.5 dB with the sign varying.
Objective moved it not at all. More of the same data is an experiment that has
been run and published as flat.

That is an argument about quantity. Everything below is about kind.

## Two defects in the corpus, both measured

### The SDR side did not clip

`training/prepare_training_data.py` applied `-1 EV` before the ACES curve,
described in the source as "slight underexpose for safety". The ACES
approximation saturates near a scene-linear input of 7.24, so halving the input
roughly doubles the radiance a pixel needs before it blows.

Measured on the corpus that produced: **median clipped fraction 0.000%**, and
**52.6% of frames with no clipped pixel anywhere**.

An inverse tone mapper is a machine for saying what was above a blown highlight.
A training set without blown highlights does not contain the question. Corpora
from v4 on render at 0 EV.

### The top code was unreachable

Underneath that, a smaller bug with a wider blast radius. `oetf_srgb(1.0)` is
`1.055 * 1**(1/2.4) - 0.055`, which in float32 lands on 0.99999994. Times 255
that is 254.99998, and `astype(uint8)` truncates it to **254**.

So the corpus SDR could not contain the value 255 at any exposure. Every fully
blown pixel was stored one code below full. Anything looking for clipping by
`== 255` found none, ever. And a model trained on it never saw the top code,
while real delivered SDR is full of it: train/serve skew at exactly the pixels
the model exists for.

Both are fixed, and `clipped_fraction` is now written into every sidecar, so the
next corpus states on its face whether it contains the phenomenon.

### What the fix is actually worth, measured

`pilot_clipping.py` on 200 frames from `E:\source_hdr`:

| | -1 EV (shipped) | 0 EV (fixed) |
|---|---:|---:|
| median clipped fraction | 0.000% | **0.000%** |
| mean clipped fraction | 0.962% | 2.873% |
| p90 clipped fraction | 2.627% | 11.619% |
| frames with no clipping at all | 54.5% | **54.5%** |

**Verdict: fail.** Read the two bold cells together. The exposure change roughly
triples the mean and quadruples the p90, so on frames that have highlights it
does exactly what it was meant to. It does not move the median or the
no-clipping fraction by a single frame, because **54.5% of that source has
nothing to clip at any exposure**.

The -1 EV was not the binding constraint for this footage. The source dynamic
range is. That matches the inventory: a median of 3.97 stops and no source above
10 000 nits cannot produce a blown highlight however it is exposed.

So the render fix is worth doing and is not sufficient. It improves the half of
the corpus that has highlights and cannot do anything for the half that does
not. Collecting footage with real range moves from "second" to "the binding
constraint", which is the opposite of where this document started, and the
pilot is why that is now known before hours of re-rendering rather than after.

### What this means for the convention

The render's exposure offset is recorded per frame and carried on the model.
`rudra.sdr2hdr.sdr_to_baseline_hdr(sdr, corpus_ev)` derives its scale as
`2**(-corpus_ev) * 203/10000`. The literal `2` it replaces was that same
expression at `-1 EV`, written down and unexplained. A checkpoint whose config
has no `corpus_ev` is legacy, which is correct for every checkpoint that existed
when this was added.

## What is on disk now

From the inventory summaries in `E:\RUDRA_v3_20260822`:

| set | files | independent scenes | median DR | above 10 000 nits |
|---|---:|---:|---:|---:|
| local | 21 970 | 12 | 3.97 stops | 0 |
| Stuttgart (HdM-HDR-2014) | 11 007 | 9 | 5.99 stops | 0 |
| NAS | 49 127 | 11 204 moving | 8.66 stops | 685 |

Three things to take from that table. The local set's ground truth has a median
dynamic range of **four stops**, which is not HDR in any sense that matters here.
Two of the three sets contain **no source above 10 000 nits** at all. And the
moving footage is nine and twelve scenes: `video_manifest_9f.jsonl.GATED` says
it plainly, *"only 2 independent training scenes (need >= 6). Temporal training
on this will diverge, exactly as the August run did."*

The NAS summary also reports a maximum dynamic range of **37.75 stops**, which is
not physical. Something in that set decodes wrong and it is in the training pool.
Resolve before the next render.

## Specification

Counted in independent scenes, not frames. 926 clips drawn from 11 scenes is 11
scenes, and the paper says so.

| class | scenes | measured DR floor | why |
|---|---:|---|---|
| Specular exteriors, sun in frame | 25 | 14 stops | the clipped-highlight case the model exists for |
| Practical lights, night interiors | 20 | 12 stops | a 200-nit lamp against a 20 000-nit sun is the ambiguity the bound is made of |
| Low dynamic range, well graded | 20 | 4 to 8 stops | the band where the model regresses: 45.2% of test against 27.2% of training draws |
| Skin under mixed light | 15 | 10 stops | the failure nobody forgives |
| Fire, explosions, muzzle flashes | 10 | 16 stops | FXTD's actual work, and absent from the corpus |
| Screens and displays in shot | 8 | 10 stops | emissive at odd chromaticities, a known iTM failure |
| Water, snow, chrome | 12 | 14 stops | dense small speculars, where the residual is worst |
| Native SDR and HDR pairs, real grade | 15 | n/a | the only class that breaks the synthetic-degradation circularity |

That last row is the one that changes what the model can claim. Every `hard`
frame today comes from one synthetic degradation pipeline, and the shadow gate is
trained against that pipeline's own label. It may be learning this generator's
signature rather than "these shadows need reconstruction", and nothing in the
current evaluation separates the two, because the test condition is generated the
same way as the training condition. Natively graded pairs are the test.

## Per-scene record

A scene without these fields does not enter the manifest.

| field | why |
|---|---|
| `scene_id` | independent scenes are the unit that matters |
| `peak_nits`, `median_nits`, `dr_stops` | measured, not claimed by the source |
| `clipped_fraction` | whether it contains the phenomenon |
| `tonemap_ev` | which convention rendered the SDR side |
| `camera`, `codec`, `grade` | provenance for the degradation model |
| `licence`, `licence_url`, `commercial_ok` | see below |
| `sha256` | so a checkpoint can pin it |

## Licence

RUDRA's weights ship under a noncommercial licence and corpus provenance is why.
The current scope is **internal and noncommercial**, which widens what can be
used but does not remove the need to record it: a corpus whose terms are not
written down cannot later be shown to permit anything.

One flag on what is already here. `E:\source_hdr\Netflix` is in the training
pool. Netflix publishes that content for testing and development, which is not
the same permission as training a model and shipping the weights. Under the
current noncommercial scope that is defensible; it would have to be revisited
before any commercial build, and `commercial_ok` in the manifest is where that
answer lives.

Every fetched source records its licence at fetch time, from the source, not from
memory.

## Candidate sources

To verify at fetch time, never assumed:

- **xDR**, a cinematic natively graded HDR and SDR dataset built for evaluating
  inverse tone mapping. The natively graded pairs are the missing class above.
- **Inverse-tone-mapped HDR video quality assessment** (ACM MM 2025), a
  broadcast-oriented dataset with SDR references.
- **Open HDRI**, 25 free 29K panoramas, for the renderer path rather than the
  photographic one.
- **HdM-HDR-2014** is already here, and is the only publicly redistributable
  part of the current corpus.

## Cleaning up

`training/survey_datasets.py` walks the dataset roots, walks the manifests, and
reports which directories are still referenced. It proposes and never deletes,
because every checkpoint pins its manifest by SHA-256: a result whose corpus has
been deleted stops being reproducible the moment you need to defend it.

```
python training/survey_datasets.py ^
    --root G:\datasets\sources --root G:\datasets\corpora ^
    --root E:\RUDRA_v3_20260822 --root E:\RUDRA_postfix_20260818 ^
    --manifest E:\RUDRA_v3_20260822\sdr_hdr_manifest.jsonl ^
    --manifest E:\RUDRA_v3_20260822\video_manifest_9f.jsonl ^
    --checkpoints checkpoints --output survey.json
```

Read the referenced column before the size column. An unreferenced directory is
either dead or its manifest was not passed in, and only one of those is safe.

## Order of work

Revised after the pilot, which changed what is first.

1. Run `pilot_clipping.py` per source set, not on all of them together. The
   answer differs by set, and which sets are worth re-rendering is itself the
   result. `E:\source_hdr` returns fail at 54.5% no-clip.
2. Collect against the specification above, licence recorded per scene. This was
   step four. The pilot moved it, because no render setting fixes a source whose
   ground truth peaks near diffuse white.
3. Re-render everything at 0 EV with the rounding fix, old and new together, and
   check `clipped_fraction` across the result.
4. Retrain and remeasure. One change at a time, so the result is attributable.
5. Survey the drives and agree what comes off. Last, not first: what is worth
   keeping is easier to judge once there is something to compare it against.
