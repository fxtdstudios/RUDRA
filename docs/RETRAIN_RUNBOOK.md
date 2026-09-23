# Retrain runbook

Supersedes the "Order of work" in `docs/CORPUS.md`. Written after the
`pairs_index.jsonl` audit, which changed what comes first.

Five phases, each with a gate. **A gate that fails stops the phase — it does not
get noted and stepped over.** The whole point of the audit was that the last run
skipped its gates and cost a retrain.

---

## Phase 0 — fix the units. ✅ done (16 Sep 2026, commit `732d0d9`)

`tonemap_ev` now travels manifest → `corpus_ev` in the checkpoint config → frame
header → compositor uniform, and `SDR2HDRNet.from_config` reads it back, so the
analytic baseline always undoes the exposure the render actually applied.
`verify_dataset.py` check 3b puts a floor on SDR clipping (commit `f3f2749`).
The historical defect this phase fixed is kept below for the record.

The corpus had two luminance defects, in opposite directions, from two ingest
paths. Re-rendering before they were fixed would have baked both into the new
corpus.

| set | pairs | symptom | cause |
|---|---:|---|---|
| PolyHaven | 2 889 | median 601 nits, p90 26 796, max 962 784; p90/median = 44.6 | relative data read as absolute |
| Chimera (`data/hdr`) | 4 071 | every pair peaks **below 1 nit** (0.0002–0.3245) | normalized 0..1 never multiplied |

Absolutely-scaled HDR is bunched — Stuttgart's p90/median is 1.2, Netflix 1.4,
HdM-HFR 1.4. PolyHaven at 44.6 is not a bad file, it is no scale at all.

**0.1** In `pipeline/hdr_io.py`, separate the two cases explicitly. Panoramic
HDRI is *relative*: it needs a stated anchor (normalise to diffuse white and
record `absolute: false` on the scene) rather than a nit figure that pretends to
be measured. The Chimera path needs its missing multiply.

**0.2** Add `absolute: true|false` to the per-scene record. A scene that does not
say whether its luminance is measured cannot be filtered on dynamic range, and
that is exactly how 19.5% of the corpus ended up mis-scaled unnoticed.

**0.3** Re-run the audit:

```
python training\quarantine_broken.py --index E:\RUDRA_v3_20260822\pairs\pairs_index.jsonl
```

> **GATE 0** — no set reports `NO ABSOLUTE SCALE` or `NOT NITS — NORMALIZED`.
> Until this passes, every dynamic-range number downstream is fiction.

---

## Phase 1 — fetch and measure

**1.1** Fetch. Plan, check, then pull:

```
python training\fetch_corpus.py --plan
python training\fetch_corpus.py --check
python training\fetch_corpus.py --fetch
```

Leave `polyhaven` off. You already hold it at 2K (E:, 11.6 GB) and 4K
(`RUDRA_v02\panoramas`, 47.3 GB); a 16K third copy adds nothing until the ingest is proven to handle it.

**1.2 — on the critical path, and it is a web form.** LIVE-TMHDR: 40 scenes a
commissioned colourist graded by hand, licensed *"for any purpose"*,
royalty-free. It is the only source that is simultaneously (a) native
HDR↔SDR pairs, (b) enough scenes to clear the temporal gate, and (c)
commercially clean. Nothing else on the list is all three.

**1.3** Measure what landed:

```
python pipeline\scan_sources.py G:\datasets\sources --out G:\datasets\_inv\inventory.jsonl
```

> **GATE 1** — `independent_moving_sources` ≥ 6 per set, and ≥ 60 across the
> corpus before temporal training is attempted again.
> `video_manifest_9f.jsonl.GATED` already recorded what happens otherwise:
> *"only 2 independent training scenes (need >= 6). Temporal training on this
> will diverge, exactly as the August run did."*

---

## Phase 2 — decide the manifest, which is a licence decision

Two scopes, two different corpora. Choose before building, not after training.

**Internal / noncommercial** — everything fetched, HdM and Stuttgart included,
`commercial_ok` recorded honestly per scene.

**Commercially clean** — only sources whose text permits training *and*
distributing the resulting weights:

| in | out |
|---|---|
| Netflix Open Content (CC BY 4.0) | HdM 2014 — paid tier grants trade shows, pipeline dev, TMO evaluation; **not training** |
| Poly Haven (CC0, AI training blessed in writing) | HdM-HFR-2017 — no licence stated at all |
| LIVE databases ("for any purpose", royalty-free) | SJTU — "no commercial use", explicit |
| Laval Photometric, paid tier — the only licence that names weight distribution | Fraunhofer — CC BY-NC-**ND**; a training tensor is a derivative |
| | LiU HDRv — CC BY-SA; ShareAlike vs weights is unsettled |
| | StEM2 — "solely for education, training, research…", silent on shipping weights |

**Deleting HdM from disk does not clean existing weights.** The shipped
checkpoints learned from it — 69% of the corpus across both sets. Clean
provenance means training from scratch, not warm-starting from those.

Whether weights are a derivative work of training data is unsettled law. This
table is a reading of licence text, not a legal opinion; a commercial build
needs a real one.

> **GATE 2** — every scene in the manifest carries `licence`, `licence_url`,
> `commercial_ok`, `sha256`, `absolute`. A scene missing any of them does not
> enter.

---

## Phase 3 — re-render. This is the decisive experiment.

**3.1** Reclaim first — orphans, then the referenced cache:

```
python pipeline\clean_pairs.py E:\RUDRA_v3_20260822           # report
python pipeline\clean_pairs.py E:\RUDRA_v3_20260822 --apply   # to _trash
```

Then delete `pairs\{sdr,hdr,meta}` and **keep `pairs_index.jsonl` and
`_ingest_config.json`** — the cache regenerates from them, and the audit reads
the index.

**3.2** Render at the corrected exposure:

```
python training\prepare_training_data.py --tonemap-ev 0
```

**3.3** Measure clipping **per set, not pooled**. Pooling is what hid this last
time:

```
python training\pilot_clipping.py --src G:\datasets\sources\netflix_sparks
python training\pilot_clipping.py --src G:\datasets\sources\live_tmhdr
```

> **GATE 3 — the one that matters.** Median `clipped_fraction` > 0, and the
> no-clipping fraction well under 50%.
>
> Today, of 35 585 correctly decoded pairs, **not one contains a clipped
> pixel**. Every clipped pixel in the corpus came from the 205 broken PolyHaven
> decodes. An inverse tone mapper is a machine for saying what was above a blown
> highlight, and it has never seen one.
>
> If the new corpus still does not clip, **stop here**. More data was never the
> answer, and Phase 4 would burn a week proving it slowly. That result is worth
> more than a retrain.

---

## Phase 4 — train

**4.1** From scratch under the commercial scope. Warm-starting inherits HdM.

**4.2** One change at a time, corpus first, architecture and objective held
fixed. The paper already measured the alternatives: six times the footage moved
the bound 0.03 dB, capacity moved it 0.5 dB with the sign varying, objective
moved it not at all. So this run tests exactly one hypothesis — that content,
not quantity, was the constraint — and it is testable because nothing else
moved.

> **GATE 4** — same bench, same metrics as the paper. A new bench makes the
> comparison meaningless.

---

## Phase 5 — remeasure, then write it down

**5.1** Score the pixels the claim rests on, not whole frames:

```
python training\measure_clipping.py --bench <bench> --score --checkpoint <new>
```

Whole-frame PSNR is mostly pixels that were never clipped. The claim is about
the clipped ones, and `--score` is the only thing that addresses it.

**5.2** Update `README.md` and `docs/` in the same pass — corpus table, results,
and the open items this run closes or fails to.

---

## What is still open regardless

- Class 4 (fire, ≥16 stops) has almost no public supply. HdM has it and the
  licence does not permit it; SJTU has Bonfire and says no commercial use.
  Under the commercial scope this class stays empty.
- The HdM exposure in the already-published checkpoints is a separate question
  from what happens next, and it does not resolve itself by changing the next
  corpus.
