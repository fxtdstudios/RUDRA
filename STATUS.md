# RUDRA — Training & Research Status

> **Updated 22 Sep 2026.** The snapshot below the line dates from 22 Aug and is
> still accurate for what it covers. Read this section first: the repository
> holds **five separate lines of work** that share a name, and "is RUDRA
> finished?" has a different answer for each.
>
> | line | what it is | state |
> |---|---|---|
> | **A. Production decoders** | distilled log-space VAE decoders, 7 backbones, ComfyUI node | **paused (23 Sep 2026)** — complete and measured; out of product scope, not maintained |
> | **B. Research pipeline (Stages 1-3)** | descriptor + FiLM + DR-gated LoRA + DRE cross-attention, the *original paper's core thesis* | **paused (23 Sep 2026)** — Stage 3 never trained; latent-specific, out of scope |
> | **C. Direct SDR-to-HDR image model** | `rudra/sdr2hdr.py`, v5, RUDRA Studio, the delivery path | **measured and written up** |
> | **D. Temporal (v02)** | rendered camera-move corpus, clip metric, the oracle gate | **CLOSED.** Exact poses +0.60 JOD, RAFT +0.34, DIS −0.07, against a +0.5 threshold fixed in advance. Nothing a plate can supply clears it; no temporal model trained, and that is the result |
>
> | **E. Corpus programme (v4b)** | 0 EV re-ingest on `G:\datasets`, gate 3b, the retrain that tests "corpus content was the constraint" | **corpus built and gated; training not started.** Three runs made between 18 and 22 Sep were on the wrong corpus and are quarantined |
>
> **22 Sep 2026 — dataset audit (line E).** What was done since 16 Sep, checked:
>
> - Sources moved to `G:\datasets\sources` (`pipeline/migrate_datasets_to_g.ps1`);
>   `E:\source_hdr` is a junction. `BUILD_CORPUS_V4.ps1` re-ingests at 0 EV
>   through the pipeline path into **`G:\datasets\corpora\corpus_v4b`**: 20,628
>   records, 783 train scenes, **check 3b passes at 29.3%** (the shipped corpus
>   was 0.57%), check 8 passes with 12 train + 2 held-out video scenes, check 9
>   WARNs (2% of targets above 40,000 nits; left at `max_hdr=4.0` on purpose).
>   The smoke run on it reads `corpus_ev 0.0` and a step-0 baseline of 48.99 dB
>   clean `psnr_log`. The corpus is sound. `docs/TRAINING_STEPS.md` has the run.
> - **Three checkpoint runs are invalid and quarantined** in
>   `checkpoints/_invalid_corpus_v4/` (README inside): `sdr2hdr_image_v4` (70k
>   steps), `sdr2hdr_image_v4_gate` (8k) and `sdr2hdr_temporal_v4` (111 steps,
>   stopped). The first two trained on the *first* `G:\corpus_v4` — old ingest,
>   no sidecars, so `corpus_ev -1.0` over a 0 EV render — and their "+13.9 dB
>   clean gain" is the baseline's one-stop error, learned (that baseline reads
>   25 dB; on v4b it reads 49). The temporal run took the invalid image model
>   over the 0 EV clips, and nothing stopped it because the video manifest never
>   carried `tonemap_ev`: `corpus_ev_of` fell back to -1 EV and agreed with the
>   wrong checkpoint by accident.
> - **Fixed, each with a test in `tests/test_temporal_corpus_ev_2026_09_22.py`:**
>   clips carry `tonemap_ev` and a sidecar path; `corpus_ev_of` reads a clip
>   row's sidecars (v4b's existing video manifest needs no rebuild); temporal
>   training exits when the image checkpoint's `corpus_ev` is not the manifest's;
>   `build_manifests.py --hold-out-scenes` pins a previous manifest's test scenes
>   to test, matched on the drive-independent tail of the id, so the paper's 429
>   frames stay held out of anything trained on v4b (`BUILD_CORPUS_V4.ps1
>   -Manifest` passes it by default); `.gitignore` stops `checkpoints/*/` and
>   `/bench/` — 816 MB of step checkpoints and exported bench PNGs were staged.
> - **Not done, now measurable:** the out-of-generator condition is exported
>   (`bench/oog`, 429 frames from `sdr2hdr_shadow_v1.pt` on the v3 test split;
>   Hable + H.264 CRF 28) but not scored. The synthetic-clip protocol
>   (`measure_clipping.py --score` on 0/+1/+2 EV re-renders) is still the only
>   measurement of the highlight claim and has not been run.
> - **`rudra deliver` failed every encode on FFmpeg 7.x** (found 23 Sep on the
>   Windows box): swscale's `out_color_matrix` takes `bt2020`; `bt2020nc` is the
>   tag's name and 7.x rejects it as an undefined constant. 4.4 (CI) accepted
>   it, so CI was green. Filter now gets `bt2020`, tags stay `bt2020nc`;
>   verified on 4.4.2 and 7.0.2, pinned by a test.
> - Repo root organised: one-shot commit/push scripts → `scripts/archive/`
>   (git-ignored), dated reports → `reports/`, `scripts/AUDIT_REPO.ps1` kept.
>   `codex/rudra-final-release` (20 Sep, +2,309 lines: batch/video delivery,
>   quality benchmark, recovery ablation, finetune scripts, 14 test files) is a
>   worktree branch off 6 Sep and has not been reviewed or merged; nothing here
>   depends on it.
>
> **23 Sep 2026, 02:00 — Step 4 ran.** Launched on v4b at 21:51 (manifest
> `ffcd8bbb…`, before the hold-out rebuild), `corpus_ev 0.0`, 50k steps in
> 4 h 08 min. `best.pt` = step 36,000: **+1.80 dB clean, +0.82 dB hard**
> `psnr_log` over its own analytic baseline (25.29 / 19.96 dB). Both signs
> positive, which v5 never managed. Acceptance waits on
> `pipeline/check_holdout_overlap.py`: if any of the paper's bench scenes are in
> this run's train split it is retrained on the rebuilt manifest (4 h), and
> `scripts/next_steps_2026-09-22.ps1` does that decision by data.
>
> **23 Sep 2026, 17:26 — v4c is the research (non-commercial) corpus**, HdM
> included, at `G:\datasets\corpora\corpus_v4c`; one model, `sdr2hdr_image_v4c`.
> The commercial build waits for more clean real footage (FXTD, LIVE-TMHDR).
> Checked on G: first: all 52,772 inventory sources resolve after the move;
> Sparks' duplicate P3-PQ rendition (13,777 files) is dropped; ACES scaled
> 1.0 = 100 nits (measured 55-91 against the PQ grade); Poly Haven venice_* /
> stuttgart_* no longer read as S-Log3 / PQ / HdM. liu_hdrv and pandora are
> still zip archives (pandora partly .part) and are not in any inventory.
>
> **23 Sep 2026, 17:00 — critical path built, running on the box.**
> `scripts/RUN_CRITICAL_PATH.bat` runs N1 → N2 + N3 → N7 unattended with
> gates and resume markers (`reports/logs/cp_*`). What is in the code now:
>
> - [x] **N2** `pipeline/sdr_render.py` + `prepare_pairs.py --sdr-render mix`:
>   six curves (ACES, Hable, Reinhard, AgX-like, camera log→709, clip), ±1.5 EV,
>   contrast/saturation/OETF jitter, JPEG/H.264/HEVC/AV1 round trips, one
>   recipe per shot, recorded per pair. ACES path bit-exact with the old render.
> - [x] **N3** `rudra.sdr2hdr.CurveHead` (`--curve-head`): per-frame exposure +
>   8-knot log2 correction of the analytic inverse, zero-init (identity), one
>   curve per frame under tiling, the same maths in the Studio shader
>   (`uCurve[9]`), gain still scored against the *analytic* baseline.
> - [x] **N7** `pipeline/licences.py`: every manifest row carries
>   `licence_source` / `commercial_ok`; `build_manifests.py --commercial-only`
>   is the rudra-studio corpus (HdM and unclassified sources out).
> - [x] Sparks: ACES 2065-1 EXRs now decode AP0 → Rec.2020 (`aces` encoding);
>   `pipeline/reclassify_inventory.py` applies it without a rescan.
> - [x] `training/paired_gate.py`: paired Δ, bootstrap CI, pass/fail.
> - [ ] Runs: v4c ingest, v4c research + studio training, benches (hours).
> - [ ] N8 desktop app: its plan moved to native Qt/C++ in `d427bec` (another
>   session); not touched here.
>
> **CPU proxy for N3 (mechanism check, not a result).** 97 bench scenes at
> 192×108, 70 train / 27 test, 12-channel model, ~900 steps. psnr_log, model
> vs analytic inverse, per test curve: with the curve head Hable **31.4 vs
> 29.8**, AgX **35.3 vs 33.4**, camera-log **33.4 vs 30.7**, clip **30.1 vs
> 27.3**; without it +0.4/+0.9/+0.6/−0.4. **But on the corpus's own ACES render
> it falls to ~31 vs 49**: from one frame it cannot tell ACES from the mix
> and applies an average correction. The real run decides whether 25% ACES in
> the mix teaches it to recognise the curve; the `aces` bench in CP7 measures
> exactly this. If it doesn't, the fix is a confidence output that falls back
> to the analytic inverse, not dropping the head.
>
> **23 Sep 2026 — scope.** RUDRA is SDR→HDR for any image or video from any
> source, pixel-domain. Lines A and B are paused (they only work inside
> specific latent models). Plan, competitive read and what is against us:
> `reports/SDR2HDR_PLAN_2026-09-23.md`.
>
> **Out-of-generator bench scored (23 Sep).** 429 frames, Hable curve + real
> H.264 CRF 28, shipped `sdr2hdr_shadow_v1.pt`: RUDRA **26.77 dB PU21 / 8.114
> JOD vs baseline 27.18 / 8.139**, paired Δ −0.41 dB (95% CI −0.49…−0.34),
> −0.024 JOD (−0.048…−0.003); worse on 270 of 429 frames. On SDR made by a
> curve and codec it never trained on, the shipped model is worse than
> doing nothing learned. This is now the top item.
>
> **23 Sep, 16:00 — hold-out check answered.** 97 of 97 paper-bench scenes are in
> v4b; 96 were already in test, **one (`fireplace`, 348 records) was in train**.
> The overnight Step 4 run trained on it, so it is not held out on the old
> bench and is retrained on the rebuilt manifest (4 h). Evidence:
> `reports/logs/ns_2_overlap_original_manifest_ffcd8bbb.json`.
>
> The rebuild then failed check 7 on test (largest scene 30.4% vs 25%):
> `cap_scene_share` sized each dominant scene against the total as it stood,
> so thinning `fireplace` after `carousel_fireworks` pushed carousel back
> over. It now caps all dominant scenes jointly and re-checks the result.
>
> **Next steps, in order. Each has the gate it must pass before the next.**
> *(`scripts/RUN_NEXT_STEPS.bat` runs 1, 2, 6, 3 and 5 unattended with markers in
> `reports/logs/`; re-run it after each training window.)*
>
> 1. **Push the 22 Sep batch** — `scripts/finalize_2026-09-22.ps1` (pytest, then
>    one commit, then push). Gate: CI green; `git ls-files checkpoints/` shows
>    only root-level weights.
> 2. **Rebuild v4b manifests with the hold-out** — `BUILD_CORPUS_V4.ps1 -Manifest
>    -Verify` (no re-ingest; minutes). Gate: the hold-out line reports > 0 scenes
>    pinned, `verify_dataset.py` exits 0 with 3b still passing, and test's
>    largest-scene share stays ≤ 25%. If the hold-out pushes test over the cap,
>    raise `--test-frac`, not the cap.
> 3. **Step 4 — image model on v4b** — ran 22–23 Sep (above). Gate: the run's
>    manifest is the rebuilt one, or the overlap check finds no old bench scene
>    in its train split; otherwise retrain (4 h). `config.json` must read
>    `corpus_ev 0.0` (it does). Note the 49 dB smoke baseline was a 40-item
>    subset; the full val baseline is 25.3 dB clean, and that is not a defect.
> 4. **Step 5 — the conditioning head**, from item 3's `best.pt`, 8k steps.
> 5. **Step 7 — acceptance on the paper's bench**, re-exported from the *same*
>    429 reference frames at 0 EV (`export_bench_pairs.py` from the v4b manifest's
>    test rows once step 2 has pinned them; `clean`, `hard`, and
>    `out-of-generator`). Score v5, shadow_v1 and the v4b model each against the
>    render it was trained to invert. Then `measure_clipping.py --score`. Gate:
>    RUDRA beats the analytic inverse on clipped pixels in stops; if it ties to
>    the digit again, corpus content was not the constraint — write that down and
>    stop the corpus programme.
> 6. **Score `bench/oog`** with the shipped `sdr2hdr_shadow_v1.pt` now (no GPU
>    time to speak of) and put the row in `docs/RESULTS.md`; it answers the
>    review's "one out-of-generator degradation" today, before any retrain.
> 7. **Step 6 — temporal on v4b**, only after 3 and 5, and only because check 8
>    passed. Line D closed on a rendered corpus; 12 real video scenes with real
>    parallax is the one thing STATUS said would reopen it. Kill it if the first
>    three evals do not beat the step-0 baseline.
> 8. **Promote or drop.** A v4b `best.pt` that passes step 5 is copied to
>    `checkpoints/` under a release name with `SHA256SUMS` and `models.json`
>    updated; `sdr2hdr_temporal_v1.pt` ("Unevaluated") and v6 (4× capacity, no
>    gain) get measured or pulled from `models.json` in the same commit.
> 9. **Sparks scale — cause found, fix in, corpus not yet rebuilt (23 Sep).** The
>    Sparks download is ACES EXRs under `netflix_sparks/`; `scan_sources.py`
>    matched the PQ keyword "netflix" before looking at the container, so
>    scene-linear values were decoded as PQ codes and 1,799 pairs "peaked below
>    1 nit". A float container is now linear regardless of dataset name (camera
>    log names still win); `tests/…_2026_09_22.py` pins it. Verify on the box:
>    `python pipeline\scan_sources.py G:\datasets\sources\netflix_sparks --out
>    reports\logs\sparks_inventory.jsonl` and read `encoding_guess` and
>    `peak_nits_estimate` (expect thousands of nits). Then a **corpus_v4c**
>    re-ingest with Sparks in, before the `rudra-studio` (no-HdM) retrain. Not
>    before this run's acceptance: one variable at a time.
> 10. **`codex/rudra-final-release` merged to main on GitHub (PR #1, `205b19b`).**
>     The review (`reports/CODEX_BRANCH_REVIEW_2026-09-23.md`, local) still
>     stands as a to-do list against what is now in main: `rudra/video.py`
>     duplicates `rudra deliver` and converts input to Rec.2020 before the
>     network; `quality_benchmark.py` / `recovery_ablation.py` /
>     `assess_finetune.py` score the analytic baseline at the legacy −1 EV;
>     `master_targets` writes EXRs to any absolute folder a request names.
>     **Fixed on integration (23 Sep):** the merge (`869ecfa`) left
>     `_render_master` resizing an undefined PIL `image` around main's
>     full-depth `decode_sdr`, so every Studio master raised
>     `UnboundLocalError`; it now uses `_fit(decoded.rgb, master_max_side)`
>     (0 = full resolution). Suite on the merged tree: 498 passed on
>     FFmpeg 8.0.1. `rudra/video.py`'s QC needs ffprobe ≥ 5
>     (`frame_side_data`).
>
> **The paper ([`paper/main.pdf`](paper/main.pdf)) is about line C.** It is not
> the earlier manuscript, which was about line B; what was withdrawn from that
> one and why is Appendix D of the paper. Line B's
> completion path is unchanged and is listed below; nothing since 22 Aug has
> advanced it.
>
> **Line C, as of 1 Sep 2026:** v5 (1,196,197 parameters) benchmarked on 429
> held-out frames — **+1.43 dB PU21-PSNR and +0.44 JOD on degraded input, 348 of
> 429 frames**; **-3.0 dB / -0.046 JOD on clean input**. Failure mode located
> (error correlates +0.46 with scene headroom), oracle bound measured (+5.84 dB
> clean), and that bound shown to be mostly unreachable from an 8-bit input
> (features explain 3% of the oracle scale's variance; 79% of its variance is
> within-condition). Capacity (4x), corpus (6x) and objective were each varied;
> none moved the bound. Three evaluation-harness defects found and fixed:
> selection on a noisy maximum, an eval reading the alphabetical front of the
> split, and a viewer presenting every frame upside down.
>
> **Line C, done since:** ExpandNet run from the authors' released weights on
> the same 429 frames (−18.56 dB, −2.03 JOD, 1 frame won on PU21 and 16 on
> CVVDP); the shadow gate scored on three seeds (+0.41 ± 0.33 dB clean,
> +1.12 ± 0.15 dB hard); `step_0072000.pt` scored, confirming the selection fix
> picks it and that it is better on the criterion the selector optimises and
> worse on CVVDP in both conditions; the LaTeX build, the arXiv package and the
> committed PDF. Related work cited (9 references + 4 standards); no `[CITE]`
> markers remain.
>
> **Line C, 16 Sep 2026 review (engineer / colourist / researcher; the full
> text is local, `reports/FINAL_REVIEW_2026-09-16.md`).** Fixed the same day, each
> with a test in `tests/test_review_fixes_2026_09_16.py`:
>
> - `rudra deliver` encoded **BT.601 chroma under a bt2020nc tag** — swscale's
>   default matrix; `-colorspace` only labels. Reproduced on ffmpeg 6.1 (red at
>   PQ' 0.75 → Y' 1044 where BT.2020 is 946). Fixed with
>   `-vf scale=out_color_matrix=bt2020nc`. **Anything delivered before this
>   date should be re-encoded.**
> - HLG export skipped the inverse OOTF (diffuse white −0.46 st, 18% grey −0.96 st).
> - `deliver` clipped at 10,000 nits and wrote MaxCLL from those pixels, so SDR
>   white (2,552 nits from the baseline) exceeded a 1,000-nit MaxMDL. It now
>   shoulders into the declared peak (hue-preserving) and measures after.
> - Rec.709 masters were labelled Rec.2020 everywhere: the Studio's ACES path fed
>   709 primaries to the 2020→AP0 matrix, the linear EXR carried no
>   chromaticities, and both CLI defaults were `rec2020`. Default is `rec709`
>   now; the linear container converts and says so in its header.
> - The Studio bound 0.0.0.0 and would `torch.load(weights_only=False)` any path
>   a request named. Loopback by default (`--host` to widen), checkpoint
>   overrides limited to the registry and `RUDRA_CHECKPOINT_ROOTS`, 256 MB body
>   cap.
> - `infer_sdr2hdr.py` ran the shadow gate per tile (seams in tiled masters) and,
>   untiled, from full-resolution features it was never trained on; the paper's
>   bench went through this path. One whole-frame weight now, as the Studio
>   does. **The bench should be re-run**; `--recovery-mode` defaults to `all`,
>   which is what was scored.
> - `tonemap_ev` was written by the corpus builder and read by nothing, so the
>   next training run would have built the model against the legacy −1 EV
>   baseline over a 0 EV corpus. It now travels manifest → `corpus_ev` in the
>   checkpoint config → frame header → compositor uniform. Phase 0 of the
>   runbook is closed.
> - Playback ran at 24 fps for every shot (`fps` was never sent); `theme.css`
>   and `shell.js` were not cache-stamped; `paper/mdtotex.py` failed the
>   fresh-clone test and is gone.
>
> **Line C, 18 Sep 2026 — the Studio is an instrument, not a dark web app.**
> Neutral surround (every grey R = G = B; a tinted one biases colour
> judgement), the titlebar folded into the menubar, the scopes moved out from
> under the viewer into a full-height right-rail column and joined by a
> vectorscope, a permanent probe readout in the left rail with the largest
> figures on the page, a measured `Frame` block that reports the share of
> pixels the SDR actually clipped beside what the network chose to act on,
> timecode on the transport, and a colour pipeline bar across the bottom that
> names all four transforms and warns when MaxCLL is over the view peak.
> Every existing element id was kept, so `compositor.js` is untouched.
> `tests/ui_smoke/press_everything.py` now runs 72 checks, 0 failed, console
> clean — eight of them new, and two assertions fixed that had been stale
> since the pannable viewer landed (zoom stopped being a class on the viewer)
> and were failing before this change too.
>
> **Still open from the review, in order:** the synthetic-clip protocol
> (`measure_clipping.py --score` on 0/+1/+2 EV re-renders — the only measurement
> of the highlight claim); one out-of-generator degradation beside "hard"
> (**code now in `export_bench_pairs.py --condition out-of-generator`** — a Hable
> curve + real H.264; the measurement is still to run);
> per-shot smoothing of the gate, anchor and chroma scalars; a BT.1886 input
> option and a conforming baseline (a retrain — belongs with the corpus
> programme); an ExpandNet row on the hard condition; §4 provenance and the
> weights licence in the paper.
>
> **Line C remaining:** nothing measurable. The blockers are the arXiv
> endorsement (a person has to say yes), the HuggingFace upload, and the
> weights licence below. Section 10 still declares two gaps honestly: three
> published methods have runnable code and have not been run, and §6's
> trimmed-PSNR table and §7's oracle sweep have no script because they need
> per-pixel statistics the benchmark does not write.
>
> ---
>
> **Line D — v02, the temporal track (opened 4 Sep 2026).** The v01 video corpus
> is 935 clips across **13 scenes**, which is why the temporal refiner is
> reported as unevaluated. `pipeline/render_hdri_moves.py` flies a virtual
> camera through the CC0 Poly Haven panoramas, one panorama being one scene:
> the corpus is now **993 scenes, 17,874 frames, 128 GB**.
>
> Three things were measured before any model was trained, in the shape of §7:
>
> - `TemporalHDRRefiner`'s receptive field is **±4 frames, ±4 pixels**, measured
>   by gradient, against **4.3–27.3 px/frame** of camera motion. It cannot fetch
>   a value from where the content was; it can only smooth. Retraining it is the
>   control, not a candidate.
> - Scoring per frame is **blind to flicker by construction**. Two clips with an
>   identical 0.0600 relative error on every frame score 10.000 and 10.000
>   per-frame, and 10.000 and 5.111 as clips. `hdr_vdp3_clip_jod` runs
>   ColorVideoVDP in video mode; `rudra/pose_warp.py` adds a temporal
>   consistency measure with no flow estimator in it, ground truth against
>   itself sitting at 0.0145 stops.
> - **The gate.** `training/gate_temporal_oracle.py`. On clean frames the
>   per-frame model already scores 9.972 JOD of 10, so nothing can be won and
>   the run says nothing; the gate has to be read on `--condition hard`.
>
> **The gate failed, and then said why (5 Sep 2026).** The 4 Sep reading of
> **+9.31 JOD achievable** does not survive. Two faults produced it, and both
> are fixed:
>
> - It scored a **bare `model(x)` forward pass**, not RUDRA as deployed. The
>   benchmark runs `predict_image(preserve_outside=True)`, where outside the
>   learned masks the output *is* the analytic baseline. That is why the floor
>   sat at −1.96 JOD against the benchmark's 7.805 on the same checkpoint and
>   the same 1280×720 frames — it was never a crop-size or resolution
>   mismatch.
> - `degrade_like_eval` **reseeds from the frame index**, so every frame of a
>   nine-frame clip got its own exposure, tone curve, white balance,
>   saturation, chroma subsampling, bit depth and JPEG quality. Averaging nine
>   independent draws of a corruption is worth √9 whether or not the frames
>   carry information. `--degradation {per-frame,per-clip,realistic,codec}`
>   makes that assumption a flag; `codec` puts the clip through a real H.264
>   round trip and is the one with a GOP in it.
>
> | degradation | clips | per-frame | achievable | ceiling |
> |---|---|---|---|---|
> | `per-frame` (4 Sep) | 2 | −2.396 JOD | **+9.397** | +9.408 |
> | `realistic` | 40 | 6.961 | **+0.033** | +0.191 |
> | `codec` | 40 | 6.864 | **+0.034** | +0.165 |
>
> Against a +0.5 JOD threshold, **the gate fails** — and even the unreachable
> bound (omniscient per-pixel selection over exactly aligned neighbours)
> reaches only +0.19.
>
> **That is a fact about the corpus, not about video.** `make_sdr` tone-maps
> every frame with one fixed curve and one fixed EV offset, and the renderer's
> camera only rotates through a static panorama — so a scene point carries the
> **same SDR code in every frame it appears in**, and what is blown in frame 3
> is blown in frame 7. The corpus contains none of the information v02's claim
> is about. `--exposure-drift` and `--exposure-jitter` put that axis back
> (defaults 0.0; the `_ingest_config` sentinel refuses to mix a drifted render
> into the existing 993 scenes).
>
> **The gate passes once exposure moves (6 Sep 2026).** 50 panoramas rendered
> twice with `PYTHONHASHSEED=0` — identical camera paths (checked to 1e-9),
> identical H.264 degradation, the SDR exposure the only difference. Scene
> list in `_gate50_scenes.txt`.
>
> | exposure | per-frame | achievable | ceiling |
> |---|---|---|---|
> | constant | 6.209 JOD | +0.110 | +0.264 JOD / +0.632 dB |
> | ±0.48 stops | 5.786 | **+0.511** | +0.851 JOD / **+4.554 dB** |
>
> **+0.511 against a +0.5 threshold, with the fixed-exposure control at +0.110
> on the very same scenes and camera paths.** The PU21 ceiling is the clearer
> signal: +0.63 dB without exposure movement, +4.55 dB with it.
>
> **Repeated on three seeds (6 Sep 2026, `training/run_drift_gate.ps1`, on the
> 4080):**
>
> | seed | flat | drift | drift ceiling | clips |
> |---|---|---|---|---|
> | 20260903 | +0.110 | +0.511 | +0.851 | 50 |
> | 20260906 | +0.035 | **+0.603** | +1.090 | 40 |
> | 20260907 | +0.046 | **+0.661** | +1.142 | 40 |
> | **mean** | **+0.064** | **+0.592** | +1.028 | |
>
> **The result holds.** Every drift arm clears the threshold; every flat arm
> is an order of magnitude below it. The ~+0.53 JOD separation between arms is
> far larger than the spread across seeds, which is what makes this a result
> rather than a run. The hypothesis behaves exactly as stated — a neighbour is
> worth something when it saw the scene at a different exposure and close to
> nothing when it did not.
>
> Two things travel with it. The 40 in the later rows is a defect:
> `stage_oracle_clips` defaulted `--count` to 40 and the sweep passed none, so
> those seeds used the first 40 of the 50 scenes — the *same* 40 in both arms,
> so each pairing is intact, and the default is now 0. And drift makes the
> per-frame job harder (6.209 → 5.786 on seed 20260903), so part of the gain
> is repairing damage the drift itself did; the oracle ceiling separates the
> two, and +0.26 JOD flat against +0.85 drift is information that exists to be
> fetched, not merely damage to undo.
>
> What it licenses is narrow: a proposition about footage whose exposure
> breathes — handheld, documentary, anything riding auto-exposure — not about
> video in general.
>
> **And then estimated alignment took it away (6 Sep 2026).** Every number
> above used the camera angles the renderer wrote down. A plate has none, so
> the same 50 drifted clips were re-scored with the correspondence estimated
> by DIS optical flow from the degraded SDR — what a deployed model holds:
>
> | arm | per-frame | control | aligned | oracle | **achievable** | ceiling |
> |---|---|---|---|---|---|---|
> | pose (exact) | 5.786 | 5.835 | 6.296 | 6.687 | **+0.511** | +0.851 |
> | flow (DIS) | 5.786 | 5.844 | 5.802 | 5.839 | **+0.016** | −0.005 |
> | flow + texture gate | 5.786 | 5.828 | 5.658 | 5.514 | **−0.128** | −0.314 |
>
> **Read the ceiling column.** Under estimated alignment the *oracle* — ground
> truth consulted per pixel, better than any architecture or weighting could
> ever do — beats the per-frame model by +0.053 JOD, while the control, which
> carries no information at all, beats it by +0.058. The oracle's whole margin
> is selection on resampling noise. **There is no headroom left to build for.**
>
> The alignment is not visibly bad, which is what makes this worth stating
> carefully: DIS matched the analytic poses to a median 0.04 px at one frame
> of separation and 0.09 px at eight, coverage within 1%, warped frames
> agreeing to under half an 8-bit code value. PU21-PSNR keeps 75% of the gain
> (+0.326 of +0.435 dB). The JOD keeps 3%. What survives a per-frame PSNR and
> not a video metric is a small, spatially coherent error that changes every
> frame — the exact artefact a temporal model exists to remove.
>
> Where it fails is legible: the six worst clips are open sky, the four best
> interiors. On the sky scene flow error in low-texture regions is **1.35 px
> against 0.15 px** in textured regions of the same frame, and
> forward-backward consistency passed **57%** of those bad pixels, because in
> a flat region any displacement round-trips perfectly. Flow fails exactly
> where the highlights are. Gating on texture energy made it worse (−0.128):
> a hard mask leaves nine-frame averaging beside one-frame averaging with a
> seam between them. A softer weighting is not worth trying — the oracle row
> bounds every weighting there is.
>
> **The learned estimator, and the close (10 Sep 2026).** DIS is a fast
> classical estimator, so the flow arm was re-run with RAFT-large — markedly
> better in exactly the low-texture regions where DIS failed, 0.35 px against
> 2.61 px on an open-sky clip. Seed 20260906, 40 drifted clips, same clips
> every row:
>
> | arm | per-frame | aligned | oracle | **achievable** | ceiling |
> |---|---|---|---|---|---|
> | pose (exact) | 5.925 | 6.528 | 7.063 | **+0.603** | +1.090 |
> | flow (DIS) | 5.925 | 5.856 | 6.115 | **−0.069** | +0.122 |
> | flow (RAFT) | 5.925 | 6.266 | 6.659 | **+0.341** | +0.671 |
>
> RAFT recovers 57% of what exact poses give and **misses the threshold** —
> +0.341 against the +0.5 set before any of this was measured. Not the flat
> zero DIS gave: the ceiling moved from +0.122 to +0.671, so there was room a
> better combiner might have reached.
>
> One combiner was declared and tried, *before* it was run — weight each
> neighbour by its forward-backward residual instead of averaging equally.
> On the same clips: **mean +0.351, confidence +0.350.** Nothing.
>
> The reason is the same wall from a third angle. RAFT's forward-backward
> drift is 0.04–0.09 px on nearly every pixel that passes, so the weight is ~1
> everywhere and the signal has no dynamic range. **The failures are not
> low-confidence matches — they are confident wrong ones**, in flat regions
> where any displacement round-trips perfectly. That is precisely what
> forward-backward consistency cannot see, and therefore what weighting by it
> cannot fix.
>
> **Line D is closed.** Three alignment arms and two combiners against a
> threshold fixed in advance: exact camera poses clear it, nothing a plate can
> supply does. The +0.603 belongs to the renderer's angles. No temporal model
> was trained because there is nothing measurable for one to learn — **that is
> the result, not the absence of one**, and it is worth more than the month it
> would have taken to find out the other way.
>
> The corpus re-render, the architecture ladder and the video split are off
> the board. What would reopen it is a corpus whose neighbouring frames carry
> information these do not — real parallax, moving subjects, genuine
> multi-exposure capture — not a better estimator and not a better
> architecture. The gate would have to be re-run from scratch on it.
>
> **The shadow gate is settled, and it stays (10 Sep 2026).** It was proposed
> for retirement on the strength of three checkpoints agreeing to 0.07 nits on
> ONE frame. `training/compare_bench_methods.py` answers it properly from the
> 429 held-out pairs already scored under `bench/results/` -- paired per frame,
> bootstrap CI, and measured against the spread between the three gate SEEDS,
> because an effect smaller than retraining noise is not an effect:
>
> | condition | metric | gate effect | worst seed-to-seed | verdict |
> |---|---|---:|---:|---|
> | hard | CVVDP | **+0.218 JOD** | 0.081 | 2.7x seed noise |
> | hard | PU21 | **+0.796 dB** | 0.278 | 2.9x |
> | clean | CVVDP | **+0.107 JOD** | 0.045 | 2.4x |
> | clean | PU21 | −0.103 dB | 0.645 | within noise |
>
> All three seeds clear zero with 95% CIs excluding it on three of four
> measures. The gate stays; the one measure it does not help, it does not hurt
> beyond noise. No GPU time was spent -- the files were already on disk.
>
> **Still open, neither a training run:** `sdr2hdr_temporal_v1.pt` ships in
> `models.json` marked *"Unevaluated"* and belongs to the closed line — pull
> it or measure it. And v6 (4× capacity, moved neither condition) is a keep
> or drop.
>
> ---
>
> **Licensing (5 Sep 2026).** The code is Apache 2.0. The **weights are not**:
> HdM-HDR-2014 and HdM-HFR-2017 are **75.6% of the training corpus** and are
> free for academic use only, with commercial use requiring a separate
> agreement with HdM Stuttgart. `checkpoints/LICENSE` now licenses the weights
> non-commercially and `NOTICE` carries the Netflix Chimera CC BY 4.0
> attribution. Academic use is permitted, so **the paper is unaffected**.
>
> The intended resolution is two model families: `rudra-research` as it stands,
> and a `rudra-studio` retrained without HdM — Netflix Chimera (14.3%, CC BY,
> median peak 7,094 nits) plus Poly Haven (10.1%, CC0) plus the 17,874 rendered
> frames plus FXTD's own footage, which is comparable corpus volume with the
> high-nit end covered. That retrain depends on nobody's permission.

---

Snapshot of what is trained, with measured values, and what remains for the paper.

## Production decoders (ComfyUI node) — ✅ complete

Distilled `RadianceTurbo/FullDecoder` (`fast_vae.py`), trained by
`training/train_turbo_decoder.py`. Metric: log-space PSNR on held-out pairs.

| Backbone | Decoder used | PSNR_log | peak step | pairs |
|---|---|---|---|---|
| Flux.1 | **full** | 29.77 | 26k | 12,281 |
| Wan | **full** | 32.45 | 16k | 13,227 |
| LTX (2.3, **32× fixed**) | **full** | 25.47 | 40k | 13,227 |
| SDXL | **turbo** | 33.86 | 18k | 963 |
| Qwen-Image | **turbo** | 26.67 | 20k | 963 |
| Flux.2 Klein (128ch/16×) | **turbo** | 28.57 | 18k | 963 |
| Z-Image | use Flux decoder | — | — | (Flux VAE) |

Turbo also exists for flux/wan/ltx/sdxl; the table lists the *best* per backbone.
Notes: SDXL/Qwen/Klein are on 963 pairs (data-starved) — regenerate multi-crop pairs to
lift them. Full is slow on 64×64-latent backbones (decodes at 512×512). Always deploy the
`*_decoder_ema_best.safetensors`.

## RUDRA research pipeline (the paper) — partial

`rudra/` package + `training/train_rudra.py`, Stages 1–3.

| Paper stage | Status | Measured value | Blocker / next |
|---|---|---|---|
| **Stage 1** — RUDRA-Lite descriptor + FiLM decoder (§3.6) | ✅ flux/wan/ltx @ 50k | ColorVideoVDP **JOD 9.0–9.2**, PSNR_tm ~25–26, EV-error ~0.05 stops, HRA ~0.18 | done |
| **Stage 2** — DR-gated LoRA (RUDRA-Lite conditioning) | ⚠️ partial (400 steps) | — | Flux-16GB OOM; runs on SDXL or 256px pairs |
| **Stage 3** — DRE transformer + cross-attention injection (§3.4–3.5, *core thesis*) | ❌ not trained (dryrun only) | — | run on **SDXL** (`research_sdxl.py --phase stage3`) |
| §7.1 component ablation | ❌ TBD | — | derived from Stage 1→3 runs |
| §7.2 descriptor channel ablation (L / L+E / L+H / L+xy / full R⁵) | ❌ TBD | — | `research_sdxl.py --phase ablate` |
| §7.3 λ conditioning-strength sweep (0.0 … 1.25) | ❌ TBD | — | `research_sdxl.py --phase sweep` |
| §6 results vs baselines (incl. LTX IC-LoRA-HDR) | ❌ TBD | — | `benchmark_hdr.py` + ColorVideoVDP |
| §8 qualitative grids / EV-over-time | ❌ TBD | — | after the above |

## Completion path

1. **Stage 3 on SDXL** — the paper's intended backbone (U-Net cross-attention; ~5× lighter
   than Flux, fits 16 GB). This is the core contribution. `research_sdxl.py --phase stage3`.
2. **λ sweep + descriptor ablation** — `--phase sweep` and `--phase ablate` (short runs).
3. **Benchmark** RUDRA vs LTX IC-LoRA-HDR with ColorVideoVDP (JOD), ΔE2000, EV-error, HRA.
4. **Fill the paper tables** (§6/§7) with the measured values; drop the placeholder numbers.
5. *(Optional)* Flux port of the conditioning — the paper's §10 generalization experiment.

## Code health

All P0/P1/P2 review items closed. Real HDR metric is
ColorVideoVDP (`rudra/hdrvdp.py`); the old hand-rolled "HDR-VDP-3 ≈ 80" numbers were a
placeholder and must not be reported. `pytest tests/` covers curve round-trips, the freeze
guarantee, conditioning, and the metric backend.
