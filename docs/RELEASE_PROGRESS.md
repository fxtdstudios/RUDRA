# Rudra completion status — 2026-09-25

## Verified results

- Windows Studio 0.3.3rc1 preview release checks passed September 27. Kept shipped
  Shadow v1 SHA256 6b7f73f0b44a6edc3e117373f7397ae3a8df12cb398ec4815597b6e2ebeb63f9;
  no experimental weight bundled. 450 source tests passed (four warnings).
  Built wheel, verified source parity, installed dependencies in fresh isolated
  Python 3.13 environments, and ran the ZIP's actual Install/Start CMD scripts.
  GPU model loaded on RTX 4080 SUPER; both installations passed 12 image/sequence
  renders across six presets, 16-bit frame transport, EXR/ICC metadata and overwrite
  rejection. pip check passed. Browser smoke passed with no console errors.
  Server now binds loopback only. ZIP internal hashes verified. Evidence:
  outputs/windows_preview_verification and outputs/windows_preview_installer_verification;
  browser image outputs/windows_preview_build/installed-studio.png.
  Separate Python/browser Studio preview, not native v0.9.0 beta replacement.
  No clean-OS, calibrated-display, macOS/Linux or temporal quality certification.
  Non-commercial weights license included. User approved prerelease publication.

- September 27 reference-bank feasibility complete: 24 proposals, 12 distinct
  training bank scenes and eight separate diagnostic scenes, excluding earlier
  guard/gradient scenes. All 24 updates rejected; 8/8 diagnostics preserved only
  because model did not change. Feasibility FAILED (no useful learning). Five
  focused tests passed; no long run or promotion. Evidence outputs/reference_bank_20260927.
  Report docs/REFERENCE_BANK_FEASIBILITY_20260927.md. Next architectural hypothesis
  is a separate correction branch with shipped reconstruction frozen; routing risk
  must be measured and is not solved by a bypass alone. Not yet implemented.

- September 27 guarded-update feasibility completed: 24 proposals on distinct
  training scenes, 10 accepted and 14 rolled back with model/optimizer restored.
  Only 2/8 separate training diagnostic scenes preserved all clean/degraded
  surrogate components, so feasibility FAILED. No longer training run or promotion.
  Six focused tests passed. Main trainer remains unchanged; experimental checkpoint
  and protocol live in outputs/guard_feasibility_20260927. Report:
  docs/GUARDED_UPDATE_FEASIBILITY_20260927.md. Next design needs a diverse training
  reference bank and separate diagnostics, not a current-batch-only guard.

- September 27 training-only gradient audit complete: 24 scenes at two checkpoints
  (48 measurements, zero updates). Shipped model: clean/degraded gradients conflict
  on 10/24; combined region loss opposes clean improvement on 6/24; degraded-region
  loss dominates 24/24. Rejected candidate: corresponding counts 6/24, 2/24, 24/24.
  These are local raw-gradient findings, not AdamW trajectory or quality guarantees.
  Five focused tests passed. No new training run or promotion. Next: short actual-
  optimizer-update guard feasibility test using training data only. Report:
  docs/GRADIENT_CONFLICT_AUDIT_20260927.md; evidence outputs/gradient_conflict_20260927.

- REGION EXPERIMENT COMPLETE AND REJECTED: outputs/region_preservation_20260927
  finished 600 steps and 204/204 comparisons. Clean deltas: PU21 -1.792811 dB,
  JOD -0.100944. Degraded deltas: PU21 +0.279202 dB, JOD +0.065059. The objective
  improved degraded means but failed clean preservation, so no promotion or repeat
  sweep. These are reused validation findings only. Diagnosis saved in evaluation/
  diagnosis.json. Shipped Shadow v1 remains active. Full software suite passed
  443 tests (four dependency warnings, 37.84 seconds); software correctness is not
  model quality. Training-only gradient conflict review and verified independent
  data remain next prerequisites. No training process remains active from this run.

- September 27 targeted model experiment launched at outputs/region_preservation_20260927
  after reviewing two worst failures with fixed SDR mapping. Highlight PU error rose
  10.9% in the clean studio and 70.8% in degraded forest while shadows improved.
  Added opt-in region-pu objective with per-image/region teacher harm, worst-condition
  PU risk and existing censored-log protection. Six focused tests pass. Fixed 600
  steps from shipped weights, then automatic 204-comparison reused validation.
  No automatic promotion; reserved source sequences remain outside training.
  Protocol: docs/REGION_PRESERVATION_EXPERIMENT_20260927.md. Check this run's
  training and evaluation status files before starting any further process.

- September 27 model upgrade audit: read both backbone evaluations, confirmed the
  latest clean JOD regression and generated a 20-entry diagnostic review list.
  Audited 32,718 pairs (802/102/102 train/val/test scenes), no missing referenced
  files or exact scene-ID split overlap. This is not content duplicate verification.
- Found four Stuttgart source sequences absent from current/original manifests;
  all 3,999 inventoried frames exist. Saved a hashed provisional holdout manifest
  with training prohibited, pending prior-use/license/color verification. NOT an
  independent evaluation set yet. User clarification requested; no training started.
  Report: docs/MODEL_UPGRADE_AUDIT_20260927.md. Evidence:
  outputs/model_upgrade_audit_20260927/{audit,source_candidates,holdout_status}.json.

- September 27 precision input follow-up supersedes the 8-bit-only limitation
  below: unsigned 16-bit grayscale/RGB PNG/TIFF now decode through OpenCV and
  retain float precision through resizing and all three inference/export paths.
  Live frame transport includes float32 SDR for 16-bit sources; WebGL uses RGB32F
  so the analytic baseline does not use an 8-bit approximation. Automatic embedded
  ICC conversion for 16-bit remains unsupported and requires explicit manual input
  interpretation; signed/float TIFF and unsupported photometric layouts are rejected.
- Full suite: 440 passed, four dependency warnings, 32.93 seconds. Tests verify
  adjacent 16-bit values reach the real model, frame payload, and master path.
  Both modified JavaScript files pass Node syntax checks. GPU Studio rendered a
  257x129 RGB16 test ramp with measured scopes and no browser console errors;
  screenshot outputs/precision_viewer_20260927.png. No calibrated display claim.

- September 27 final color/export regression: 436 tests passed, four dependency
  warnings, in 32.66 seconds (CPU, outputs/regression_final_20260927). Studio now
  rejects higher-bit-depth PNG/TIFF before Pillow can silently reduce precision;
  five new tests cover automatic/manual input and RGB16 PNG. UI and README state
  the current 8-bit input limitation. Precision-preserving input remains future work.
- Linear Rec.2020 EXR masters now carry chromaticities/white-point metadata,
  verified in real sequence renders alongside existing ACES metadata/pixel checks.
  No new model promoted, release tagged, cross-platform clean install or calibrated
  HDR display verification performed in this pass.

- Auto color setup added: default ACES Studio config loads at startup when OCIO
  is available. Browser-local settings persist config/mappings, export preset,
  render destination/name/range. Reload verification restored an OCIO view preset;
  restored normal ACES EXR preset after the test. Untagged bundled image displayed
  “No embedded ICC — assumed sRGB.” Screenshot: outputs/auto_color_20260927.png.
- Shared input decoder now honors embedded ICC via LittleCMS in Auto mode before
  inference, frame fields and export, bypassing a second OCIO input transform.
  Missing profiles are explicit assumptions; invalid profiles error with manual
  override guidance. Manual mode remains available and ignores embedded ICC.
  Input interpretation is recorded in sidecars and frame response metadata.
  28 focused tests passed, including real LAB ICC conversion, untagged fallback,
  invalid-profile handling and prior OCIO/preview/export checks. No untagged log
  detection, monitor profiling, or new reconstruction weights claimed.

- Optional OCIO support added September 27. Installed OpenColorIO 2.5.2 in the
  working Studio Python; packaging extra is `[ocio]`. Loads bundled ACES 1.3 Studio
  config or local .ocio path, with explicit input/model-sRGB/linear-Rec.2020 bridge
  selections, output space and display/view. Input conversion is used for inference,
  viewport fields and masters. OCIO output EXR is float32; view PNG is 8-bit with
  config identity/selection recorded in sidecar. No automatic ICC input detection.
- Export preview uses the same backend writer as PNG/TIFF delivery. Seven OCIO
  tests passed (including real EXR/PNG pixels, optional dependency absence, config
  change rejection and HTTP preview/export pixel identity), alongside 18 existing
  export tests. GPU Studio browser preview confirmed the default sRGB ACES view;
  screenshot `outputs/ocio_preview_20260927.png`. Main HDR viewer/scopes retain their
  original transform; OCIO preview is separate. P3/PQ/HLG browser appearance and
  native HDR delivery are not certified. Custom bridge mappings are user supplied.

- September 27 export update: save-folder browser and output preset selector added.
  Existing ACES AP0 / Rec.2020 EXR paths retained; new 8-bit sRGB PNG/TIFF use an
  explicit fixed Reinhard luminance transform, Rec.2020-to-sRGB conversion and ICC
  embedding. Sequence extensions and overwrite checks use the selected format.
  18 focused tests passed including real-model SDR exports. Browser selected sRGB
  and navigated/confirmed the output directory. Screenshot: outputs/export_controls_20260927.png.
  Input remains assumed sRGB; preview remains HDR (clearly labeled); P3, ACEScg,
  input ICC override/detection and export soft-proof are not included in this step.

- ROBUST BACKBONE RUN COMPLETE: 600/600 training steps and 204/204 validation
  comparisons. Candidate failed the unchanged four-check criterion: clean PU21
  +0.16550 dB, clean JOD -0.005223; degraded PU21 +0.04231 dB, degraded JOD
  +0.000325. These are reused validation averages, not independent confirmation.
  Three positive means do not justify promotion when the fourth regresses.
- Diagnosis saved to `outputs/robust_backbone_20260925/evaluation/diagnosis.json`.
  Clean JOD regresses on 52/102 frames (some near floating-point equality); worst
  wooden_studio_13 loses 0.26336 JOD. Degraded JOD regresses on 37/102; worst
  nature_reserve_forest loses 1.02724 JOD and 1.14314 dB PU21. Descriptive bootstrap
  intervals include zero for both degraded metric averages and clean JOD; no
  reliable general improvement claim follows. The log-error training surrogate
  did not ensure perceptual preservation. No further threshold/epoch sweep started.
- Shipped Shadow v1 remains active; candidate is experimental and non-commercial.
  Release still needs unused licensed paired SDR/HDR scenes and consecutive
  sequences with provenance, plus clean-machine and calibrated-display review.
  Pausing recurring follow-ups after this completed run and diagnosis, pending
  user-supplied independent data. No commit, push, tag or model promotion performed.

- USER AUTHORIZED NEXT MODEL WORK. New active experiment:
  `outputs/robust_backbone_20260925`. This fine-tunes reconstruction weights from
  shipped Shadow v1, unlike the earlier recovery selectors. Frame gates remain
  frozen because random crops do not represent whole-frame gate statistics.
  Fixed 600 steps, lr 2e-6, 256px crops, two scenes/batch sampled with equal scene
  weight; paired clean/degraded inputs every step. Objective is censored log error
  plus 4x per-image regression penalty relative to a frozen shipped teacher.
  This surrogate cannot guarantee PU21/JOD improvement. Three objective/split tests
  passed and a separate two-step CUDA smoke run completed with finite losses.
- After training, the same process automatically evaluates its fixed final
  checkpoint on the 102 reused validation scenes, both clean and hard, using real
  CVVDP and PU21 at native resolution, 512/64 tiled inference. Shipped measurements
  are reused with checkpoint/source hashes checked. No test images, validation
  threshold tuning, Studio promotion or release publication is authorized by a
  training-loss improvement. Check both training status and `evaluation/status.json`.
- Independent paired data, temporal review and clean-machine verification are still
  missing release evidence. Existing shipped weights stay active. Recurring checks
  resume for this new run; do not start duplicate training or another sweep.

- EXPANDED EXPERIMENT COMPLETE: all 1,808 measurements and 900 epochs finished;
  **0 of 90 candidates passed all four checks**. No model was promoted. CPU review
  and checkpoint replay diagnosis completed in `quality_policy_expanded_20260925`.
  The diagnostic best-mean-JOD candidate (seed 20260925, epoch 300) changes clean
  validation by +2.02244 dB PU21 / +0.05482 JOD, but degraded validation by
  -0.16074 dB / -0.02766 JOD. On degraded validation, 37/102 frames regress PU21
  and 26/102 regress JOD. Degraded training also regresses (-0.06936 dB,
  -0.02525 JOD), so expanding scenes did not resolve the selector's failure.
  These are reused exploratory validation results, not independent evaluation.
- No further sweep launched: the selector still chooses highlights too often on
  degraded inputs (51/102 validation frames). Even a hindsight PU21-optimal mode
  selector loses mean JOD on degraded data, showing metric disagreement. Next model
  design must address degradation robustness and both objectives, not relax gates.
- User input needed for release evidence: provide a local path to unused, licensed
  paired SDR/HDR scenes and real consecutive-frame sequences, with provenance and
  confirmation they were not used in training/selection. No such independent set
  has been identified. Existing test data was used in earlier model work and must
  not be relabeled independent. Clean-machine and calibrated-display checks remain
  unperformed. Recurring follow-up is being paused pending that input; no release
  commit, push or tag has been made. Non-commercial weights remain unchanged.

- Rebuilt verification wheel with the latest scope fixes and installed it into an
  isolated target directory. Confirmed Studio, inference and Rudra modules import
  from the installed package, all six referenced local UI assets exist, and the
  packaged app includes corrected training wording and PQ vectorscope math.
  Installed Studio command help also ran. Evidence:
  `outputs/package_scope_check_20260925/verification.json`; wheel SHA-256
  `ed82f3fd507e321765f062ed9a28946678ec12a00ecb8c7ce098cb85b3f3362c`.
  This reuses existing dependencies; it is not clean-machine verification. Version
  0.3.2 is a local verification build, not a newly published or tagged release.
- Expanded experiment observed at 1,468/1,808 measurement comparisons, still running.
  No duplicate run started and shipped weights remain active. Candidate evaluation,
  independent data, temporal review and clean-install checks still gate release.

- Full project suite after scope and model-experiment additions: **412 passed,
  4 warnings in 42.22s**. Tests ran with CUDA hidden to avoid competing with the
  active GPU experiment; this does not constitute a full GPU suite. JavaScript
  syntax check also passed. Browser reload confirmed the corrected historical
  assessment message. Latest experiment observation: 1,306/1,808 comparisons.

- Browser visual check passed for the corrected vectorscope on the bundled sunset
  in a separate tab at 1280x720: waveform, histogram, six chroma targets and trace
  are visible; clamped/invalid counts both zero. Shipped model ran on RTX 4080 SUPER.
  Existing user tab preserved. This is ordinary browser QA, not HDR display calibration.
- Corrected misleading startup log: the September 20 archived candidate assessment
  is now explicitly historical, not a claim that current training is complete.
- Expanded run has reached 1,298/1,808 comparisons; still measuring, no new model
  selected or promoted. Independent evaluation data and real temporal review remain
  release requirements after a candidate passes the existing quality checks.

- Vectorscope correction: absolute reconstructed Rec.2020 RGB is now PQ encoded
  before BT.2100 NCL Cb/Cr calculation (BT.2100-3 Tables 4 and 6). Both axes use
  equal scale; 75% signal targets are calculated from RGB instead of arbitrary
  positions. Removed circular rejection of valid saturated colors and artificial
  hue coloring. Every occupied bin is shown; clamped/invalid sample counts appear.
  This is a diagnostic mapping without display tone mapping, not a gamut legality
  or calibrated display certification. Five focused scope tests and JavaScript
  syntax check passed. Browser visual acceptance remains pending.
- Expanded experiment status at this check: 1,079/1,808 comparisons, still running.
  Shipped weights remain unchanged. No new candidate quality result is available.

- Browser sequence acceptance completed in a separate Studio tab: imported two
  synthetic SDR fixtures, chose an absolute render folder, sequence output and
  start frame 1001, then rendered ACES EXR through the actual Master EXR button.
  UI confirmed both saved files at native 64x48 and 80x64. A second click correctly
  stopped at 0/2 with overwrite refusal. The user's existing tab was preserved.
- Independent OpenEXR reader decoded both files with finite RGB pixels exactly
  matching the project reader; chromaticities were present and sidecars reported
  native dimensions, linear transfer and diffuse white 203 nits. Evidence is in
  `outputs/browser_sequence_check_20260925/verification.json`; masters are in its
  `render` subfolder. This verifies file-reader interoperability, not Resolve/Nuke
  application color management, formal ACES conformance or real-video flicker.

- Added real two-frame sequence export checks for both linear Rec.2020 and ACES
  AP0 using shipped weights on CPU. Verified numbered output paths, native dimensions,
  decoded half-float pixels against reconstruction at the requested recovery mode
  and strength, AP0 chromaticities, sidecar transfer/white/model/grade fields, and
  overwrite rejection. Together with exposure checks: **24 passed**. This covers
  backend reconstruction-to-file behavior on synthetic fixtures; it does not replace
  browser sequence interaction, external-reader acceptance, or temporal validation.
- Expanded GPU run remains active: 301/1,808 comparisons at this check.

- CURRENT ACTIVE RUN: `outputs/quality_policy_expanded_20260925`.
  Dataset inventory found 802 train scenes (31,794 frames), 102 validation scenes
  (465 frames), and 102 test scenes (459 frames). Expanded from 256 to all 802
  train scenes: 546 additional scenes. One deterministic frame per scene and
  clean/hard conditions produce 1,808 comparisons including the unchanged 102
  validation scenes. Preflight confirmed all selected SDR/HDR files exist and
  zero selected test rows. Joint PU21/JOD objective, three seeds, 300 epochs each.
  This expands an experimental recovery selector, NOT reconstruction backbone
  training. Existing validation is reused, not independent evidence. Monitor this
  run first; do not duplicate it. No candidate may replace shipped weights without
  the existing quality criteria and subsequent independent/temporal review.

- Gain-prediction experiment finished all 716 feature extractions. None of its
  15 penalty/margin combinations passed the reserved-training selection criterion;
  no candidate was selected and final validation scoring was not run. Do not promote
  or rerun this same sweep. The feature changes alone have not established an upgrade.
  Latest focused suite: 14 passed. Future model work needs a broader training/data
  design rather than further threshold sweeps on these same scenes. Both the
  regressor and selector paths remain experimental; no model training is active.

- New gain-prediction experiment launched in `outputs/quality_gain_20260925`.
  Follow-ups should check its status and error log first. `rudra/quality_gain.py`
  adds 20 native SDR descriptors (clipping, gradients, local residual, boundary
  differences and chroma statistics). `training/train_gain_policy.py` augments the
  saved encoder features, fits ridge regressors to both baseline-relative quality
  gains using fit-training scenes, and selects penalty/margin on reserved training
  scenes. Validation is consulted only after the selection is frozen, but remains
  reused exploratory data. Predicted gains are NOT guaranteed actual gains.
  Five focused feature/fallback/partition tests passed. This checkpoint has a new
  regression format and MUST NOT be passed to Studio or the old policy loader.
  The shipped model is unchanged. No new GPU reconstruction measurements are needed.

- Conservative failure inspection completed via `training/inspect_conservative_failures.py`.
  It verifies saved source hashes, replays the candidate's published validation
  scores, and writes `failure_cases.json`/`.md` beside the experiment. Of 11 changes
  from shipped recovery, six regress at least one metric. All five degraded-image
  switches regress; confidence ranges about 0.902–0.988. One clean scene loses
  9.588 dB PU21 despite a small JOD gain. Thus high softmax confidence is not a
  reliable measure of improvement, and average gains can hide large individual losses.
- Two paired scenes (montorfano and wooden_studio_11) switch to the same mode on
  clean and degraded inputs: helpful on clean, harmful on degraded. This motivates
  testing degradation-sensitive features on training scenes, plus per-scene
  regression limits and calibration of predicted quality gains. These are research
  directions, not proven remedies. Do not patch scene names or tune thresholds
  against these validation failures. Test pixels were not read; no weights changed.

- Conservative cached experiment COMPLETE in `outputs/quality_policy_conservative_20260925`.
  Three seeds, 300 epochs each; 205 training scenes fit weights and 51 reserved
  training scenes select checkpoint/threshold. Only then was the fixed candidate
  evaluated on reused validation. A 0.9 softmax threshold falls back to all-recovery
  for uncertain cases; it is not a statistical safety guarantee. Selected seed 2,
  epoch 300: clean deltas +0.10641 dB / +0.00421 JOD; degraded deltas -0.01522 dB /
  -0.00656 JOD. Retained all-recovery on 193/204 validation cases. Still FAILED;
  not promoted. 14 focused tests passed, including scene separation and saved
  checkpoint fallback behavior. Review/diagnosis now honor checkpoint thresholds.
- Avoid further threshold tuning on the same validation set. Next model research
  should examine failure cases and feature limitations, with a predeclared new
  independent evaluation set before making improvement claims. GUI/packaging work
  can continue with the shipped model. No training is currently running.

- Waveform now renders measured luminance-density bins rather than filled percentile
  envelopes. Zero-density gaps remain empty. An executable JS fixture checks sample
  counts, horizontal positions, bimodal gaps, and invalid-value exclusion. Four
  focused scope tests and JavaScript syntax check passed.
- Live GPU Studio verification at localhost:8431 with the bundled sunset sample:
  shipped shadow model loaded on RTX 4080 SUPER; 1024x1024 inference succeeded;
  density waveform and both diffuse-white/MaxCLL markers visibly rendered.
  Found and fixed narrow-window panel collapse using a two-row responsive grid;
  verified waveform, histogram, vectorscope panel and measurements visible at the
  browser's 1280x720 viewport. Measurements scroll when needed. This is UI verification,
  not calibrated HDR-display or vectorscope numerical verification. Server remains
  available at port 8431; launch logs are `outputs/studio_scope_check*.log`.

- Latest full suite after objective, numerical-scope and packaging changes:
  **403 passed, 4 warnings** in 32.08 seconds. No clean-machine claim.

- Joint experiment is COMPLETE: 716 measurements, 300 epochs, zero passing
  candidates. Review and diagnosis generated in `outputs/quality_policy_joint_20260925`.
  Diagnostic candidate seed 20260925 epoch 50 has clean deltas +1.23120 dB PU21 /
  +0.03366 JOD, but degraded deltas -0.17652 dB / -0.03036 JOD. This reduces the
  earlier regressions but still fails acceptance. No active model training or promotion.
- Packaging fix: wheel now contains `ui` and `training` packages, Studio browser
  assets including `scope_math.js`, and a `rudra-studio` entry point. Built a local
  0.3.2 verification wheel in `outputs/package_check` and installed it into a separate
  target directory. Imports resolved to that target outside the repository and all
  five required browser assets were present. Existing runtime dependencies were
  reused; this is NOT clean-Windows installation proof or a final release artifact.
  Weights remain external and need an explicit checkpoint path for wheel installs.

- Scope correction: browser waveform/histogram now use scene-linear Rec.2020
  luminance, matching the server, while existing maximum-channel light-level
  metrics remain separate. Added executable JavaScript numerical tests for RGB
  primaries, neutral gray and black against server output. Three focused scope
  tests passed and `node --check ui/app.js` passed. Diffuse-white marker now remains
  visible alongside MaxCLL. UI script version bumped. Full visual/browser QA is
  pending; this does not complete the density waveform/vectorscope upgrade.
- Earlier joint experiment progress: 559/716; superseded by completion above.

- The completed experiment in `outputs/quality_policy_joint_20260925` used
  `--joint-quality`. Check this run first on subsequent follow-ups; do not restart
  the completed PU21-only experiment. It measures real CVVDP on training scenes,
  computes train-only per-condition normalization, uses the lesser normalized
  baseline-relative gain across PU21/JOD as action utility, and minimizes worst-group
  expected regret plus 0.1 times mean group regret. Full training batches include
  both conditions. This is experimental, not a proven fix. It reuses validation;
  independent confirmation is still pending. No automatic model promotion.
- New objective, regression diagnosis, workflow, resume and review tests: 14 passed.
  When the joint run completes, run `training/review_quality_policy.py` against its
  directory and inspect the result before considering integration.

- Recovery-policy experiment completed 716 measured cases and 300 total training epochs across three seeds.
- Evaluated 30 saved candidates against 102 validation scenes in clean and degraded conditions.
- No candidate passed the no-regression requirement for both PU21 and real image ColorVideoVDP in both conditions. No promotion or Studio model replacement.
- Best average perceptual candidate: seed 3, epoch 20. Clean PU21 51.44995 vs shipped 49.13633; clean JOD 9.87955 vs 9.81814. Degraded PU21 27.90470 vs 28.25490; degraded JOD 7.98818 vs 8.05252. These are exploratory validation scores, not independent release evidence.
- Resuming measurement reuses verified completed comparisons; source/checkpoint changes are rejected. Atomic JSON writes retry transient access errors.
- Full project suite: **396 passed, 4 warnings** using the working Python environment on this machine. Fixed pytest collection to target `tests`, excluding the temporary audit checkout.
- Automatic review written to `outputs/quality_policy_20260925/review.md` and `review.json`.

## Remaining work

### Completed selector diagnosis

`training/diagnose_quality_policy.py` replays the diagnostic candidate on cached
features, verifies its validation deltas against the assessment, and writes
`outputs/quality_policy_20260925/diagnosis.json`. No image/test pixels were read.

- Degraded training PU21 already regresses by 0.21097 dB, so this is not solely validation overfitting.
- On degraded validation the selector chooses highlights-only for 62/102 scenes;
  the reference-informed PU21 optimum chooses that mode for only 16/102.
  The selector regresses PU21 on 51/102 scenes and perceptual JOD on 40/102.
- Even the reference-informed PU21 optimum on degraded validation improves PU21
  by 0.07833 dB but reduces JOD by 0.07507. Optimizing PU21 alone is therefore
  insufficient for the current two-metric release criterion. This oracle needs
  HDR ground truth and cannot be deployed.
- Clean training gains (~2.03 dB) dominate the mean objective while degraded
  training loses quality. Next experiment should measure perceptual scores on
  training scenes and investigate condition-balanced, baseline-relative losses
  with conservative fallback. Do not use validation perceptual labels to fit it.
- All diagnosis is retrospective on previously used train/validation scenes;
  independent confirmation remains necessary. No new model is approved.

1. Diagnose degraded-scene regressions using the existing measurements and feature cache before spending GPU time on another experiment. Keep the current model as baseline; do not weaken acceptance criteria to promote a candidate.
2. Any revised candidate requires explicit reporting of validation reuse, independent evaluation data, visual inspection, and sequence stability checks. Current single-frame experiment does not establish temporal quality.
3. Integrate a passing selector into preview and render paths with checkpoint binding and consistent per-frame decisions. No selector is approved yet.
4. Audit and correct HDR scope numerical behavior; current appearance/string tests do not establish waveform or vectorscope accuracy.
5. Package a versioned Studio distribution with the correct assets, launchers, dependencies and model manifest. Verify GPU startup, image/sequence render paths, and output metadata.
6. Verify installation in a clean Windows environment; current-machine pytest results are not a clean-machine test or calibrated HDR-display inspection.
7. Include model limitations, usage instructions and licensing. Source code is Apache 2.0; distributed weights and derivatives remain non-commercial under `checkpoints/LICENSE` and `NOTICE`.
8. Commit/push/tag only a verified, accurately described release. No new release tag or push performed in this completion pass.

## Continuation

User explicitly approved recurring checks every 15 minutes, including fixes, validation, packaging, and commit/push/tag only after release checks pass. The task heartbeat `complete-rudra-model-and-release` is ACTIVE. It continues this work, reports meaningful progress or blockers, and stops when completed or user input is required. Preserve existing permissions and release acceptance criteria.
