# Commit and push the 5-10 Sep 2026 work. Run from PowerShell on the Windows
# host -- NOT from an agent shell.
#
# `git add` from the sandbox mount cannot unlink its own temp objects: it
# leaves .git\index.lock behind and stages NOTHING. AGENTS.md records that
# happening four times in August; it happened again on 6 Sep when I tried, and
# the sandbox also has no git identity, so the commit would have carried the
# wrong author. Hence this file.
#
#   powershell -ExecutionPolicy Bypass -File COMMIT_AND_PUSH.ps1
#   powershell -ExecutionPolicy Bypass -File COMMIT_AND_PUSH.ps1 -NoPush
#
# Then delete it.

param([switch] $NoPush)

$ErrorActionPreference = "Continue"
Set-Location "D:\A.I\Devlopments\rudra"

if (Test-Path ".git\index.lock") {
  Write-Host "A stale .git\index.lock is present; moving it aside." -ForegroundColor Yellow
  New-Item -ItemType Directory -Force -Path ".git\_stale_locks" | Out-Null
  Move-Item ".git\index.lock" ".git\_stale_locks\index.lock.$(Get-Random)"
}

Write-Host "Before:" -ForegroundColor Cyan
git status --short

git add `
  .gitignore `
  pytest.ini `
  README.md `
  STATUS.md `
  train.bat `
  train.sh `
  pipeline/render_hdri_moves.py `
  rudra/flow_warp.py `
  training/gate_temporal_oracle.py `
  training/stage_oracle_clips.py `
  training/train_from_footage.py `
  training/run_drift_gate.ps1 `
  training/run_raft_gate.ps1 `
  training/run_combiner.ps1 `
  tests/test_gate_degradation_2026_09_05.py `
  tests/test_render_exposure_2026_09_05.py `
  tests/test_flow_warp_2026_09_06.py `
  tests/test_gate_resume_2026_09_07.py `
  docs/gate_v02_2026-09-05
if ($LASTEXITCODE -ne 0) { Write-Host "git add failed" -ForegroundColor Red; exit 1 }

$message = @'
Close the v02 temporal line, and make training one command

THE GATE. v02 rested on a claim: neighbouring frames carry information a
single 8-bit frame does not. gate_temporal_oracle.py measures what a
PERFECTLY aligned temporal model could win before one is built -- exact
correspondence from the renderer's camera poses, omniscient per-pixel
selection among aligned neighbours. Threshold +0.5 JOD achievable, fixed
before anything was measured.

The 4 Sep reading of +9.31 JOD does not survive. Two faults produced it.

It scored a bare model(x) pass rather than RUDRA as deployed. The
benchmark runs predict_image with preserve_outside=True, where outside
the learned masks the output IS the analytic baseline -- which is why the
floor sat at -1.96 JOD against the benchmark's 7.805 on the same
checkpoint and the same 1280x720 frames. Never a crop size or a
resolution mismatch.

And degrade_like_eval reseeds from the frame index, so every frame of a
nine-frame clip got its own exposure, tone curve, white balance,
saturation, chroma subsampling, bit depth and JPEG quality. Averaging
nine independent draws of a corruption is worth sqrt(9) whether or not
the frames carry information. --degradation {per-frame,per-clip,
realistic,codec} makes that assumption a flag; codec is a real H.264
round trip and is the only mode with a GOP in it.

Corrected: 40 clips give +0.033 and +0.034 JOD against +0.5.

EXPOSURE WAS THE MISSING AXIS. make_sdr tone-maps every frame with one
fixed curve and the camera only rotates through a static panorama, so a
scene point carries the same 8-bit code in every frame it appears in --
what is blown in frame 3 is blown in frame 7. --exposure-drift and
--exposure-jitter put that back (default 0.0, so the existing 993 scenes
reproduce byte for byte; the _ingest_config sentinel refuses to mix).
Rendering 50 panoramas twice, identical camera paths to 1e-9, exposure
the only difference:

  constant       +0.110 JOD achievable
  +/-0.48 stops  +0.511 JOD achievable

Held on three seeds: flat +0.110/+0.035/+0.046, drift +0.511/+0.603/
+0.661. The separation between arms is far larger than the spread across
seeds.

THEN ESTIMATED ALIGNMENT TOOK IT BACK. Every number above uses camera
angles the renderer wrote down, and a plate has none. rudra/flow_warp.py
estimates correspondence from the degraded SDR -- what a deployed model
holds -- with DIS or RAFT-large, forward-backward consistency, optional
texture gating, and a --flow-scale that keeps RAFT inside a laptop. Seed
20260906, 40 clips, the same clips every row:

  pose (exact)   ACHIEVABLE +0.603   ceiling +1.090
  flow (DIS)                -0.069           +0.122
  flow (RAFT)               +0.341           +0.671
  RAFT, confidence combiner +0.350

RAFT recovers 57% of what exact poses give and misses the threshold. One
combiner was declared and tried before it was run -- weighting each
neighbour by its forward-backward residual -- and moved the number by
0.001.

The reason is the same wall from three angles. RAFT's forward-backward
drift is 0.04-0.09 px on nearly every pixel that passes, so the weight is
~1 everywhere. The failures are not low-confidence matches, they are
CONFIDENT WRONG ones, in flat regions where any displacement round-trips
perfectly -- exactly where the blown highlights are, and exactly what
forward-backward consistency cannot see.

LINE D IS CLOSED. Three alignment arms, two combiners, a threshold fixed
in advance. No temporal model was trained because there is nothing
measurable for one to learn, and that is the result rather than the
absence of one.

TRAINING IS NOW ONE COMMAND. train.bat / train.sh -> training/
train_from_footage.py: preflight, scan, pairs, manifests, verify,
backbone, shadow gate, benchmark. Resumable per stage, --dry-run,
--from. It refuses SDR-only footage rather than training a model that
learned the identity function, and warns when the corpus is too small to
mean anything. README gains a "Train on your own footage" section
including the six ways we wasted a week.

Also: the gate writes a row after every clip and --resume skips them, so
an interrupted run costs one clip instead of the run -- two multi-hour
runs were lost to an OOM kill and a reclaimed sandbox before that
existed. stage_oracle_clips.py builds clip directories from a manifest or
straight from a render. --device now reaches ColorVideoVDP, which is
nearly all of the gate's cost. 66 tests, flake8 clean, per-clip results
under docs/gate_v02_2026-09-05/.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KjKPk9qJVd4kgcRY4JXCgF
'@

$file = [IO.Path]::GetTempFileName()
[IO.File]::WriteAllText($file, $message)
git commit -F $file
$ok = $LASTEXITCODE
Remove-Item $file -Force
if ($ok -ne 0) { Write-Host "git commit failed" -ForegroundColor Red; exit 1 }

Write-Host ""
git --no-pager log --oneline -1
Write-Host ""
Write-Host "After:" -ForegroundColor Cyan
git status --short

if ($NoPush) {
  Write-Host ""
  Write-Host "Committed. Not pushed (-NoPush)." -ForegroundColor Yellow
  exit 0
}

$branch = (git rev-parse --abbrev-ref HEAD).Trim()
Write-Host ""
Write-Host "Pushing $branch ..." -ForegroundColor Cyan
git rev-parse --abbrev-ref "$branch@{upstream}" *> $null
if ($LASTEXITCODE -ne 0) {
  Write-Host "  no upstream set; using -u origin $branch"
  git push -u origin $branch
} else {
  git push
}
if ($LASTEXITCODE -ne 0) {
  Write-Host "git push failed. The commit is safe locally -- fix the remote and push again." -ForegroundColor Red
  exit 1
}
Write-Host "Pushed." -ForegroundColor Green
