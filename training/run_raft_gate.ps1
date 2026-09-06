# The flow arm with a learned estimator, on the GPU.
#
# WHY. The v02 gate reads +0.511 JOD achievable with the renderer's exact
# camera poses and +0.016 with DIS optical flow estimated from the plate --
# and the oracle bound under DIS is -0.005, meaning no architecture can
# recover it. The failure is specific: DIS drifts in flat regions (2.61 px
# against 0.17 px in textured parts of the same frame on an open-sky clip)
# and forward-backward consistency cannot detect it, because any displacement
# round-trips through a constant area.
#
# RAFT does not have that failure. Measured 6 Sep 2026, four frames apart,
# median endpoint error against the analytic poses in the flattest quarter:
#
#     scene                         DIS    RAFT-small   RAFT-large
#     drackenstein_quarry_puresky   2.61      0.95         0.35
#     ostrich_road                  0.60      0.34         0.11
#     billiard_hall                 0.21       --          0.10
#
# So this re-runs the whole 50-clip drift arm with RAFT-large. It is the
# single experiment that decides whether v02 continues: if the achievable
# gain comes back near the pose arm's +0.511, the temporal model is buildable
# and the alignment is part of it. If it stays near zero, v02 as specified is
# finished whatever the estimator.
#
# ON THE GPU, because RAFT-large on a CPU is about 20 s per pair at 720p and
# one nine-frame clip needs 72 of them -- 20 hours for 50 clips. On a 4080 it
# is minutes.
#
#   powershell -ExecutionPolicy Bypass -File training\run_raft_gate.ps1
#
# Needs the drifted clips from run_drift_gate.ps1 (seed 20260903 is the
# 50-clip set the pose and DIS numbers were measured on). torchvision
# downloads the RAFT weights on first use, about 20 MB.

param(
  [string] $Clips      = "D:\A.I\Devlopments\RUDRA_v02\_gate_seeds\seed20260903_drift_clips",
  [string] $Checkpoint = "D:\A.I\Devlopments\rudra\checkpoints\sdr2hdr_shadow_v1.pt",
  [string] $Out        = "D:\A.I\Devlopments\RUDRA_v02\_gate_seeds\seed20260903_drift_raft.json",
  [string] $Device     = "cuda",
  [string] $Python     = "D:\A.I\Devlopments\rudra\.venv\Scripts\python.exe"
)

# See pipeline/build_moves_corpus.ps1: under Stop, Windows PowerShell 5.1
# turns python's progress bars into NativeCommandError.
$ErrorActionPreference = "Continue"
$repo = "D:\A.I\Devlopments\rudra"

function Fail($m) { Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }

foreach ($path in @($Python, $Checkpoint)) {
  if (-not (Test-Path $path)) { Fail "not found: $path" }
}
if (-not (Test-Path $Clips)) {
  Fail ("no clips at $Clips. Run training\run_drift_gate.ps1 first, or point " +
        "-Clips at another drifted set.")
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
  Fail "ffmpeg is not on PATH and --degradation codec needs it."
}
& $Python -c "import torchvision, sys; sys.exit(0)"
if ($LASTEXITCODE -ne 0) { Fail "torchvision is not installed in that venv; RAFT needs it." }
& $Python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)"
if ($Device -eq "cuda" -and $LASTEXITCODE -ne 0) {
  Fail "torch reports no CUDA device. -Device cpu works but takes about 20 hours."
}

$count = (Get-ChildItem $Clips -Directory).Count
Write-Host "   clips      : $count"
Write-Host "   device     : $Device"

& $Python "$repo\training\gate_temporal_oracle.py" `
    --clips $Clips --checkpoint $Checkpoint --condition hard `
    --degradation codec --alignment flow --flow-backend raft `
    --flow-device $Device --device $Device --out $Out
if ($LASTEXITCODE -ne 0) { Fail "gate failed" }

Write-Host ""
Write-Host "Compare against docs\gate_v02_2026-09-05\, same 50 clips:" -ForegroundColor Green
Write-Host "  pose (exact)      ACHIEVABLE +0.511   ceiling +0.851"
Write-Host "  flow (DIS)        ACHIEVABLE +0.016   ceiling -0.005"
Write-Host "Near +0.5 reopens v02. Near zero closes it whatever the estimator."
