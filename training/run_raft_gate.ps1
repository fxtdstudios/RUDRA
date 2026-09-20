# The flow arm with a learned estimator, on the GPU. Fully paired.
#
# WHY. The v02 gate reads a large achievable gain with the renderer's exact
# camera poses and almost none with DIS optical flow estimated from the plate
# -- and the oracle bound under DIS is negative, meaning no architecture can
# recover it. The failure is specific: DIS drifts in flat regions (2.61 px
# against 0.17 px in textured parts of the same frame on an open-sky clip) and
# forward-backward consistency cannot detect it, because any displacement
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
# This runs DIS and RAFT over the SAME clips and prints both beside the pose
# arm already measured on them, so all three numbers are paired. Cross-seed
# comparison is not valid here: the flat arm moved +0.035 to +0.110 across
# seeds, which is the size of the effect being looked for.
#
# Read it as: near the pose arm reopens v02; near zero closes it whatever the
# estimator.
#
# ON THE GPU, because RAFT-large on a CPU is about 20 s per pair at 720p and
# one nine-frame clip needs 72 of them. On a 4080 it is minutes.
#
#   powershell -ExecutionPolicy Bypass -File training\run_raft_gate.ps1
#
# Resumable: the gate writes a row after every clip and --resume skips the
# ones already in the output file, so an interruption costs one clip.

param(
  [string] $Seed       = "20260906",
  [string] $Work       = "D:\A.I\Devlopments\RUDRA_v02\_gate_seeds",
  [string] $Checkpoint = "D:\A.I\Devlopments\rudra\checkpoints\sdr2hdr_shadow_v1.pt",
  [string] $Device     = "cuda",
  [switch] $SkipDis,
  [string] $Python     = "D:\A.I\Devlopments\rudra\.venv\Scripts\python.exe"
)

# See pipeline/build_moves_corpus.ps1: under Stop, Windows PowerShell 5.1
# turns python's progress bars into NativeCommandError.
$ErrorActionPreference = "Continue"
$repo = "D:\A.I\Devlopments\rudra"

function Fail($m) { Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }

$clips = Join-Path $Work "seed${Seed}_drift_clips"
$pose  = Join-Path $Work "seed${Seed}_drift.json"
$outD  = Join-Path $Work "seed${Seed}_drift_dis.json"
$outR  = Join-Path $Work "seed${Seed}_drift_raft.json"

foreach ($path in @($Python, $Checkpoint)) {
  if (-not (Test-Path $path)) { Fail "not found: $path" }
}
if (-not (Test-Path $clips)) {
  Write-Host "ERROR: no clips at $clips" -ForegroundColor Red
  Write-Host "Drifted clip sets present under ${Work}:"
  Get-ChildItem $Work -Directory -Filter "*_drift_clips" -ErrorAction SilentlyContinue |
    ForEach-Object { "  -Seed " + ($_.Name -replace '^seed', '' -replace '_drift_clips$', '') +
                     "   ($((Get-ChildItem $_.FullName -Directory).Count) clips)" }
  Write-Host "Or run training\run_drift_gate.ps1 to build one."
  exit 1
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

$count = (Get-ChildItem $clips -Directory).Count
Write-Host "   seed       : $Seed"
Write-Host "   clips      : $count"
Write-Host "   device     : $Device"

function Run-Arm($label, $out, $extra) {
  Write-Host ""
  Write-Host "== $label ==" -ForegroundColor Cyan
  $args = @("$repo\training\gate_temporal_oracle.py",
            "--clips", $clips, "--checkpoint", $Checkpoint, "--condition", "hard",
            "--degradation", "codec", "--alignment", "flow",
            "--device", $Device, "--resume", "--out", $out) + $extra
  & $Python $args
  if ($LASTEXITCODE -ne 0) { Fail "$label failed" }
}

if (-not $SkipDis) { Run-Arm "DIS"  $outD @("--flow-backend", "dis") }
Run-Arm "RAFT" $outR @("--flow-backend", "raft", "--flow-device", $Device)

function Summarise($label, $path) {
  if (-not (Test-Path $path)) { return }
  $rows = Get-Content $path -Raw | ConvertFrom-Json
  $pf  = ($rows | ForEach-Object { $_.per_frame.cvvdp_jod }    | Measure-Object -Average).Average
  $am  = ($rows | ForEach-Object { $_.aligned_mean.cvvdp_jod } | Measure-Object -Average).Average
  $ct  = ($rows | ForEach-Object { $_.control.cvvdp_jod }      | Measure-Object -Average).Average
  $orc = ($rows | ForEach-Object { $_.oracle.cvvdp_jod }       | Measure-Object -Average).Average
  "{0,-16} {1,3} clips   per-frame {2,7:0.000}   ACHIEVABLE {3,8:+0.000;-0.000}   ceiling {4,8:+0.000;-0.000}" -f `
    $label, $rows.Count, $pf, ($am - $pf), ($orc - $ct)
}

Write-Host ""
Write-Host "== seed $Seed, same clips, alignment is the only difference ==" -ForegroundColor Green
Summarise "pose (exact)" $pose
Summarise "flow (DIS)"   $outD
Summarise "flow (RAFT)"  $outR
Write-Host ""
Write-Host "Threshold is +0.5 JOD achievable. If RAFT lands near the pose row,"
Write-Host "estimated alignment is good enough and v02 continues. If it lands"
Write-Host "near zero like DIS, the gain belongs to the renderer's poses and"
Write-Host "not to anything a plate can supply."
