# The one declared follow-up: does a confidence-weighted combiner cross +0.5?
#
# 10 Sep 2026, seed 20260906, 40 drifted clips, alignment the only variable:
#
#     pose (exact)   ACHIEVABLE +0.603   ceiling +1.090
#     flow (DIS)                -0.069           +0.122
#     flow (RAFT)               +0.341           +0.671
#
# RAFT missed the +0.5 threshold set before any of this was measured. But the
# ceiling moved: under DIS there was nothing for any combiner to find, under
# RAFT there is roughly twice the achievable figure sitting above it. The
# aligned mean was always declared as the SIMPLEST thing a model could learn,
# a lower bound -- not the definition of achievable.
#
# So this re-runs the same clips with one alternative combiner, weighting each
# neighbour by exp(-(drift/sigma)^2/2) on its forward-backward residual
# instead of averaging them equally. Nothing a model could not compute at
# inference. Under exact poses it reduces to the mean exactly, which is what
# keeps the pose arm a fixed reference (pinned by a test).
#
# ONE alternative, declared before it was run. If it does not cross +0.5,
# line D closes -- it does not get a second combiner.
#
#   powershell -ExecutionPolicy Bypass -File training\run_combiner.ps1

param(
  [string] $Seed       = "20260906",
  [string] $Work       = "D:\A.I\Devlopments\RUDRA_v02\_gate_seeds",
  [string] $Checkpoint = "D:\A.I\Devlopments\rudra\checkpoints\sdr2hdr_shadow_v1.pt",
  [double] $Sigma      = 0.75,
  [string] $Device     = "cuda",
  [string] $Python     = "D:\A.I\Devlopments\rudra\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Continue"
$repo  = "D:\A.I\Devlopments\rudra"
$clips = Join-Path $Work "seed${Seed}_drift_clips"
$out   = Join-Path $Work "seed${Seed}_drift_raft_conf.json"

function Fail($m) { Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $clips)) { Fail "no clips at $clips" }
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { Fail "ffmpeg not on PATH" }

& $Python "$repo\training\gate_temporal_oracle.py" `
    --clips $clips --checkpoint $Checkpoint --condition hard `
    --degradation codec --alignment flow --flow-backend raft `
    --flow-device $Device --device $Device `
    --combiner confidence --fb-sigma $Sigma --resume --out $out
if ($LASTEXITCODE -ne 0) { Fail "gate failed" }

function Summarise($label, $path) {
  if (-not (Test-Path $path)) { return }
  $rows = Get-Content $path -Raw | ConvertFrom-Json
  $pf  = ($rows | ForEach-Object { $_.per_frame.cvvdp_jod }    | Measure-Object -Average).Average
  $am  = ($rows | ForEach-Object { $_.aligned_mean.cvvdp_jod } | Measure-Object -Average).Average
  $ct  = ($rows | ForEach-Object { $_.control.cvvdp_jod }      | Measure-Object -Average).Average
  $orc = ($rows | ForEach-Object { $_.oracle.cvvdp_jod }       | Measure-Object -Average).Average
  "{0,-22} {1,3} clips   ACHIEVABLE {2,8:+0.000;-0.000}   ceiling {3,8:+0.000;-0.000}" -f `
    $label, $rows.Count, ($am - $pf), ($orc - $ct)
}

Write-Host ""
Write-Host "== seed $Seed, 40 drifted clips ==" -ForegroundColor Green
Summarise "pose, mean"        (Join-Path $Work "seed${Seed}_drift.json")
Summarise "RAFT, mean"        (Join-Path $Work "seed${Seed}_drift_raft.json")
Summarise "RAFT, confidence"  $out
Write-Host ""
Write-Host "Crosses +0.5 -> v02 is live and the confidence weighting is part"
Write-Host "of the architecture. Does not -> close line D."
