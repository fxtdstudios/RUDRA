param(
    [int]$Steps = 50000,
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
$Manifest = Join-Path $Repo "hdrdata\sdr_hdr_manifest.jsonl"
$Output = Join-Path $Repo "hdrdata\checkpoints\sdr2hdr_image_v2"
$Initial = Join-Path $Repo "hdrdata\checkpoints\sdr2hdr_image_50k\step_0050000.pt"

if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $Manifest)) { throw "Manifest not found: $Manifest" }
if (-not (Test-Path -LiteralPath $Initial)) { throw "Initial checkpoint not found: $Initial" }
New-Item -ItemType Directory -Force -Path $Output | Out-Null

$TrainArgs = @(
    "training\train_sdr2hdr.py",
    "--mode", "image",
    "--manifest", $Manifest,
    "--output-dir", $Output,
    "--steps", $Steps,
    "--batch-size", 2,
    "--grad-accum", 2,
    "--crop-size", 384,
    "--workers", 2,
    "--base-channels", 32,
    "--lr", "2e-5",
    "--weight-decay", "1e-4",
    "--augmentation-strength", "0.75",
    "--degradation-probability", "0.35",
    "--video-sample-fraction", "0.25",
    "--shadow-chroma-weight", "0.15",
    "--shadow-smoothness-weight", "0.02",
    "--best-metric", "loss",
    "--eval-every", 500,
    "--eval-batches", 32,
    "--save-every", 2000,
    "--log-every", 20,
    "--device", $Device
)

$Latest = Get-ChildItem -LiteralPath $Output -Filter "step_*.pt" -File -ErrorAction SilentlyContinue |
    Sort-Object Name | Select-Object -Last 1
if ($Latest) {
    $TrainArgs += @("--resume", $Latest.FullName)
    Write-Host "Resuming image v2 from $($Latest.FullName)"
} else {
    $TrainArgs += @("--init-checkpoint", $Initial)
    Write-Host "Starting image v2 from the 50K model with a fresh optimizer"
}

Set-Location $Repo
& $Python @TrainArgs
if ($LASTEXITCODE -ne 0) { throw "Image-v2 training failed with exit code $LASTEXITCODE" }

Write-Host "Training complete. Running held-out test and guarded Radiance deployment."
& (Join-Path $PSScriptRoot "postprocess_sdr2hdr_v2.ps1")
if ($LASTEXITCODE -ne 0) { throw "Post-training quality gate failed with exit code $LASTEXITCODE" }
