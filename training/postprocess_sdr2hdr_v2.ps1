param(
    [int]$TrainingProcessId = 0
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
$RunDir = Join-Path $Repo "hdrdata\checkpoints\sdr2hdr_image_v2"
$Final = Join-Path $RunDir "step_0050000.pt"
$Best = Join-Path $RunDir "best.pt"
$EvalCsv = Join-Path $Repo "hdrdata\eval\sdr2hdr_image_v2_test.csv"
$EvalJson = [IO.Path]::ChangeExtension($EvalCsv, ".json")
$Deploy = "D:\A.I\ComfyUI\models\radiance\sdr2hdr_pixel_image.pt"
$DeployBackup = "D:\A.I\ComfyUI\models\radiance\sdr2hdr_pixel_image_v1_50k.pt"
$StatusPath = Join-Path $RunDir "deployment_status.json"

if ($TrainingProcessId -gt 0 -and $TrainingProcessId -ne $PID) {
    Wait-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue
}
Set-Location $Repo

if (-not (Test-Path -LiteralPath $Final)) {
    @{ status = "training_incomplete"; expected = $Final; timestamp = (Get-Date).ToString("o") } |
        ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
    exit 2
}

& $Python training\evaluate_sdr2hdr.py `
    --manifest hdrdata\sdr_hdr_manifest.jsonl `
    --checkpoint $Best `
    --split test `
    --crop-size 512 `
    --output $EvalCsv `
    --device cuda
if ($LASTEXITCODE -ne 0) { throw "Held-out evaluation failed with exit code $LASTEXITCODE" }

$Metrics = Get-Content -LiteralPath $EvalJson -Raw | ConvertFrom-Json
$Passed = (
    [double]$Metrics.model_psnr_log -gt [double]$Metrics.baseline_psnr_log -and
    [double]$Metrics.model_psnr_tm -gt [double]$Metrics.baseline_psnr_tm -and
    [double]$Metrics.model_highlight_log_l1 -lt [double]$Metrics.baseline_highlight_log_l1
)

if ($Passed) {
    if ((Test-Path -LiteralPath $Deploy) -and -not (Test-Path -LiteralPath $DeployBackup)) {
        Copy-Item -LiteralPath $Deploy -Destination $DeployBackup -Force
    }
    Copy-Item -LiteralPath $Best -Destination $Deploy -Force
    $Status = "deployed"
} else {
    $Status = "rejected_by_test_gate"
}

@{
    status = $Status
    checkpoint = $Best
    deployed_to = if ($Passed) { $Deploy } else { $null }
    metrics = $Metrics
    timestamp = (Get-Date).ToString("o")
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $StatusPath -Encoding UTF8
