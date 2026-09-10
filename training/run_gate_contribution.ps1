#Requires -Version 5.1
<#
    run_gate_contribution.ps1 -- does the shadow gate earn its place?

    The gate predicts one scalar per frame and costs a training stage, a
    checkpoint, a registry entry and a README section. This answers whether it
    is doing anything the picture can tell, on the real test split, before
    spending four CVVDP bench passes on the question.

        cd D:\A.I\Devlopments\rudra
        powershell -ExecutionPolicy Bypass -File training\run_gate_contribution.ps1

    Reads SDR frames only -- no ground truth, no scoring. Minutes, not hours.

    NOTE: if the bench has already scored the checkpoints you care about,
    compare_bench_methods.py answers the same question from bench/results/
    with no inference at all, and answers it better -- paired, per frame,
    against the seed-to-seed spread. Reach for that first.
#>
[CmdletBinding()]
param(
    [string]$Repo    = "D:\A.I\Devlopments\rudra",
    [string]$Frames  = "",          # folder of test SDR frames, or a .jsonl manifest
    [string]$Python  = "python",
    [string]$Device  = "cuda",
    [int]   $MaxSide = 1600,
    [int]   $Limit   = 0,
    [string]$Out     = ""
)

$ErrorActionPreference = "Continue"   # native stderr must never end the run
Set-Location -LiteralPath $Repo

if (-not $Frames) {
    # The test split as run_bench.ps1 exports it. Point -Frames somewhere else
    # if your export lives elsewhere; the script needs SDR frames, nothing more.
    $candidates = @(
        "E:\RUDRA_v3_20260822\bench\clean\sdr",
        "E:\RUDRA_v3_20260822\bench\hard\sdr",
        "D:\A.I\Devlopments\RUDRA_v02\bench\clean\sdr"
    )
    $Frames = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $Frames) {
        Write-Host "No SDR frames found. Looked in:" -ForegroundColor Red
        $candidates | ForEach-Object { Write-Host "   $_" }
        Write-Host "Pass -Frames <folder or manifest.jsonl>." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "Using $Frames"
}

$ck = Join-Path $Repo "checkpoints"
$gated    = Join-Path $ck "sdr2hdr_shadow_v1.pt"
$ungated  = Join-Path $ck "sdr2hdr_image_v5.pt"
$seed2    = Join-Path $ck "sdr2hdr_shadow_s2.pt"
$seed3    = Join-Path $ck "sdr2hdr_shadow_s3.pt"
foreach ($p in @($gated, $ungated, $seed2, $seed3)) {
    if (-not (Test-Path -LiteralPath $p)) { Write-Host "missing: $p" -ForegroundColor Red; exit 1 }
}
if (-not $Out) { $Out = Join-Path $Repo ("docs\gate_contribution_" + (Get-Date -Format "yyyy-MM-dd") + ".json") }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Out) | Out-Null

$argv = @("training\measure_gate_contribution.py",
          "--frames", $Frames, "--gated", $gated, "--ungated", $ungated,
          "--also", $seed2, $seed3,
          "--device", $Device, "--max-side", $MaxSide, "--out", $Out)
if ($Limit -gt 0) { $argv += @("--limit", $Limit) }

& $Python @argv
if ($LASTEXITCODE -ne 0) { Write-Host "measurement failed" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "Wrote $Out" -ForegroundColor Green
Write-Host ""
Write-Host "What to do with it:" -ForegroundColor Cyan
Write-Host "  shadow_weight cv_pct under ~5%   -> the gate is a constant. Replace it"
Write-Host "                                      with its mean, confirm parity with"
Write-Host "                                      score_checkpoint.ps1, delete the module."
Write-Host "  gate_vs_constant pu21_db over 60 -> it moves the picture by less than any"
Write-Host "                                      measurement in the bench can resolve."
Write-Host "  either of those, and the gate is a row in models.json that costs more than"
Write-Host "  it returns. Neither, and it stays -- with this file as the reason."
