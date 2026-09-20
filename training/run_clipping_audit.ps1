#Requires -Version 5.1
<#
    run_clipping_audit.ps1 -- how much of the corpus clips, and who recovers it.

    The claim "RUDRA reconstructs the range" rests on pixels the SDR clipped.
    Every number in RESULTS.md is a whole-frame score, and a whole frame is
    mostly pixels that were never clipped -- so none of them answer it.

        cd D:\A.I\Devlopments\rudra
        powershell -ExecutionPolicy Bypass -File training\run_clipping_audit.ps1

    The census is seconds. -Score adds inference on the frames that do clip and
    is the part that takes minutes.
#>
[CmdletBinding()]
param(
    [string]$Repo       = "D:\A.I\Devlopments\rudra",
    [string]$Bench      = "E:\RUDRA_v3_20260822\bench",
    [string]$Python     = "python",
    [string]$Device     = "cuda",
    [switch]$Score,
    [double]$MinClipPct = 0.1,
    [int]   $MaxFrames  = 40,
    [string]$Out        = ""
)

$ErrorActionPreference = "Continue"   # native stderr must never end the run
Set-Location -LiteralPath $Repo

if (-not (Test-Path -LiteralPath $Bench)) {
    Write-Host "No bench at $Bench. Pass -Bench <path to the folder holding clean\ and hard\>." -ForegroundColor Red
    exit 1
}
if (-not $Out) {
    $Out = Join-Path $Repo ("docs\clipping_audit_" + (Get-Date -Format "yyyy-MM-dd") + ".json")
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Out) | Out-Null

$argv = @("training\measure_clipping.py", "--bench", $Bench,
          "--device", $Device, "--min-clip-pct", $MinClipPct,
          "--max-frames", $MaxFrames, "--out", $Out)
if ($Score) { $argv += "--score" }

& $Python @argv
if ($LASTEXITCODE -ne 0) { Write-Host "audit failed" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "Wrote $Out" -ForegroundColor Green
Write-Host ""
Write-Host "Reading it:" -ForegroundColor Cyan
Write-Host "  median clipped % near zero      -> the corpus barely contains the thing the"
Write-Host "                                     model is meant to reverse, and no benchmark"
Write-Host "                                     built on it can measure recovery."
Write-Host "  RUDRA minus baseline about 0    -> the recovery is the analytic curve. The"
Write-Host "                                     network is not what puts the highlights back."
Write-Host "  RUDRA minus baseline clearly    -> the network earns the claim, and this file"
Write-Host "  negative, on most frames           is the evidence for it."
