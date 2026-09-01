#Requires -Version 5.1
<#
    score_checkpoint.ps1 -- put one more checkpoint on the existing benchmark.

    run_bench.ps1 writes ref/ and baseline/ once; those are the expensive half
    and they do not depend on the checkpoint. This adds a single prediction
    tree beside them in both conditions and scores it against the same
    reference, so a new model costs two exports and two scorings instead of a
    full re-run.

        cd D:\A.I\Devlopments\rudra
        powershell -ExecutionPolicy Bypass -File training\score_checkpoint.ps1 `
            -Checkpoint E:\RUDRA_v3_20260822\checkpoints\sdr2hdr_image_v5\step_0072000.pt `
            -Name v5s72k

    Resumable in the same way: each stage is skipped if its output exists.
    Prints every method already scored, so the new row lands in context.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [Parameter(Mandatory = $true)][string]$Name,
    [string]$Repo   = "D:\A.I\Devlopments\rudra",
    [string]$Data   = "E:\RUDRA_v3_20260822",
    [string]$Python = "python",
    [int]   $Limit  = 0,
    [ValidateSet("all","highlights","shadows","off")][string]$RecoveryMode = "all",
    [double]$RecoveryStrength = 1.0,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# Native stderr must never kill this script. $ErrorActionPreference = "Stop"
# turns a NativeCommandError -- which is what PowerShell makes of ANY line a
# native command writes to stderr -- into a TERMINATING error, so a single
# benign warning ends the run. CVVDP emits exactly such a warning ("the mean
# color value is less than 1") on dark frames, and on 1 Sep 2026 it stopped a
# benchmark mid-scoring that was working perfectly. Exit codes are what decide
# success here, and they are checked explicitly after every call.
function Invoke-Tool([string]$exe, [string[]]$argv, [string]$log) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try   { & $exe @argv 2>&1 | Tee-Object -FilePath $log }
    finally { $ErrorActionPreference = $previous }
}


if ($Name -notmatch '^[A-Za-z0-9_.-]+$') { throw "-Name must be a plain directory name, got '$Name'" }
foreach ($reserved in @("ref", "test", "baseline")) {
    if ($Name -eq $reserved) { throw "-Name '$Name' is reserved; pick something else" }
}

$manifest = Join-Path $Data "sdr_hdr_manifest.jsonl"
$bench    = Join-Path $Data "bench"
$results  = Join-Path $bench "results"
$logs     = Join-Path $bench "logs"
$exporter = Join-Path $Repo "training\export_bench_pairs.py"
$suffix   = if ($Limit -gt 0) { "_limit$Limit" } else { "" }

function Say([string]$t, [string]$c = "White") { Write-Host $t -ForegroundColor $c }

Say ""
Say ("=" * 72) Cyan
Say ("   scoring '$Name' on the existing benchmark") Cyan
Say ("=" * 72) Cyan

foreach ($p in @($Checkpoint, $manifest, $exporter)) {
    if (-not (Test-Path -LiteralPath $p)) { throw "missing: $p" }
}
foreach ($d in @($results, $logs)) {
    if (-not (Test-Path -LiteralPath $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

$conditions = @(
    @{ Cond = "clean"; Root = (Join-Path $bench "clean$suffix") },
    @{ Cond = "hard";  Root = (Join-Path $bench "hard$suffix")  }
)
foreach ($c in $conditions) {
    if (-not (Test-Path -LiteralPath (Join-Path $c.Root "ref"))) {
        throw ("no reference tree at {0}\ref -- run run_bench.ps1 first" -f $c.Root)
    }
}

$drive = (Get-Item -LiteralPath $Data).PSDrive
Say ("   free space : {0:N1} GB on {1}:  (one tree per condition, roughly 3 GB)" -f ($drive.Free / 1GB), $drive.Name)
if ($Limit -gt 0) { Say "   LIMIT      : $Limit frames per condition (smoke run)" Yellow }

Push-Location $Repo
try {
    & $Python -c "import torch; print('   torch      : cuda=%s' % torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { throw "python/torch is not usable from this shell" }
    Say ""

    foreach ($c in $conditions) {
        $marker = Join-Path $c.Root ("export_" + $Name + ".json")
        if ((Test-Path -LiteralPath $marker) -and (-not $Force)) {
            Say ("-- export {0,-6} already done" -f $c.Cond) DarkGray
        } else {
            Say ("-- export {0}" -f $c.Cond) Green
            $a = @($exporter, "--checkpoint", $Checkpoint, "--manifest", $manifest,
                   "--out", $c.Root, "--split", "test", "--condition", $c.Cond,
                   "--test-name", $Name, "--only-test",
                   "--recovery-mode", $RecoveryMode,
                   "--recovery-strength", "$RecoveryStrength")
            if ($Limit -gt 0) { $a += @("--limit", "$Limit") }
            $log = Join-Path $logs ("export_" + $c.Cond + "_" + $Name + ".log")
            $t0 = Get-Date
            Invoke-Tool $Python $a $log
            if ($LASTEXITCODE -ne 0) { throw "export $($c.Cond) failed (exit $LASTEXITCODE); see $log" }
            Say ("   done in {0:hh\:mm\:ss}" -f ((Get-Date) - $t0)) DarkGray
        }

        $json = Join-Path $results ($c.Cond + "_" + $Name + $suffix + ".json")
        if ((Test-Path -LiteralPath $json) -and (-not $Force)) {
            Say ("-- bench  {0,-6} already done" -f $c.Cond) DarkGray
        } else {
            Say ("-- bench  {0}" -f $c.Cond) Green
            $a = @("-m", "rudra.delivery.cli", "bench", $c.Root,
                   "--nits-scale", "203", "--test-dir", $Name, "--output", $json)
            $log = Join-Path $logs ("bench_" + $c.Cond + "_" + $Name + ".log")
            $t0 = Get-Date
            Invoke-Tool $Python $a $log
            if ($LASTEXITCODE -ne 0) { throw "bench $($c.Cond) failed (exit $LASTEXITCODE); see $log" }
            Say ("   done in {0:hh\:mm\:ss}" -f ((Get-Date) - $t0)) DarkGray
        }
    }

    Say ""
    Say ("=" * 72) Cyan
    Say "   every method scored so far" Cyan
    Say ("=" * 72) Cyan

    $rows = @()
    foreach ($cond in @("clean", "hard")) {
        Get-ChildItem -LiteralPath $results -Filter ($cond + "_*" + $suffix + ".json") |
            Where-Object { $suffix -ne "" -or $_.Name -notmatch '_limit\d+\.json$' } |
            Sort-Object Name | ForEach-Object {
                $j = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
                $psnr = if ($null -eq $j.pu_psnr_db_mean) { "" } else { "{0:N3}" -f $j.pu_psnr_db_mean }
                $jod  = if ($null -eq $j.cvvdp_jod_mean)  { "" } else { "{0:N4}" -f $j.cvvdp_jod_mean }
                $rows += [pscustomobject]@{
                    Condition = $cond
                    Method    = $j.test_dir
                    Pairs     = $j.pairs
                    'PU21-PSNR dB' = $psnr
                    'CVVDP JOD'    = $jod
                }
            }
    }
    # Format-Table renders nothing in a host with no console width (a redirected
    # or headless shell), and this table IS the deliverable, so it is built by hand.
    Write-Host ("  {0,-10} {1,-14} {2,6} {3,16} {4,13}" -f `
                "Condition", "Method", "Pairs", "PU21-PSNR dB", "CVVDP JOD")
    Write-Host ("  " + ("-" * 62))
    foreach ($r in $rows) {
        Write-Host ("  {0,-10} {1,-14} {2,6} {3,16} {4,13}" -f `
                    $r.Condition, $r.Method, $r.Pairs, $r.'PU21-PSNR dB', $r.'CVVDP JOD')
    }
    Write-Host ""
    Say ("   per-frame CSVs : {0}" -f $results) Cyan
    Say ""
}
finally {
    Pop-Location
}
