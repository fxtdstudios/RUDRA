#Requires -Version 5.1
<#
    run_bench.ps1 -- the full paired benchmark, end to end.

    Four exports (v5 and v6, clean and hard) then six `rudra bench` scorings
    (baseline / v5 / v6 in each condition), against one shared reference tree.

    Resumable: every stage writes a marker file and is skipped if that marker
    already exists, so a run that dies at hour three picks up where it stopped.
    Pass -Force to redo everything.

        cd D:\A.I\Devlopments\rudra
        powershell -ExecutionPolicy Bypass -File training\run_bench.ps1

    Smoke it first on a handful of frames:

        powershell -ExecutionPolicy Bypass -File training\run_bench.ps1 -Limit 6
#>
[CmdletBinding()]
param(
    [string]$Repo   = "D:\A.I\Devlopments\rudra",
    [string]$Data   = "E:\RUDRA_v3_20260822",
    [string]$Python = "python",
    [int]   $Limit  = 0,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$manifest = Join-Path $Data "sdr_hdr_manifest.jsonl"
$ckptV5   = Join-Path $Data "checkpoints\sdr2hdr_image_v5\shipped_v5_step81000.pt"
$ckptV6   = Join-Path $Data "checkpoints\sdr2hdr_image_v6_c64\best.pt"
$bench    = Join-Path $Data "bench"
$results  = Join-Path $bench "results"
$logs     = Join-Path $bench "logs"
$exporter = Join-Path $Repo "training\export_bench_pairs.py"

$suffix = if ($Limit -gt 0) { "_limit$Limit" } else { "" }
$outClean = Join-Path $bench "clean$suffix"
$outHard  = Join-Path $bench "hard$suffix"

function Say([string]$text, [string]$colour = "White") {
    Write-Host $text -ForegroundColor $colour
}

# ---------------------------------------------------------------- preflight
Say ""
Say ("=" * 72) Cyan
Say "   RUDRA paired benchmark" Cyan
Say ("=" * 72) Cyan

foreach ($p in @($manifest, $ckptV5, $ckptV6, $exporter)) {
    if (-not (Test-Path -LiteralPath $p)) { throw "missing: $p" }
}
foreach ($d in @($bench, $results, $logs)) {
    if (-not (Test-Path -LiteralPath $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

$drive = (Get-Item -LiteralPath $Data).PSDrive
Say ("   free space : {0:N1} GB on {1}:  (the export needs roughly 20 GB)" -f ($drive.Free / 1GB), $drive.Name)
if ($Limit -gt 0) { Say "   LIMIT      : $Limit frames per condition (smoke run)" Yellow }

Push-Location $Repo
try {
    & $Python -c "import torch, sys; print('   torch      : %s  cuda=%s  %s' % (torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'))"
    if ($LASTEXITCODE -ne 0) { throw "python/torch is not usable from this shell" }

    & $Python -c "import pycvvdp; print('   cvvdp      : available')" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Say "   cvvdp      : NOT importable -- PU21-PSNR will still be scored, JOD will not" Yellow
        Say "                (pip install cvvdp)" Yellow
    }
    Say ""

    # ------------------------------------------------------------- exports
    $exports = @(
        @{ Name = "clean/v5"; Out = $outClean; Ckpt = $ckptV5; Cond = "clean"; Tree = "v5"; Only = $false },
        @{ Name = "clean/v6"; Out = $outClean; Ckpt = $ckptV6; Cond = "clean"; Tree = "v6"; Only = $true  },
        @{ Name = "hard/v5";  Out = $outHard;  Ckpt = $ckptV5; Cond = "hard";  Tree = "v5"; Only = $false },
        @{ Name = "hard/v6";  Out = $outHard;  Ckpt = $ckptV6; Cond = "hard";  Tree = "v6"; Only = $true  }
    )

    foreach ($e in $exports) {
        $marker = Join-Path $e.Out ("export_" + $e.Tree + ".json")
        if ((Test-Path -LiteralPath $marker) -and (-not $Force)) {
            Say ("-- export {0,-9} already done ({1})" -f $e.Name, $marker) DarkGray
            continue
        }
        Say ("-- export {0}" -f $e.Name) Green
        $a = @(
            $exporter,
            "--checkpoint", $e.Ckpt,
            "--manifest",   $manifest,
            "--out",        $e.Out,
            "--split",      "test",
            "--condition",  $e.Cond,
            "--test-name",  $e.Tree
        )
        if ($e.Only)      { $a += "--only-test" }
        if ($Limit -gt 0) { $a += @("--limit", "$Limit") }

        $log = Join-Path $logs ("export_" + $e.Cond + "_" + $e.Tree + ".log")
        $started = Get-Date
        & $Python @a 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { throw "export $($e.Name) failed (exit $LASTEXITCODE); see $log" }
        Say ("   done in {0:hh\:mm\:ss}" -f ((Get-Date) - $started)) DarkGray
    }

    # -------------------------------------------------------------- scoring
    $scorings = @(
        @{ Cond = "clean"; Root = $outClean; Tree = "baseline" },
        @{ Cond = "clean"; Root = $outClean; Tree = "v5" },
        @{ Cond = "clean"; Root = $outClean; Tree = "v6" },
        @{ Cond = "hard";  Root = $outHard;  Tree = "baseline" },
        @{ Cond = "hard";  Root = $outHard;  Tree = "v5" },
        @{ Cond = "hard";  Root = $outHard;  Tree = "v6" }
    )

    foreach ($s in $scorings) {
        $json = Join-Path $results ($s.Cond + "_" + $s.Tree + $suffix + ".json")
        if ((Test-Path -LiteralPath $json) -and (-not $Force)) {
            Say ("-- bench  {0,-14} already done" -f ($s.Cond + "/" + $s.Tree)) DarkGray
            continue
        }
        Say ("-- bench  {0}" -f ($s.Cond + "/" + $s.Tree)) Green
        $a = @(
            "-m", "rudra.delivery.cli", "bench", $s.Root,
            "--nits-scale", "203",
            "--test-dir",   $s.Tree,
            "--output",     $json
        )
        $log = Join-Path $logs ("bench_" + $s.Cond + "_" + $s.Tree + ".log")
        $started = Get-Date
        & $Python @a 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { throw "bench $($s.Cond)/$($s.Tree) failed (exit $LASTEXITCODE); see $log" }
        Say ("   done in {0:hh\:mm\:ss}" -f ((Get-Date) - $started)) DarkGray
    }

    # -------------------------------------------------------------- summary
    Say ""
    Say ("=" * 72) Cyan
    Say "   results" Cyan
    Say ("=" * 72) Cyan

    $rows = @()
    $md   = @("| Condition | Method | Pairs | PU21-PSNR (dB) | CVVDP (JOD) |",
              "| --- | --- | ---: | ---: | ---: |")
    foreach ($cond in @("clean", "hard")) {
        foreach ($tree in @("baseline", "v5", "v6")) {
            $json = Join-Path $results ($cond + "_" + $tree + $suffix + ".json")
            if (-not (Test-Path -LiteralPath $json)) { continue }
            $j = Get-Content -LiteralPath $json -Raw | ConvertFrom-Json
            $psnr = if ($null -eq $j.pu_psnr_db_mean) { "" } else { "{0:N3}" -f $j.pu_psnr_db_mean }
            $jod  = if ($null -eq $j.cvvdp_jod_mean)  { "" } else { "{0:N4}" -f $j.cvvdp_jod_mean }
            $rows += [pscustomobject]@{
                Condition = $cond
                Method    = $tree
                Pairs     = $j.pairs
                'PU21-PSNR dB' = $psnr
                'CVVDP JOD'    = $jod
                Backend   = $j.cvvdp_backend
            }
            $md += ("| {0} | {1} | {2} | {3} | {4} |" -f $cond, $tree, $j.pairs, $psnr, $(if ($jod) { $jod } else { "n/a" }))
        }
    }
    $rows | Format-Table -AutoSize | Out-String | Write-Host

    $mdPath = Join-Path $bench ("RESULTS" + $suffix + ".md")
    $header = @(
        "# RUDRA paired benchmark",
        "",
        ('Generated {0}. Held-out `test` split, native 1280x720, scene-linear,' -f (Get-Date -Format "yyyy-MM-dd HH:mm")),
        'diffuse white = 1.0, scored at `--nits-scale 203`. `baseline` is the analytic',
        'inverse-ACES tone map; `v5` is base_channels 32 step 81000; `v6` is',
        'base_channels 64 best.',
        ""
    )
    ($header + $md) -join "`r`n" | Set-Content -LiteralPath $mdPath -Encoding UTF8
    Say ""
    Say ("   markdown table : {0}" -f $mdPath) Cyan
    Say ("   per-frame CSVs : {0}" -f $results) Cyan
    Say ""
}
finally {
    Pop-Location
}
