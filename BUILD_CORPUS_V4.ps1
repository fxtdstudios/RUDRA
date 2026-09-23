# RUDRA -- build the v4 corpus the way the gate expects, and push the gate fix.
#
#   powershell -ExecutionPolicy Bypass -File BUILD_CORPUS_V4.ps1 -Push
#   powershell -ExecutionPolicy Bypass -File BUILD_CORPUS_V4.ps1 -Scan -Ingest
#
# WHY G:\corpus_v4 FAILED VERIFICATION
#
# It was built with training/prepare_training_data.py, which is the OLD
# ingest. That module clips the HDR target at 203/10000 -- the defect
# pipeline/prepare_pairs.py exists to fix -- and it writes neither
# _ingest_config.json (check 1) nor clipped_fraction into an index
# (check 3). Then build_sdr_hdr_manifest.py built the manifest from the
# sdr/ and hdr/ folders by filename, so nothing measured ever reached it.
#
# The path the gate is written for:
#   scan_sources.py   -> source_inventory.jsonl  (what is out there)
#   prepare_pairs.py  -> pairs + pairs_index.jsonl + _ingest_config.json
#   build_manifests.py-> sdr_hdr_manifest.jsonl  (copies index rows whole)
#   verify_dataset.py -> the gate
#
# G:\corpus_v4 as it stands cannot be salvaged into that shape. Re-ingest.
param(
    [switch]$Push,
    [switch]$Scan,
    [switch]$Ingest,
    [switch]$Manifest,
    [switch]$Verify,
    [string]$Src = "G:\datasets\sources",
    [string]$Extra = "",
    [string]$Dst = "G:\datasets\corpora\corpus_v4b",
    [double]$TonemapEv = 0.0,
    # The previous corpus's manifest: its TEST scenes are pinned to test here so
    # the paper's 429-frame bench stays held out for anything trained on v4b.
    [string]$HoldOut = "E:\RUDRA_v3_20260822\sdr_hdr_manifest.jsonl"
)
$ErrorActionPreference = "Continue"
Set-Location -LiteralPath "D:\A.I\Devlopments\rudra"

if ($Push) {
    git reset --quiet
    git add -f -- "pipeline/prepare_pairs.py" "pipeline/verify_dataset.py" "tests/test_review_fixes_2026_09_16.py"
    $staged = git --no-optional-locks diff --cached --name-only
    if (-not $staged) { Write-Host "nothing to commit" -ForegroundColor Yellow }
    else {
        $staged | ForEach-Object { Write-Host "   $_" -ForegroundColor Cyan }
        $f = Join-Path $env:TEMP "rudra_msg_gate.txt"
        @"
The corpus gate gets the floor it was missing

Check 3 capped the share of records whose HDR TARGET clips at the storage
ceiling. Nothing ever checked the other direction, and the two quantities
have nearly the same name: the SDR INPUT clipping at the top code is what
an inverse tone mapper exists to undo, and the shipped corpus had a median
of 0.000% of it. 0.57% of records touched the top code at all. Every check
passed.

prepare_pairs.py now takes --tonemap-ev (the exposure make_sdr applies
before the curve, which is what decides whether the SDR side clips at
all), records it and the SDR clipped fraction per pair, and puts the
exposure in the ingest sentinel so one directory cannot mix two renders.
verify_dataset.py gains check 3b with --min-clipped-records, defaulting to
20%: below that the highlight loss sees almost nothing and the retrain is
not worth running.

The manifest builder needed no change -- pipeline/build_manifests.py
copies index rows whole, so both new fields reach the manifest and
corpus_ev_of picks the exposure up from there.

Three tests in test_review_fixes_2026_09_16.py, including the one that
fails a corpus shaped like the one we shipped.
"@ | Set-Content -LiteralPath $f -Encoding ASCII
        git commit -F $f
        Remove-Item -LiteralPath $f -Force -ErrorAction SilentlyContinue
        git fetch origin
        $behind = (git --no-optional-locks rev-list --count HEAD..origin/main).Trim()
        if ($behind -ne "0") { Write-Host "origin is $behind ahead. Rebase, then push." -ForegroundColor Red }
        else { git push; if ($LASTEXITCODE -eq 0) { Write-Host "Pushed." -ForegroundColor Green } }
    }
}

$inv = Join-Path $Dst "_inv\source_inventory.jsonl"

if ($Scan) {
    Write-Host "`nScanning sources. This reads headers only; it does not decode." -ForegroundColor Cyan
    New-Item -ItemType Directory -Path (Split-Path $inv) -Force | Out-Null
    python pipeline\scan_sources.py $Src --out $inv
    if (Test-Path -LiteralPath $Extra) {
        $inv2 = Join-Path $Dst "_inv\source_inventory_extra.jsonl"
        python pipeline\scan_sources.py $Extra --out $inv2
        Get-Content $inv2 | Add-Content $inv
        Write-Host "appended $Extra to the inventory" -ForegroundColor DarkGray
    }
    $n = (Get-Content $inv | Measure-Object -Line).Lines
    Write-Host "inventory: $n sources" -ForegroundColor Green
}

if ($Ingest) {
    if (-not (Test-Path -LiteralPath $inv)) { Write-Host "no inventory; run -Scan first" -ForegroundColor Red; exit 1 }
    Write-Host "`nIngesting at $TonemapEv EV. Long job." -ForegroundColor Cyan
    python pipeline\prepare_pairs.py --inventory $inv --dst $Dst --mode log2_extended --crops 3 --video-stride 8 --tonemap-ev $TonemapEv
}

if ($Manifest) {
    # 0.25 matches verify_dataset's check 7. Passing 0.35 here, as this script
    # did on its first run, thins the dominant scene to a share the gate then
    # rejects -- which is exactly how corpus_v4b failed on 34.7% of test.
    $holdArgs = @()
    if ($HoldOut -and (Test-Path -LiteralPath $HoldOut)) { $holdArgs = @("--hold-out-scenes", $HoldOut) }
    else { Write-Host "no hold-out manifest at '$HoldOut' -- the old bench scenes may land in train" -ForegroundColor Red }
    python pipeline\build_manifests.py --pairs-dir $Dst --out-dir $Dst --max-eval-scene-share 0.25 @holdArgs
}

if ($Verify) {
    python pipeline\verify_dataset.py --pairs-dir $Dst --manifest (Join-Path $Dst "sdr_hdr_manifest.jsonl") --video-manifest (Join-Path $Dst "video_manifest_9f.jsonl")
}

if (-not ($Push -or $Scan -or $Ingest -or $Manifest -or $Verify)) {
    Write-Host @"

Nothing selected. The order is:

  -Push                 commit and push the gate fix (do this first)
  -Scan                 $Src (+ $Extra) -> inventory
  -Ingest               inventory -> $Dst, rendered at $TonemapEv EV
  -Manifest             $Dst -> sdr_hdr_manifest.jsonl + video_manifest_9f.jsonl
  -Verify               the gate

Or all at once:
  .\BUILD_CORPUS_V4.ps1 -Scan -Ingest -Manifest -Verify

Check 3b is the one that matters. If it fails at 0 EV, the sources
themselves peak near diffuse white and no exposure will fix them -- which
is a result worth more than a retrain.
"@ -ForegroundColor Yellow
}
