# Three-seed repeat of the exposure-drift gate, on the GPU.
#
# WHY THIS EXISTS. On 6 Sep 2026 the paired gate came back at +0.511 JOD
# achievable with exposure drift against +0.110 without it, on 50 scenes and
# identical camera paths. The threshold is +0.5. Eleven thousandths over the
# line on ONE seed is not a result to re-render a 993-scene corpus on, so this
# runs the same pairing at two further seeds.
#
# It also exists because that run cost three and a half hours on two CPU
# cores. Nearly all of it was ColorVideoVDP, which runs wherever the tensors
# it is handed live -- so with -Device cuda on the 4080 this should be minutes.
#
# What one seed does: renders the 50 named panoramas TWICE, at constant
# exposure and with drift, from the same seed. PYTHONHASHSEED is pinned
# because the renderer draws its per-clip RNG from hash() of a string, which
# Python salts per process -- without it the two conditions get different
# camera paths and the pairing is worthless.
#
#   powershell -ExecutionPolicy Bypass -File training\run_drift_gate.ps1
#
# Resumable: every stage skips work already on disk, so re-running after an
# interruption picks up where it stopped.

param(
  [int[]] $Seeds       = @(20260906, 20260907),
  [double] $Drift      = 0.12,
  [string] $Panoramas  = "D:\A.I\Devlopments\RUDRA_v02\panoramas",
  [string] $SceneList  = "D:\A.I\Devlopments\rudra\docs\gate_v02_2026-09-05\_gate50_scenes.txt",
  [string] $Work       = "D:\A.I\Devlopments\RUDRA_v02\_gate_seeds",
  [string] $Checkpoint = "D:\A.I\Devlopments\rudra\checkpoints\sdr2hdr_shadow_v1.pt",
  [string] $Device     = "cuda",
  [string] $Python     = "D:\A.I\Devlopments\rudra\.venv\Scripts\python.exe"
)

# NOT "Stop". Windows PowerShell 5.1 turns any native stderr into a
# NativeCommandError under Stop, and python writes progress bars to stderr --
# so Stop aborts on output that is not an error. Every step checks
# $LASTEXITCODE instead. Same reason as pipeline/build_moves_corpus.ps1.
$ErrorActionPreference = "Continue"

$repo = "D:\A.I\Devlopments\rudra"
$env:PYTHONHASHSEED = "0"
$env:OPENCV_IO_ENABLE_OPENEXR = "1"

function Fail($message) { Write-Host "ERROR: $message" -ForegroundColor Red; exit 1 }

# ---- preflight ------------------------------------------------------------
foreach ($path in @($Python, $Checkpoint, $SceneList, $Panoramas)) {
  if (-not (Test-Path $path)) { Fail "not found: $path" }
}
# --degradation codec shells out to ffmpeg. Find that out now, not 40 minutes
# into a render.
# Get-Command, not `& ffmpeg`: a missing executable raises
# CommandNotFoundException and leaves $LASTEXITCODE holding whatever the
# previous native command set, so testing the exit code would pass.
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
  Fail ("ffmpeg is not on PATH, and --degradation codec needs it. " +
        "Install it (winget install Gyan.FFmpeg) and open a new shell.")
}
& $Python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)"
if ($Device -eq "cuda" -and $LASTEXITCODE -ne 0) {
  Fail "torch reports no CUDA device. Re-run with -Device cpu (slow: hours)."
}

$scenes = (Get-Content $SceneList | Where-Object { $_.Trim() }).Count
Write-Host "   scenes     : $scenes"
Write-Host "   seeds      : $($Seeds -join ', ')"
Write-Host "   device     : $Device"
New-Item -ItemType Directory -Force -Path $Work | Out-Null

# ---- the sweep ------------------------------------------------------------
$summary = @()
foreach ($seed in $Seeds) {
  foreach ($arm in @(@{ name = "flat"; drift = 0.0 }, @{ name = "drift"; drift = $Drift })) {
    $tag     = "seed$seed" + "_" + $arm.name
    $render  = Join-Path $Work "$tag`_render"
    $clips   = Join-Path $Work "$tag`_clips"
    $done    = Join-Path $Work "$tag`_done.txt"
    $out     = Join-Path $Work "$tag`.json"

    if (Test-Path $out) { Write-Host "   $tag already scored, skipping"; $summary += $out; continue }

    Write-Host ""
    Write-Host "== $tag : render ==" -ForegroundColor Cyan
    & $Python "$repo\pipeline\render_hdri_moves.py" `
        --hdri-dir $Panoramas --scene-list $SceneList --dst $render `
        --clips-per-hdri 1 --seed $seed --exposure-drift $arm.drift `
        --done-file $done
    if ($LASTEXITCODE -ne 0) { Fail "render failed for $tag" }

    Write-Host "== $tag : clips ==" -ForegroundColor Cyan
    & $Python "$repo\training\stage_oracle_clips.py" `
        --from-render --meta-dir "$render\meta" --dest $clips
    if ($LASTEXITCODE -ne 0) { Fail "clip staging failed for $tag" }

    Write-Host "== $tag : gate ==" -ForegroundColor Cyan
    & $Python "$repo\training\gate_temporal_oracle.py" `
        --clips $clips --checkpoint $Checkpoint --condition hard `
        --degradation codec --device $Device --out $out
    if ($LASTEXITCODE -ne 0) { Fail "gate failed for $tag" }
    $summary += $out
  }
}

# ---- read it back ---------------------------------------------------------
Write-Host ""
Write-Host "== achievable JOD, per arm ==" -ForegroundColor Green
foreach ($path in $summary) {
  $rows = Get-Content $path -Raw | ConvertFrom-Json
  $pf  = ($rows | ForEach-Object { $_.per_frame.cvvdp_jod }    | Measure-Object -Average).Average
  $am  = ($rows | ForEach-Object { $_.aligned_mean.cvvdp_jod } | Measure-Object -Average).Average
  $ct  = ($rows | ForEach-Object { $_.control.cvvdp_jod }      | Measure-Object -Average).Average
  $orc = ($rows | ForEach-Object { $_.oracle.cvvdp_jod }       | Measure-Object -Average).Average
  # "+6" is not a valid alignment; a signed custom format is. Getting this
  # wrong throws at the very end of a run that took hours.
  "{0,-28} per-frame {1,7:0.000}   ACHIEVABLE {2,7:+0.000;-0.000}   ceiling {3,7:+0.000;-0.000}" -f `
    [IO.Path]::GetFileNameWithoutExtension($path), $pf, ($am - $pf), ($orc - $ct)
}
Write-Host ""
Write-Host "The threshold is +0.5 JOD achievable. 6 Sep, seed 20260903:"
Write-Host "  flat +0.110, drift +0.511. A drift arm that does not clear +0.5"
Write-Host "  on the other seeds means the 6 Sep result was noise."
