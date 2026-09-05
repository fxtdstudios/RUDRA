<#
  Build the v02 temporal corpus: fetch Poly Haven HDRIs, render virtual camera
  moves, write the video manifest.

  The v01 video corpus is 935 clips across 13 SCENES, split 11 train / 1 val /
  1 test, which is why the temporal refiner is reported as unevaluated. One
  panorama renders as one scene, and Poly Haven publishes 993 of them under
  CC0, so this turns 13 scenes into ~993.

      powershell -ExecutionPolicy Bypass -File pipeline\build_moves_corpus.ps1

  Safe to stop and re-run. The renderer appends every panorama it finishes to
  rendered.txt and the fetcher skips those, so an interrupted run resumes
  where it stopped rather than starting over.

  Measured on 4 September 2026:
      4k EXR          50 MB mean   ->  48 GB for 993 panoramas
      rendered pair   4.69 MB      ->  82 GB for 993 x 2 clips x 9 frames
      D: had 302 GB free, E: had 40 GB. Hence the D: default.

  4k is the smallest resolution the renderer accepts: 4096 px equirect gives
  853 source px across a 1280-wide 75 deg frame, a 1.5x upscale, inside
  --max-upscale's default of 2.0. 2k gives 3.0x and is correctly refused.

  NOTE ON ERROR HANDLING. $ErrorActionPreference is deliberately NOT "Stop"
  here. Under Windows PowerShell 5.1 that setting turns ANY native command's
  stderr output into a NativeCommandError, so a single Python warning aborts
  a multi-hour corpus build. Every step below checks $LASTEXITCODE instead,
  which is what actually says whether the step failed.
#>
param(
    [string]$Root         = "D:\A.I\Devlopments\RUDRA_v02",
    [string]$Repo         = "D:\A.I\Devlopments\rudra",
    [string]$Python       = "python",
    [int]   $ClipsPerHdri = 2,
    [int]   $Frames       = 9,
    [int]   $BatchSize    = 50,
    [int]   $Limit        = 0,          # 0 = every published HDRI
    [switch]$DropSource                 # delete each panorama after it renders
)

$ErrorActionPreference = "Continue"

$panoramas = Join-Path $Root "panoramas"
$pairs     = Join-Path $Root "pairs_moves"
$doneFile  = Join-Path $Root "rendered.txt"
$manifest  = Join-Path $Root "video_manifest_moves.jsonl"

New-Item -ItemType Directory -Force -Path $panoramas | Out-Null
New-Item -ItemType Directory -Force -Path $pairs     | Out-Null

Push-Location $Repo
try {
    if ($Limit -gt 0) {
        $total = $Limit
    } else {
        $total = [int](& $Python pipeline\fetch_polyhaven.py --count)
        if ($LASTEXITCODE -ne 0 -or $total -le 0) {
            Write-Error "could not reach the Poly Haven API"; exit 1
        }
    }

    Write-Host "   corpus root : $Root"
    Write-Host "   panoramas   : $total at 4k"
    Write-Host "   clips       : $ClipsPerHdri per panorama x $Frames frames"
    Write-Host ""

    # Fetch and render in batches. One pass over everything would park 48 GB of
    # panoramas before a single frame is written, and would lose the lot to any
    # interruption.
    for ($skip = 0; $skip -lt $total; $skip += $BatchSize) {
        $n = [Math]::Min($BatchSize, $total - $skip)
        Write-Host "== batch $skip..$($skip + $n - 1) of $total =="

        & $Python pipeline\fetch_polyhaven.py --dest $panoramas --res 4k `
            --skip $skip --limit $n --exclude-file $doneFile
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "fetch reported failures in this batch; rendering what arrived"
        }

        $render = @("pipeline\render_hdri_moves.py",
                    "--hdri-dir", $panoramas, "--dst", $pairs,
                    "--clips-per-hdri", $ClipsPerHdri, "--frames", $Frames,
                    "--done-file", $doneFile)
        if ($DropSource) { $render += "--drop-source" }
        & $Python @render
        if ($LASTEXITCODE -ne 0) {
            Write-Error "render failed in the batch starting at $skip"; exit 1
        }
    }

    Write-Host ""
    Write-Host "== manifest =="
    & $Python training\build_video_manifest.py `
        --hdr-dir (Join-Path $pairs "hdr") --sdr-dir (Join-Path $pairs "sdr") `
        --output $manifest --clip-length $Frames --frame-step 1
    if ($LASTEXITCODE -ne 0) { Write-Error "manifest build failed"; exit 1 }

    Write-Host ""
    Write-Host "   manifest : $manifest"
    Write-Host "   next     : python pipeline\verify_dataset.py --pairs-dir $pairs --video-manifest $manifest"
}
finally { Pop-Location }
