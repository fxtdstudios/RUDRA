# RUDRA -- fix the two CI failures and push.  (v2)
#   powershell -ExecutionPolicy Bypass -File FIX_CI.ps1
#
# WHAT FAILED IN CI
#
# Run 95091897100, both pytest jobs, same two tests:
#
#   test_an_empty_sequence_is_refused   expected 'no frames'
#   test_a_non_image_frame_is_refused   expected '(H, W, 3)'
#   both got  'ffmpeg is not on PATH, so nothing can be encoded.'
#
# The runner has no ffmpeg and encode_sequence probed for it BEFORE it looked
# at the frames, so an empty sequence reported the wrong problem. Fixed by
# ordering: arguments, then the environment, then anything with side effects.
#
# WHAT V1 OF THIS SCRIPT GOT WRONG
#
# It ran pytest without --basetemp, so pytest fell back to its default,
# %TEMP%\pytest-of-Ahmed -- and that folder is not readable on this machine:
#
#   PermissionError: [WinError 5] Access is denied:
#     'C:\Users\Ahmed\AppData\Local\Temp\pytest-of-Ahmed'
#
# Every failure was at SETUP of the tmp_path fixture, before a single test body
# ran, so it said nothing about the code. That folder almost certainly got
# restrictive ACLs from a pytest run under an elevated shell; once it exists,
# every later non-elevated run trips on it. This version gives both runs their
# own fresh basetemp and never touches the default, and it offers to clear the
# stale folder.
#
# It also refuses to report a pass from the hidden-ffmpeg run unless ffmpeg is
# genuinely gone from PATH -- a strip that silently failed would have produced
# a green run that proved nothing.
$ErrorActionPreference = "Continue"
Set-Location -LiteralPath "D:\A.I\Devlopments\rudra"

$tempRoot = $env:TEMP
if (-not $tempRoot) { $tempRoot = [System.IO.Path]::GetTempPath() }

# ---- 0. the stale basetemp ----------------------------------------------
$stale = Join-Path $tempRoot "pytest-of-$env:USERNAME"
if (Test-Path -LiteralPath $stale) {
    $readable = $true
    try { [void](Get-ChildItem -LiteralPath $stale -ErrorAction Stop) }
    catch { $readable = $false }
    if (-not $readable) {
        Write-Host "Stale pytest basetemp is unreadable:" -ForegroundColor Yellow
        Write-Host "   $stale" -ForegroundColor Yellow
        Write-Host "Nothing below uses it, so this is not blocking. To clear it:" -ForegroundColor Yellow
        Write-Host "   takeown /f `"$stale`" /r /d y" -ForegroundColor Cyan
        Write-Host "   icacls `"$stale`" /grant `"$env:USERNAME`":F /t" -ForegroundColor Cyan
        Write-Host "   rmdir /s /q `"$stale`"" -ForegroundColor Cyan
        Write-Host ""
    }
}

function Fresh-Basetemp {
    param([string]$Name)
    $p = Join-Path $tempRoot $Name
    if (Test-Path -LiteralPath $p) {
        Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction SilentlyContinue
    }
    return $p
}

$before = (git --no-optional-locks rev-parse HEAD).Trim()
Write-Host "HEAD is $before" -ForegroundColor DarkGray

# ---- 1. the delivery tests with ffmpeg hidden, the way CI sees them ------
Write-Host ""
Write-Host "Delivery tests with ffmpeg hidden from PATH." -ForegroundColor Cyan
$saved = $env:PATH
$kept = @()
foreach ($dir in ($saved -split ';')) {
    if (-not $dir) { continue }
    $has = $false
    try { $has = Test-Path -LiteralPath (Join-Path $dir "ffmpeg.exe") } catch { $has = $false }
    if (-not $has) { $kept += $dir }
}
$env:PATH = ($kept -join ';')

# Prove the strip worked. Reporting a pass from a run that still had ffmpeg
# would be worse than not running it at all.
$stillThere = Get-Command ffmpeg -ErrorAction SilentlyContinue
if ($stillThere) {
    $env:PATH = $saved
    Write-Host "ffmpeg is still reachable at $($stillThere.Source)." -ForegroundColor Red
    Write-Host "Cannot reproduce the CI condition here. Push and let CI judge it." -ForegroundColor Red
    exit 1
}

$bt1 = Fresh-Basetemp "rudra-noffmpeg"
python -m pytest tests/test_delivery_video_2026_09_11.py -q --basetemp="$bt1"
$hidden = $LASTEXITCODE
$env:PATH = $saved
if ($hidden -ne 0) {
    Write-Host ""
    Write-Host "Still failing without ffmpeg. Not committing." -ForegroundColor Red
    exit 1
}
Write-Host "Passes without ffmpeg -- which is the condition CI runs in." -ForegroundColor Green

# ---- 2. the full suite, ffmpeg back -------------------------------------
Write-Host ""
Write-Host "Full suite, with ffmpeg available." -ForegroundColor Cyan
$bt2 = Fresh-Basetemp "rudra-pytest"
python -m pytest tests/ -q --basetemp="$bt2"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Full suite fails. Not committing." -ForegroundColor Red
    exit 1
}

# ---- 3. commit and push --------------------------------------------------
git reset --quiet
git add -- "rudra/delivery/video.py"
if ($LASTEXITCODE -ne 0) { Write-Host "git add failed." -ForegroundColor Red; exit 1 }
$staged = git --no-optional-locks diff --cached --name-only
if (-not $staged) {
    Write-Host "Nothing changed -- already committed?" -ForegroundColor Yellow
    git reset --quiet; exit 0
}

$msg = @"
Validate the arguments before probing for ffmpeg

Both CI pytest jobs failed on the same two tests:

  test_an_empty_sequence_is_refused   expected 'no frames'
  test_a_non_image_frame_is_refused   expected '(H, W, 3)'
  both got 'ffmpeg is not on PATH, so nothing can be encoded.'

The runner has no ffmpeg, and encode_sequence checked for it before it looked
at the frames, so an empty sequence reported the wrong problem.

Worth fixing rather than skipping. A caller who passes nothing to encode should
be told they passed nothing, whether or not a codec is installed -- being sent
to install ffmpeg when the real fault is an empty list costs someone an
afternoon. The test file already marks @needs_ffmpeg on the tests that genuinely
need it and leaves these two unmarked, so the intent was always that argument
validation is environment-independent.

Reordered to: arguments, then the environment, then anything with side effects.
Consuming the first frame is also what validates it, so the shape check now runs
before the output directory is created -- a call that could never succeed no
longer leaves a folder behind.

Verified with shutil.which stubbed to None, and again by running the delivery
tests with ffmpeg stripped from PATH: both messages correct, the unknown target
still caught first, a valid call still reports ffmpeg missing, no directory
created.
"@
$f = Join-Path $tempRoot "rudra_msg_ci.txt"
$msg | Set-Content -LiteralPath $f -Encoding ASCII
git commit -F $f
if ($LASTEXITCODE -ne 0) { Write-Host "commit failed." -ForegroundColor Red; exit 1 }
Remove-Item -LiteralPath $f -Force -ErrorAction SilentlyContinue

git fetch origin
$behind = (git --no-optional-locks rev-list --count HEAD..origin/main).Trim()
if ($behind -ne "0") {
    Write-Host "origin is $behind ahead. Rebase first." -ForegroundColor Red
    exit 1
}
git push
if ($LASTEXITCODE -ne 0) { Write-Host "push failed; the commit stands." -ForegroundColor Red; exit 1 }
Write-Host ""
Write-Host "Pushed." -ForegroundColor Green
git --no-optional-locks log --oneline -3
