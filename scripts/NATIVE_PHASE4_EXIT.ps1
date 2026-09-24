<#
.SYNOPSIS
  Phase 4 exit for the native app on this Windows machine: a movie in, an
  HDR10, HLG or ProRes master out with its audio, checked before it is
  published, from rudra-native, from the queue and from RUDRA.exe, with no
  Python behind any of them.

.DESCRIPTION
  Steps:
    1. Dependencies and the build, as NATIVE_PHASE3_EXIT.ps1 (Qt 6.8, ONNX
       Runtime with DirectML, OpenCV), into build\native_phase4, plus the
       native tests.
    2. ffmpeg: found on PATH (or -Ffmpeg, the folder with ffmpeg.exe and
       ffprobe.exe); rudra-native ffmpeg-check asks it for libx265, prores_ks,
       zscale and alphaextract and runs a 16-frame HDR10 export through it.
    3. The native tests of the video pipeline (probe, decode, predictor
       parts, mastering, the encoder command, QC, sequences, the queue),
       against the goldens the Python wrote.
    4. rudra-native video on -Clip: HDR10 with its audio, and ProRes 422 HQ;
       each published only after QC, its report read back.
    5. A three-job queue (HDR10, HLG, ProRes 4444 when -Clip has alpha, else
       ProRes 422 HQ) through rudra-native batch run, stopped hard as the
       second job starts and run again: the first verified and skipped, the
       rest finished.
    6. RUDRA.exe --workflow-check with --movie -Clip on a PATH of the app's
       folder, the ffmpeg folder and Windows only: the Phase 3 workflow on
       -Frames, then the movie opened as a shot,
       every frame scrubbed, and HDR10 exported through the app's queue.
       -Frames defaults to the clip's first 48 frames.
    7. The table, a report in reports\, and the by-hand checklist.

  Needs: what NATIVE_PHASE3_EXIT.ps1 needs, and an ffmpeg with libx265 and
  zscale (the gyan.dev "full" build: winget install Gyan.FFmpeg).

.EXAMPLE
  .\scripts\NATIVE_PHASE4_EXIT.ps1 -Clip D:\shots\sh010.mov
  .\scripts\NATIVE_PHASE4_EXIT.ps1 -Clip D:\shots\sh010.mov -Ffmpeg C:\ffmpeg\bin -SkipBuild
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Clip,
    [string]$Frames = "",
    [string]$Package = "dist\models\sdr2hdr_shadow_v1",
    [string]$Checkpoint = "checkpoints\sdr2hdr_shadow_v1.pt",
    [string]$Backend = "onnxruntime/directml",
    [string]$Ffmpeg = "",
    [string]$Python = "python",
    [string]$QtVersion = "6.8.3",
    [string]$OrtVersion = "1.22.0",
    [string]$OpenCvVersion = "4.10.0",
    [switch]$SkipBuild,
    [switch]$SkipTests,
    [switch]$InstallBuildTools
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Deps = Join-Path $Repo "tmp\native_deps"
$QtRoot = Join-Path $Deps "Qt\$QtVersion\msvc2022_64"
$Build = Join-Path $Repo "build\native_phase4"
$Stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$Reports = Join-Path $Repo "reports"
$Work = Join-Path $Repo "tmp\phase4_$Stamp"
New-Item -ItemType Directory -Force -Path $Deps, $Reports, $Work | Out-Null

function Say($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
. (Join-Path $PSScriptRoot "native_toolchain.ps1")

$Package = (Resolve-Path $Package -ErrorAction SilentlyContinue).Path
if (-not $Package -or -not (Test-Path (Join-Path $Package "manifest.json"))) { Fail "no model package (run NATIVE_GATE_A.ps1 first, or pass -Package)" }
$Clip = (Resolve-Path $Clip -ErrorAction SilentlyContinue).Path
if (-not $Clip -or -not (Test-Path $Clip -PathType Leaf)) { Fail "no movie at -Clip" }
$Runtime, $Device = $Backend.Split("/")

# ffmpeg: -Ffmpeg, else PATH.
if ($Ffmpeg) {
    if (-not (Test-Path (Join-Path $Ffmpeg "ffmpeg.exe"))) { Fail "no ffmpeg.exe in -Ffmpeg $Ffmpeg" }
    $FfDir = (Resolve-Path $Ffmpeg).Path
} else {
    $ff = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if (-not $ff) { Fail "ffmpeg is not on PATH (winget install Gyan.FFmpeg, or pass -Ffmpeg)" }
    $FfDir = Split-Path $ff.Source
}
if (-not (Test-Path (Join-Path $FfDir "ffprobe.exe"))) { Fail "no ffprobe.exe beside ffmpeg in $FfDir" }
$env:PATH = "$FfDir;$env:PATH"
Write-Host "ffmpeg from $FfDir"
# The Phase 3 workflow runs on a folder of frames: -Frames, else the clip's first 48.
if (-not $Frames) {
    $Frames = Join-Path $Work "frames"
    New-Item -ItemType Directory -Force -Path $Frames | Out-Null
    & (Join-Path $FfDir "ffmpeg.exe") -v error -nostdin -i $Clip -frames:v 48 (Join-Path $Frames "f_%04d.png")
    if ($LASTEXITCODE -ne 0) { Fail "could not take frames from -Clip" }
}
$Frames = (Resolve-Path $Frames -ErrorAction SilentlyContinue).Path
if (-not $Frames -or -not (Test-Path $Frames -PathType Container)) { Fail "no folder of frames at -Frames" }

# ---------------------------------------------------------------------------
Say "Dependencies"
if (-not (Test-Path (Join-Path $QtRoot "bin\qsb.exe"))) {
    & $Python -m pip install --quiet --upgrade aqtinstall
    if ($LASTEXITCODE -ne 0) { Fail "pip install aqtinstall" }
    & $Python -m aqt install-qt windows desktop $QtVersion win64_msvc2022_64 -m qtshadertools -O (Join-Path $Deps "Qt")
    if ($LASTEXITCODE -ne 0) { Fail "aqt install-qt" }
}
Write-Host "Qt at $QtRoot"

function Get-Nupkg($id, $version, $dest) {
    if (-not (Test-Path $dest)) {
        $zip = Join-Path $Deps "$id.$version.zip"
        $url = "https://api.nuget.org/v3-flatcontainer/$($id.ToLower())/$version/$($id.ToLower()).$version.nupkg"
        Write-Host "download $url"
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
        Expand-Archive -Path $zip -DestinationPath $dest -Force
        Remove-Item $zip
    }
    return $dest
}
$OrtPkg = Get-Nupkg "Microsoft.ML.OnnxRuntime.DirectML" $OrtVersion (Join-Path $Deps "ort-dml-$OrtVersion")
$nuspec = Get-ChildItem $OrtPkg -Filter *.nuspec | Select-Object -First 1
[xml]$spec = Get-Content $nuspec.FullName
$dmlDep = $spec.SelectNodes("//*[local-name()='dependency' and @id='Microsoft.AI.DirectML']") | Select-Object -First 1
$DmlVersion = if ($dmlDep) { $dmlDep.version.Trim("[]() ").Split(",")[0] } else { "1.15.4" }
$DmlPkg = Get-Nupkg "Microsoft.AI.DirectML" $DmlVersion (Join-Path $Deps "directml-$DmlVersion")
$OrtRoot = Join-Path $Deps "ort-root-$OrtVersion"
New-Item -ItemType Directory -Force -Path "$OrtRoot\include", "$OrtRoot\lib" | Out-Null
Copy-Item "$OrtPkg\build\native\include\*" "$OrtRoot\include\" -Force
Copy-Item "$OrtPkg\runtimes\win-x64\native\*" "$OrtRoot\lib\" -Force
$DmlDll = Get-ChildItem "$DmlPkg\bin\x64-win" -Filter DirectML.dll | Select-Object -First 1
if (-not $DmlDll) { Fail "DirectML.dll not found in $DmlPkg" }
Write-Host "ONNX Runtime $OrtVersion (DirectML), DirectML $DmlVersion"

# OpenCV: the official package is a 7-Zip self-extractor.
$CvRoot = Join-Path $Deps "opencv-$OpenCvVersion"
$CvBuild = Join-Path $CvRoot "opencv\build"
if (-not (Test-Path (Join-Path $CvBuild "OpenCVConfig.cmake"))) {
    $exe = Join-Path $Deps "opencv-$OpenCvVersion-windows.exe"
    $url = "https://github.com/opencv/opencv/releases/download/$OpenCvVersion/opencv-$OpenCvVersion-windows.exe"
    Write-Host "download $url"
    Invoke-WebRequest -Uri $url -OutFile $exe -UseBasicParsing
    $p = Start-Process -FilePath $exe -ArgumentList @("-o`"$CvRoot`"", "-y") -Wait -PassThru
    if ($p.ExitCode -ne 0 -or -not (Test-Path (Join-Path $CvBuild "OpenCVConfig.cmake"))) { Fail "OpenCV extract" }
    Remove-Item $exe
}
# The pack's top-level OpenCVConfig.cmake picks binaries by MSVC_VERSION and
# knows nothing newer than VS 2022 (19.4x), so VS 2026 (19.5x) finds "no
# compatible binaries". The MSVC ABI is stable since 2015: use the newest vcNN
# folder's own config directly.
$CvLib = Get-ChildItem (Join-Path $CvBuild "x64") -Directory | Where-Object { $_.Name -match "^vc\d+$" } |
         Sort-Object { [int]($_.Name.Substring(2)) } -Descending | Select-Object -First 1
if (-not $CvLib -or -not (Test-Path (Join-Path $CvLib.FullName "lib\OpenCVConfig.cmake"))) { Fail "no vcNN\lib\OpenCVConfig.cmake under $CvBuild\x64" }
$CvConfigDir = Join-Path $CvLib.FullName "lib"
$CvDll = Get-ChildItem (Join-Path $CvBuild "x64") -Recurse -Filter "opencv_world*.dll" |
         Where-Object { $_.Name -notmatch "d\.dll$" } | Select-Object -First 1
if (-not $CvDll) { Fail "opencv_world DLL not found under $CvBuild" }
Write-Host "OpenCV $OpenCvVersion ($($CvDll.Name), $($CvLib.Name))"

# ---------------------------------------------------------------------------
if (-not $SkipBuild) {
    Say "Configure and build RUDRA, rudra-native and the app tests (Visual Studio, Release)"
    $tc = Find-NativeToolchain
    if (-not $tc -and $InstallBuildTools) {
        if (-not (Install-NativeBuildTools)) { Fail "Build Tools install" }
        $tc = Find-NativeToolchain
    }
    if (-not $tc) { Fail "no C++ toolchain (see above)" }
    $cmake = $tc.CMake
    Reset-StaleCMakeCache $Build $tc.Generator
    & $cmake -S native -B $Build -G $tc.Generator -A x64 `
        -DRUDRA_BUILD_TESTS=ON -DRUDRA_BUILD_CLI=ON -DRUDRA_BUILD_APP=ON -DRUDRA_BUILD_RENDER=ON `
        -DRUDRA_WITH_ONNXRUNTIME=ON "-DONNXRUNTIME_ROOT=$OrtRoot" -DRUDRA_WITH_LIBTORCH=OFF `
        -DRUDRA_WITH_OPENCV=ON "-DOpenCV_DIR=$CvConfigDir" "-DCMAKE_PREFIX_PATH=$QtRoot"
    if ($LASTEXITCODE -ne 0) { Fail "cmake configure" }
    & $cmake --build $Build --config Release --parallel --target RUDRA rudra-native rudra_app_tests rudra_tests
    if ($LASTEXITCODE -ne 0) { Fail "build" }
}
$App = Join-Path $Build "app\Release\RUDRA.exe"
$Cli = Join-Path $Build "cli\Release\rudra-native.exe"
$AppTests = Join-Path $Build "app\Release\rudra_app_tests.exe"
$Tests = Join-Path $Build "tests\Release\rudra_tests.exe"
foreach ($e in @($App, $Cli)) { if (-not (Test-Path $e)) { Fail "not built: $e" } }

Say "Deploy"
foreach ($e in @($App, $AppTests, $Tests)) {
    if (Test-Path $e) { & (Join-Path $QtRoot "bin\windeployqt.exe") --release --no-translations --no-compiler-runtime $e | Out-Null }
}
# windeployqt deploys the "windows" platform plugin only; the offscreen tests need theirs.
$Platforms = Join-Path (Split-Path $App) "platforms"
New-Item -ItemType Directory -Force -Path $Platforms | Out-Null
Copy-Item (Join-Path $QtRoot "plugins\platforms\qoffscreen.dll") $Platforms -Force
foreach ($dir in @((Split-Path $App), (Split-Path $Cli), (Split-Path $Tests))) {
    if (-not (Test-Path $dir)) { continue }
    Copy-Item "$OrtRoot\lib\*.dll" $dir -Force
    Copy-Item $DmlDll.FullName $dir -Force
    Copy-Item $CvDll.FullName $dir -Force
}
Write-Host "RUDRA.exe and rudra-native.exe with their DLLs beside them"


$results = [ordered]@{}
function Note($name, $ok, $what) { $results[$name] = [pscustomobject]@{ Step = $name; Result = $(if ($ok) { "PASS" } else { "FAIL" }); What = $what } }
function Run-Quiet($exe, $argv) {
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    $out = & $exe @argv 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    return @{ Code = $code; Text = @($out) }
}

# ---------------------------------------------------------------------------
Say "ffmpeg on this machine"
$r = Run-Quiet $Cli @("ffmpeg-check", "--force")
$r.Text | Write-Host
$caps = ($r.Text -join "`n") | ConvertFrom-Json -ErrorAction SilentlyContinue
Note "ffmpeg" ($r.Code -eq 0) $(if ($caps) { "$($caps.version); self-test $([math]::Round($caps.self_test.seconds, 1)) s" } else { "no report" })

# ---------------------------------------------------------------------------
if (-not $SkipTests -and (Test-Path $Tests)) {
    Say "The native tests of the video pipeline"
    $r = Run-Quiet $Tests @("--gtest_brief=1", "--gtest_filter=Process*:Video*:Sequence*:SequenceEncode*:FfmpegCheck*:Queue*")
    $r.Text | Set-Content -Encoding utf8 (Join-Path $Reports "native_phase4_tests_$Stamp.txt")
    $r.Text | Where-Object { $_ -match "Failure|FAILED|PASSED" } | Write-Host
    Note "tests" ($r.Code -eq 0) (($r.Text | Where-Object { $_ -match "PASSED|FAILED" } | Select-Object -Last 1))
}
if (-not $SkipTests -and (Test-Path $AppTests)) {
    Say "The app's Qt tests (offscreen)"
    $env:QT_QPA_PLATFORM = "offscreen"
    $r = Run-Quiet $AppTests @("--gtest_brief=1")
    Remove-Item Env:QT_QPA_PLATFORM
    $r.Text | Set-Content -Encoding utf8 (Join-Path $Reports "native_phase4_app_tests_$Stamp.txt")
    $r.Text | Where-Object { $_ -match "Failure|FAILED|PASSED" } | Write-Host
    Note "app tests" ($r.Code -eq 0) (($r.Text | Where-Object { $_ -match "PASSED|FAILED" } | Select-Object -Last 1))
}

# ---------------------------------------------------------------------------
Say "rudra-native video: HDR10 with its audio, and ProRes 422 HQ"
$stem = [IO.Path]::GetFileNameWithoutExtension($Clip)
foreach ($f in @(@{ Format = "hdr10"; Ext = ".mp4" }, @{ Format = "prores422hq"; Ext = ".mov" })) {
    $out = Join-Path $Work "$stem`_$($f.Format)$($f.Ext)"
    $t = [Diagnostics.Stopwatch]::StartNew()
    $r = Run-Quiet $Cli @("video", $Package, $Clip, "--output", $out, "--format", $f.Format, "--runtime", $Runtime, "--device", $Device)
    $t.Stop()
    $r.Text | Select-Object -Last 3 | Write-Host
    $side = "$out.json"
    $ok = ($r.Code -eq 0) -and (Test-Path $out) -and (Test-Path $side)
    $what = "exit $($r.Code)"
    if (Test-Path $side) {
        $rep = Get-Content $side -Raw | ConvertFrom-Json
        $ok = $ok -and $rep.qc.passed
        $what = "$($rep.timing.frames) frames at $($rep.timing.fps), MaxCLL $($rep.max_cll), MaxFALL $($rep.max_fall), $($rep.qc.audio_streams) audio, QC passed, $([math]::Round($t.Elapsed.TotalSeconds, 1)) s"
    } else { $what += ": " + (($r.Text | Select-Object -Last 1) -join "") }
    Note "video $($f.Format)" $ok $what
}

# ---------------------------------------------------------------------------
Say "The queue: three jobs, stopped hard half way, resumed"
$qdir = Join-Path $Work "queue"
New-Item -ItemType Directory -Force -Path $qdir | Out-Null
$alpha = (& (Join-Path $FfDir "ffprobe.exe") -v error -select_streams v:0 -show_entries stream=pix_fmt -of csv=p=0 $Clip) -match "a"
$third = if ($alpha) { @{ output = "c_prores4444.mov"; options = @{ format = "prores4444"; alpha_mode = "straight" } } }
         else { @{ output = "c_prores422hq.mov"; options = @{ format = "prores422hq" } } }
$queue = [ordered]@{
    version = 1
    defaults = @{ checkpoint = (Resolve-Path $Checkpoint).Path; device = "cpu" }
    jobs = @(
        @{ input = $Clip; output = "a_hdr10.mp4"; options = @{ format = "hdr10" } },
        @{ input = $Clip; output = "b_hlg.mp4"; options = @{ format = "hlg" } },
        (@{ input = $Clip } + $third))
}
$qfile = Join-Path $qdir "queue.json"
# No BOM: the Python's json.loads refuses one, and the queue must run on either side.
[IO.File]::WriteAllText($qfile, ($queue | ConvertTo-Json -Depth 6), (New-Object Text.UTF8Encoding $false))
# Stop it hard as the second job starts (a crash, a closed laptop): the first job's report is watched for.
$argv = @("batch", "run", $qfile, "--package", $Package, "--runtime", $Runtime, "--device", $Device)
$p = Start-Process -FilePath $Cli -ArgumentList $argv -PassThru -NoNewWindow -RedirectStandardOutput (Join-Path $qdir "first.log")
$deadline = (Get-Date).AddMinutes(60)
while (-not $p.HasExited -and -not (Test-Path (Join-Path $qdir "a_hdr10.mp4.json")) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 200 }
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
$p.WaitForExit()
$mid = Get-Content "$qfile.state.json" -Raw | ConvertFrom-Json
$midStates = ($mid.jobs | ForEach-Object { $_.status }) -join ", "
$r = Run-Quiet $Cli ($argv)
$r.Text | Where-Object { $_ -match "^Job|error" } | Write-Host
$end = Get-Content "$qfile.state.json" -Raw | ConvertFrom-Json
$endStates = ($end.jobs | ForEach-Object { $_.status }) -join ", "
# A job left "running" by the stop is started again by run_queue itself; nothing was published for it.
Note "queue" (($r.Code -eq 0) -and ($endStates -eq "complete, complete, complete")) "stopped at [$midStates], resumed to [$endStates]"

# ---------------------------------------------------------------------------
Say "The workflow in RUDRA.exe with a movie, no Python on the PATH"
$clean = @((Split-Path $App), $FfDir, "$env:SystemRoot\System32", $env:SystemRoot) -join ";"
$saved = @{ PATH = $env:PATH; PYTHONHOME = $env:PYTHONHOME; PYTHONPATH = $env:PYTHONPATH }
$env:PATH = $clean
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$report = Join-Path $Reports "native_phase4_workflow_$Stamp.json"
$masters = Join-Path $Work "app"
$p = Start-Process -FilePath $App -Wait -PassThru -NoNewWindow -ArgumentList @(
    "--workflow-check", "`"$report`"", "--package", "`"$Package`"", "--frames", "`"$Frames`"",
    "--backend", $Backend, "--out", "`"$masters`"", "--movie", "`"$Clip`"")
$env:PATH = $saved.PATH
if ($saved.PYTHONHOME) { $env:PYTHONHOME = $saved.PYTHONHOME }
if ($saved.PYTHONPATH) { $env:PYTHONPATH = $saved.PYTHONPATH }
if (Test-Path $report) {
    $w = Get-Content $report -Raw | ConvertFrom-Json
    Note "app workflow" ($w.verdict -eq "PASS") "model, open, scrub, grade, compare, measure, master: $(@('model','open','scrub','grade','compare','measure','master' | Where-Object { $w.$_.ok }).Count) of 7"
    $m = $w.movie
    Note "app movie" ([bool]$m.ok) "$($m.delivered)/$($m.frames) frames scrubbed in $([math]::Round($m.scrub_s, 1)) s, HDR10 through the queue in $([math]::Round($m.export_s, 1)) s, MaxCLL $($m.max_cll), QC $(if ($m.qc_passed) { 'passed' } else { 'FAILED' })"
} else { Note "app workflow" $false "RUDRA.exe wrote no report (exit $($p.ExitCode))" }

# ---------------------------------------------------------------------------
$table = $results.Values | Format-Table -AutoSize | Out-String
$table | Write-Host
@("RUDRA native Phase 4 exit, $Stamp", "package $Package", "clip $Clip", "backend $Backend", "ffmpeg $FfDir", "",
  $table) | Set-Content -Encoding utf8 (Join-Path $Reports "native_phase4_exit_$Stamp.txt")

Say "By hand, on the HDR display (tick in STATUS.md)"
@(
    "  1. Drop the movie on RUDRA.exe: the title, frame count and rate are the movie's; scrub with , and . and Space.",
    "  2. Grade a frame; Export (Ctrl+E or the toolbar): the HDR10 tile is live; Export queues it and the Queue window opens.",
    "  3. Watch the Queue window to Complete; Stop a second export (HLG) half way, then Resume it.",
    "  4. Open the HDR10 master in the player you deliver to (HDR on): the highlights, the audio in sync.",
    "  5. Quit and reopen: the Queue window lists the exports with their states."
) | Write-Host

$all = -not ($results.Values | Where-Object { $_.Result -ne "PASS" })
Write-Host ("`nPhase 4 exit (scripted): " + $(if ($all) { "PASS" } else { "FAIL" })) -ForegroundColor $(if ($all) { "Green" } else { "Red" })
if ($all) { exit 0 } else { exit 1 }
