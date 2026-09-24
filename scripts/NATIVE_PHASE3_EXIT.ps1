<#
.SYNOPSIS
  Phase 3 exit for the native app on this Windows machine: the whole Studio
  workflow in RUDRA.exe with no Python behind it, scripted, its masters held
  to the Studio-held CLI path.

.DESCRIPTION
  Steps:
    1. Dependencies into tmp\native_deps (nothing system-wide): Qt 6.8 with
       Shader Tools (aqtinstall, build time only), ONNX Runtime with DirectML
       (NuGet), OpenCV 4.10 (the official Windows package from GitHub).
    2. Configure and build native\ into build\native_phase3: RUDRA.exe,
       rudra-native.exe and the app's Qt tests, ONNX Runtime and OpenCV on.
    3. Deploy: windeployqt, and the ONNX Runtime, DirectML and OpenCV DLLs
       next to both executables, so the folder runs on its own.
    4. The app's Qt tests (offscreen).
    5. RUDRA.exe --workflow-check on -Frames with a PATH of the app's own
       folder and Windows only (no Python, no PYTHONHOME/PYTHONPATH): open the
       package and the folder, scrub every frame (each checked against its own
       decode every 24), grade and undo, compare, probe and measure, master
       the first, middle and last frames.
    6. rudra-native master-compare: each of those masters against the CLI's
       own master of the same frame and parameters (the path master-check
       holds to the Studio), within 1 half-float ulp, headers and sidecars
       equal. The same backend as the app.
    7. The table, a report in reports\, and the by-hand checklist.

  Needs: Visual Studio 2022 or 2026 with the C++ tools (see NATIVE_GATE_B.ps1),
  a model package (NATIVE_GATE_A.ps1 exports dist\models\sdr2hdr_shadow_v1),
  and a folder of frames (-Frames, 240 of them for the exit).

.EXAMPLE
  .\scripts\NATIVE_PHASE3_EXIT.ps1 -Frames D:\shots\sh010
  .\scripts\NATIVE_PHASE3_EXIT.ps1 -Frames D:\shots\sh010 -Backend onnxruntime/cpu
  .\scripts\NATIVE_PHASE3_EXIT.ps1 -Frames D:\shots\sh010 -SkipBuild
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Frames,
    [string]$Package = "dist\models\sdr2hdr_shadow_v1",
    [string]$Backend = "onnxruntime/directml",
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
$Build = Join-Path $Repo "build\native_phase3"
$Stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$Reports = Join-Path $Repo "reports"
New-Item -ItemType Directory -Force -Path $Deps, $Reports | Out-Null

function Say($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
. (Join-Path $PSScriptRoot "native_toolchain.ps1")

$Package = (Resolve-Path $Package -ErrorAction SilentlyContinue).Path
if (-not $Package -or -not (Test-Path (Join-Path $Package "manifest.json"))) { Fail "no model package (run NATIVE_GATE_A.ps1 first, or pass -Package)" }
$Frames = (Resolve-Path $Frames -ErrorAction SilentlyContinue).Path
if (-not $Frames -or -not (Test-Path $Frames -PathType Container)) { Fail "no folder of frames at -Frames" }
$Runtime, $Device = $Backend.Split("/")

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
    & $cmake --build $Build --config Release --parallel --target RUDRA rudra-native rudra_app_tests
    if ($LASTEXITCODE -ne 0) { Fail "build" }
}
$App = Join-Path $Build "app\Release\RUDRA.exe"
$Cli = Join-Path $Build "cli\Release\rudra-native.exe"
$AppTests = Join-Path $Build "app\Release\rudra_app_tests.exe"
foreach ($e in @($App, $Cli)) { if (-not (Test-Path $e)) { Fail "not built: $e" } }

Say "Deploy"
foreach ($e in @($App, $AppTests)) {
    if (Test-Path $e) { & (Join-Path $QtRoot "bin\windeployqt.exe") --release --no-translations --no-compiler-runtime $e | Out-Null }
}
foreach ($dir in @((Split-Path $App), (Split-Path $Cli))) {
    Copy-Item "$OrtRoot\lib\*.dll" $dir -Force
    Copy-Item $DmlDll.FullName $dir -Force
    Copy-Item $CvDll.FullName $dir -Force
}
Write-Host "RUDRA.exe and rudra-native.exe with their DLLs beside them"

# ---------------------------------------------------------------------------
$testsOk = $true
if (-not $SkipTests -and (Test-Path $AppTests)) {
    Say "The app's Qt tests (offscreen)"
    $env:QT_QPA_PLATFORM = "offscreen"
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    $text = & $AppTests "--gtest_brief=1" 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    Remove-Item Env:QT_QPA_PLATFORM
    $text | Where-Object { $_ -match "PASSED|FAILED|Failure" } | Write-Host
    $testsOk = $code -eq 0
    Write-Host ("  => " + $(if ($testsOk) { "PASS" } else { "FAIL" }))
}

# ---------------------------------------------------------------------------
Say "The workflow in RUDRA.exe, no Python on the PATH"
$clean = @((Split-Path $App), "$env:SystemRoot\System32", $env:SystemRoot) -join ";"
$saved = @{ PATH = $env:PATH; PYTHONHOME = $env:PYTHONHOME; PYTHONPATH = $env:PYTHONPATH }
$env:PATH = $clean
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$py = Get-Command python, python3, py -ErrorAction SilentlyContinue | Where-Object { $_.CommandType -eq "Application" }
if ($py) { Write-Host "python is still reachable: $($py.Source -join ', ')" -ForegroundColor Yellow }
$report = Join-Path $Reports "native_phase3_workflow_$Stamp.json"
$masters = Join-Path $Repo "tmp\phase3_masters_$Stamp"
$p = Start-Process -FilePath $App -Wait -PassThru -NoNewWindow -ArgumentList @(
    "--workflow-check", "`"$report`"", "--package", "`"$Package`"", "--frames", "`"$Frames`"",
    "--backend", $Backend, "--out", "`"$masters`"")
$env:PATH = $saved.PATH
if ($saved.PYTHONHOME) { $env:PYTHONHOME = $saved.PYTHONHOME }
if ($saved.PYTHONPATH) { $env:PYTHONPATH = $saved.PYTHONPATH }
if (-not (Test-Path $report)) { Fail "RUDRA.exe wrote no report (exit $($p.ExitCode))" }
$w = Get-Content $report -Raw | ConvertFrom-Json
$rows = @()
foreach ($s in @("model", "open", "scrub", "grade", "compare", "measure", "master")) {
    $o = $w.$s
    $what = switch ($s) {
        "model" { "$($o.backend), $($o.device), loaded and checked in $([math]::Round($o.load_s, 1)) s" }
        "open" { "$($o.frames) frames" }
        "scrub" { "$($o.delivered)/$($o.frames) delivered, $($o.checked_against_decode) checked against their decode, median $([math]::Round($o.median_ms)) ms, p95 $([math]::Round($o.p95_ms)) ms" }
        "grade" { "$($o.moves) moves, undo and redo restore params()" }
        "compare" { "wipe, flip, layers" }
        "measure" { "MaxCLL $($o.maxcll), MaxFALL $($o.maxfall), p99 $([math]::Round($o.p99_nits, 1)) nits" }
        "master" { "$(@($o.masters).Count) masters in $($o.folder)" }
    }
    $rows += [pscustomobject]@{ Step = $s; Result = $(if ($o.ok) { "PASS" } else { "FAIL" }); What = $what }
}
$rows | Format-Table -AutoSize | Out-String | Write-Host
$workflowOk = $w.verdict -eq "PASS"

Say "The app's masters against rudra-native master ($Backend)"
$prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
$text = & $Cli master-compare $Package $report --runtime $Runtime --device $Device 2>&1 | ForEach-Object { "$_" }
$compareOk = $LASTEXITCODE -eq 0
$ErrorActionPreference = $prev
$text | Write-Host

$log = @("RUDRA native Phase 3 exit, $Stamp", "package $Package", "frames $Frames", "backend $Backend",
         "PATH for the app: $clean", "") + ($rows | Format-Table -AutoSize | Out-String) + $text
$log | Set-Content -Encoding utf8 (Join-Path $Reports "native_phase3_exit_$Stamp.txt")

Say "By hand, on the HDR display (tick in STATUS.md)"
@(
    "  1. RUDRA.exe with no arguments: the first-run check shows the HDR card and names the display's peak.",
    "  2. File > Model packages: the package in use; switch to another and back; the grade stays.",
    "  3. Drop the folder on the window; scrub with , and . and the scrub bar; Space plays.",
    "  4. Grade (Reconstruct and the Region EV rows), undo and redo with Z and Y.",
    "  5. W for the wipe, drag it; hold B for the baseline; the Layer buttons.",
    "  6. Probe on: hover a highlight; the rail and the floating box agree with the Frame panel.",
    "  7. Deliver: Master EXR of the frame and of the sequence; Copy delivery metadata.",
    "  8. Quit and reopen: the window, rails, tab and Render fields come back; Open recent lists the shot."
) | Write-Host

$all = $testsOk -and $workflowOk -and $compareOk
Write-Host ("`nPhase 3 exit (scripted): " + $(if ($all) { "PASS" } else { "FAIL" })) -ForegroundColor $(if ($all) { "Green" } else { "Red" })
if ($all) { exit 0 } else { exit 1 }
