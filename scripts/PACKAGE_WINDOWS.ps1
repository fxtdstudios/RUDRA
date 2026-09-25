<#
.SYNOPSIS
  The RUDRA beta for Windows x64: a self-contained folder and its ZIP.

.DESCRIPTION
  Builds RUDRA.exe and rudra-native.exe (Release; ONNX Runtime with DirectML,
  OpenCV still decode, the QRhi viewer on Qt 6.8), then makes
    dist\beta\RUDRA-<version>-windows-x64\
      RUDRA.exe, rudra-native.exe, Qt (windeployqt), the MSVC runtime,
      onnxruntime.dll, DirectML.dll, opencv_world*.dll,
      models\   the package(s) from dist\models
      LICENSE, NOTICE, LICENSE-weights, "Read me first.md"
  and dist\beta\RUDRA-<version>-windows-x64.zip with its SHA-256.

  Needs what NATIVE_PHASE3_EXIT.ps1 needs: Visual Studio 2022 or 2026 (or the
  Build Tools) with the C++ tools, a Python for aqtinstall, and a model package
  (NATIVE_GATE_A.ps1 exports dist\models\sdr2hdr_shadow_v1). Movies need an
  ffmpeg with libx265, prores_ks and zscale on the PATH (the gyan.dev "full"
  build); it is not bundled.

.EXAMPLE
  .\scripts\PACKAGE_WINDOWS.ps1
  .\scripts\PACKAGE_WINDOWS.ps1 -Tests        # also build and run the app's Qt tests
  .\scripts\PACKAGE_WINDOWS.ps1 -SkipBuild    # package the last build
#>
[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$QtVersion = "6.8.3",
    [string]$OrtVersion = "1.22.0",
    [string]$OpenCvVersion = "4.10.0",
    [switch]$SkipBuild,
    [switch]$Tests,
    [switch]$InstallBuildTools
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Deps = Join-Path $Repo "tmp\native_deps"
$QtRoot = Join-Path $Deps "Qt\$QtVersion\msvc2022_64"
$Build = Join-Path $Repo "build\native_release"
New-Item -ItemType Directory -Force -Path $Deps | Out-Null

function Say($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
. (Join-Path $PSScriptRoot "native_toolchain.ps1")

$Models = Join-Path $Repo "dist\models"
if (-not (Get-ChildItem $Models -Recurse -Filter manifest.json -ErrorAction SilentlyContinue)) {
    Fail "no model package in dist\models (run NATIVE_GATE_A.ps1 first)"
}

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
$CvLib = Get-ChildItem (Join-Path $CvBuild "x64") -Directory | Where-Object { $_.Name -match "^vc\d+$" } |
         Sort-Object { [int]($_.Name.Substring(2)) } -Descending | Select-Object -First 1
if (-not $CvLib) { Fail "no vcNN folder under $CvBuild\x64" }
$CvConfigDir = Join-Path $CvLib.FullName "lib"
$CvDll = Get-ChildItem (Join-Path $CvBuild "x64") -Recurse -Filter "opencv_world*.dll" |
         Where-Object { $_.Name -notmatch "d\.dll$" } | Select-Object -First 1
if (-not $CvDll) { Fail "opencv_world DLL not found under $CvBuild" }

# ---------------------------------------------------------------------------
if (-not $SkipBuild) {
    Say "Configure and build (Visual Studio, Release)"
    $tc = Find-NativeToolchain
    if (-not $tc -and $InstallBuildTools) {
        if (-not (Install-NativeBuildTools)) { Fail "Build Tools install" }
        $tc = Find-NativeToolchain
    }
    if (-not $tc) { Fail "no C++ toolchain (see above)" }
    $cmake = $tc.CMake
    Reset-StaleCMakeCache $Build $tc.Generator
    $testsFlag = if ($Tests) { "ON" } else { "OFF" }
    & $cmake -S native -B $Build -G $tc.Generator -A x64 `
        "-DRUDRA_BUILD_TESTS=$testsFlag" -DRUDRA_BUILD_CLI=ON -DRUDRA_BUILD_APP=ON -DRUDRA_BUILD_RENDER=ON `
        -DRUDRA_WITH_ONNXRUNTIME=ON "-DONNXRUNTIME_ROOT=$OrtRoot" -DRUDRA_WITH_LIBTORCH=OFF `
        -DRUDRA_WITH_OPENCV=ON "-DOpenCV_DIR=$CvConfigDir" "-DCMAKE_PREFIX_PATH=$QtRoot"
    if ($LASTEXITCODE -ne 0) { Fail "cmake configure" }
    $targets = @("RUDRA", "rudra-native")
    if ($Tests) { $targets += "rudra_app_tests" }
    & $cmake --build $Build --config Release --parallel --target $targets
    if ($LASTEXITCODE -ne 0) { Fail "build" }
}
$App = Join-Path $Build "app\Release\RUDRA.exe"
$Cli = Join-Path $Build "cli\Release\rudra-native.exe"
foreach ($e in @($App, $Cli)) { if (-not (Test-Path $e)) { Fail "not built: $e" } }

if ($Tests) {
    Say "The app's Qt tests (offscreen)"
    $AppTests = Join-Path $Build "app\Release\rudra_app_tests.exe"
    & (Join-Path $QtRoot "bin\windeployqt.exe") --release --no-translations $AppTests | Out-Null
    $plat = Join-Path (Split-Path $AppTests) "platforms"
    New-Item -ItemType Directory -Force -Path $plat | Out-Null
    Copy-Item (Join-Path $QtRoot "plugins\platforms\qoffscreen.dll") $plat -Force
    Copy-Item "$OrtRoot\lib\*.dll", $DmlDll.FullName, $CvDll.FullName (Split-Path $AppTests) -Force
    $env:QT_QPA_PLATFORM = "offscreen"
    & $AppTests
    $code = $LASTEXITCODE
    Remove-Item Env:QT_QPA_PLATFORM
    if ($code -ne 0) { Fail "app tests (exit $code)" }
}

# ---------------------------------------------------------------------------
$Version = (Get-Item $App).VersionInfo.ProductVersion
if (-not $Version) { Fail "RUDRA.exe has no version resource" }
$Name = "RUDRA-$Version-windows-x64"
$Out = Join-Path $Repo "dist\beta\$Name"
$Zip = "$Out.zip"
Say "Package $Name"
if (Test-Path $Out) { Remove-Item $Out -Recurse -Force }
if (Test-Path $Zip) { Remove-Item $Zip -Force }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Copy-Item $App, $Cli $Out
# Qt, its plugins and the MSVC runtime beside the executables.
& (Join-Path $QtRoot "bin\windeployqt.exe") --release --no-translations --compiler-runtime (Join-Path $Out "RUDRA.exe")
if ($LASTEXITCODE -ne 0) { Fail "windeployqt" }
# ONNX Runtime and DirectML beside the exe: Windows' own older onnxruntime.dll
# in System32 would otherwise be loaded first.
Copy-Item "$OrtRoot\lib\*.dll", $DmlDll.FullName, $CvDll.FullName $Out -Force
$OutModels = Join-Path $Out "models"
New-Item -ItemType Directory -Force -Path $OutModels | Out-Null
Copy-Item "$Models\*" $OutModels -Recurse -Force
Get-ChildItem $OutModels -Recurse -Include *.safetensors, *.config.json | Remove-Item -Force
Copy-Item LICENSE, NOTICE $Out
if (Test-Path checkpoints\LICENSE) { Copy-Item checkpoints\LICENSE (Join-Path $Out "LICENSE-weights") }
if (Test-Path docs\BETA.md) { Copy-Item docs\BETA.md (Join-Path $Out "Read me first.md") }

Say "Check the package runs from where it is"
$pkg = Get-ChildItem $OutModels -Directory | Where-Object { Test-Path (Join-Path $_.FullName "manifest.json") } | Select-Object -First 1
& (Join-Path $Out "rudra-native.exe") info $pkg.FullName | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "rudra-native in the package cannot read its model" }
$check = Join-Path $env:TEMP "rudra-theme-check.json"
$p = Start-Process -FilePath (Join-Path $Out "RUDRA.exe") -ArgumentList @("--theme-check", "`"$check`"") -Wait -PassThru
if ($p.ExitCode -ne 0) { Fail "RUDRA.exe in the package does not start (exit $($p.ExitCode))" }

Say "ZIP"
Compress-Archive -Path $Out -DestinationPath $Zip -CompressionLevel Optimal
$hash = (Get-FileHash $Zip -Algorithm SHA256).Hash.ToLower()
"$hash  $Name.zip" | Set-Content -Encoding ascii "$Zip.sha256"
Write-Host "$hash  $Name.zip"
Write-Host "`nBuilt: $Zip"
