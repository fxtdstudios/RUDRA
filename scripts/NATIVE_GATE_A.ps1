<#
.SYNOPSIS
  Gate A for the native app on this Windows machine: does the C++ build compute
  what Python computes, on every inference backend this box has?

.DESCRIPTION
  One run, one table. Steps:
    1. Export the model package with the Python in this shell (tools/export_model.py),
       optionally adding the bench frames to the parity check (-BenchDir).
    2. Download ONNX Runtime with DirectML (NuGet) and the DirectML runtime it
       depends on, into tmp/native_deps (git-ignored). Nothing is installed.
    3. Configure and build native/ with Visual Studio 2022:
         - LibTorch from this Python's torch (CUDA if the CUDA toolkit is present)
         - ONNX Runtime DirectML
    4. Run rudra-native diff on: LibTorch CPU, LibTorch CUDA, ONNX Runtime CPU,
       ONNX Runtime DirectML. Each is compared with the package's golden frames
       (eager PyTorch outputs) at the package's own tolerances.
    5. Write reports/native_gate_a_<date>.txt and print the summary.

  Needs: Visual Studio 2022 with "Desktop development with C++", CMake 3.24+
  (VS ships one), and a Python with torch, onnx, onnxruntime, numpy, opencv.
  For LibTorch CUDA, the CUDA toolkit matching torch.version.cuda must be
  installed (CUDA_PATH set); without it the CUDA row is skipped, not failed.

.EXAMPLE
  .\scripts\NATIVE_GATE_A.ps1
  .\scripts\NATIVE_GATE_A.ps1 -BenchDir D:\bench\sdr -Checkpoint checkpoints\sdr2hdr_shadow_v1.pt
#>
[CmdletBinding()]
param(
    [string]$Checkpoint = "checkpoints\sdr2hdr_shadow_v1.pt",
    [string]$BenchDir = "",
    [int]$BenchLimit = 0,
    [string]$Python = "python",
    [string]$OrtVersion = "1.22.0",
    [switch]$SkipExport,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Deps = Join-Path $Repo "tmp\native_deps"
$Build = Join-Path $Repo "build\native_gate_a"
$Stem = [IO.Path]::GetFileNameWithoutExtension($Checkpoint)
$Package = Join-Path $Repo "dist\models\$Stem"
$Stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$Report = Join-Path $Repo "reports\native_gate_a_$Stamp.txt"
New-Item -ItemType Directory -Force -Path $Deps, (Split-Path $Report) | Out-Null

function Say($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------------------
Say "Python and torch"
$info = & $Python -c "import torch, sys; print(torch.__version__); print(torch.version.cuda or ''); print(int(torch.cuda.is_available())); print(torch.utils.cmake_prefix_path); print(sys.executable)"
if ($LASTEXITCODE -ne 0) { Fail "python with torch not found ($Python)" }
$TorchVersion, $TorchCuda, $CudaAvail, $TorchCMake, $PyExe = $info
$TorchLib = Join-Path (Split-Path (Split-Path $TorchCMake)) "lib"
Write-Host "torch $TorchVersion  cuda '$TorchCuda'  gpu available $CudaAvail"
Write-Host "python $PyExe"
& $Python -c "import onnx, onnxruntime, numpy, cv2" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "installing onnx, onnxruntime into this Python"
    & $Python -m pip install -q onnx onnxruntime
    if ($LASTEXITCODE -ne 0) { Fail "pip install onnx onnxruntime" }
}

$CudaToolkit = $env:CUDA_PATH
$HasToolkit = $CudaToolkit -and (Test-Path (Join-Path $CudaToolkit "bin\nvcc.exe"))
$WithCuda = ($TorchCuda -ne "") -and ($CudaAvail -eq "1") -and $HasToolkit
$LibTorchPrefix = $TorchCMake
$WithLibTorch = $true
if ($TorchCuda -ne "" -and -not $HasToolkit) {
    # A CUDA build of torch makes CMake look for the CUDA toolkit, and that
    # cannot be switched off from outside. Without the toolkit, use the CPU
    # LibTorch of the same version instead: the CPU row still runs, the CUDA
    # row is reported as skipped.
    Write-Host "CUDA toolkit not found (CUDA_PATH): using CPU LibTorch $TorchVersion; the CUDA row is skipped." -ForegroundColor Yellow
    Write-Host "Install CUDA $TorchCuda and re-run to measure LibTorch CUDA." -ForegroundColor Yellow
    $base = $TorchVersion.Split("+")[0]
    $ltDir = Join-Path $Deps "libtorch-cpu-$base"
    if (-not (Test-Path "$ltDir\libtorch\share\cmake\Torch")) {
        $zip = Join-Path $Deps "libtorch-cpu-$base.zip"
        $url = "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-$base%2Bcpu.zip"
        try {
            Write-Host "download $url"
            Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
            Expand-Archive -Path $zip -DestinationPath $ltDir -Force
            Remove-Item $zip
        } catch {
            Write-Host "CPU LibTorch $base could not be downloaded: LibTorch rows skipped." -ForegroundColor Yellow
            $WithLibTorch = $false
        }
    }
    if ($WithLibTorch) {
        $LibTorchPrefix = "$ltDir\libtorch\share\cmake"
        $TorchLib = "$ltDir\libtorch\lib"
    }
}

# ---------------------------------------------------------------------------
if (-not $SkipExport) {
    Say "Export the model package"
    $exportArgs = @("tools\export_model.py", $Checkpoint, "--out", "dist\models")
    if ($BenchDir) { $exportArgs += @("--bench-dir", $BenchDir); if ($BenchLimit) { $exportArgs += @("--bench-limit", $BenchLimit) } }
    & $Python @exportArgs
    if ($LASTEXITCODE -ne 0) { Fail "export_model.py (Python-side parity failed or export error)" }
}
if (-not (Test-Path (Join-Path $Package "manifest.json"))) { Fail "no package at $Package" }

# ---------------------------------------------------------------------------
Say "ONNX Runtime $OrtVersion with DirectML"
function Get-Nupkg($id, $version, $dest) {
    $zip = Join-Path $Deps "$id.$version.zip"
    if (-not (Test-Path $dest)) {
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

# The NuGet layout, rearranged into the include/ + lib/ shape CMake expects.
$OrtRoot = Join-Path $Deps "ort-root-$OrtVersion"
New-Item -ItemType Directory -Force -Path "$OrtRoot\include", "$OrtRoot\lib" | Out-Null
Copy-Item "$OrtPkg\build\native\include\*" "$OrtRoot\include\" -Force
Copy-Item "$OrtPkg\runtimes\win-x64\native\*" "$OrtRoot\lib\" -Force
$DmlDll = Get-ChildItem "$DmlPkg\bin\x64-win" -Filter DirectML.dll | Select-Object -First 1
if (-not $DmlDll) { Fail "DirectML.dll not found in $DmlPkg" }
Write-Host "ONNX Runtime $OrtVersion (DirectML), DirectML $DmlVersion"

# ---------------------------------------------------------------------------
if (-not $SkipBuild) {
    Say "Configure and build native/ (Visual Studio 2022, Release)"
    $cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
    if (-not $cmake) {
        $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
        $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        $cmake = Join-Path $vs "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    }
    if (-not (Test-Path $cmake)) { Fail "cmake not found (install VS 2022 C++ workload or CMake)" }
    $cfg = @("-S", "native", "-B", $Build, "-G", "Visual Studio 17 2022", "-A", "x64",
             "-DRUDRA_BUILD_TESTS=OFF", "-DRUDRA_BUILD_APP=OFF",
             "-DRUDRA_WITH_ONNXRUNTIME=ON", "-DONNXRUNTIME_ROOT=$OrtRoot")
    if ($WithLibTorch) { $cfg += @("-DRUDRA_WITH_LIBTORCH=ON", "-DCMAKE_PREFIX_PATH=$LibTorchPrefix") }
    else { $cfg += @("-DRUDRA_WITH_LIBTORCH=OFF") }
    & $cmake @cfg
    if ($LASTEXITCODE -ne 0) { Fail "cmake configure" }
    & $cmake --build $Build --config Release --parallel
    if ($LASTEXITCODE -ne 0) { Fail "build" }
}
$Exe = Join-Path $Build "cli\Release\rudra-native.exe"
if (-not (Test-Path $Exe)) { Fail "rudra-native.exe not built" }

# The ONNX Runtime and DirectML DLLs go next to the exe: Windows ships an older
# onnxruntime.dll in System32 that would otherwise be loaded first.
Copy-Item "$OrtRoot\lib\*.dll" (Split-Path $Exe) -Force
Copy-Item $DmlDll.FullName (Split-Path $Exe) -Force
$env:PATH = "$TorchLib;$env:PATH"

# ---------------------------------------------------------------------------
Say "Gate A"
$rows = @(
    @{ Name = "LibTorch CPU";          Runtime = "libtorch";    Device = "cpu";      Run = $WithLibTorch },
    @{ Name = "LibTorch CUDA";         Runtime = "libtorch";    Device = "cuda";     Run = $WithCuda },
    @{ Name = "ONNX Runtime CPU";      Runtime = "onnxruntime"; Device = "cpu";      Run = $true },
    @{ Name = "ONNX Runtime DirectML"; Runtime = "onnxruntime"; Device = "directml"; Run = $true }
)
$log = @("RUDRA native Gate A, $Stamp", "package $Package", "torch $TorchVersion (cuda '$TorchCuda'), ONNX Runtime $OrtVersion, DirectML $DmlVersion", "")
& $Exe info $Package | Tee-Object -Variable infoOut | Out-Null
$log += $infoOut; $log += ""
$summary = @()
foreach ($r in $rows) {
    if (-not $r.Run) { $summary += [pscustomobject]@{ Backend = $r.Name; Result = "skipped"; "Worst |d|" = "" }; continue }
    $out = & $Exe diff $Package --runtime $r.Runtime --device $r.Device 2>&1
    $code = $LASTEXITCODE
    $log += "---- $($r.Name)"; $log += $out; $log += ""
    $worst = ($out | Select-String "max \|d\| ([0-9.e+-]+)" -AllMatches | ForEach-Object { $_.Matches } | ForEach-Object { [double]$_.Groups[1].Value } | Measure-Object -Maximum).Maximum
    $result = switch ($code) { 0 { "PASS" } 1 { "FAIL" } default { "ERROR" } }
    if ($result -eq "ERROR") { $worst = ($out | Select-Object -Last 2) -join " " }
    $summary += [pscustomobject]@{ Backend = $r.Name; Result = $result; "Worst |d|" = $worst }
}
$log += ($summary | Format-Table -AutoSize | Out-String)
$log | Set-Content -Encoding utf8 $Report

Say "Result"
$summary | Format-Table -AutoSize
Write-Host "Full log: $Report"
if ($summary | Where-Object { $_.Result -in @("FAIL", "ERROR") }) { exit 1 }
