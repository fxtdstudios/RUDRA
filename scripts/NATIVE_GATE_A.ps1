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
         - LibTorch from this Python's torch, CUDA included, no CUDA toolkit needed
         - ONNX Runtime DirectML
    4. Run rudra-native diff on: LibTorch CPU, LibTorch CUDA, ONNX Runtime CPU,
       ONNX Runtime DirectML. Each is compared with the package's golden frames
       (eager PyTorch outputs) at the package's own tolerances.
    5. Write reports/native_gate_a_<date>.txt and print the summary.

  Needs: Visual Studio 2022 or 2026 (or its Build Tools) with the C++ tools,
  which ship CMake; -InstallBuildTools installs the Build Tools with winget if
  none is found. And a Python with torch, onnx, onnxruntime, numpy, opencv.
  The LibTorch CUDA row runs when this torch is a CUDA build and sees a GPU.

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
    [switch]$SkipBuild,
    [switch]$NoBench,
    [switch]$InstallBuildTools
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
. (Join-Path $PSScriptRoot "native_toolchain.ps1")

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

# LibTorch comes from this Python's torch, imported directly (RUDRA_TORCH_ROOT)
# rather than through TorchConfig: a CUDA torch's TorchConfig demands the CUDA
# toolkit at torch's version, wired into this Visual Studio, to build a program
# that compiles no CUDA. The run needs only torch's own DLLs, loaded from PATH.
$TorchRoot = Split-Path $TorchLib
$WithLibTorch = Test-Path (Join-Path $TorchRoot "include\torch\script.h")
if (-not $WithLibTorch) { Write-Host "torch headers not found under $TorchRoot`: LibTorch rows skipped." -ForegroundColor Yellow }
$WithCuda = $WithLibTorch -and ($TorchCuda -ne "") -and ($CudaAvail -eq "1")

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
    Say "Configure and build native/ (Visual Studio, Release)"
    $tc = Find-NativeToolchain
    if (-not $tc -and $InstallBuildTools) {
        if (-not (Install-NativeBuildTools)) { Fail "Build Tools install" }
        $tc = Find-NativeToolchain
    }
    if (-not $tc) { Fail "no C++ toolchain (see above)" }
    $cmake = $tc.CMake
    Reset-StaleCMakeCache $Build $tc.Generator
    $cfg = @("-S", "native", "-B", $Build, "-G", $tc.Generator, "-A", "x64",
             "-DRUDRA_BUILD_TESTS=OFF", "-DRUDRA_BUILD_APP=OFF",
             "-DRUDRA_WITH_ONNXRUNTIME=ON", "-DONNXRUNTIME_ROOT=$OrtRoot")
    if ($WithLibTorch) { $cfg += @("-DRUDRA_WITH_LIBTORCH=ON", "-DRUDRA_TORCH_ROOT=$TorchRoot") }
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
$prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
$infoOut = & $Exe info $Package 2>&1 | ForEach-Object { "$_" }
$ErrorActionPreference = $prev
$log += $infoOut; $log += ""
$summary = @()
foreach ($r in $rows) {
    if (-not $r.Run) { $summary += [pscustomobject]@{ Backend = $r.Name; Result = "skipped"; "Worst |d|" = "" }; continue }
    # Windows PowerShell turns any stderr line of a native program into a
    # terminating error under "Stop"; a runtime's warning is not a failure,
    # the exit code is. Collect both streams as text and judge by the code.
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    $out = & $Exe diff $Package --runtime $r.Runtime --device $r.Device 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    $log += "---- $($r.Name)"; $log += $out; $log += ""
    $worst = ($out | Select-String "max \|d\| ([0-9.e+-]+)" -AllMatches | ForEach-Object { $_.Matches } | ForEach-Object { [double]$_.Groups[1].Value } | Measure-Object -Maximum).Maximum
    $result = switch ($code) { 0 { "PASS" } 1 { "FAIL" } default { "ERROR" } }
    if ($result -eq "ERROR") { $worst = ($out | Select-Object -Last 2) -join " " }
    $summary += [pscustomobject]@{ Backend = $r.Name; Result = $result; "Worst |d|" = $worst }
}
$log += ($summary | Format-Table -AutoSize | Out-String)

# ---------------------------------------------------------------------------
# Inference time at 1080p on every backend that passed, for the budget table
# (NATIVE_ARCHITECTURE.md 6.6). Wall time to fields in host memory.
$bench = @()
if (-not $NoBench) {
    Say "Inference time, 1920x1080, fp32 (median of 5)"
    foreach ($r in $rows) {
        $row = $summary | Where-Object { $_.Backend -eq $r.Name }
        if ($row.Result -ne "PASS") { continue }
        $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        $out = & $Exe bench $Package --runtime $r.Runtime --device $r.Device --size 1920x1080 --iters 5 2>&1 | ForEach-Object { "$_" }
        $ErrorActionPreference = $prev
        $log += "---- bench $($r.Name)"; $log += $out; $log += ""
        $ms = @{}
        foreach ($l in ($out | Where-Object { $_ -match "^BENCH " })) { $f = $l -split " "; $ms[$f[4]] = [double]$f[5] }
        $bench += [pscustomobject]@{ Backend = $r.Name; "untiled ms" = $ms["untiled"]; "tiled 512/64 ms" = $ms["tiled"] }
    }
    $bench | Format-Table -AutoSize | Out-String | Write-Host
    $log += ($bench | Format-Table -AutoSize | Out-String)

    # The viewer's CPU work after a slider move (Phase 2 step 12): the sample,
    # the measurements, the waveform and histogram, the vectorscope.
    Say "Viewer measurements and scopes on the CPU"
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    $out = & $Exe bench-scopes 2>&1 | ForEach-Object { "$_" }
    $ErrorActionPreference = $prev
    $out | Where-Object { $_ -notmatch "^BENCH " } | Write-Host
    $log += "---- bench-scopes"; $log += $out; $log += ""
}
$log | Set-Content -Encoding utf8 $Report

Say "Result"
$summary | Format-Table -AutoSize
Write-Host "Full log: $Report"
if ($summary | Where-Object { $_.Result -in @("FAIL", "ERROR") }) { exit 1 }
