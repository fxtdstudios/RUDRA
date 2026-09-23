<#
.SYNOPSIS
  Gate B for the native app on this Windows machine: does an HDR swapchain
  carry a 1 000-nit patch above SDR white, through QRhi on D3D12?

.DESCRIPTION
  One run, one report per API. Steps:
    1. Fetch Qt 6.8 (MSVC 2022, with Qt Shader Tools) into tmp/native_deps/Qt
       with aqtinstall from this Python. Nothing is installed system-wide.
    2. Configure and build native/ with only rudra-hdr-probe enabled.
    3. Run the probe on D3D12 (scRGB, then HDR10) and D3D11 (scRGB), each for
       a few seconds. Each run opens a window with the test card, reads the
       swapchain back and writes reports/native_gate_b_<api>_<format>_<date>.json.
    4. Print the table.

  PASS means the swapchain carried the 1 000-nit patch at least a stop above
  SDR white. It is necessary, not sufficient: Windows HDR must be ON for the
  display (Settings > System > Display > Use HDR) and the patch has to look,
  or measure, brighter than the 203-nit one. Leave -Frames 0 to keep the
  window open and look; Esc closes it.

  Needs: Visual Studio 2022 or 2026 (or its Build Tools) with the C++ tools;
  -InstallBuildTools installs the Build Tools with winget if none is found.
  And a Python that can pip install aqtinstall.

.EXAMPLE
  .\scripts\NATIVE_GATE_B.ps1
  .\scripts\NATIVE_GATE_B.ps1 -Frames 0          # keep the window open
  .\scripts\NATIVE_GATE_B.ps1 -Screen 1          # the HDR display, if it is not the primary
#>
[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$QtVersion = "6.8.3",
    [int]$Frames = 240,
    [int]$Screen = -1,
    [switch]$SkipBuild,
    [switch]$InstallBuildTools
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Deps = Join-Path $Repo "tmp\native_deps"
$QtRoot = Join-Path $Deps "Qt\$QtVersion\msvc2022_64"
$Build = Join-Path $Repo "build\native_gate_b"
$Stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$Reports = Join-Path $Repo "reports"
New-Item -ItemType Directory -Force -Path $Deps, $Reports | Out-Null

function Say($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "FAILED: $m" -ForegroundColor Red; exit 1 }
. (Join-Path $PSScriptRoot "native_toolchain.ps1")

# ---------------------------------------------------------------------------
if (-not (Test-Path (Join-Path $QtRoot "bin\qsb.exe"))) {
    Say "Qt $QtVersion (msvc2022_64, qtshadertools) into tmp\native_deps\Qt"
    & $Python -m pip install --quiet --upgrade aqtinstall
    if ($LASTEXITCODE -ne 0) { Fail "pip install aqtinstall" }
    & $Python -m aqt install-qt windows desktop $QtVersion win64_msvc2022_64 -m qtshadertools -O (Join-Path $Deps "Qt")
    if ($LASTEXITCODE -ne 0) { Fail "aqt install-qt" }
}
Write-Host "Qt at $QtRoot"

# ---------------------------------------------------------------------------
if (-not $SkipBuild) {
    Say "Configure and build rudra-hdr-probe (Visual Studio, Release)"
    $tc = Find-NativeToolchain
    if (-not $tc -and $InstallBuildTools) {
        if (-not (Install-NativeBuildTools)) { Fail "Build Tools install" }
        $tc = Find-NativeToolchain
    }
    if (-not $tc) { Fail "no C++ toolchain (see above)" }
    $cmake = $tc.CMake
    Reset-StaleCMakeCache $Build $tc.Generator
    & $cmake -S native -B $Build -G $tc.Generator -A x64 `
        -DRUDRA_BUILD_TESTS=OFF -DRUDRA_BUILD_CLI=OFF -DRUDRA_BUILD_APP=OFF `
        -DRUDRA_BUILD_HDR_PROBE=ON "-DCMAKE_PREFIX_PATH=$QtRoot"
    if ($LASTEXITCODE -ne 0) { Fail "cmake configure" }
    & $cmake --build $Build --config Release --parallel --target rudra-hdr-probe
    if ($LASTEXITCODE -ne 0) { Fail "build" }
}
$Exe = Join-Path $Build "render\probe\Release\rudra-hdr-probe.exe"
if (-not (Test-Path $Exe)) { Fail "probe not built: $Exe" }
& (Join-Path $QtRoot "bin\windeployqt.exe") --release --no-translations --no-compiler-runtime $Exe | Out-Null

# ---------------------------------------------------------------------------
Say "Displays"
& $Exe --list-screens | Write-Host
$screenArgs = if ($Screen -ge 0) { @("--screen", $Screen) } else { @() }

$runs = @(@("d3d12", "scrgb"), @("d3d12", "hdr10"), @("d3d11", "scrgb"))
$rows = @()
$hint = $null
foreach ($r in $runs) {
    $api, $fmt = $r
    Say "rudra-hdr-probe --api $api --format $fmt"
    $json = Join-Path $Reports "native_gate_b_${api}_${fmt}_$Stamp.json"
    & $Exe --api $api --format $fmt --frames $Frames --report $json @screenArgs | Out-Null
    if (Test-Path $json) {
        $d = Get-Content $json -Raw | ConvertFrom-Json
        $p = @{}; foreach ($x in $d.patches) { $p[[string]$x.target_nits] = $x.swapchain_nits }
        $peak = if ($d.hdr_info.limits -eq "nits") { [math]::Round([double]$d.hdr_info.max_luminance) } else { "" }
        if ($d.hint) { $hint = $d.hint }
        $rows += [pscustomobject]@{ API = $api; Asked = $fmt; Got = $d.output_path; Screen = $d.screen_model;
                                    WinHDR = $(if ($null -ne $d.windows_output.windows_hdr_on) { if ($d.windows_output.windows_hdr_on) { "on" } else { "off" } } else { "" });
                                    "203" = $p["203"]; "1000" = $p["1000"]; "2000" = $p["2000"];
                                    PeakNits = $peak; SdrWhite = $d.hdr_info.sdr_white_level; Verdict = $d.verdict }
    } else {
        $rows += [pscustomobject]@{ API = $api; Asked = $fmt; Got = ""; Screen = ""; WinHDR = ""; "203" = ""; "1000" = "";
                                    "2000" = ""; PeakNits = ""; SdrWhite = ""; Verdict = "ERROR" }
    }
}

Say "Gate B"
$rows | Format-Table -AutoSize | Out-String | Write-Host
if ($hint) { Write-Host $hint -ForegroundColor Yellow }
Write-Host "Reports in $Reports. PASS is the swapchain half of the gate; confirm on the glass."
if ($rows | Where-Object { $_.API -eq "d3d12" -and $_.Verdict -eq "PASS" }) { exit 0 } else { exit 1 }
