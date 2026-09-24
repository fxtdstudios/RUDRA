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
    5. Day 8: run rudra-gpu-parity on D3D12, D3D11, Vulkan and OpenGL, the
       composite shader against the C++ composite on the composite goldens.

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
    & $cmake --build $Build --config Release --parallel --target rudra-hdr-probe rudra-gpu-parity rudra-viewer-check
    if ($LASTEXITCODE -ne 0) { Fail "build" }
}
$Exe = Join-Path $Build "render\probe\Release\rudra-hdr-probe.exe"
if (-not (Test-Path $Exe)) { Fail "probe not built: $Exe" }
$Parity = Join-Path (Split-Path $Exe) "rudra-gpu-parity.exe"
$Viewer = Join-Path (Split-Path $Exe) "rudra-viewer-check.exe"
foreach ($e in @($Exe, $Parity, $Viewer)) {
    if (Test-Path $e) { & (Join-Path $QtRoot "bin\windeployqt.exe") --release --no-translations --no-compiler-runtime $e | Out-Null }
}

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
$gateB = [bool]($rows | Where-Object { $_.API -eq "d3d12" -and $_.Verdict -eq "PASS" })

# ---------------------------------------------------------------------------
# Day 8: the composite shader on this GPU against the C++ reference, on every
# API the viewer can run on here. fp32 target within 1e-6 + 2e-4 |ref|, fp16
# target within 2 half-float ulp. Then one composite pass timed at 1080p and
# 4K (GPU timestamps; budgets 4 ms and 12 ms for composite and view).
Say "GPU composite parity (day 8)"
$parityRows = @()
if (Test-Path $Parity) {
    foreach ($api in @("d3d12", "d3d11", "vulkan", "gl")) {
        $json = Join-Path $Reports "native_gpu_parity_${api}_$Stamp.json"
        $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        $text = & $Parity --api $api --report $json --bench 2>&1 | ForEach-Object { "$_" }
        $code = $LASTEXITCODE
        $ErrorActionPreference = $prev
        $text | Where-Object { $_ -match "^GPU composite parity" } | Write-Host
        if (Test-Path $json) {
            $d = Get-Content $json -Raw | ConvertFrom-Json
            $w32 = ($d.cases | Measure-Object -Property fp32_max_abs -Maximum).Maximum
            $w16 = ($d.cases | Measure-Object -Property fp16_max_ulp -Maximum).Maximum
            $wv = if ($d.views) { ($d.views | Measure-Object -Property max_code -Maximum).Maximum } else { "" }
            $t = @{}; foreach ($b in $d.bench) {
                $key = if ($b.pass) { "$($b.pass) $($b.size)" } else { $b.size }
                $t[$key] = if ($null -ne $b.gpu_ms) { "{0:f3}" -f $b.gpu_ms } else { "wall {0:f2}" -f $b.wall_ms } }
            $parityRows += [pscustomobject]@{ API = $api; Device = $d.device; "fp32 max|d|" = "{0:e2}" -f $w32;
                                              "fp16 ulp" = $w16; "view codes" = $wv; "1080p ms" = $t["1920x1080"]; "4K ms" = $t["3840x2160"];
                                              "+view 1080p" = $t["composite+view 1920x1080"]; "+view 4K" = $t["composite+view 3840x2160"];
                                              Result = $(if ($d.pass) { "PASS" } else { "FAIL" }) }
        } else {
            $why = ($text | Select-Object -Last 1)
            $parityRows += [pscustomobject]@{ API = $api; Device = ""; "fp32 max|d|" = ""; "fp16 ulp" = "";
                                              Result = $(if ($code -eq 2) { "n/a: $why" } else { "ERROR" }) }
        }
    }
    $parityRows | Format-Table -AutoSize | Out-String | Write-Host
}
$parityOk = [bool]($parityRows | Where-Object { $_.API -eq "d3d12" -and $_.Result -eq "PASS" }) -and
            -not ($parityRows | Where-Object { $_.Result -in @("FAIL", "ERROR") })

# ---------------------------------------------------------------------------
# Phase 2 step 9: the viewer window itself. Its SDR swapchain read back against
# core/view.cpp (fit and 2x, image, false colour, wipe), then Gate B through
# the real display pass: a card of 10 to 2 000 nits on the HDR swapchain, every
# patch at its luminance up to the display's peak and clipped above it.
Say "Viewer window (Phase 2 step 9)"
$viewerRows = @()
if (Test-Path $Viewer) {
    foreach ($api in @("d3d12", "d3d11", "vulkan", "gl")) {
        foreach ($mode in @("parity", "card")) {
            $json = Join-Path $Reports "native_viewer_${mode}_${api}_$Stamp.json"
            $vargs = @("--api", $api, "--report", $json)
            if ($mode -eq "card") { $vargs += "--card" } else { $vargs += @("--dump", (Join-Path $Reports "native_viewer_dump_${api}_$Stamp")) }
            $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
            $text = & $Viewer @vargs 2>&1 | ForEach-Object { "$_" }
            $code = $LASTEXITCODE
            $ErrorActionPreference = $prev
            $text | Where-Object { $_ -match "^Viewer window|^Gate B through|patch|=>|FAIL$" } | Write-Host
            if (Test-Path $json) {
                $d = Get-Content $json -Raw | ConvertFrom-Json
                $worst = if ($d.cases) { ($d.cases | Measure-Object -Property max_code -Maximum).Maximum } else { "" }
                $p = @{}; foreach ($x in $d.patches) { $p[[string]$x.target_nits] = $x.swapchain_nits }
                $viewerRows += [pscustomobject]@{ API = $api; Check = $mode; Backend = $d.backend; Swapchain = $d.swapchain;
                                                  Peak = [math]::Round([double]$d.peak_nits); From = $d.peak_from;
                                                  DPR = $d.device_pixel_ratio; "max code" = $worst;
                                                  "203" = $p["203"]; "1000" = $p["1000"]; "2000" = $p["2000"]; Verdict = $d.verdict }
            } else {
                $viewerRows += [pscustomobject]@{ API = $api; Check = $mode; Verdict = $(if ($code -eq 2) { "n/a" } else { "ERROR" }) }
            }
        }
    }
    $viewerRows | Format-Table -AutoSize | Out-String | Write-Host
}
$viewerOk = [bool]($viewerRows | Where-Object { $_.API -eq "d3d12" -and $_.Check -eq "parity" -and $_.Verdict -eq "PASS" }) -and
            [bool]($viewerRows | Where-Object { $_.API -eq "d3d12" -and $_.Check -eq "card" -and $_.Verdict -eq "PASS" })
if ($gateB -and $parityOk -and $viewerOk) { exit 0 } else { exit 1 }
