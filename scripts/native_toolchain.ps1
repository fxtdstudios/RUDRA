# Dot-sourced by NATIVE_GATE_A.ps1 and NATIVE_GATE_B.ps1: finds the C++
# toolchain for native/ on Windows.
#
# Any Visual Studio with the C++ tools counts: 2022 (17.x) or 2026 (18.x),
# Community, Professional, Enterprise or the standalone Build Tools. The CMake
# generator follows the version that is installed, and the CMake that ships
# with that Visual Studio is preferred, because a CMake from PATH (a conda env,
# say) may predate the generator. Qt's msvc2022_64 binaries link with either:
# the MSVC ABI has been stable since 2015.

function Find-NativeToolchain {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $vs = $null
    if (Test-Path $vswhere) {
        $found = & $vswhere -latest -products * -prerelease `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format json | ConvertFrom-Json
        if ($found) { $vs = @($found)[0] }
    }
    if (-not $vs) {
        Write-Host ""
        Write-Host "No Visual Studio with the C++ tools was found. Install the Build Tools (about 3 GB):" -ForegroundColor Yellow
        Write-Host '  winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"'
        Write-Host "or rerun this script with -InstallBuildTools to run that command. Then run the script again."
        return $null
    }
    $major = [int]($vs.installationVersion.Split('.')[0])
    $generator = switch ($major) {
        17 { "Visual Studio 17 2022" }
        18 { "Visual Studio 18 2026" }
        default { $null }
    }
    if (-not $generator) {
        Write-Host "Visual Studio $($vs.installationVersion) at $($vs.installationPath) is not 2022 or 2026." -ForegroundColor Yellow
        return $null
    }
    $cmake = Join-Path $vs.installationPath "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    if (-not (Test-Path $cmake)) { $cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source }
    if (-not $cmake) {
        Write-Host "Visual Studio found but no CMake: add the 'C++ CMake tools for Windows' component, or install CMake 3.24+." -ForegroundColor Yellow
        return $null
    }
    Write-Host "Visual Studio $($vs.installationVersion) ($($vs.displayName)), generator '$generator'"
    Write-Host "cmake $cmake"
    return @{ Generator = $generator; CMake = $cmake; Path = $vs.installationPath }
}

function Install-NativeBuildTools {
    Write-Host "Installing Visual Studio 2022 Build Tools with the C++ workload (winget, a few minutes)..."
    winget install --id Microsoft.VisualStudio.2022.BuildTools --accept-package-agreements --accept-source-agreements `
        --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
    return ($LASTEXITCODE -eq 0)
}

# A configure that failed, or ran with another Visual Studio, leaves a cache
# CMake refuses to reuse under a different generator. Clear just that.
function Reset-StaleCMakeCache($BuildDir, $Generator) {
    $cache = Join-Path $BuildDir "CMakeCache.txt"
    if (-not (Test-Path $cache)) { return }
    $line = Select-String -Path $cache -Pattern '^CMAKE_GENERATOR:INTERNAL=(.*)$' | Select-Object -First 1
    $cached = if ($line) { $line.Matches[0].Groups[1].Value } else { "" }
    if ($cached -ne $Generator) {
        Write-Host "clearing the CMake cache in $BuildDir (was '$cached')"
        Remove-Item -Force $cache
        Remove-Item -Recurse -Force (Join-Path $BuildDir "CMakeFiles") -ErrorAction SilentlyContinue
    }
}
