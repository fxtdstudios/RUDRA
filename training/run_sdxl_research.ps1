<#
  run_sdxl_research.ps1 - launch the RUDRA SDXL research program with auto-resume.

  Survives crashes / OOM / reboots: it re-invokes research_sdxl.py --resume in a
  loop. Because each stage checkpoints (Stage 2/3 every N steps) and --resume picks
  up the newest checkpoint and SKIPS phases already at their target step count, a
  restart loses at most the steps since the last save, not the whole run.

  Usage (from the repo root, in the ComfyUI conda env):
      conda activate comfyui
      powershell -ExecutionPolicy Bypass -File training\run_sdxl_research.ps1
      # only the core thesis:
      powershell -ExecutionPolicy Bypass -File training\run_sdxl_research.ps1 -Phase stage3

  Stop it: Ctrl-C, or delete the .keep_running flag it creates.
#>

param(
    [string]$Phase    = "all",     # all | prereq | stage2 | stage3 | sweep | ablate | report
    [string]$SdxlCkpt = "",        # path to the SDXL base .safetensors (sets SDXL_CKPT)
    [int]$MaxRetries  = 50,        # give up after this many crash-restarts
    [int]$CooldownSec = 20         # wait between restarts (lets the GPU settle after OOM)
)

$ErrorActionPreference = "Continue"

# Point the trainer at an SDXL checkpoint outside the ComfyUI root (e.g. G:\models).
if ($SdxlCkpt) {
    if (-not (Test-Path $SdxlCkpt)) { Write-Host "[run] SdxlCkpt not found: $SdxlCkpt" -ForegroundColor Red; exit 1 }
    $env:SDXL_CKPT = $SdxlCkpt
    Write-Host "[run] SDXL_CKPT = $SdxlCkpt" -ForegroundColor Cyan
}
$Repo  = Split-Path -Parent $PSScriptRoot
$Train = Join-Path $PSScriptRoot "research_sdxl.py"
$Flag  = Join-Path $Repo ".keep_running"
$Log   = Join-Path $Repo ("sdxl_research_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

# ComfyUI custom-nodes + radiance on PYTHONPATH so comfy/radiance import.
$ComfyRoot = if ($env:COMFY_ROOT) { $env:COMFY_ROOT } else { "D:\A.I\ComfyUI" }
$env:PYTHONPATH = "$ComfyRoot\custom_nodes;$ComfyRoot\custom_nodes\radiance;$env:PYTHONPATH"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

New-Item -ItemType File -Path $Flag -Force | Out-Null
Write-Host "[run] phase=$Phase  log=$Log  (delete $Flag to stop)" -ForegroundColor Cyan

for ($i = 1; $i -le $MaxRetries; $i++) {
    if (-not (Test-Path $Flag)) { Write-Host "[run] stop flag removed - exiting."; break }
    Write-Host "[run] attempt $i/$MaxRetries - python research_sdxl.py --phase $Phase --resume" -ForegroundColor Yellow

    & python $Train --phase $Phase --resume 2>&1 | Tee-Object -FilePath $Log -Append
    $code = $LASTEXITCODE

    if ($code -eq 0) {
        Write-Host "[run] completed cleanly (exit 0)." -ForegroundColor Green
        Remove-Item $Flag -ErrorAction SilentlyContinue
        break
    }
    if ($code -eq 1) {
        # research_sdxl.py returns 1 for missing prereqs / config errors - NOT transient.
        # Retrying won't help (e.g. SDXL weights or pairs missing). Stop and tell the user.
        Write-Host "[run] fatal prereq/config error (exit 1) - not retrying. Fix the cause above." -ForegroundColor Red
        Write-Host "      e.g. download SDXL:  python training\download_models.py --only sd_xl_base_1.0.safetensors" -ForegroundColor Red
        Write-Host "      or point to it:     `$env:SDXL_CKPT = '<path-to-sdxl>.safetensors'" -ForegroundColor Red
        Remove-Item $Flag -ErrorAction SilentlyContinue
        break
    }
    Write-Host "[run] crashed (exit $code) - resuming in $CooldownSec s..." -ForegroundColor Red
    Start-Sleep -Seconds $CooldownSec
}

if (Test-Path $Flag) {
    Write-Host "[run] hit MaxRetries=$MaxRetries without clean completion. Check the log." -ForegroundColor Red
    Remove-Item $Flag -ErrorAction SilentlyContinue
}
