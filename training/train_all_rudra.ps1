# train_all_rudra_fixed.ps1 — Safer sequential RUDRA training pipeline (Flux & Wan)
# ==============================================================================
# Recommended workflow:
#   1) Run with small STEPS first.
#   2) Validate outputs and previews.
#   3) Increase STEPS only after stable loss and checkpoints.
#
# Example:
#   powershell -ExecutionPolicy Bypass -File .\train_all_rudra_fixed.ps1
# ==============================================================================

$ErrorActionPreference = "Stop"

# -----------------------------
# User paths
# -----------------------------
$COMFY_DIR    = "D:\A.I\ComfyUI"
$RUDRA_ROOT   = "D:\A.I\Devlopments\rudra"
$SCRIPTS_DIR  = "$RUDRA_ROOT\training"

$HDR_DIR      = "D:\HDR"
$FLUX_PAIRS   = "D:\A.I\Devlopments\rudra\hdrdata\hdr_pairs"
$WAN_PAIRS    = "D:\A.I\Devlopments\rudra\hdrdata\wan_hdr_pairs"
$OUTPUT_DIR   = "D:\HDR\checkpoints"
$COMFY_VAE    = "$COMFY_DIR\models\vae"

$FLUX_VAE     = "$COMFY_VAE\ae.safetensors"
$WAN_VAE      = "$COMFY_VAE\wan_2.1_vae.safetensors"

# -----------------------------
# Training safety defaults
# -----------------------------
# Start small. Increase after the first successful run.
$PAIR_SIZE        = 512
$TEST_STEPS       = 2000
$FULL_STEPS       = 50000
$USE_TEST_STEPS   = $true       # Set to $false only after smoke test is stable.
$RUN_FLUX         = $true
$RUN_WAN          = $true
$RUN_TURBO        = $true
$RUN_FULL         = $false      # Keep false for first pass. Full decoder costs more VRAM/time.
$REBUILD_DATASET  = $false      # Set true only when you need to regenerate pairs.

$STEPS = if ($USE_TEST_STEPS) { $TEST_STEPS } else { $FULL_STEPS }

# -----------------------------
# Helpers
# -----------------------------
function Assert-PathExists {
    param([string]$PathToCheck, [string]$Label)
    if (!(Test-Path $PathToCheck)) {
        throw "$Label not found: $PathToCheck"
    }
}

function New-DirIfMissing {
    param([string]$DirPath)
    if (!(Test-Path $DirPath)) {
        New-Item -ItemType Directory -Force -Path $DirPath | Out-Null
    }
}

function Run-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    Write-Host "`n==================================================================" -ForegroundColor DarkGray
    Write-Host $Name -ForegroundColor Cyan
    Write-Host "==================================================================" -ForegroundColor DarkGray

    $start = Get-Date
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed with exit code $LASTEXITCODE : $Name"
    }
    $elapsed = (Get-Date) - $start
    Write-Host "Completed: $Name in $($elapsed.ToString())" -ForegroundColor Green
}

# -----------------------------
# Preflight
# -----------------------------
Write-Host "==================================================================" -ForegroundColor Green
Write-Host "         STARTING SAFE RUDRA TRAINING PIPELINE" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green

Assert-PathExists $SCRIPTS_DIR "Training scripts directory"
Assert-PathExists "$SCRIPTS_DIR\dataset_hdr.py" "dataset_hdr.py"
Assert-PathExists "$SCRIPTS_DIR\train_rudra.py" "train_rudra.py"
Assert-PathExists $HDR_DIR "HDR dataset directory"

if ($RUN_FLUX) { Assert-PathExists $FLUX_VAE "FLUX VAE" }
if ($RUN_WAN)  { Assert-PathExists $WAN_VAE  "WAN VAE"  }

New-DirIfMissing $OUTPUT_DIR
New-DirIfMissing "$OUTPUT_DIR\logs"

Write-Host "`nConfiguration:" -ForegroundColor Yellow
Write-Host "  Steps:          $STEPS"
Write-Host "  Test mode:      $USE_TEST_STEPS"
Write-Host "  Run FLUX:       $RUN_FLUX"
Write-Host "  Run WAN:        $RUN_WAN"
Write-Host "  Run Turbo:      $RUN_TURBO"
Write-Host "  Run Full:       $RUN_FULL"
Write-Host "  Rebuild pairs:  $REBUILD_DATASET"

# -----------------------------
# Dataset generation
# -----------------------------
if ($RUN_FLUX -and $REBUILD_DATASET) {
    Run-Step "[FLUX] Generating training pairs" {
        python "$SCRIPTS_DIR\dataset_hdr.py" `
            --exr_dir "$HDR_DIR" `
            --output_dir "$FLUX_PAIRS" `
            --vae_path "$FLUX_VAE" `
            --vae_type flux `
            --size $PAIR_SIZE
    }
}

if ($RUN_WAN -and $REBUILD_DATASET) {
    Run-Step "[WAN] Generating training pairs" {
        python "$SCRIPTS_DIR\dataset_hdr.py" `
            --exr_dir "$HDR_DIR" `
            --output_dir "$WAN_PAIRS" `
            --vae_path "$WAN_VAE" `
            --vae_type wan `
            --size $PAIR_SIZE
    }
}

if ($RUN_FLUX) { Assert-PathExists $FLUX_PAIRS "FLUX pair directory" }
if ($RUN_WAN)  { Assert-PathExists $WAN_PAIRS  "WAN pair directory"  }

# -----------------------------
# Training
# -----------------------------
if ($RUN_FLUX -and $RUN_TURBO) {
    Run-Step "[FLUX] Training TURBO decoder" {
        python "$SCRIPTS_DIR\train_rudra.py" `
            --pair_dir "$FLUX_PAIRS" `
            --output_dir "$OUTPUT_DIR\flux_turbo" `
            --model_type flux `
            --model_size turbo `
            --stage decoder `
            --steps $STEPS `
            --multi_curve
    }
}

if ($RUN_FLUX -and $RUN_FULL) {
    Run-Step "[FLUX] Training FULL decoder" {
        python "$SCRIPTS_DIR\train_rudra.py" `
            --pair_dir "$FLUX_PAIRS" `
            --output_dir "$OUTPUT_DIR\flux_full" `
            --model_type flux `
            --model_size full `
            --stage decoder `
            --steps $STEPS `
            --multi_curve
    }
}

if ($RUN_WAN -and $RUN_TURBO) {
    Run-Step "[WAN] Training TURBO decoder" {
        python "$SCRIPTS_DIR\train_rudra.py" `
            --pair_dir "$WAN_PAIRS" `
            --output_dir "$OUTPUT_DIR\wan_turbo" `
            --model_type wan `
            --model_size turbo `
            --stage decoder `
            --steps $STEPS `
            --multi_curve
    }
}

if ($RUN_WAN -and $RUN_FULL) {
    Run-Step "[WAN] Training FULL decoder" {
        python "$SCRIPTS_DIR\train_rudra.py" `
            --pair_dir "$WAN_PAIRS" `
            --output_dir "$OUTPUT_DIR\wan_full" `
            --model_type wan `
            --model_size full `
            --stage decoder `
            --steps $STEPS `
            --multi_curve
    }
}

Write-Host "`n==================================================================" -ForegroundColor Green
Write-Host "         RUDRA TRAINING PIPELINE COMPLETE" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
