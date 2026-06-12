# RUDRA Stage 1 Resume & LTX 2.3 Integration Script
# ==============================================================================
# This script handles the next phase of Stage 1 training:
#   1. Resumes the interrupted `flux2_full` training from step 15,000 to 100,000.
#   2. Generates LTX 2.3 `.npz` training pairs (128 channels) via diffusers.
#   3. Trains the LTX 2.3 Turbo Decoder (50,000 steps).
# ==============================================================================

$ErrorActionPreference = "Stop"

$RUDRA_ROOT   = "D:\A.I\Devlopments\rudra"
$OUTPUT_ROOT  = "D:\HDR\checkpoints"
$PAIR_ROOT    = "$RUDRA_ROOT\hdrdata"

Set-Location $RUDRA_ROOT
$env:PYTHONPATH = "d:\A.I\ComfyUI\custom_nodes;d:\A.I\ComfyUI\custom_nodes\radiance;$env:PYTHONPATH"

# Helper function to run steps
function Run-TrainingStep {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    Write-Host "`n==================================================================" -ForegroundColor Cyan
    Write-Host " STARTING: $Name" -ForegroundColor Cyan
    Write-Host "==================================================================" -ForegroundColor Cyan
    $start = Get-Date
    & $Command
    $elapsed = (Get-Date) - $start
    Write-Host "COMPLETED: $Name in $($elapsed.ToString())" -ForegroundColor Green
}

# ------------------------------------------------------------------------------
# PHASE 1: RESUME FLUX 2 FULL DECODER
# ------------------------------------------------------------------------------
Run-TrainingStep "Resume Flux 2 Full Decoder (Step 15,000 -> 100,000)" {
    $FinalCheckpoint = "$OUTPUT_ROOT\flux2_full\flux_rudra_stage1\flux_rudra_stage1\rudra_full_decoder_step100000.pth"
    if (Test-Path $FinalCheckpoint) {
        Write-Host "Flux 2 Full Decoder training is already completed up to step 100,000! Skipping Phase 1." -ForegroundColor Yellow
        return
    }

    $Flux2FullCheckpoint = "$OUTPUT_ROOT\flux2_full\flux_rudra_stage1\rudra_full_decoder_step015000.pth"
    if (!(Test-Path $Flux2FullCheckpoint)) {
        Write-Host "Error: Checkpoint not found at $Flux2FullCheckpoint" -ForegroundColor Red
        throw "Missing flux2_full checkpoint"
    }

    python training\train_rudra.py `
        --stage decoder `
        --pair_dir "$PAIR_ROOT\hdr_pairs" `
        --output_dir "$OUTPUT_ROOT\flux2_full\flux_rudra_stage1" `
        --model_type flux `
        --model_size full `
        --steps 100000 `
        --batch_size 1 `
        --lr 1e-4 `
        --val_split 0 `
        --patience 0 `
        --multi_curve `
        --resume $Flux2FullCheckpoint
}

# ------------------------------------------------------------------------------
# PHASE 2: GENERATE LTX 2.3 TRAINING PAIRS (128 channels)
# ------------------------------------------------------------------------------
Run-TrainingStep "Generate LTX 2.3 Training Pairs" {
    $LtxPairDir = "$PAIR_ROOT\ltx_pairs"
    
    # Generate 5,000 pairs (with 6 crops per image)
    python training\dataset_hdr.py `
        --exr_dir "G:\data\hdr" `
        --output_dir $LtxPairDir `
        --vae_path "dummy" `
        --vae_type "ltx-video" `
        --size 512 `
        --crops_per_image 6 `
        --target_count 5000 `
        --device cuda
}

# ------------------------------------------------------------------------------
# PHASE 3: TRAIN LTX 2.3 TURBO DECODER (50,000 steps)
# ------------------------------------------------------------------------------
Run-TrainingStep "Train LTX 2.3 Turbo Decoder" {
    python training\train_rudra.py `
        --stage decoder `
        --pair_dir "$PAIR_ROOT\ltx_pairs" `
        --output_dir "$OUTPUT_ROOT\ltx_turbo\ltx_rudra_stage1" `
        --model_type "ltx-video" `
        --model_size "turbo" `
        --steps 50000 `
        --batch_size 4 `
        --lr 3e-4 `
        --val_split 0 `
        --patience 0 `
        --multi_curve `
        --device cuda
}

Write-Host "`n==================================================================" -ForegroundColor Green
Write-Host " RUDRA Resume & LTX 2.3 Integration Completed Successfully!" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
