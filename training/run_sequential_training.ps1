# RUDRA Sequential Training Orchestration Script
# ==============================================================================
# This script executes Stage 3 Spatial DRE training followed automatically by 
# Stage 2 Gated LoRA training (resuming from step 2,000 to 10,000).
# ==============================================================================
$ErrorActionPreference = "Stop"

$RUDRA_ROOT   = "D:\A.I\Devlopments\rudra"
$OUTPUT_ROOT  = "D:\HDR\checkpoints"
$PAIR_ROOT    = "$RUDRA_ROOT\hdrdata"

Set-Location $RUDRA_ROOT

# Setup PYTHONPATH to import ComfyUI components cleanly
$env:PYTHONPATH = "d:\A.I\ComfyUI\custom_nodes;d:\A.I\ComfyUI\custom_nodes\radiance;$env:PYTHONPATH"

function Run-TrainingPhase {
    param(
        [string]$Name,
        [scriptblock]$Command
    )
    Write-Host "`n==================================================================" -ForegroundColor Cyan
    Write-Host " STARTING PHASE: $Name" -ForegroundColor Cyan
    Write-Host "==================================================================" -ForegroundColor Cyan
    $start = Get-Date
    & $Command
    $elapsed = (Get-Date) - $start
    Write-Host "SUCCESSFULLY COMPLETED: $Name in $($elapsed.ToString())" -ForegroundColor Green
}

# ------------------------------------------------------------------------------
# STEP 1: STAGE 3 SPATIAL DRE TRAINING (10,000 steps)
# ------------------------------------------------------------------------------
Run-TrainingPhase "Stage 3 Spatial DRE (10,000 steps)" {
    python training\train_rudra.py `
        --stage dre `
        --pair_dir "$PAIR_ROOT\hdr_pairs" `
        --output_dir "$OUTPUT_ROOT\flux2_full_lora\flux_rudra_stage3" `
        --model_path "D:/A.I/ComfyUI/models/diffusion_models/flux1-dev.safetensors" `
        --model_type flux `
        --steps 10000 `
        --batch_size 2 `
        --lr 2e-4 `
        --device cuda
}

# ------------------------------------------------------------------------------
# STEP 2: STAGE 2 GATED LORA TRAINING (Resume Step 2,000 -> 10,000)
# ------------------------------------------------------------------------------
Run-TrainingPhase "Stage 2 Gated LoRA (Step 2,000 -> 10,000)" {
    python training\train_hdr_lora.py `
        --cache_dir "$PAIR_ROOT\hdr_pairs" `
        --model_path "D:/A.I/ComfyUI/models/diffusion_models/flux1-dev.safetensors" `
        --output_dir "$OUTPUT_ROOT\flux2_full_lora\flux_rudra_stage2" `
        --model_name flux `
        --rank 16 `
        --steps 10000 `
        --resume "$OUTPUT_ROOT\flux2_full_lora\flux_rudra_stage2\rudra_stage2_step002000.pth" `
        --device cuda
}

Write-Host "`n==================================================================" -ForegroundColor Green
Write-Host " RUDRA ALL BACKBONE TRAINING PHASES COMPLETE!" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
