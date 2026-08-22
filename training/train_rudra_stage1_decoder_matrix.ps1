# RUDRA Stage 1 Decoder Training Matrix
# Train decoders separately for:
#   Flux 1 Turbo / Flux 1 Full
#   Flux 2 Turbo / Flux 2 Full
#   Wan 2.1 Turbo / Wan 2.1 Full
#   Wan 2.2 Turbo / Wan 2.2 Full
#
# Run from:
#   D:\A.I\Devlopments\rudra
#
# IMPORTANT:
# - This trains RUDRA decoder only, not LoRA and not DRE.
# - Replace pair_dir paths if your dataset folders are different.
# - Start with TEST_STEPS first. After success, set $USE_TEST_STEPS = $false.

$ErrorActionPreference = "Stop"

$RUDRA_ROOT = "D:\A.I\Devlopments\rudra"
$OUTPUT_ROOT  = "D:\HDR\checkpoints"
$PAIR_ROOT    = "D:\A.I\Devlopments\rudra\hdrdata"

$USE_TEST_STEPS = $false

if ($USE_TEST_STEPS) {
    $STEPS_TURBO = 2000
    $STEPS_FULL  = 2000
} else {
    $STEPS_TURBO = 50000
    $STEPS_FULL  = 100000
}

$BATCH_TURBO = 1
$BATCH_FULL  = 1

$LR_TURBO = "3e-4"
$LR_FULL  = "1e-4"

Set-Location $RUDRA_ROOT

function Train-RudraDecoder {
    param(
        [string]$Name,
        [string]$ModelType,
        [string]$ModelSize,
        [string]$PairDir,
        [string]$OutputDir,
        [int]$Steps,
        [int]$BatchSize,
        [string]$LR
    )

    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Cyan
    Write-Host " Training $Name | model_type=$ModelType | model_size=$ModelSize" -ForegroundColor Cyan
    Write-Host " PairDir:   $PairDir"
    Write-Host " OutputDir: $OutputDir"
    Write-Host " Steps:     $Steps"
    Write-Host "==================================================================" -ForegroundColor Cyan

    if (!(Test-Path $PairDir)) {
        Write-Host "Missing pair_dir: $PairDir" -ForegroundColor Red
        throw "Missing pair_dir"
    }

    python training\train_rudra.py `
        --stage decoder `
        --pair_dir $PairDir `
        --output_dir $OutputDir `
        --model_type $ModelType `
        --model_size $ModelSize `
        --steps $Steps `
        --batch_size $BatchSize `
        --lr $LR `
        --val_split 0 `
        --patience 0 `
        --multi_curve

    Write-Host "Completed $Name" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# PHASE A: TURBO DECODERS
# ---------------------------------------------------------------------------

Train-RudraDecoder `
    -Name "Flux 1 Turbo Decoder" `
    -ModelType "flux" `
    -ModelSize "turbo" `
    -PairDir "$PAIR_ROOT\hdr_pairs" `
    -OutputDir "$OUTPUT_ROOT\flux1_turbo" `
    -Steps $STEPS_TURBO `
    -BatchSize $BATCH_TURBO `
    -LR $LR_TURBO

Train-RudraDecoder `
    -Name "Flux 2 Turbo Decoder" `
    -ModelType "flux" `
    -ModelSize "turbo" `
    -PairDir "$PAIR_ROOT\hdr_pairs" `
    -OutputDir "$OUTPUT_ROOT\flux2_turbo" `
    -Steps $STEPS_TURBO `
    -BatchSize $BATCH_TURBO `
    -LR $LR_TURBO

Train-RudraDecoder `
    -Name "Wan 2.1 Turbo Decoder" `
    -ModelType "wan" `
    -ModelSize "turbo" `
    -PairDir "$PAIR_ROOT\wan_hdr_pairs" `
    -OutputDir "$OUTPUT_ROOT\wan21_turbo" `
    -Steps $STEPS_TURBO `
    -BatchSize $BATCH_TURBO `
    -LR $LR_TURBO

Train-RudraDecoder `
    -Name "Wan 2.2 Turbo Decoder" `
    -ModelType "wan" `
    -ModelSize "turbo" `
    -PairDir "$PAIR_ROOT\wan_hdr_pairs" `
    -OutputDir "$OUTPUT_ROOT\wan22_turbo" `
    -Steps $STEPS_TURBO `
    -BatchSize $BATCH_TURBO `
    -LR $LR_TURBO

# ---------------------------------------------------------------------------
# PHASE B: FULL DECODERS
# ---------------------------------------------------------------------------

Train-RudraDecoder `
    -Name "Flux 1 Full Decoder" `
    -ModelType "flux" `
    -ModelSize "full" `
    -PairDir "$PAIR_ROOT\hdr_pairs" `
    -OutputDir "$OUTPUT_ROOT\flux1_full" `
    -Steps $STEPS_FULL `
    -BatchSize $BATCH_FULL `
    -LR $LR_FULL

Train-RudraDecoder `
    -Name "Flux 2 Full Decoder" `
    -ModelType "flux" `
    -ModelSize "full" `
    -PairDir "$PAIR_ROOT\hdr_pairs" `
    -OutputDir "$OUTPUT_ROOT\flux2_full" `
    -Steps $STEPS_FULL `
    -BatchSize $BATCH_FULL `
    -LR $LR_FULL

Train-RudraDecoder `
    -Name "Wan 2.1 Full Decoder" `
    -ModelType "wan" `
    -ModelSize "full" `
    -PairDir "$PAIR_ROOT\wan_hdr_pairs" `
    -OutputDir "$OUTPUT_ROOT\wan21_full" `
    -Steps $STEPS_FULL `
    -BatchSize $BATCH_FULL `
    -LR $LR_FULL

Train-RudraDecoder `
    -Name "Wan 2.2 Full Decoder" `
    -ModelType "wan" `
    -ModelSize "full" `
    -PairDir "$PAIR_ROOT\wan_hdr_pairs" `
    -OutputDir "$OUTPUT_ROOT\wan22_full" `
    -Steps $STEPS_FULL `
    -BatchSize $BATCH_FULL `
    -LR $LR_FULL

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Green
Write-Host " RUDRA Stage 1 Decoder Matrix Complete" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
