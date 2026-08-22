@echo off
REM ============================================================
REM  Radiance Training Pipeline - Wan 2.1 on RTX 4080 Super 16GB
REM  Phase 1: FullDecoder training pairs + training
REM  Phase 2: HDR LoRA training
REM ============================================================
REM
REM  PREREQUISITES:
REM    1. ComfyUI running with radiance custom node installed
REM    2. Wan VAE at: D:\A.I\ComfyUI\models\vae\wan_2.1_vae.safetensors
REM    3. Wan model at: D:\A.I\ComfyUI\models\diffusion_models\Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors
REM    4. HDR source data at: G:\data\hdr\  (24000+ PNG files already prepared)
REM
REM  ADJUST paths below to match your setup.
REM ============================================================

set RUDRA_ROOT=D:\A.I\Devlopments\rudra
set WAN_VAE=D:\A.I\ComfyUI\models\vae\wan_2.1_vae.safetensors
set WAN_MODEL=D:\A.I\ComfyUI\models\diffusion_models\Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors
set FLUX_VAE=D:\A.I\ComfyUI\models\vae\ae.safetensors
set FLUX_MODEL=D:\A.I\ComfyUI\models\diffusion_models\flux1-dev-kontext_fp8_scaled.safetensors
set HDR_SOURCE=G:\data\hdr
set OUT_BASE=G:\data\checkpoints

echo ============================================================
echo  RADIANCE TRAINING PIPELINE
echo  RTX 4080 Super 16GB
echo ============================================================
echo.
echo Available phases:
echo   1a - Generate Wan FullDecoder training pairs
echo   1b - Train Wan FullDecoder (200k steps)
echo   1c - Resume Wan FullDecoder training (from checkpoint)
echo   1d - Validate Wan FullDecoder
echo   2a - Generate Flux TurboDecoder training pairs
echo   2b - Train Flux TurboDecoder (50k steps)
echo   3a - Build Wan HDR LoRA latent cache (via ComfyUI)
echo   3b - Train Wan HDR LoRA (3000 steps, fp8)
echo   3c - Train Wan HDR LoRA (5000 steps, gradient checkpointing)
echo   4a - Build Flux HDR LoRA latent cache (via ComfyUI)
echo   4b - Train Flux HDR LoRA (5000 steps, fp8)
echo.

REM ============================================================
REM PHASE 1a: Generate Wan FullDecoder training pairs
REM ============================================================
:phase1a
echo [Phase 1a] Generating Wan FullDecoder training pairs...
echo   Source: %HDR_SOURCE%
echo   Output: %OUT_BASE%\wan_hdr_pairs\
echo   VAE:    %WAN_VAE%
echo.

python %RUDRA_ROOT%\training\dataset_hdr.py ^
    --exr_dir %HDR_SOURCE% ^
    --output_dir D:\A.I\Devlopments\rudra\hdrdata\wan_hdr_pairs ^
    --vae_path %WAN_VAE% ^
    --vae_type wan ^
    --size 512 ^
    --crops_per_image 6 ^
    --log_curve "ARRI LogC4" ^
    --target_count 5000 ^
    --device cuda

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Pair generation failed. Check paths and VAE.
    pause
    exit /b 1
)

echo [Phase 1a] Done. Check output at: %OUT_BASE%\wan_hdr_pairs\
echo.
echo Run 'validate_pairs.bat' to check pair quality, then proceed to Phase 1b.
pause

REM ============================================================
REM PHASE 1b: Train Wan FullDecoder
REM ============================================================
:phase1b
echo [Phase 1b] Training Wan FullDecoder...
echo   Pairs:  %OUT_BASE%\wan_hdr_pairs
echo   Output: %OUT_BASE%\wan_full_decoder
echo   Model:  wan, full, 200k steps, batch 4
echo   VRAM:   ~8 GB (fits your RTX 4080 Super 16GB easily)
echo.

python %RUDRA_ROOT%\training\train_turbo_decoder.py ^
    --pair_dir D:\A.I\Devlopments\rudra\hdrdata\wan_hdr_pairs ^
    --output_dir %OUT_BASE%\wan_full_decoder ^
    --model_type wan ^
    --model_size full ^
    --steps 200000 ^
    --batch_size 4 ^
    --lr 3e-4 ^
    --highlight_weight 2.0 ^
    --knee 0.96 ^
    --ema_decay 0.999 ^
    --val_split 0.05 ^
    --patience 10 ^
    --save_every 5000 ^
    --eval_every 2000 ^
    --log_every 100 ^
    --device cuda

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Training failed.
    pause
    exit /b 1
)

echo [Phase 1b] Done! Copy best checkpoint to ComfyUI:
echo   copy %OUT_BASE%\wan_full_decoder\full_decoder_ema_best.safetensors D:\A.I\ComfyUI\models\radiance\full_decoder_wan_ema.safetensors
pause

REM ============================================================
REM PHASE 1c: Resume Wan FullDecoder training (from existing checkpoint)
REM ============================================================
:phase1c
echo [Phase 1c] Resuming Wan FullDecoder training from checkpoint...
echo   This continues from the last saved checkpoint.
echo.

REM Auto-detect latest checkpoint
set RESUME_FROM=
for /f "delims=" %%i in ('dir /b /o:n "%OUT_BASE%\wan_full_decoder\full_decoder_step*.pth" 2^>nul') do set RESUME_FROM=%OUT_BASE%\wan_full_decoder\%%i

if "%RESUME_FROM%"=="" (
    echo [ERROR] No checkpoint found in %OUT_BASE%\wan_full_decoder\
    echo   Run Phase 1b first to generate initial checkpoints.
    pause
    exit /b 1
)

echo   Resuming from: %RESUME_FROM%
echo.

python %RUDRA_ROOT%\training\train_turbo_decoder.py ^
    --pair_dir D:\A.I\Devlopments\rudra\hdrdata\wan_hdr_pairs ^
    --output_dir %OUT_BASE%\wan_full_decoder ^
    --model_type wan ^
    --model_size full ^
    --steps 200000 ^
    --batch_size 4 ^
    --lr 3e-4 ^
    --highlight_weight 2.0 ^
    --knee 0.96 ^
    --ema_decay 0.999 ^
    --val_split 0.05 ^
    --patience 10 ^
    --save_every 5000 ^
    --eval_every 2000 ^
    --log_every 100 ^
    --resume "%RESUME_FROM%" ^
    --device cuda

echo [Phase 1c] Resume complete.
pause

REM ============================================================
REM PHASE 1d: Validate Wan FullDecoder
REM ============================================================
:phase1d
echo [Phase 1d] Validating Wan FullDecoder...
echo   Checking training log for best PSNR...
echo.

if exist "%OUT_BASE%\wan_full_decoder\train_log.jsonl" (
    echo Last 5 eval entries:
    findstr /C:"eval" "%OUT_BASE%\wan_full_decoder\train_log.jsonl" | more +0
) else (
    echo No train_log.jsonl found at %OUT_BASE%\wan_full_decoder\
)

echo.
echo Check files in: %OUT_BASE%\wan_full_decoder\
dir /b "%OUT_BASE%\wan_full_decoder\*.safetensors" 2>nul
dir /b "%OUT_BASE%\wan_full_decoder\*.pth" 2>nul
echo.
echo Copy best EMA weights to ComfyUI models dir:
echo   copy %OUT_BASE%\wan_full_decoder\full_decoder_ema_best.safetensors D:\A.I\ComfyUI\models\radiance\full_decoder_wan_ema.safetensors
pause

REM ============================================================
REM PHASE 2a: Generate Flux TurboDecoder training pairs
REM ============================================================
:phase2a
echo [Phase 2a] Generating Flux TurboDecoder training pairs...
echo   Source: %HDR_SOURCE%
echo   Output: %OUT_BASE%\flux_hdr_pairs\
echo   VAE:    %FLUX_VAE%
echo.

python %RUDRA_ROOT%\training\dataset_hdr.py ^
    --exr_dir %HDR_SOURCE% ^
    --output_dir D:\A.I\Devlopments\rudra\hdrdata\hdr_pairs ^
    --vae_path %FLUX_VAE% ^
    --vae_type flux ^
    --size 512 ^
    --crops_per_image 6 ^
    --log_curve "ARRI LogC4" ^
    --target_count 5000 ^
    --device cuda

echo [Phase 2a] Done. Pairs at: %OUT_BASE%\flux_hdr_pairs\
pause

REM ============================================================
REM PHASE 2b: Train Flux TurboDecoder
REM ============================================================
:phase2b
echo [Phase 2b] Training Flux TurboDecoder...
echo   Pairs:  %OUT_BASE%\flux_hdr_pairs
echo   Output: %OUT_BASE%\flux_turbo_decoder
echo   Model:  flux, turbo, 50k steps, batch 8
echo   VRAM:   ~4 GB (very lightweight)
echo.

python %RUDRA_ROOT%\training\train_turbo_decoder.py ^
    --pair_dir D:\A.I\Devlopments\rudra\hdrdata\hdr_pairs ^
    --output_dir %OUT_BASE%\flux_turbo_decoder ^
    --model_type flux ^
    --model_size turbo ^
    --steps 50000 ^
    --batch_size 8 ^
    --lr 3e-4 ^
    --highlight_weight 2.0 ^
    --knee 0.96 ^
    --ema_decay 0.999 ^
    --val_split 0.05 ^
    --patience 8 ^
    --save_every 5000 ^
    --eval_every 2000 ^
    --log_every 100 ^
    --device cuda

echo [Phase 2b] Done! Copy best EMA weights:
echo   copy %OUT_BASE%\flux_turbo_decoder\turbo_decoder_ema_best.safetensors D:\A.I\ComfyUI\models\radiance\turbo_decoder_flux_ema.safetensors
pause

REM ============================================================
REM PHASE 3a: Build Wan HDR LoRA latent cache
REM NOTE: This must be run from WITHIN ComfyUI (needs ComfyUI VAE)
REM ============================================================
:phase3a
echo [Phase 3a] Building Wan HDR LoRA latent cache...
echo.
echo   IMPORTANT: This step requires ComfyUI's VAE and cannot run standalone.
echo   You have TWO options:
echo.
echo   OPTION A - Use ComfyUI node (recommended):
echo     1. Open ComfyUI
echo     2. Add RadianceDatasetGenerator node
echo     3. Connect Wan VAE, set model_name=wan, compression_ratio=0.6
echo     4. Set input_directory=G:\data\hdr
echo     5. Set output_directory=G:\data\checkpoints\wan_lora_cache
echo.
echo   OPTION B - Use standalone script (run inside ComfyUI Python env):
echo.

python %RUDRA_ROOT%\training\dataset_hdr_lora.py ^
    --exr_dirs %HDR_SOURCE% ^
    --cache_dir D:\A.I\Devlopments\rudra\hdrdata\wan_lora_cache ^
    --vae_path %WAN_VAE% ^
    --vae_type wan ^
    --model_name wan ^
    --size 512 ^
    --n_frames 1 ^
    --device cuda

echo [Phase 3a] Cache at: %OUT_BASE%\wan_lora_cache\
pause

REM ============================================================
REM PHASE 3b: Train Wan HDR LoRA (fp8 model, batch 1, 16GB safe)
REM ============================================================
:phase3b
echo [Phase 3b] Training Wan HDR LoRA...
echo   Cache:  %OUT_BASE%\wan_lora_cache
echo   Model:  %WAN_MODEL% (fp8)
echo   Output: %OUT_BASE%\wan_hdr_lora
echo   VRAM:   ~15 GB (fp8 model + LoRA + optimizer)
echo   Steps:   3000 (quick baseline)
echo   Rank:    16
echo.

python %RUDRA_ROOT%\training\train_hdr_lora.py ^
    --cache_dir D:\A.I\Devlopments\rudra\hdrdata\wan_lora_cache ^
    --model_path %WAN_MODEL% ^
    --output_dir %OUT_BASE%\wan_hdr_lora ^
    --model_name wan ^
    --rank 16 ^
    --alpha 16.0 ^
    --steps 3000 ^
    --batch_size 1 ^
    --lr 1e-4 ^
    --highlight_weight 0.5 ^
    --ema_decay 0.999 ^
    --grad_clip 1.0 ^
    --gradient_checkpointing ^
    --use_8bit_adam ^
    --save_every 500 ^
    --eval_every 250 ^
    --log_every 50 ^
    --device cuda

echo [Phase 3b] Done! LoRA checkpoint at:
echo   %OUT_BASE%\wan_hdr_lora\radiance_hdr_lora_ema_best.safetensors
echo.
echo Load this in ComfyUI via:
echo   RadianceHDRLoRALoader node (strength=1.0)
pause

REM ============================================================
REM PHASE 3c: Train Wan HDR LoRA (longer run, same settings)
REM ============================================================
:phase3c
echo [Phase 3c] Training Wan HDR LoRA (5000 steps - production quality)...
echo.

python %RUDRA_ROOT%\training\train_hdr_lora.py ^
    --cache_dir D:\A.I\Devlopments\rudra\hdrdata\wan_lora_cache ^
    --model_path %WAN_MODEL% ^
    --output_dir %OUT_BASE%\wan_hdr_lora_5k ^
    --model_name wan ^
    --rank 16 ^
    --alpha 16.0 ^
    --steps 5000 ^
    --batch_size 1 ^
    --lr 1e-4 ^
    --highlight_weight 0.5 ^
    --ema_decay 0.999 ^
    --grad_clip 1.0 ^
    --gradient_checkpointing ^
    --use_8bit_adam ^
    --save_every 500 ^
    --eval_every 250 ^
    --log_every 50 ^
    --device cuda

echo [Phase 3c] Done!
pause

REM ============================================================
REM PHASE 4a: Build Flux HDR LoRA latent cache
REM ============================================================
:phase4a
echo [Phase 4a] Building Flux HDR LoRA latent cache...
echo.

python %RUDRA_ROOT%\training\dataset_hdr_lora.py ^
    --exr_dirs %HDR_SOURCE% ^
    --cache_dir D:\A.I\Devlopments\rudra\hdrdata\flux_lora_cache ^
    --vae_path %FLUX_VAE% ^
    --vae_type flux ^
    --model_name flux ^
    --size 512 ^
    --n_frames 1 ^
    --device cuda

echo [Phase 4a] Cache at: %OUT_BASE%\flux_lora_cache\
pause

REM ============================================================
REM PHASE 4b: Train Flux HDR LoRA (fp8 model, batch 1)
REM ============================================================
:phase4b
echo [Phase 4b] Training Flux HDR LoRA...
echo   Model:  %FLUX_MODEL% (fp8 ~11GB)
echo   VRAM:   ~15 GB with gradient_checkpointing + 8bit Adam
echo.

python %RUDRA_ROOT%\training\train_hdr_lora.py ^
    --cache_dir D:\A.I\Devlopments\rudra\hdrdata\flux_lora_cache ^
    --model_path %FLUX_MODEL% ^
    --output_dir %OUT_BASE%\flux_hdr_lora ^
    --model_name flux ^
    --rank 16 ^
    --alpha 16.0 ^
    --steps 5000 ^
    --batch_size 1 ^
    --lr 1e-4 ^
    --highlight_weight 0.5 ^
    --ema_decay 0.999 ^
    --grad_clip 1.0 ^
    --gradient_checkpointing ^
    --use_8bit_adam ^
    --quantize_base nf4 ^
    --save_every 500 ^
    --eval_every 250 ^
    --log_every 50 ^
    --device cuda

echo [Phase 4b] Done!
echo   %OUT_BASE%\flux_hdr_lora\radiance_hdr_lora_ema_best.safetensors
pause