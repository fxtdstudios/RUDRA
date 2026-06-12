@echo off
set RUDRA_ROOT=D:\A.I\Devlopments\rudra
cd /d %RUDRA_ROOT%
REM ============================================================
REM  RADIANCE TRAINING REFERENCE — ALL MODELS
REM  RTX 4080 Super 16GB
REM  Generated: 2026-05-25
REM ============================================================
REM
REM  KEY PATHS:
REM    HDR Source:    G:\data\hdr                    (24,076 files)
REM    Checkpoints:   G:\data\checkpoints
REM    Diffusion MD:  D:\A.I\ComfyUI\models\diffusion_models
REM    VAEs:          D:\A.I\ComfyUI\models\vae
REM    RUDRA Root:    D:\A.I\Devlopments\rudra
REM
REM  NOTES:
REM    - log_curve is AUTO-SELECTED per model from MODEL_VAE_CONFIG
REM    - scale_factor is AUTO-SELECTED per model from MODEL_VAE_CONFIG
REM    - latent_channels is AUTO-SELECTED per model from MODEL_VAE_CONFIG
REM    - See the MODEL CONFIGURATION TABLE below for details
REM ============================================================

echo ============================================================
echo  RADIANCE TRAINING — ALL MODELS REFERENCE
echo  RTX 4080 Super 16GB
echo ============================================================
echo.
echo  MODEL CONFIGURATION TABLE (from MODEL_VAE_CONFIG):
echo  ============================================================
echo  Model            Type         Ch  Scale_Factor Log_Curve    Noise_Sched  VAE File
echo  ---------------  -----------  --  ------------ -----------  -----------  -----------------------
echo  flux             flux1-dev    16  0.18215      ARRI LogC4  flow         ae.safetensors
echo  flux             flux2        16  0.18215      ARRI LogC4  flow         flux2-vae.safetensors
echo  wan              wan2.1       16  0.18215      ARRI LogC4  flow         wan_2.1_vae.safetensors
echo  wan              wan2.2       16  0.18215      ARRI LogC4  flow         wan2.2_vae.safetensors
echo  hunyuanvideo     hunyuan      16  0.18215      ARRI LogC4  flow         (needs hunyuan_vae)
echo  ltx-video        ltx-2.3      128 1.00000      Sony S-Log3 flow         (needs ltx_vae)
echo  sd3              sd3.5        16  1.53050      ARRI LogC4  flow         (needs sd3_vae)
echo  sdxl             sdxl         4   0.18215      ARRI LogC3  ddpm         vae-ft-mse
echo  sd15             sd1.5        4   0.18215      ARRI LogC3  ddpm         vae-ft-mse
echo  cogvideox        cogvideox    16  0.18215      ARRI LogC4  flow         (needs cogvideox_vae)
echo  lumina2          lumina2      16  0.18215      ARRI LogC4  flow         (needs lumina_vae)
echo  pixart           pixart       4   0.18215      ARRI LogC3  ddpm         vae-ft-mse
echo  kolors           kolors       4   0.18215      ARRI LogC3  ddpm         (needs kolors_vae)
echo  aura_flow        auraflow     4   0.18215      ARRI LogC3  ddpm         (needs aura_vae)
echo.
echo  AVAILABLE MODEL FILES ON YOUR DISK:
echo  ============================================================
echo  FLUX 1:
echo    flux1-dev.safetensors                          22.17 GB  (bf16, needs 24GB+ VRAM)
echo    flux1-dev-kontext_fp8_scaled.safetensors       11.09 GB  (fp8, fits 16GB)
echo    flux1-krea-dev_fp8_scaled.safetensors         11.09 GB  (fp8, fits 16GB)
echo  FLUX 2:
echo    flux-2-klein-4b-fp8.safetensors                3.79 GB  (fp8, fits 16GB)
echo    flux-2-klein-4b.safetensors                    7.22 GB  (bf16, fits 16GB)
echo    flux-2-klein-9b-fp8.safetensors                8.79 GB  (fp8, fits 16GB)
echo    flux-2-klein-base-9b-fp8.safetensors           8.91 GB  (fp8, fits 16GB)
echo  WAN 2.1:
echo    Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors    15.96 GB  (fp8, fits 16GB)
echo    wan2.1_fun_camera_v1.1_14B_bf16.safetensors   31.48 GB  (bf16, needs 32GB+)
echo  WAN 2.2:
echo    wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors  13.31 GB (fp8, fits 16GB)
echo    wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors   13.31 GB (fp8, fits 16GB)
echo    wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors  13.31 GB (fp8, fits 16GB)
echo    wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors   13.31 GB (fp8, fits 16GB)
echo    wan2.2_s2v_14B_fp8_scaled.safetensors             15.27 GB (fp8, tight 16GB)
echo    wan2.2_fun_camera_high_noise_14B_fp8_scaled.safetensors  14.25 GB (fp8)
echo    wan2.2_fun_camera_low_noise_14B_fp8_scaled.safetensors   14.25 GB (fp8)
echo    wan2.2_Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors    17.14 GB (fp8, OVER 16GB!)
echo    wan2.2_ti2v_5B_fp16.safetensors                    9.31 GB (bf16, fits 16GB)
echo  HUNYUANVIDEO:
echo    humo_17B_fp8_e4m3fn.safetensors               15.89 GB  (fp8, tight 16GB)
echo  LTX 2.3:
echo    ltx-2.3-22b-dev-fp8.safetensors               27.14 GB  (fp8, needs 24GB+)
echo    ltx-2.3-22b-distilled-lora-384.safetensors     7.08 GB  (LoRA only)
echo  Z-IMAGE (Qwen):
echo    qwen_image_fp8_e4m3fn.safetensors              19.03 GB  (fp8, needs 24GB+)
echo    z_image_bf16.safetensors                        11.46 GB  (bf16, fits 16GB)
echo    z_image_turbo_bf16.safetensors                  11.46 GB  (bf16, fits 16GB)
echo  SDXL:
echo    (no SDXL model found on disk)
echo  SD 1.5:
echo    (no SD1.5 model found on disk)
echo.
echo  AVAILABLE VAEs ON YOUR DISK:
echo  ============================================================
echo    ae.safetensors                              (Flux 1 VAE, 16ch)
echo    flux2-vae.safetensors                       (Flux 2 VAE, 16ch)
echo    wan_2.1_vae.safetensors                    (Wan 2.1 VAE, 16ch)
echo    wan2.2_vae.safetensors                     (Wan 2.2 VAE, 16ch)
echo    qwen_image_vae.safetensors                 (Qwen/Z-Image VAE, 16ch)
echo    vae-ft-mse-840000-ema-pruned.safetensors   (SD/SDXL VAE, 4ch)
echo    diffusion_pytorch_model.safetensors         (SDXL VAE, 4ch)
echo.
pause

REM ============================================================
REM  1. FLUX 1 DEV — TurboDecoder
REM ============================================================
:flux1_turbo_decoder
echo [FLUX 1] Generating TurboDecoder training pairs...
set FLUX1_VAE=D:\A.I\ComfyUI\models\vae\ae.safetensors
set FLUX1_MODEL=D:\A.I\ComfyUI\models\diffusion_models\flux1-dev-kontext_fp8_scaled.safetensors
set OUT=G:\data\checkpoints

python %RUDRA_ROOT%\training\dataset_hdr.py ^
    --exr_dir G:\data\hdr ^
    --output_dir D:\A.I\Devlopments\rudra\hdrdata\hdr_pairs ^
    --vae_path %FLUX1_VAE% ^
    --vae_type flux ^
    --size 512 ^
    --crops_per_image 6 ^
    --target_count 5000 ^
    --device cuda

echo [FLUX 1] Training TurboDecoder (50k steps)...
python %RUDRA_ROOT%\training\train_turbo_decoder.py ^
    --pair_dir D:\A.I\Devlopments\rudra\hdrdata\hdr_pairs ^
    --output_dir %OUT%\flux1_turbo_decoder ^
    --model_type flux ^
    --model_size turbo ^
    --steps 50000 ^
    --batch_size 8 ^
    --highlight_weight 2.0 ^
    --val_split 0.05 ^
    --patience 8 ^
    --device cuda
echo [FLUX 1] Done! Deploy:
echo   copy %OUT%\flux1_turbo_decoder\turbo_decoder_ema_best.safetensors D:\A.I\ComfyUI\models\radiance\turbo_decoder_flux_ema.safetensors
pause

REM ============================================================
REM  2. FLUX 1 DEV — HDR LoRA (fp8 model, 16GB safe)
REM ============================================================
:flux1_lora
echo [FLUX 1] Building LoRA latent cache...
python %RUDRA_ROOT%\training\dataset_hdr_lora.py ^
    --exr_dirs G:\data\hdr ^
    --cache_dir D:\A.I\Devlopments\rudra\hdrdata\flux1_lora_cache ^
    --vae_path %FLUX1_VAE% ^
    --vae_type flux ^
    --model_name flux ^
    --size 512 ^
    --device cuda

echo [FLUX 1] Training HDR LoRA (5000 steps, fp8 + nf4)...
python %RUDRA_ROOT%\training\train_hdr_lora.py ^
    --cache_dir D:\A.I\Devlopments\rudra\hdrdata\flux1_lora_cache ^
    --model_path %FLUX1_MODEL% ^
    --output_dir %OUT%\flux1_hdr_lora ^
    --model_name flux ^
    --rank 16 ^
    --steps 5000 ^
    --batch_size 1 ^
    --gradient_checkpointing ^
    --use_8bit_adam ^
    --quantize_base nf4 ^
    --device cuda
pause

REM ============================================================
REM  3. FLUX 2 KLEIN — TurboDecoder + LoRA
REM ============================================================
:flux2_turbo_decoder
echo [FLUX 2] Generating TurboDecoder training pairs...
set FLUX2_VAE=D:\A.I\ComfyUI\models\vae\flux2-vae.safetensors
set FLUX2_MODEL=D:\A.I\ComfyUI\models\diffusion_models\flux-2-klein-9b-fp8.safetensors

python %RUDRA_ROOT%\training\dataset_hdr.py ^
    --exr_dir G:\data\hdr ^
    --output_dir D:\A.I\Devlopments\rudra\hdrdata\hdr_pairs ^
    --vae_path %FLUX2_VAE% ^
    --vae_type flux ^
    --size 512 ^
    --crops_per_image 6 ^
    --target_count 5000 ^
    --device cuda

echo [FLUX 2] Training TurboDecoder...
python %RUDRA_ROOT%\training\train_turbo_decoder.py ^
    --pair_dir D:\A.I\Devlopments\rudra\hdrdata\hdr_pairs ^
    --output_dir %OUT%\flux2_turbo_decoder ^
    --model_type flux ^
    --model_size turbo ^
    --steps 50000 ^
    --batch_size 8 ^
    --highlight_weight 2.0 ^
    --val_split 0.05 ^
    --patience 8 ^
    --device cuda
pause

REM ============================================================
REM  3b. FLUX 2 KLEIN — HDR LoRA
REM ============================================================
:flux2_lora
echo [FLUX 2] Building LoRA cache...
python %RUDRA_ROOT%\training\dataset_hdr_lora.py ^
    --exr_dirs G:\data\hdr ^
    --cache_dir D:\A.I\Devlopments\rudra\hdrdata\flux2_lora_cache ^
    --vae_path %FLUX2_VAE% ^
    --vae_type flux ^
    --model_name flux ^
    --size 512 ^
    --device cuda

echo [FLUX 2] Training HDR LoRA (5000 steps)...
python %RUDRA_ROOT%\training\train_hdr_lora.py ^
    --cache_dir D:\A.I\Devlopments\rudra\hdrdata\flux2_lora_cache ^
    --model_path %FLUX2_MODEL% ^
    --output_dir %OUT%\flux2_hdr_lora ^
    --model_name flux ^
    --rank 16 ^
    --steps 5000 ^
    --batch_size 1 ^
    --gradient_checkpointing ^
    --use_8bit_adam ^
    --device cuda
pause

REM ============================================================
REM  4. WAN 2.1 — FullDecoder + LoRA (BEST ON YOUR GPU)
REM ============================================================
:wan21_full_decoder
set WAN21_VAE=D:\A.I\ComfyUI\models\vae\wan_2.1_vae.safetensors
set WAN21_MODEL=D:\A.I\ComfyUI\models\diffusion_models\Wan2_1-I2V-ATI-14B_fp8_e4m3fn.safetensors

echo [WAN 2.1] Generating FullDecoder training pairs...
python %RUDRA_ROOT%\training\dataset_hdr.py ^
    --exr_dir G:\data\hdr ^
    --output_dir D:\A.I\Devlopments\rudra\hdrdata\wan_hdr_pairs ^
    --vae_path %WAN21_VAE% ^
    --vae_type wan ^
    --size 512 ^
    --crops_per_image 6 ^
    --target_count 5000 ^
    --device cuda

echo [WAN 2.1] Training FullDecoder (200k steps)...
python %RUDRA_ROOT%\training\train_turbo_decoder.py ^
    --pair_dir D:\A.I\Devlopments\rudra\hdrdata\wan_hdr_pairs ^
    --output_dir %OUT%\wan21_full_decoder ^
    --model_type wan ^
    --model_size full ^
    --steps 200000 ^
    --batch_size 4 ^
    --highlight_weight 2.0 ^
    --knee 0.96 ^
    --val_split 0.05 ^
    --patience 10 ^
    --device cuda
echo [WAN 2.1] Done! Deploy:
echo   copy %OUT%\wan21_full_decoder\full_decoder_ema_best.safetensors D:\A.I\ComfyUI\models\radiance\full_decoder_wan_ema.safetensors
pause

:wan21_lora
echo [WAN 2.1] Building LoRA cache...
python %RUDRA_ROOT%\training\dataset_hdr_lora.py ^
    --exr_dirs G:\data\hdr ^
    --cache_dir D:\A.I\Devlopments\rudra\hdrdata\wan_lora_cache ^
    --vae_path %WAN21_VAE% ^
    --vae_type wan ^
    --model_name wan ^
    --size 512 ^
    --device cuda

echo [WAN 2.1] Training HDR LoRA (3000 steps, fp8)...
python %RUDRA_ROOT%\training\train_hdr_lora.py ^
    --cache_dir D:\A.I\Devlopments\rudra\hdrdata\wan_lora_cache ^
    --model_path %WAN21_MODEL% ^
    --output_dir %OUT%\wan21_hdr_lora ^
    --model_name wan ^
    --rank 16 ^
    --steps 3000 ^
    --batch_size 1 ^
    --gradient_checkpointing ^
    --use_8bit_adam ^
    --device cuda
pause

REM ============================================================
REM  5. WAN 2.2 — FullDecoder + LoRA
REM ============================================================
:wan22_full_decoder
set WAN22_VAE=D:\A.I\ComfyUI\models\vae\wan2.2_vae.safetensors
set WAN22_MODEL=D:\A.I\ComfyUI\models\diffusion_models\wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors

echo [WAN 2.2] Generating FullDecoder training pairs...
python %RUDRA_ROOT%\training\dataset_hdr.py ^
    --exr_dir G:\data\hdr ^
    --output_dir D:\A.I\Devlopments\rudra\hdrdata\wan_hdr_pairs ^
    --vae_path %WAN22_VAE% ^
    --vae_type wan ^
    --size 512 ^
    --crops_per_image 6 ^
    --target_count 5000 ^
    --device cuda

echo [WAN 2.2] Training FullDecoder (200k steps)...
python %RUDRA_ROOT%\training\train_turbo_decoder.py ^
    --pair_dir D:\A.I\Devlopments\rudra\hdrdata\wan_hdr_pairs ^
    --output_dir %OUT%\wan22_full_decoder ^
    --model_type wan ^
    --model_size full ^
    --steps 200000 ^
    --batch_size 4 ^
    --highlight_weight 2.0 ^
    --knee 0.96 ^
    --val_split 0.05 ^
    --patience 10 ^
    --device cuda
pause

:wan22_lora
echo [WAN 2.2] Building LoRA cache...
python %RUDRA_ROOT%\training\dataset_hdr_lora.py ^
    --exr_dirs G:\data\hdr ^
    --cache_dir D:\A.I\Devlopments\rudra\hdrdata\wan22_lora_cache ^
    --vae_path %WAN22_VAE% ^
    --vae_type wan ^
    --model_name wan ^
    --size 512 ^
    --device cuda

echo [WAN 2.2] Training HDR LoRA (3000 steps, fp8)...
python %RUDRA_ROOT%\training\train_hdr_lora.py ^
    --cache_dir D:\A.I\Devlopments\rudra\hdrdata\wan22_lora_cache ^
    --model_path %WAN22_MODEL% ^
    --output_dir %OUT%\wan22_hdr_lora ^
    --model_name wan ^
    --rank 16 ^
    --steps 3000 ^
    --batch_size 1 ^
    --gradient_checkpointing ^
    --use_8bit_adam ^
    --device cuda
pause

REM ============================================================
REM  6. HUNYUANVIDEO — TurboDecoder + LoRA (tight on 16GB!)
REM ============================================================
:hunyuan_turbo_decoder
REM NOTE: You need a hunyuan_vae.safetensors — download if missing
set HUNYUAN_VAE=D:\A.I\ComfyUI\models\vae\hunyuan_vae.safetensors
set HUNYUAN_MODEL=D:\A.I\ComfyUI\models\diffusion_models\humo_17B_fp8_e4m3fn.safetensors

echo [HUNYUAN] Generating TurboDecoder training pairs...
echo   NOTE: You need hunyuan_vae.safetensors in your VAE directory
python %RUDRA_ROOT%\training\dataset_hdr.py ^
    --exr_dir G:\data\hdr ^
    --output_dir %OUT%\hunyuan_pairs ^
    --vae_path %HUNYUAN_VAE% ^
    --vae_type hunyuanvideo ^
    --size 512 ^
    --crops_per_image 6 ^
    --target_count 5000 ^
    --device cuda

echo [HUNYUAN] Training TurboDecoder (50k steps)...
python %RUDRA_ROOT%\training\train_turbo_decoder.py ^
    --pair_dir %OUT%\hunyuan_pairs ^
    --output_dir %OUT%\hunyuan_turbo_decoder ^
    --model_type hunyuanvideo ^
    --model_size turbo ^
    --steps 50000 ^
    --batch_size 8 ^
    --highlight_weight 2.0 ^
    --val_split 0.05 ^
    --patience 8 ^
    --device cuda
pause

:hunyuan_lora
echo [HUNYUAN] Building LoRA cache (VERY TIGHT on 16GB, use --batch_size 1 + all optimizations)
echo   WARNING: HunyuanVideo 17B fp8 = 15.9 GB. LoRA training may not fit on 16GB.
echo   Consider using nf4 quantization or a 24GB GPU.

python %RUDRA_ROOT%\training\dataset_hdr_lora.py ^
    --exr_dirs G:\data\hdr ^
    --cache_dir %OUT%\hunyuan_lora_cache ^
    --vae_path %HUNYUAN_VAE% ^
    --vae_type hunyuanvideo ^
    --model_name hunyuanvideo ^
    --size 512 ^
    --n_frames 5 ^
    --device cuda

echo [HUNYUAN] Training HDR LoRA (3000 steps, VERY TIGHT on 16GB!)
python %RUDRA_ROOT%\training\train_hdr_lora.py ^
    --cache_dir %OUT%\hunyuan_lora_cache ^
    --model_path %HUNYUAN_MODEL% ^
    --output_dir %OUT%\hunyuan_hdr_lora ^
    --model_name hunyuanvideo ^
    --rank 8 ^
    --steps 3000 ^
    --batch_size 1 ^
    --gradient_checkpointing ^
    --use_8bit_adam ^
    --quantize_base nf4 ^
    --device cuda
pause

REM ============================================================
REM  7. Z-IMAGE (Qwen) — TurboDecoder + LoRA
REM ============================================================
:zimage_turbo_decoder
set ZIMAGE_VAE=D:\A.I\ComfyUI\models\vae\qwen_image_vae.safetensors
set ZIMAGE_MODEL=D:\A.I\ComfyUI\models\diffusion_models\z_image_bf16.safetensors

echo [Z-IMAGE] Generating TurboDecoder training pairs...
python %RUDRA_ROOT%\training\dataset_hdr.py ^
    --exr_dir G:\data\hdr ^
    --output_dir %OUT%\zimage_pairs ^
    --vae_path %ZIMAGE_VAE% ^
    --vae_type lumina2 ^
    --size 512 ^
    --crops_per_image 6 ^
    --target_count 5000 ^
    --device cuda

echo [Z-IMAGE] Training TurboDecoder (50k steps)...
python %RUDRA_ROOT%\training\train_turbo_decoder.py ^
    --pair_dir %OUT%\zimage_pairs ^
    --output_dir %OUT%\zimage_turbo_decoder ^
    --model_type lumina2 ^
    --model_size turbo ^
    --steps 50000 ^
    --batch_size 8 ^
    --highlight_weight 2.0 ^
    --val_split 0.05 ^
    --patience 8 ^
    --device cuda
pause

:zimage_lora
echo [Z-IMAGE] Building LoRA cache...
python %RUDRA_ROOT%\training\dataset_hdr_lora.py ^
    --exr_dirs G:\data\hdr ^
    --cache_dir %OUT%\zimage_lora_cache ^
    --vae_path %ZIMAGE_VAE% ^
    --vae_type lumina2 ^
    --model_name lumina2 ^
    --size 512 ^
    --device cuda

echo [Z-IMAGE] Training HDR LoRA (5000 steps)...
python %RUDRA_ROOT%\training\train_hdr_lora.py ^
    --cache_dir %OUT%\zimage_lora_cache ^
    --model_path %ZIMAGE_MODEL% ^
    --output_dir %OUT%\zimage_hdr_lora ^
    --model_name lumina2 ^
    --rank 16 ^
    --steps 5000 ^
    --batch_size 1 ^
    --gradient_checkpointing ^
    --use_8bit_adam ^
    --device cuda
pause

REM ============================================================
REM  8. LTX 2.3 — TurboDecoder (128ch!) — NEEDS 24GB+ FOR LoRA
REM ============================================================
:ltx_turbo_decoder
REM NOTE: LTX 2.3 22B model = 27 GB fp8. CANNOT train LoRA on 16GB.
REM TurboDecoder training is fine (only ~4-8GB).
REM LTX uses 128 latent channels!
set LTX_MODEL=D:\A.I\ComfyUI\models\checkpoints\ltx-2.3-22b-dev-fp8.safetensors

echo [LTX 2.3] Generating TurboDecoder training pairs (128ch!)...
REM NOTE: LTX VAE needs to be downloaded separately or use diffusers
echo   Requires ltx_vae.safetensors — currently not in your VAE directory
echo   Download from: https://huggingface.co/Lightricks/LTX-Video/tree/main/vae
echo.
echo   Once you have the VAE:
echo   python dataset_hdr.py --exr_dir G:\data\hdr --output_dir %OUT%\ltx_pairs --vae_path ltx_vae.safetensors --vae_type ltx-video --size 512 --crops_per_image 6 --target_count 5000 --device cuda
echo.
echo   Then train:
echo   python train_turbo_decoder.py --pair_dir %OUT%\ltx_pairs --output_dir %OUT%\ltx_turbo_decoder --model_type ltx-video --model_size turbo --steps 50000 --batch_size 2 --device cuda
echo.
echo   LTX LoRA training CANNOT fit on 16GB (model is 27GB fp8 alone).
echo   Need 40GB+ for LoRA training.
pause

REM ============================================================
REM  9. SDXL — TurboDecoder + LoRA (EASY, fits 16GB comfortably)
REM ============================================================
:sdxl_turbo_decoder
set SDXL_VAE=D:\A.I\ComfyUI\models\vae\vae-ft-mse-840000-ema-pruned.safetensors

echo [SDXL] Generating TurboDecoder training pairs...
python %RUDRA_ROOT%\training\dataset_hdr.py ^
    --exr_dir G:\data\hdr ^
    --output_dir %OUT%\sdxl_pairs ^
    --vae_path %SDXL_VAE% ^
    --vae_type sdxl ^
    --size 512 ^
    --crops_per_image 6 ^
    --target_count 5000 ^
    --device cuda

echo [SDXL] Training TurboDecoder (50k steps)...
python %RUDRA_ROOT%\training\train_turbo_decoder.py ^
    --pair_dir %OUT%\sdxl_pairs ^
    --output_dir %OUT%\sdxl_turbo_decoder ^
    --model_type sdxl ^
    --model_size turbo ^
    --steps 50000 ^
    --batch_size 8 ^
    --highlight_weight 2.0 ^
    --val_split 0.05 ^
    --patience 8 ^
    --device cuda
pause

REM ============================================================
REM  10. SD 1.5 — TurboDecoder (very fast, 4ch)
REM ============================================================
:sd15_turbo_decoder
echo [SD 1.5] Generating TurboDecoder training pairs...
python %RUDRA_ROOT%\training\dataset_hdr.py ^
    --exr_dir G:\data\hdr ^
    --output_dir %OUT%\sd15_pairs ^
    --vae_path %SDXL_VAE% ^
    --vae_type sd15 ^
    --size 512 ^
    --crops_per_image 6 ^
    --target_count 5000 ^
    --device cuda

echo [SD 1.5] Training TurboDecoder (50k steps, very fast!)...
python %RUDRA_ROOT%\training\train_turbo_decoder.py ^
    --pair_dir %OUT%\sd15_pairs ^
    --output_dir %OUT%\sd15_turbo_decoder ^
    --model_type sd15 ^
    --model_size turbo ^
    --steps 50000 ^
    --batch_size 16 ^
    --highlight_weight 2.0 ^
    --val_split 0.05 ^
    --patience 8 ^
    --device cuda
pause