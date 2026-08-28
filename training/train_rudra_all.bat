@echo off
cd /d D:\A.I\Devlopments\rudra
REM RUDRA Training Pipeline — Turbo then Full for all models with existing data
REM RTX 4080 Super 16GB
REM Run from: D:\A.I\Devlopments\rudra

echo ============================================================
echo   RUDRA TRAINING PIPELINE
echo   Turbo 50K + Full 200K for all models with existing pairs
echo ============================================================
echo.

REM ── Wan 2.1 RUDRA Turbo 50K ──────────────────────────────────
echo [1/4] Wan 2.1 RUDRA Turbo (50K steps, ~1.2h)
echo ------------------------------------------------------------
python training/train_rudra.py --pair_dir D:\A.I\Devlopments\rudra\hdrdata\wan_hdr_pairs --output_dir D:\HDR\checkpoints\wan_rudra_turbo --model_type wan --model_size turbo --dr_dim 64 --multi_curve --steps 50000 --batch_size 4 --val_split 0.05 --patience 8
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Wan turbo failed. Continuing...
) else (
    echo [OK] Wan turbo complete.
)
echo.

REM ── Wan 2.1 RUDRA Full 200K ──────────────────────────────────
echo [2/4] Wan 2.1 RUDRA Full (200K steps, ~14h)
echo ------------------------------------------------------------
python training/train_rudra.py --pair_dir D:\A.I\Devlopments\rudra\hdrdata\wan_hdr_pairs --output_dir D:\HDR\checkpoints\wan_rudra_full --model_type wan --model_size full --dr_dim 64 --multi_curve --steps 200000 --batch_size 2 --val_split 0.05 --patience 8
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Wan full failed. Continuing...
) else (
    echo [OK] Wan full complete.
)
echo.

REM ── Flux RUDRA Turbo 50K ────────────────────────────────────
echo [3/4] Flux RUDRA Turbo (50K steps, ~2.1h)
echo ------------------------------------------------------------
python training/train_rudra.py --pair_dir D:\A.I\Devlopments\rudra\hdrdata\hdr_pairs --output_dir D:\HDR\checkpoints\flux_rudra_turbo --model_type flux --model_size turbo --dr_dim 64 --multi_curve --steps 50000 --batch_size 8 --val_split 0.05 --patience 8
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Flux turbo failed. Continuing...
) else (
    echo [OK] Flux turbo complete.
)
echo.

REM ── Flux RUDRA Full 200K ─────────────────────────────────────
echo [4/4] Flux RUDRA Full (200K steps, ~25h)
echo ------------------------------------------------------------
python training/train_rudra.py --pair_dir D:\A.I\Devlopments\rudra\hdrdata\hdr_pairs --output_dir D:\HDR\checkpoints\flux_rudra_full --model_type flux --model_size full --dr_dim 64 --multi_curve --steps 200000 --batch_size 4 --val_split 0.05 --patience 8
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Flux full failed. Continuing...
) else (
    echo [OK] Flux full complete.
)
echo.

echo ============================================================
echo   ALL RUDRA TRAINING COMPLETE
echo ============================================================
echo.
echo Checkpoints:
echo   D:\HDR\checkpoints\wan_rudra_turbo\rudra_turbo_decoder_ema_best.safetensors
echo   D:\HDR\checkpoints\wan_rudra_full\rudra_full_decoder_ema_best.safetensors
echo   D:\HDR\checkpoints\flux_rudra_turbo\rudra_turbo_decoder_ema_best.safetensors
echo   D:\HDR\checkpoints\flux_rudra_full\rudra_full_decoder_ema_best.safetensors
echo.
echo Estimated total time: ~42 hours
echo.
pause