@echo off
REM ===================================================================
REM  RUDRA corrected training pipeline -- 22 Aug 2026
REM  Every step gates the next. Do not skip verify_dataset.
REM
REM  Edit SRC and DST, then run from the repo root:
REM      pipeline\run_pipeline.bat
REM ===================================================================
setlocal

set SRC=Z:\08_Research
set WORK=E:\RUDRA_v3
set PAIRS=%WORK%\pairs
set MODE=log2_extended
set CROPS=3

REM Storage ceiling. log2_extended keeps the full scene range; the network
REM caps at max_hdr * 10000 nits, so set LOSS_CEILING to match (see README).
set LOSS_CEILING=40000

echo.
echo [0/5] inventory the sources
python pipeline\scan_sources.py "%SRC%" --out "%WORK%\source_inventory.jsonl" || exit /b 1

echo.
echo [1/5] prepare pairs  (correct HDR target storage)
python pipeline\prepare_pairs.py --inventory "%WORK%\source_inventory.jsonl" ^
    --dst "%PAIRS%" --mode %MODE% --crops %CROPS% || exit /b 1

echo.
echo [2/5] build manifests  (scene-safe, video in every split)
python pipeline\build_manifests.py --pairs-dir "%PAIRS%" --out-dir "%WORK%" ^
    --min-video-share 0.25 --clip-length 9 || exit /b 1

echo.
echo [3/5] VERIFY  -- training will not start if this fails
python pipeline\verify_dataset.py --pairs-dir "%PAIRS%" ^
    --manifest "%WORK%\sdr_hdr_manifest.jsonl" ^
    --video-manifest "%WORK%\video_manifest_9f.jsonl" || (
    echo.
    echo Dataset verification FAILED. Fix the corpus before training.
    exit /b 1
)

echo.
echo [4/5] patch the temporal split guard  (one-time, idempotent)
python pipeline\patch_video_split.py --apply || exit /b 1

echo.
echo [5/5] train the image model
python training\train_sdr2hdr.py --mode image ^
    --manifest "%WORK%\sdr_hdr_manifest.jsonl" ^
    --output-dir "%WORK%\checkpoints\sdr2hdr_image_v3" ^
    --steps 50000 --batch-size 2 --crop-size 384 --grad-accum 2 ^
    --eval-batches 128 --eval-every 1000 --best-metric loss ^
    --scene-balanced-sampling --video-sample-fraction 0.4 ^
    --device cuda || exit /b 1

echo.
echo Image model done. Temporal is INTENTIONALLY not chained here --
echo check the image model's held-out video numbers first, then run:
echo.
echo   python training\train_sdr2hdr.py --mode temporal ^^
echo       --manifest "%WORK%\video_manifest_9f.jsonl" ^^
echo       --image-checkpoint "%WORK%\checkpoints\sdr2hdr_image_v3\best.pt" ^^
echo       --output-dir "%WORK%\checkpoints\sdr2hdr_temporal_v3" ^^
echo       --steps 30000 --batch-size 2 --crop-size 256 --temporal-weight 0.5
echo.
echo Watch the first three evals. If none beats the step-0 baseline, STOP --
echo that is the August 2026 failure signature, not a slow start.
endlocal
