@echo off
REM ===================================================================
REM  RUDRA v3 data rebuild + image retrain -- tailored 22 Aug 2026
REM  Paths pre-filled for this machine. Run from the repo root:
REM      pipeline\run_v3_now.bat
REM
REM  BEFORE RUNNING, two decisions already made for you:
REM    - storage mode log2_extended (nothing clips, 27x shadow precision)
REM    - loss ceiling 40000 nits (matches SDR2HDRNet max_hdr=4.0)
REM  ONE thing still on you: temporal needs >= 6 independent video scenes.
REM  The share currently holds 2 (Chimera). Drop the Stuttgart HDR set's
REM  26 videos into %SRC% first, or temporal stays gated (image trains fine).
REM ===================================================================
setlocal

set SRC=\\192.168.100.200\Data\08_Research
set WORK=E:\RUDRA_v3_20260822
set PAIRS=%WORK%\pairs
set MODE=log2_extended
set CROPS=3

echo.
echo [0/5] inventory the sources  (NAS share -- may take a while)
python pipeline\scan_sources.py "%SRC%" --out "%WORK%\source_inventory.jsonl" || exit /b 1

echo.
echo [1/5] prepare pairs  (correct HDR target storage: %MODE%, %CROPS% crops)
python pipeline\prepare_pairs.py --inventory "%WORK%\source_inventory.jsonl" ^
    --dst "%PAIRS%" --mode %MODE% --crops %CROPS% || exit /b 1

echo.
echo [2/5] build manifests  (scene-safe, video forced into every split)
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
echo [5/5] train the image model  (fresh init -- see note below)
python training\train_sdr2hdr.py --mode image ^
    --manifest "%WORK%\sdr_hdr_manifest.jsonl" ^
    --output-dir "%WORK%\checkpoints\sdr2hdr_image_v3" ^
    --steps 50000 --batch-size 2 --crop-size 384 --grad-accum 2 ^
    --eval-batches 128 --eval-every 1000 --best-metric loss ^
    --scene-balanced-sampling --video-sample-fraction 0.4 ^
    --device cuda || exit /b 1

echo.
echo Image model done.
echo.
echo   NOTE fresh init was chosen over warm-starting from
echo   E:\RUDRA_postfix_20260818\...\sdr2hdr_image_postfix_v1\best.pt --
echo   that model learned fidelity to 10k-nit-clipped, linear-quantised
echo   targets; starting from it biases the new run toward the old defects.
echo.
echo Temporal is INTENTIONALLY not chained. Check the image model's held-out
echo video numbers first, then:
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
