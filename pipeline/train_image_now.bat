@echo off
REM ===================================================================
REM  Resume from the EXISTING pairs and train the image model.
REM  E:\RUDRA_v3_20260822\pairs already holds a verified 19,275-pair
REM  corpus (963 stills x 3 crops + strided Chimera/HFR video), so
REM  scan + prepare are skipped -- they would take hours and produce
REM  byte-identical output.
REM
REM  Run from the repo root:  pipeline\train_image_now.bat
REM ===================================================================
setlocal
set OPENCV_IO_ENABLE_OPENEXR=1
set WORK=E:\RUDRA_v3_20260822
set PAIRS=%WORK%\pairs

echo.
set DRIVE=E
set NEEDGB=5

REM --- free-space preflight -----------------------------------------
REM torch.save dies with "ios_base::badbit / unexpected pos" when the
REM drive is full (23 Aug 2026: the step-0 baseline failed 1.7 MB into
REM a write). Checkpoints + logs need a few GB; fail here, not later.
for /f %%a in ('powershell -NoProfile -Command "[math]::Floor((Get-PSDrive E).Free/1GB)"') do set FREEGB=%%a
echo   free space on E: %FREEGB% GB
if %FREEGB% LSS 5 (
    echo.
    echo   NOT ENOUGH SPACE: %FREEGB% GB free, need at least 5 GB.
    echo   Reclaim it with:
    echo     rmdir /s /q E:\RUDRA_v3_20260822\pairs_stride1_old
    echo     rmdir /s /q E:\RUDRA_postfix_20260818\data
    echo     rmdir /s /q E:\RUDRA_postfix_20260818\hdrdata\checkpoints\sdr2hdr_temporal_postfix_v1
    exit /b 1
)

echo [1/3] rebuild manifests from the existing pairs
REM Expect three lines of subtraction before the manifest table, all correct:
REM   - 2 records with non-finite source pixels (two Poly Haven crops)
REM   - 4,071 pairs below 1 nit: the scan pulled 08_Research\data\hdr into the
REM     inventory, which is ALREADY NORMALISED August output. Re-ingested as if
REM     it were scene-linear source it lands ~5,000x too dark, and each frame
REM     became its own single-frame "scene" -- 1,357 of them, which is why the
REM     old gate reported 2,324 scenes when the corpus really holds 967.
REM   - val/test scene-share caps: with only FOUR video scenes, whole-scene
REM     holdout put all 6,201 Bar records into val (95.6% of it). Capped now.
REM Real corpus: 963 Poly Haven stills + 4 video scenes (Bar, Fire, Chimera x2).
python pipeline\build_manifests.py --pairs-dir "%PAIRS%" --out-dir "%WORK%" ^
    --min-video-share 0.25 --clip-length 9 || exit /b 1

echo.
echo [2/3] VERIFY
REM Relaxed DELIBERATELY, for the image run only:
REM   --min-temporal-scenes 0  video corpus is 2 training scenes (needs 6).
REM     Temporal stays blocked by the .GATED marker; the image model uses no
REM     clips, so this must not stop it.
REM   --max-scene-share 0.35   applies to val and test, where a scene's share
REM     IS its weight in every number you quote. Train is a WARN, not a FAIL:
REM     its sampler is scene-balanced, so record counts do not set influence
REM     there. Expect "train: largest scene 41.8%%" as a warning -- that is the
REM     Bar scene, and more footage is the fix, not a lower bar.
python pipeline\verify_dataset.py --pairs-dir "%PAIRS%" ^
    --manifest "%WORK%\sdr_hdr_manifest.jsonl" ^
    --video-manifest "%WORK%\video_manifest_9f.jsonl" ^
    --min-temporal-scenes 0 --max-scene-share 0.35 || (
    echo.
    echo Dataset verification FAILED. Do not train. Paste the report.
    exit /b 1
)

echo.
echo [2b/3] TARGET SCALE CHECK -- the guard the 24 Aug run did not have
python pipeline\check_target_scale.py --manifest "%WORK%\sdr_hdr_manifest.jsonl" || (
    echo.
    echo Targets are not in the model's units. DO NOT TRAIN. Paste the output.
    exit /b 1
)

echo.
echo [3/3] train the image model  (fresh init, 50k steps)
REM New output dir. The v3 run of 24 Aug 2026 trained 50,000 steps against
REM undecoded log2 targets (loss 1.83, psnr_log 6.12 dB); its best.pt and
REM train.jsonl are wrong in a way that is not visible from the file, so they
REM stay where they are and nothing new is written beside them.
REM
REM Watch the FIRST line the dataset prints. It must say:
REM   [image/train] HDR targets: log2, 0.005..1,000,000 nits ... -> network units
REM If it says "no _ingest_config.json found", STOP -- targets are being misread
REM again and 50,000 steps will be wasted a second time.
python training\train_sdr2hdr.py --mode image ^
    --manifest "%WORK%\sdr_hdr_manifest.jsonl" ^
    --output-dir "%WORK%\checkpoints\sdr2hdr_image_v3b" ^
    --steps 50000 --batch-size 2 --crop-size 384 --grad-accum 2 ^
    --eval-batches 128 --eval-every 1000 --best-metric composite_gain --best-eval hard ^
    --video-sample-fraction 0.4 ^
    --device cuda || exit /b 1

REM Every eval now prints TWO conditions and the number that matters:
REM   clean_gain_db  model minus analytic inverse-ACES baseline on the untouched
REM                  ACES output. Near ZERO is correct: prepare_pairs made that
REM                  SDR with the very curve the baseline inverts, so there is
REM                  almost nothing to learn. It is a do-no-harm check.
REM   hard_gain_db   the same records under a seeded camera/codec degradation --
REM                  unknown tone curve, 4:2:0 chroma, banding, JPEG. THIS is
REM                  the deployment condition, and it selects best.pt.
REM Watch hard_gain_db. If it is still <= 0 after three evals, stop: the model
REM is not beating the analytic baseline it is built on top of.

echo.
echo Image model done. Temporal remains gated (2 training scenes of 6) --
echo adding multi-scene footage is what unblocks it, and it is also what fixes
echo the single-scene val/test video measurement. python pipeline\fetch_stuttgart.py
echo --list-root is the next source.
endlocal
