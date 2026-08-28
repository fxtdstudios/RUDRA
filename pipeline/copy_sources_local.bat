@echo off
REM ===================================================================
REM  One-time fix: mirror the NAS training sources onto E:\source_hdr
REM  so the ingest never reads over SMB again (a transient NAS drop
REM  silently skipped all 3,677 network sources on 23 Aug 2026).
REM  robocopy retries per-file and resumes if interrupted - safe to
REM  re-run anytime. When the copies finish, the full pipeline runs.
REM ===================================================================
setlocal
set NAS=\\192.168.100.200\Data\08_Research\Source_HDR
set LOCAL=E:\source_hdr

echo.
echo [1/3] PolyHaven HDRI stills (963 EXRs, a few GB)
robocopy "%NAS%\PolyHaven" "%LOCAL%\PolyHaven" *.exr /R:5 /W:3 /NP /NFL
if %ERRORLEVEL% GEQ 8 ( echo PolyHaven copy FAILED & exit /b 1 )

echo.
echo [2/3] Netflix Chimera TIFF sequences (large - go get coffee)
robocopy "%NAS%\Netflix" "%LOCAL%\Netflix" /E /R:5 /W:3 /NP /NFL
if %ERRORLEVEL% GEQ 8 ( echo Netflix copy FAILED & exit /b 1 )

echo.
echo [3/3] sources are local - rebuilding corpus and starting training
call "%~dp0run_v3_now.bat"
endlocal
