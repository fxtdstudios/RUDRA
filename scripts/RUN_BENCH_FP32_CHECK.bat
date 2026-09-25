@echo off
rem Re-export four CP7 rows in fp32 and compare with bf16. See BENCH_FP32_CHECK.ps1.
where pwsh >nul 2>nul
if %errorlevel%==0 (
    pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0BENCH_FP32_CHECK.ps1" %*
) else (
    rem conda and some shells drop System32\WindowsPowerShell from PATH; use the full path
    "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0BENCH_FP32_CHECK.ps1" %*
)
pause
