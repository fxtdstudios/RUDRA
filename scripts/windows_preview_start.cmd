@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run Install Studio.cmd first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m ui.server --checkpoint "%CD%\checkpoints\sdr2hdr_shadow_v1.pt" --device cuda --port 8431 %*
if errorlevel 1 pause
