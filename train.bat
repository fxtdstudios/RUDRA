@echo off
REM Train RUDRA on your own footage. Double-click, or:
REM     train.bat "D:\path\to\your_hdr_footage"
REM
REM Everything else has a sensible default. See training\train_from_footage.py
REM for the flags if you want something different.
setlocal

set "REPO=%~dp0"
set "FOOTAGE=%~1"

if "%FOOTAGE%"=="" (
  set /p FOOTAGE="Folder holding your HDR footage: "
)
if "%FOOTAGE%"=="" (
  echo No folder given. Nothing to do.
  pause
  exit /b 1
)

REM Prefer the repo's own venv, then conda, then whatever python is on PATH.
set "PY=%REPO%.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" -c "import torch, cv2, numpy" 2>nul
if errorlevel 1 (
  echo.
  echo Installing requirements. This happens once.
  "%PY%" -m pip install -r "%REPO%requirements.txt"
  if errorlevel 1 (
    echo.
    echo Could not install the requirements. Check the errors above.
    pause
    exit /b 1
  )
)

"%PY%" "%REPO%training\train_from_footage.py" "%FOOTAGE%" %2 %3 %4 %5 %6 %7 %8 %9
set "CODE=%ERRORLEVEL%"

echo.
if "%CODE%"=="0" (
  echo Finished.
) else (
  echo Stopped. See the message above.
)
pause
exit /b %CODE%
