@echo off
setlocal EnableExtensions
title RUDRA Studio

rem  Start RUDRA Studio.
rem
rem  The first run builds a virtual environment beside this file and installs
rem  what the project needs. Every run after that checks the install is intact
rem  and goes straight to the server, so this is the only file you need.
rem
rem    run_studio.bat                          start the server
rem    run_studio.bat --setup                  reinstall, then start
rem    run_studio.bat --port 9000              anything else goes to the server
rem    run_studio.bat --device cpu --preload
rem
rem  Set RUDRA_PYTHON to a Python you already have -- a ComfyUI environment,
rem  say -- and this uses that instead of building a venv here and downloading
rem  a second copy of torch.

cd /d "%~dp0"
set "VENV=%CD%\.venv"
set "STAMP=%VENV%\.rudra-setup"
set "FORCE="
set "ARGS="
set "BASE="

rem  --setup is ours. Everything else is passed straight to the server.
:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--setup" (
    set "FORCE=1"
) else (
    set ARGS=%ARGS% "%~1"
)
shift
goto parse
:parsed

rem ---------------------------------------------------------------- interpreter
if defined RUDRA_PYTHON (
    set "PY=%RUDRA_PYTHON%"
    echo Using RUDRA_PYTHON: %RUDRA_PYTHON%
    goto have_python
)

set "PY=%VENV%\Scripts\python.exe"
if exist "%PY%" goto have_python

echo No environment yet. Building one in .venv ...
for %%P in ("py -3.12" "py -3.11" "py -3.13" "py -3.10" "py -3" "python") do call :try_python %%P
if not defined BASE (
    echo.
    echo   Could not find Python 3.10 to 3.13 on this machine.
    echo   Install it from https://www.python.org/downloads/ and tick
    echo   "Add python.exe to PATH", then run this file again.
    goto fail
)
echo Using %BASE%
%BASE% -m venv "%VENV%"
if errorlevel 1 (
    echo   Could not create the virtual environment.
    goto fail
)
set "FORCE=1"

:have_python
"%PY%" -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo   The Python at "%PY%" does not run. Delete the .venv folder and try again.
    goto fail
)

rem ------------------------------------------------------------------- up to date
rem  Reinstall when pyproject.toml has changed since the last successful setup,
rem  or when an import that should already work does not.
set "WANT="
for /f "skip=1 delims=" %%H in ('certutil -hashfile pyproject.toml MD5 2^>nul') do (
    if not defined WANT set "WANT=%%H"
)
if defined WANT set "WANT=%WANT: =%"
if not defined WANT set "WANT=nohash"

set "HAVE="
if exist "%STAMP%" set /p HAVE=<"%STAMP%"

if defined FORCE goto install
if not "%WANT%"=="%HAVE%" goto install
"%PY%" -c "import numpy, cv2, PIL, yaml, imageio, safetensors, tqdm, rudra" >nul 2>&1
if errorlevel 1 goto install
goto check_gpu

rem ---------------------------------------------------------------------- install
:install
echo.
echo Installing RUDRA and its dependencies. The first run downloads torch,
echo which is a couple of gigabytes, so give it a few minutes.
echo.
"%PY%" -m pip install --upgrade pip --quiet
"%PY%" -m pip install -e .
if errorlevel 1 (
    echo.
    echo   Install failed. The error is above.
    goto fail
)
> "%STAMP%" echo %WANT%

rem -------------------------------------------------------------------------- gpu
:check_gpu
"%PY%" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 3)" >nul 2>&1
if errorlevel 3 goto no_cuda
if errorlevel 1 goto no_torch
echo GPU: CUDA is available.
goto run

:no_cuda
echo.
echo   torch is installed but sees no CUDA device, so the server will run on
echo   the CPU. To get the GPU build for your card:
echo.
echo     .venv\Scripts\python -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
echo.
goto run

:no_torch
echo.
echo   torch is not installed here, so the page will open in demo mode.
echo   Run this file with --setup to install it.
echo.
goto run

rem ------------------------------------------------------------------------- run
:run
"%PY%" ui\server.py%ARGS%
if errorlevel 1 goto fail
endlocal
exit /b 0

rem --------------------------------------------------------------------- helpers
:try_python
if defined BASE goto :eof
%~1 -c "import sys;raise SystemExit(0 if sys.version_info[:2] in [(3,10),(3,11),(3,12),(3,13)] else 1)" >nul 2>&1
if errorlevel 1 goto :eof
set "BASE=%~1"
goto :eof

:fail
echo.
pause
endlocal
exit /b 1
