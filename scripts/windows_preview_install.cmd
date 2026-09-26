@echo off
setlocal
cd /d "%~dp0"
py -3.13 -c "import struct; assert struct.calcsize('P')==8" || goto fail
if not exist ".venv\Scripts\python.exe" py -3.13 -m venv .venv
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.lock
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install --no-deps rudra_hdr-0.3.3rc1-py3-none-any.whl
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto fail
echo Installation complete. Open Start Studio.cmd.
pause
exit /b 0
:fail
echo Installation failed. Python 3.13 x64 and internet access are required. Read the error above.
pause
exit /b 1
