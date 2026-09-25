@echo off
rem Why v4b/v4c beat the inverse in training and lose on the bench: the metric, or where.
rem Reads the CP7 trees in bench\cp_*; no GPU, no inference. See tools\diagnose_bench_gap.py.
cd /d "%~dp0.."
set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe
set OPENCV_IO_ENABLE_OPENEXR=1
if not exist reports\logs mkdir reports\logs
call :run aces v4b
call :run aces v4c
call :run oog v4c
call :run mix v4c
echo.
echo Done. Logs: reports\logs\gap_*.log
pause
exit /b 0
:run
echo.
echo == %1 / %2
%PY% tools\diagnose_bench_gap.py bench\cp_%1 --test %2 --stride 2 --json reports\logs\gap_%1_%2.json > reports\logs\gap_%1_%2.log 2>&1
type reports\logs\gap_%1_%2.log
exit /b 0
