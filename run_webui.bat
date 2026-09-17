@echo off
title Farmer Crop Advisory System

echo ============================================================
echo   Launching Farmer Crop Advisory System via WSL
echo ============================================================
echo.

REM Launch Streamlit inside the WSL environment where all packages reside
wsl -d Ubuntu-26.04 bash -lc "cd /mnt/d/PROJECT && (source .venv/bin/activate 2>/dev/null || source /home/tg/chronus-gpu/bin/activate 2>/dev/null || source ~/venvs/chronus-gpu/bin/activate 2>/dev/null || true) && python3 -m streamlit run main.py --server.address 0.0.0.0"

if errorlevel 1 (
    echo.
    echo Attempting launch via default WSL distribution...
    wsl bash -lc "cd /mnt/d/PROJECT && (source .venv/bin/activate 2>/dev/null || source /home/tg/chronus-gpu/bin/activate 2>/dev/null || true) && python3 -m streamlit run main.py --server.address 0.0.0.0"
)

pause
