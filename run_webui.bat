@echo off
title Farmer Crop Advisory System

REM Launch Streamlit in default WSL distribution (Ubuntu)
wsl bash -lc "cd /mnt/d/PROJECT && (source /home/tg/chronus-gpu/bin/activate 2>/dev/null || source ~/venvs/chronus-gpu/bin/activate 2>/dev/null || true) && python3 -m streamlit run main.py --server.address 0.0.0.0"

if errorlevel 1 (
    echo.
    echo WSL launch failed or was stopped. Checking local Windows environment fallback...
    streamlit run main.py
)

pause
