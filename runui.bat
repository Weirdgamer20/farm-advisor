@echo off
setlocal
title Farmer Crop Advisory System [WSL]

echo ============================================================
echo   Launching Farmer Crop Advisory System via WSL
echo ============================================================
echo.

REM 1. Set directory to project root
cd /d "%~dp0"

REM 2. Verify WSL is installed and available
where wsl >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Windows Subsystem for Linux - WSL - was not found.
    echo Please install WSL to run this application in Linux.
    echo Native Windows execution is intentionally disabled.
    echo.
    pause
    exit /b 1
)

REM 3. Detect WSL distribution: prefer Ubuntu-26.04, then Ubuntu, or default
set "WSL_DISTRO="
wsl -d Ubuntu-26.04 true >nul 2>&1
if not errorlevel 1 (
    set "WSL_DISTRO=Ubuntu-26.04"
) else (
    wsl -d Ubuntu true >nul 2>&1
    if not errorlevel 1 (
        set "WSL_DISTRO=Ubuntu"
    )
)

if defined WSL_DISTRO (
    echo [*] Selected WSL Distribution: %WSL_DISTRO%
    set "WSL_CMD=wsl -d %WSL_DISTRO%"
) else (
    echo [*] Selected WSL Distribution: Default WSL instance
    set "WSL_CMD=wsl"
)

echo [*] Starting Streamlit inside WSL Linux...
echo [*] Local web UI: http://localhost:8501
echo [*] Waiting for server to initialize, then opening browser automatically...
start "" powershell -NoProfile -WindowStyle Hidden -Command "$u = 'http://localhost:8501'; for ($i = 0; $i -lt 90; $i++) { try { $res = (Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 1).StatusCode; if ($res -eq 200) { Start-Process $u; exit } } catch {} Start-Sleep -Milliseconds 800 }"

echo [*] Press Ctrl+C in this terminal to stop the application.
echo ============================================================
echo.

REM 4. Run Streamlit headless inside WSL
%WSL_CMD% bash -c "cd /mnt/d/PROJECT 2>/dev/null || true; if [ -f .venv/bin/activate ]; then source .venv/bin/activate; elif [ -f /home/tg/chronus-gpu/bin/activate ]; then source /home/tg/chronus-gpu/bin/activate; elif [ -f ~/venvs/chronus-gpu/bin/activate ]; then source ~/venvs/chronus-gpu/bin/activate; fi; python3 -m streamlit run main.py --server.address 0.0.0.0 --server.port 8501 --server.headless true --server.enableCORS false --server.enableXsrfProtection false --server.maxUploadSize 200"

echo.
echo ============================================================
echo   WSL application process ended.
echo ============================================================
echo.
pause
