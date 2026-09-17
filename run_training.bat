@echo off
setlocal

REM ============================================================
REM AI 3000 - Model Training Launcher
REM Trains the leaf vision model, then the soil suitability model.
REM ============================================================

title AI 3000 - Model Training

echo.
echo ============================================================
echo              AI 3000 MODEL TRAINING
echo ============================================================
echo.

REM Project location inside WSL.
set "PROJECT=/mnt/d/PROJECT"

REM Existing WSL GPU Python environment used by the project.
set "VENV=~/venvs/chronus-gpu"

echo [1/4] Checking WSL...
wsl.exe bash -lc "echo WSL OK"
if errorlevel 1 (
    echo.
    echo ERROR: WSL could not be started.
    pause
    exit /b 1
)

echo.
echo [2/4] Checking project directory...
wsl.exe bash -lc "test -d %PROJECT%"
if errorlevel 1 (
    echo.
    echo ERROR: Project directory not found:
    echo        %PROJECT%
    echo.
    echo Edit PROJECT in this script if your project is elsewhere.
    pause
    exit /b 1
)

echo.
echo [3/4] Training leaf disease model...
wsl.exe bash -lc "cd %PROJECT% && (source /home/tg/chronus-gpu/bin/activate 2>/dev/null || source %VENV%/bin/activate 2>/dev/null || true) && echo Python: && which python && echo TensorFlow: && python -c 'import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices(\"GPU\"))' && python train_image_model.py"
if errorlevel 1 (
    echo.
    echo ============================================================
    echo LEAF MODEL TRAINING FAILED
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo [4/4] Training soil suitability model...
wsl.exe bash -lc "cd %PROJECT% && (source /home/tg/chronus-gpu/bin/activate 2>/dev/null || source %VENV%/bin/activate 2>/dev/null || true) && python train_soil_model.py"
if errorlevel 1 (
    echo.
    echo ============================================================
    echo SOIL MODEL TRAINING FAILED
    echo ============================================================
    pause
    exit /b 1
)

echo.
echo ============================================================
echo              TRAINING COMPLETED
echo ============================================================
echo.
echo Model artifacts should now be available in:
echo     D:\PROJECT\models\
echo.
pause
endlocal
