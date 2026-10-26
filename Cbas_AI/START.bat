@echo off
setlocal
cd /d "%~dp0"
title CBAS — Cognitive Behavior Analysis System

echo.
echo  CBAS — Local Cognitive Behavior Analysis
echo.

where py >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON=py -3"
) else (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python 3.10 or newer was not found.
        echo Install Python from https://www.python.org/downloads/ and enable Add Python to PATH.
        pause
        exit /b 1
    )
    set "PYTHON=python"
)

%PYTHON% -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
    echo [ERROR] CBAS requires Python 3.10 or newer.
    pause
    exit /b 1
)

%PYTHON% -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing required analysis dependency: NumPy...
    %PYTHON% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed. Check your internet connection and Python pip setup.
        pause
        exit /b 1
    )
)

echo Starting CBAS at http://127.0.0.1:8000
%PYTHON% server.py --host 127.0.0.1 --port 8000 --open-browser
pause
