@echo off
title CBAS v2 — Cognitive Behavior Analysis System

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║   CBAS v2  —  Cognitive Behavior Analysis   ║
echo  ║   System Startup                            ║
echo  ╚══════════════════════════════════════════════╝
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found.
    echo  Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

:: Check numpy
python -c "import numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo  Installing numpy...
    pip install numpy
)

echo  Starting CBAS v2 server on http://localhost:8000
echo  Open  index.html  in your browser, then click BACKEND button.
echo.
echo  Press Ctrl+C to stop the server.
echo.

python server.py --port 8000

pause
