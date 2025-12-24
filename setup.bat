@echo off
REM Arabic Pronunciation Assessment System - Windows Setup Script
REM This script creates a virtual environment and installs all dependencies

echo ================================
echo Arabic Pronunciation Assessment
echo Setup Script for Windows
echo ================================
echo.

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [1/6] Checking Python installation...
python --version

REM Check if virtual environment already exists
if exist "words\" (
    echo.
    echo WARNING: Virtual environment 'words' already exists.
    set /p CONTINUE="Do you want to delete and recreate it? (y/n): "
    if /i "%CONTINUE%"=="y" (
        echo Deleting existing virtual environment...
        rmdir /s /q words
    ) else (
        echo Skipping virtual environment creation.
        goto :activate_venv
    )
)

echo.
echo [2/6] Creating virtual environment 'words'...
python -m venv words
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

:activate_venv
echo.
echo [3/6] Activating virtual environment...
call words\Scripts\activate.bat

echo.
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [5/6] Installing dependencies from requirements.txt...
echo This may take several minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo.
    echo Troubleshooting:
    echo - Make sure you have an internet connection
    echo - Try running as Administrator
    echo - Install Visual C++ Build Tools if PyAudio fails
    pause
    exit /b 1
)

echo.
echo [6/6] Verifying installation...
python -c "import streamlit; print('✓ Streamlit:', streamlit.__version__)"
python -c "import whisper; print('✓ Whisper installed successfully')"
python -c "import numpy; print('✓ NumPy:', numpy.__version__)"
python -c "import librosa; print('✓ Librosa:', librosa.__version__)"

echo.
echo ================================
echo Setup completed successfully!
echo ================================
echo.
echo To run the application:
echo   1. Activate the virtual environment:
echo      words\Scripts\activate
echo.
echo   2. Run the application:
echo      streamlit run app.py
echo.
echo Your browser will automatically open to http://localhost:8501
echo.
echo NOTE: Make sure FFmpeg is installed for audio processing
echo Download from: https://ffmpeg.org/download.html
echo.

pause
