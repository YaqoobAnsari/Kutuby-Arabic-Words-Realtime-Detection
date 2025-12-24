#!/bin/bash

# Arabic Pronunciation Assessment System - Unix/Linux/macOS Setup Script
# This script creates a virtual environment and installs all dependencies

echo "================================"
echo "Arabic Pronunciation Assessment"
echo "Setup Script for Unix/Linux/macOS"
echo "================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "[1/6] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python 3 is not installed${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓ Found: $PYTHON_VERSION${NC}"

# Check if virtual environment already exists
if [ -d "words" ]; then
    echo ""
    echo -e "${YELLOW}WARNING: Virtual environment 'words' already exists.${NC}"
    read -p "Do you want to delete and recreate it? (y/n): " CONTINUE
    if [ "$CONTINUE" = "y" ] || [ "$CONTINUE" = "Y" ]; then
        echo "Deleting existing virtual environment..."
        rm -rf words
    else
        echo "Skipping virtual environment creation."
        source words/bin/activate
        SKIP_VENV=1
    fi
fi

# Create virtual environment
if [ -z "$SKIP_VENV" ]; then
    echo ""
    echo "[2/6] Creating virtual environment 'words'..."
    python3 -m venv words
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Failed to create virtual environment${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Virtual environment created${NC}"

    # Activate virtual environment
    echo ""
    echo "[3/6] Activating virtual environment..."
    source words/bin/activate
fi

# Upgrade pip
echo ""
echo "[4/6] Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo ""
echo "[5/6] Installing dependencies from requirements.txt..."
echo "This may take several minutes..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to install dependencies${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "- Make sure you have an internet connection"
    echo "- On macOS, install portaudio: brew install portaudio"
    echo "- On Linux, install portaudio: sudo apt-get install portaudio19-dev"
    exit 1
fi

# Verify installation
echo ""
echo "[6/6] Verifying installation..."
python -c "import streamlit; print('✓ Streamlit:', streamlit.__version__)" || echo -e "${RED}✗ Streamlit not installed${NC}"
python -c "import whisper; print('✓ Whisper installed successfully')" || echo -e "${RED}✗ Whisper not installed${NC}"
python -c "import numpy; print('✓ NumPy:', numpy.__version__)" || echo -e "${RED}✗ NumPy not installed${NC}"
python -c "import librosa; print('✓ Librosa:', librosa.__version__)" || echo -e "${RED}✗ Librosa not installed${NC}"

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "To run the application:"
echo "  1. Activate the virtual environment:"
echo "     source words/bin/activate"
echo ""
echo "  2. Run the application:"
echo "     streamlit run app.py"
echo ""
echo "Your browser will automatically open to http://localhost:8501"
echo ""
echo -e "${YELLOW}NOTE: Make sure FFmpeg is installed for audio processing${NC}"
echo "  macOS:   brew install ffmpeg"
echo "  Linux:   sudo apt-get install ffmpeg"
echo ""
