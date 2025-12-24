# Quick Start Guide

Get the Arabic Pronunciation Assessment System running in 5 minutes!

## Prerequisites

- Python 3.8 or higher installed
- Internet connection (for initial setup)
- Microphone or audio files for testing

## Installation

### Windows

```bash
# 1. Run the setup script
.\setup.bat

# 2. Activate virtual environment
words\Scripts\activate

# 3. Start the application
streamlit run app.py
```

### macOS / Linux

```bash
# 1. Make setup script executable and run it
chmod +x setup.sh
./setup.sh

# 2. Activate virtual environment
source words/bin/activate

# 3. Start the application
streamlit run app.py
```

## Usage

1. **Open your browser** to `http://localhost:8501`

2. **Select an Arabic letter** from the dropdown menu
   - Easy: أ, ب, ت, د, ر, س, ل, م, ن
   - Medium: ث, ج, ز, ش, ف, ك, ه, و, ي
   - Hard: ح, خ, ذ, ص, ض, ط, ظ, ع, غ, ق

3. **Record your pronunciation**
   - **Option 1 (Recommended)**: Upload an audio file
   - **Option 2**: Use browser recording

4. **Click "Analyze Pronunciation"** to get instant feedback

5. **Review your results**
   - Overall score and letter grade
   - Pronunciation accuracy
   - Detailed error analysis
   - Personalized practice recommendations

## First Test

Try this simple test:

1. Select the letter **"ب"** (easy category)
2. Upload or record yourself saying "baa" (as in "bat")
3. Click analyze
4. You should see scores and feedback

## Troubleshooting

### "Python not found"
- Install Python from [python.org](https://www.python.org/downloads/)
- Make sure to check "Add Python to PATH" during installation

### "PyAudio installation failed"
- **Windows**: Install Visual C++ Build Tools
- **macOS**: Run `brew install portaudio`
- **Linux**: Run `sudo apt-get install portaudio19-dev`

### "FFmpeg not found"
- **Windows**: Install via Chocolatey: `choco install ffmpeg`
- **macOS**: Run `brew install ffmpeg`
- **Linux**: Run `sudo apt-get install ffmpeg`

### "Recording doesn't work"
- Use the file upload option instead
- Check microphone permissions in system settings

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Visit the [GitHub repository](https://github.com/YaqoobAnsari/Kutuby-Arabic-Words-Realtime-Detection) for updates

## Support

- Report issues: [GitHub Issues](https://github.com/YaqoobAnsari/Kutuby-Arabic-Words-Realtime-Detection/issues)
- Email: ansarimohammedyaqoob01@gmail.com

---

**Happy Learning! 🎯**
