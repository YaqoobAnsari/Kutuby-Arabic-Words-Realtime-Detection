# Quick Start Guide
Get Arabic Word Recognition running in 5 minutes!

## Prerequisites
- Python 3.8+ installed
- Microphone (for live recording)
- Internet connection (for model download)

## Installation

### Windows
```bash
# Run setup script
.\setup.bat

# Activate environment
words\Scripts\activate

# Launch app
streamlit run app.py
```

### macOS / Linux
```bash
# Run setup script
chmod +x setup.sh
./setup.sh

# Activate environment
source words/bin/activate

# Launch app
streamlit run app.py
```

## First Use

1. **Open Browser**: Navigate to `http://localhost:8501`
2. **Wait for Model**: First launch downloads Wav2Vec2 model (~1.2GB, takes 2-5 min)
3. **Set Duration**: Choose 2-5 seconds recording time
4. **Record**: Click "Start Recording" button
5. **Speak**: Say an Arabic word clearly (try "السلام" - As-salaam)
6. **View Result**: See the recognized Arabic text

## Quick Test

Try these common Arabic words:
- **السلام** (As-salaam) - Peace
- **شكرا** (Shukran) - Thank you
- **مرحبا** (Marhaban) - Hello
- **الله** (Allah) - God
- **قال** (Qala) - He said

## Troubleshooting

### "PyAudio installation failed"
**Windows:**
```bash
pip install pipwin
pipwin install pyaudio
```

**macOS:**
```bash
brew install portaudio
pip install pyaudio
```

**Linux:**
```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

### "Model download failed"
- Check internet connection
- Retry - model is large (1.2GB)
- Use VPN if HuggingFace is blocked

### "Microphone not working"
- Check system microphone permissions
- Test microphone in other apps first
- Try increasing recording duration to 5 seconds

### "No speech detected"
- Speak louder and closer to microphone
- Use quiet environment
- Check audio levels in system settings

## Performance Tips

- **Best Audio Quality**: Use external USB microphone
- **Best Accuracy**: Speak Modern Standard Arabic (MSA)
- **Best Results**: Quranic vocabulary (95.3% accuracy)
- **Quiet Environment**: Background noise < 5dB recommended

## Next Steps

- Read full [README.md](README.md) for detailed documentation
- Check model performance metrics
- Explore API usage for custom integration
- Visit [GitHub](https://github.com/YaqoobAnsari/Kutuby-Arabic-Words-Realtime-Detection) for updates

## Support

- **Issues**: [GitHub Issues](https://github.com/YaqoobAnsari/Kutuby-Arabic-Words-Realtime-Detection/issues)
- **Email**: ansarimohammedyaqoob01@gmail.com

---

**Happy Recognizing! 🎤**
