# Arabic Pronunciation Assessment System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/YaqoobAnsari/arabic-pronunciation-assessment)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, AI-powered Arabic pronunciation assessment system that provides real-time feedback on Arabic letter pronunciation using OpenAI's Whisper model for automatic speech recognition (ASR) and advanced phonetic analysis.

## Overview

This system addresses the challenge of learning Arabic pronunciation, particularly for non-native speakers, by providing objective, automated assessment of spoken Arabic letters. The application leverages state-of-the-art machine learning models to transcribe, analyze, and evaluate pronunciation accuracy across multiple linguistic dimensions.

### Key Features

- **AI-Powered Transcription**: Utilizes OpenAI's Whisper model for accurate Arabic speech-to-text conversion
- **Multi-Dimensional Assessment**: Evaluates pronunciation across four key dimensions:
  - Pronunciation Accuracy (40% weight)
  - Fluency & Timing (25% weight)
  - Accent & Prosody (20% weight)
  - Voice Quality (15% weight)
- **IPA (International Phonetic Alphabet) Analysis**: Converts Arabic text to IPA for precise phonetic comparison
- **Real-time Feedback**: Provides immediate, actionable feedback on pronunciation errors
- **Difficulty-Aware Scoring**: Adjusts scoring based on phoneme difficulty (emphatic consonants, pharyngeal sounds, etc.)
- **Comprehensive Logging**: CSV-based logging system for tracking progress over time
- **Interactive Visualization**: Radar charts and waveform displays for intuitive result interpretation

## Technical Architecture

### Components

1. **Speech Recognition Module**
   - OpenAI Whisper (base model) for Arabic ASR
   - Dual-mode transcription (Arabic-specific + auto-detect) for optimal accuracy
   - Sample rate: 16kHz, mono audio

2. **Phonetic Analysis Engine**
   - Arabic-to-IPA mapping covering all 28 Arabic letters
   - Levenshtein distance calculation for edit-distance metrics
   - Character-level and phoneme-level similarity scoring
   - Difflib sequence matching for alignment

3. **Assessment Algorithm**
   - Weighted multi-dimensional scoring system
   - Penalty system for:
     - Difficult Arabic sounds (ħ, ʕ, sˤ, dˤ, tˤ, ðˤ, q, x, ɣ)
     - Gibberish/non-Arabic speech detection
     - Over-transcription (saying more than the target)
   - Phonetic similarity boost for correct pronunciation transcribed as English

4. **User Interface**
   - Streamlit-based web application
   - Two recording options: browser-based and file upload
   - Real-time audio visualization
   - Responsive design with custom CSS styling

### Methodology

The pronunciation assessment follows this pipeline:

```
Audio Input → Whisper Transcription → Text/IPA Comparison →
Similarity Metrics → Penalty Application → Multi-dimensional Scoring →
Feedback Generation → Visualization & Logging
```

**Scoring Formula:**
```
Overall Score = (Pronunciation × 0.4) + (Fluency × 0.25) +
                (Accent × 0.2) + (Voice Quality × 0.15)

Pronunciation Accuracy = (Base Accuracy × 100) - Difficulty Penalty - Gibberish Penalty

Base Accuracy = (Char Similarity × 0.4) + (Levenshtein Accuracy × 0.4) +
                (IPA Similarity × 0.2)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (required for audio processing)
- Microphone (for live recording) or audio files

### System-Specific Setup

#### Windows
```bash
# Install FFmpeg using Chocolatey
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

#### macOS
```bash
brew install ffmpeg
brew install portaudio  # Required for PyAudio
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg portaudio19-dev python3-dev
```

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/YaqoobAnsari/Kutuby-Arabic-Words-Realtime-Detection.git
cd Kutuby-Arabic-Words-Realtime-Detection
```

2. **Run the automated setup script**

**Windows:**
```bash
.\setup.bat
```

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a virtual environment named `words`
- Install all required dependencies
- Download the Whisper base model
- Verify the installation

3. **Manual Installation (Alternative)**
```bash
# Create virtual environment
python -m venv words

# Activate virtual environment
# Windows:
words\Scripts\activate
# macOS/Linux:
source words/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import whisper; print('Whisper installed successfully')"
```

## Usage

### Running the Application

1. **Activate the virtual environment**
```bash
# Windows
words\Scripts\activate

# macOS/Linux
source words/bin/activate
```

2. **Launch the Streamlit app**
```bash
streamlit run app.py
```

3. **Access the application**
   - Open your browser to `http://localhost:8501`
   - The application will automatically load

### Using the System

1. **Select a Target Letter**
   - Choose from three difficulty categories (Easy, Medium, Hard)
   - Select an Arabic letter to practice

2. **Record Your Pronunciation**
   - **Option 1 (Recommended)**: Upload an audio file (WAV, MP3, M4A, FLAC, OGG)
   - **Option 2**: Use browser-based recording (may have compatibility issues)

3. **Analyze Results**
   - Click "Analyze Pronunciation"
   - Review your multi-dimensional scores
   - Read personalized feedback and practice recommendations

4. **Track Progress**
   - All analyses are logged to `pronunciation_analysis_log.csv`
   - Monitor improvement over time

## Academic Foundation

### Phonetic Representation

The system uses IPA (International Phonetic Alphabet) for precise phonetic representation of Arabic sounds:

| Arabic Letter | IPA Symbol | Description |
|--------------|------------|-------------|
| أ | ʔ | Glottal stop |
| ح | ħ | Voiceless pharyngeal fricative |
| ع | ʕ | Voiced pharyngeal fricative |
| ص | sˤ | Pharyngealized voiceless alveolar fricative |
| ق | q | Voiceless uvular plosive |

### Evaluation Metrics

1. **Character Similarity**: Sequence matching ratio (difflib)
2. **Levenshtein Distance**: Minimum edit distance between strings
3. **IPA Similarity**: Phoneme-level sequence matching
4. **Difficulty Weighting**: Empirically-determined penalties for challenging phonemes

### Validation

The system has been tested with:
- Native Arabic speakers (baseline calibration)
- Non-native learners (assessment accuracy)
- Various recording conditions (robustness testing)

## Deployment

### Hugging Face Spaces

This application is designed for deployment on Hugging Face Spaces:

1. **Create a new Space**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Create a new Space with Streamlit SDK

2. **Upload files**
   - `app.py` (main application)
   - `requirements.txt`
   - `README.md`

3. **Configure Space**
   - The Space will automatically detect and run the Streamlit app
   - No additional configuration needed

### Local Deployment

For production deployment:
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Data Privacy

- **No data is transmitted externally** except to load the Whisper model
- All audio processing is done locally
- Logs are stored locally in CSV format
- No personal data is collected or stored

## Limitations

1. **Whisper Model Limitations**
   - Optimized for full utterances, may struggle with isolated phonemes
   - Occasional English transcription of Arabic sounds (handled via phonetic similarity)

2. **Audio Quality Requirements**
   - Clear recording environment recommended
   - Minimum amplitude threshold: 0.01
   - Background noise may affect accuracy

3. **Phonetic Coverage**
   - Currently supports 28 Arabic letters (no diacritics/tashkeel)
   - Dialectal variations not distinguished

## Future Enhancements

- [ ] Support for full words and sentences
- [ ] Dialectal variation detection (MSA, Egyptian, Gulf, Levantine)
- [ ] Diacritical mark (tashkeel) support
- [ ] Mobile application development
- [ ] Integration with Arabic learning curricula
- [ ] Multi-user progress tracking dashboard
- [ ] Advanced phonetic error classification

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

See [QUICKSTART.md](QUICKSTART.md) for quick setup instructions.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{arabic_pronunciation_assessment,
  title={Arabic Pronunciation Assessment System},
  author={Yaqoob Ansari},
  year={2024},
  url={https://github.com/YaqoobAnsari/Kutuby-Arabic-Words-Realtime-Detection},
  version={2.0}
}
```

## References

1. Radford, A., et al. (2022). "Robust Speech Recognition via Large-Scale Weak Supervision." arXiv:2212.04356.
2. International Phonetic Association. (2020). "Handbook of the International Phonetic Association."
3. Levenshtein, V. I. (1966). "Binary codes capable of correcting deletions, insertions, and reversals."

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for the Whisper model
- Streamlit for the web framework
- The Arabic linguistics community for phonetic research

## Contact

For questions, issues, or collaboration opportunities:
- GitHub Issues: [https://github.com/YaqoobAnsari/Kutuby-Arabic-Words-Realtime-Detection/issues](https://github.com/YaqoobAnsari/Kutuby-Arabic-Words-Realtime-Detection/issues)
- Email: ansarimohammedyaqoob01@gmail.com

---

**Version**: 2.0
**Last Updated**: 2024
**Author**: Yaqoob Ansari
**Status**: Production Ready
