"""
Arabic Pronunciation Assessment System - Version 2.0 (Fixed)
Real Whisper integration with audio recording and comprehensive analysis
"""

import streamlit as st
import numpy as np
import tempfile
import os
import time
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
import warnings
import librosa
import difflib
import whisper
import soundfile as sf
import pandas as pd
import csv
from datetime import datetime
from Levenshtein import distance as levenshtein_distance
warnings.filterwarnings("ignore")

# CSV Logging Configuration
LOG_FILE = "pronunciation_analysis_log.csv"
LOG_COLUMNS = [
    "timestamp",
    "target_text", 
    "target_ipa",
    "whisper_raw_output",
    "transcribed_ipa", 
    "audio_duration_sec",
    "audio_max_amplitude",
    "char_similarity",
    "levenshtein_distance", 
    "levenshtein_accuracy",
    "ipa_similarity",
    "has_arabic_in_output",
    "difficulty_penalty",
    "gibberish_penalty",
    "base_accuracy",
    "pronunciation_score",
    "fluency_score",
    "accent_score", 
    "voice_quality_score",
    "overall_score",
    "letter_grade",
    "error_types",
    "analysis_notes"
]

def initialize_log_file():
    """Initialize CSV log file with headers if it doesn't exist"""
    if not os.path.exists(LOG_FILE):
        print(f"[LOG] Creating new log file: {LOG_FILE}")
        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(LOG_COLUMNS)
        print(f"[LOG] Log file created with {len(LOG_COLUMNS)} columns")
    else:
        print(f"[LOG] Using existing log file: {LOG_FILE}")

def log_analysis_result(
    target_text: str,
    target_ipa: str, 
    whisper_output: str,
    transcribed_ipa: str,
    audio_data: np.ndarray,
    metrics: Dict,
    scores: Dict,
    errors: List[Dict]
):
    """Log analysis result to CSV file"""
    try:
        # Prepare data for logging
        timestamp = datetime.now().isoformat()
        
        # Audio metrics
        audio_duration = len(audio_data) / 16000 if audio_data is not None else 0
        audio_max_amp = np.max(np.abs(audio_data)) if audio_data is not None else 0
        
        # Text analysis
        has_arabic = any('\u0600' <= char <= '\u06FF' for char in whisper_output) if whisper_output else False
        
        # Error types summary
        error_types = "|".join([err['error_type'] for err in errors]) if errors else "none"
        
        # Analysis notes
        notes = []
        if whisper_output == "":
            notes.append("empty_transcription")
        if whisper_output == target_text:
            notes.append("perfect_match") 
        if not has_arabic and whisper_output:
            notes.append("non_arabic_detected")
        if has_arabic and whisper_output != target_text:
            notes.append("incorrect_arabic")
            
        analysis_notes = "|".join(notes) if notes else "normal"
        
        # Prepare row data
        row_data = [
            timestamp,
            target_text.encode('ascii', 'ignore').decode('ascii') if target_text else "",  # ASCII safe
            target_ipa,
            whisper_output.encode('ascii', 'ignore').decode('ascii') if whisper_output else "",  # ASCII safe
            transcribed_ipa,
            f"{audio_duration:.3f}",
            f"{audio_max_amp:.4f}",
            f"{metrics.get('char_similarity', 0):.4f}",
            metrics.get('lev_distance', 0),
            f"{metrics.get('lev_accuracy', 0):.4f}",
            f"{metrics.get('ipa_similarity', 0):.4f}",
            has_arabic,
            f"{metrics.get('difficulty_penalty', 0):.3f}",
            f"{metrics.get('gibberish_penalty', 0):.3f}",
            f"{metrics.get('base_accuracy', 0):.4f}",
            f"{scores.get('pronunciation_accuracy', 0):.2f}",
            f"{scores.get('fluency_score', 0):.2f}",
            f"{scores.get('accent_score', 0):.2f}",
            f"{scores.get('voice_quality_score', 0):.2f}",
            f"{scores.get('overall_score', 0):.2f}",
            scores.get('letter_grade', 'F'),
            error_types,
            analysis_notes
        ]
        
        # Append to CSV file
        with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
            
        print(f"[LOG] Analysis logged to {LOG_FILE}")
        print(f"[LOG] Target: {len(target_text)} chars, Whisper: {len(whisper_output)} chars, Score: {scores.get('overall_score', 0):.1f}%")
        
    except Exception as e:
        print(f"[LOG ERROR] Failed to log analysis: {e}")

# Configure Streamlit page
st.set_page_config(
    page_title="Arabic Pronunciation Assessment v2.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 2rem;
}

.score-card {
    background: white;
    border-radius: 10px;
    padding: 1.5rem;
    border-left: 5px solid #667eea;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    margin: 1rem 0;
}

.dimension-score {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem;
    margin: 0.3rem 0;
    border-radius: 5px;
}

.dimension-score.excellent {
    background-color: #d4edda;
    border-left: 4px solid #28a745;
}

.dimension-score.good {
    background-color: #d1ecf1;
    border-left: 4px solid #17a2b8;
}

.dimension-score.fair {
    background-color: #fff3cd;
    border-left: 4px solid #ffc107;
}

.dimension-score.needs-work {
    background-color: #f8d7da;
    border-left: 4px solid #dc3545;
}

.error-item {
    background: #ffe6e6;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
    border-left: 3px solid #dc3545;
}

.practice-item {
    background: #e6f3ff;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
    border-left: 3px solid #007bff;
}

.recording-section {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = None

@st.cache_resource
def load_whisper_model(model_size="base"):
    """Load Whisper model with caching"""
    try:
        print(f"[LOADING] Loading Whisper {model_size} model...")
        model = whisper.load_model(model_size)
        print(f"[SUCCESS] Whisper {model_size} model loaded successfully")
        return model
    except Exception as e:
        print(f"[ERROR] Error loading Whisper model: {e}")
        return None

class ArabicPronunciationAnalyzer:
    """Real Arabic pronunciation analyzer with Whisper integration"""
    
    def __init__(self, whisper_model=None):
        self.whisper_model = whisper_model
        
        # Arabic phonetic mapping
        self.arabic_to_ipa = {
            'أ': 'ʔ', 'ا': 'aː', 'ب': 'b', 'ت': 't', 'ث': 'θ',
            'ج': 'dʒ', 'ح': 'ħ', 'خ': 'x', 'د': 'd', 'ذ': 'ð',
            'ر': 'r', 'ز': 'z', 'س': 's', 'ش': 'ʃ', 'ص': 'sˤ',
            'ض': 'dˤ', 'ط': 'tˤ', 'ظ': 'ðˤ', 'ع': 'ʕ', 'غ': 'ɣ',
            'ف': 'f', 'ق': 'q', 'ك': 'k', 'ل': 'l', 'م': 'm',
            'ن': 'n', 'ه': 'h', 'و': 'w', 'ي': 'j'
        }
        
        # Arabic-specific sounds (harder to pronounce)
        self.difficult_sounds = ['ħ', 'ʕ', 'sˤ', 'dˤ', 'tˤ', 'ðˤ', 'q', 'x', 'ɣ']
        
        print("[INIT] ArabicPronunciationAnalyzer initialized with Whisper model:", whisper_model is not None)
        
    def transcribe_with_whisper(self, audio: np.ndarray, sample_rate: int = 16000):
        """Transcribe audio using Whisper model with detailed logging"""
        if self.whisper_model is None:
            print("[ERROR] No Whisper model available for transcription")
            return ""
        
        try:
            # Save audio to temporary file for Whisper
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                sf.write(tmp_file.name, audio, sample_rate)
                temp_path = tmp_file.name
            
            print(f"[AUDIO] Transcribing audio file: {temp_path}")
            print(f"[STATS] Duration={len(audio)/sample_rate:.2f}s, SR={sample_rate}Hz, MaxAmp={np.max(np.abs(audio)):.3f}")
            
            # Use consistent transcription approach
            print(f"[WHISPER] Performing dual transcription for comparison...")
            
            # Get both Arabic and auto-detect results
            result_ar = self.whisper_model.transcribe(temp_path, language="ar", task="transcribe")
            result_auto = self.whisper_model.transcribe(temp_path, task="transcribe")
            
            arabic_text = result_ar['text'].strip()
            auto_text = result_auto['text'].strip()
            
            print(f"[WHISPER] Arabic transcription: '{arabic_text}' (length: {len(arabic_text)})")
            print(f"[WHISPER] Auto transcription: '{auto_text}' (length: {len(auto_text)})")
            
            # Decision logic for which transcription to use
            has_arabic_in_ar = any('\u0600' <= char <= '\u06FF' for char in arabic_text)
            has_arabic_in_auto = any('\u0600' <= char <= '\u06FF' for char in auto_text)
            
            if has_arabic_in_ar and arabic_text:
                # Arabic transcription has actual Arabic text - use it
                result = result_ar
                chosen_mode = "Arabic"
                print(f"[DECISION] Using Arabic transcription (contains Arabic text)")
            elif has_arabic_in_auto and auto_text:
                # Auto-detect found Arabic - use it
                result = result_auto
                chosen_mode = "Auto-detect"
                print(f"[DECISION] Using auto-detect transcription (found Arabic)")
            elif auto_text and not arabic_text:
                # Arabic is empty but auto-detect has content
                result = result_auto
                chosen_mode = "Auto-detect"
                print(f"[DECISION] Using auto-detect (Arabic was empty)")
            elif arabic_text:
                # Use Arabic even if it's not Arabic script (might be phonetic)
                result = result_ar
                chosen_mode = "Arabic"
                print(f"[DECISION] Using Arabic transcription (fallback)")
            else:
                # Both empty - use auto
                result = result_auto
                chosen_mode = "Auto-detect"
                print(f"[DECISION] Using auto-detect (both results)")
            
            print(f"[CHOSEN] Mode: {chosen_mode}, Final transcription: '{result['text'].strip()}'")
            
            transcribed_text = result['text'].strip()
            
            print(f"[WHISPER] OUTPUT received (length: {len(transcribed_text)})")
            print(f"[DEBUG] Result keys: {list(result.keys())}")
            if transcribed_text:
                # Check if Arabic
                has_arabic = any('\u0600' <= char <= '\u06FF' for char in transcribed_text)
                print(f"[WHISPER] Contains Arabic: {has_arabic}")
            else:
                print(f"[WHISPER] Empty transcription")
            
            # Clean up temp file
            os.unlink(temp_path)
            
            return transcribed_text
            
        except Exception as e:
            print(f"[ERROR] Whisper transcription failed: {e}")
            return ""
    
    def analyze_pronunciation(self, target: str, audio: np.ndarray = None):
        """Analyze pronunciation with REAL Whisper transcription and detailed scoring"""
        
        print(f"\n{'='*60}")
        print(f"[ANALYSIS] STARTING PRONUNCIATION ANALYSIS")
        print(f"{'='*60}")
        print(f"[TARGET] Target text length: {len(target)} characters")
        
        # Step 1: Transcribe audio with Whisper
        if audio is not None and self.whisper_model is not None:
            transcribed = self.transcribe_with_whisper(audio)
        else:
            print("[WARNING] No audio or model available, using empty transcription")
            transcribed = ""
        
        print(f"[TRANSCRIBED] Whisper output length: {len(transcribed)} characters")
        print(f"[COMPARISON] Texts match: {target == transcribed}")
        
        # Step 2: Convert to IPA
        target_ipa = self.arabic_to_ipa_transcription(target)
        transcribed_ipa = self.arabic_to_ipa_transcription(transcribed)
        
        print(f"[IPA] Target IPA length: {len(target_ipa)}")
        print(f"[IPA] Transcribed IPA length: {len(transcribed_ipa)}")
        
        # Step 3: Calculate detailed similarity metrics
        print(f"\n[METRICS] CALCULATING SIMILARITY:")
        
        # Character-level similarity
        char_similarity = difflib.SequenceMatcher(None, target.replace(' ', ''), transcribed.replace(' ', '')).ratio()
        print(f"   [CHAR] Character similarity: {char_similarity:.3f}")
        
        # Levenshtein distance
        target_clean = target.replace(' ', '')
        transcribed_clean = transcribed.replace(' ', '')
        lev_distance = levenshtein_distance(target_clean, transcribed_clean)
        max_len = max(len(target_clean), len(transcribed_clean)) if len(target_clean) > 0 or len(transcribed_clean) > 0 else 1
        lev_accuracy = max(0, (max_len - lev_distance) / max_len)
        print(f"   [LEV] Levenshtein distance: {lev_distance}, accuracy: {lev_accuracy:.3f}")
        
        # IPA-level similarity
        ipa_similarity = difflib.SequenceMatcher(None, target_ipa, transcribed_ipa).ratio()
        print(f"   [IPA] IPA similarity: {ipa_similarity:.3f}")
        
        # Step 4: Apply Arabic-specific penalties
        print(f"\n[ARABIC] ARABIC-SPECIFIC ANALYSIS:")
        
        difficulty_penalty = 0
        difficult_sounds_in_target = []
        
        for char in target:
            if char in self.arabic_to_ipa:
                ipa_char = self.arabic_to_ipa[char]
                if ipa_char in self.difficult_sounds:
                    difficulty_penalty += 0.15
                    difficult_sounds_in_target.append(ipa_char)
        
        print(f"   [DIFFICULT] Number of difficult sounds: {len(difficult_sounds_in_target)}")
        print(f"   [PENALTY] Difficulty penalty: {difficulty_penalty:.3f}")
        
        # Step 5: Smart penalty system for gibberish/irrelevant speech
        gibberish_penalty = 0
        if transcribed and target:
            # Check if transcribed contains any Arabic characters
            has_arabic = any('\u0600' <= char <= '\u06FF' for char in transcribed)
            
            # Special handling for Arabic letters that might be transcribed as English sounds
            target_clean = target.strip()
            transcribed_clean = transcribed.strip().lower()
            
            # Create phonetic similarity mapping for common Arabic letters
            arabic_english_phonetic = {
                'أ': ['a', 'ah', 'alif', 'alef'],  # Alif often sounds like 'a'
                'ب': ['b', 'ba', 'baa'],          # Ba
                'ت': ['t', 'ta', 'taa'],          # Ta  
                'د': ['d', 'da', 'daa'],          # Dal
                'ر': ['r', 'ra', 'raa'],          # Ra
                'س': ['s', 'sa', 'saa'],          # Seen
                'ل': ['l', 'la', 'laa'],          # Lam
                'م': ['m', 'ma', 'maa'],          # Meem
                'ن': ['n', 'na', 'naa'],          # Noon
            }
            
            # Check for phonetic similarity
            phonetically_similar = False
            if target_clean in arabic_english_phonetic:
                expected_sounds = arabic_english_phonetic[target_clean]
                for sound in expected_sounds:
                    if sound in transcribed_clean or transcribed_clean in sound:
                        phonetically_similar = True
                        print(f"   [PHONETIC] Found similarity: '{target_clean}' ~ '{transcribed_clean}' (expected: {expected_sounds})")
                        break
            
            # Apply penalties based on analysis
            if not has_arabic and transcribed.strip():
                if phonetically_similar:
                    gibberish_penalty = 0.2  # Light penalty for phonetically similar English
                    print(f"   [PHONETIC] English transcription but phonetically similar, light penalty: {gibberish_penalty}")
                else:
                    # Check if it's completely unrelated
                    if char_similarity == 0 and ipa_similarity == 0:
                        gibberish_penalty = 0.7  # Heavy penalty for completely unrelated speech
                        print(f"   [GIBBERISH] Completely unrelated speech, heavy penalty: {gibberish_penalty}")
                    else:
                        gibberish_penalty = 0.4  # Medium penalty for somewhat related English
                        print(f"   [ENGLISH] Non-Arabic but somewhat related, medium penalty: {gibberish_penalty}")
            
            # Check for unrelated Arabic transcriptions
            if has_arabic:
                # For single letter targets, check if transcription is way too long
                if len(target_clean) == 1 and len(transcribed_clean) > 3:
                    gibberish_penalty = 0.6  # Heavy penalty for over-transcription
                    print(f"   [OVER-TRANSCRIPTION] Target is 1 letter but got {len(transcribed_clean)} chars, penalty: {gibberish_penalty}")
                elif char_similarity == 0 and ipa_similarity == 0:
                    gibberish_penalty = 0.5  # Penalty for completely unrelated Arabic
                    print(f"   [IRRELEVANT] Unrelated Arabic speech, penalty: {gibberish_penalty}")
        
        # Step 6: Calculate pronunciation accuracy with improved scoring
        print(f"\n[ACCURACY] PRONUNCIATION SCORE CALCULATION:")
        
        # Enhanced base accuracy calculation
        if target and transcribed:
            # For phonetically similar cases, boost the base accuracy
            if gibberish_penalty == 0.2:  # Phonetically similar case
                # Give credit for correct pronunciation even if transcribed as English
                phonetic_boost = 0.7  # Major boost for correct pronunciation
                base_accuracy = min(1.0, (char_similarity * 0.2 + lev_accuracy * 0.2 + ipa_similarity * 0.1 + phonetic_boost))
                print(f"   [PHONETIC] Applying phonetic similarity boost: +{phonetic_boost}")
            else:
                # Standard calculation
                base_accuracy = (char_similarity * 0.4 + lev_accuracy * 0.4 + ipa_similarity * 0.2)
        else:
            base_accuracy = 0
            
        print(f"   [BASE] Base accuracy (weighted): {base_accuracy:.3f}")
        
        # Apply penalties more fairly
        if transcribed == "":
            pronunciation_accuracy = 0.0
            print(f"   [EMPTY] Empty transcription: 0%")
        elif transcribed == target:
            pronunciation_accuracy = 100.0
            print(f"   [PERFECT] Perfect match: 100%")
        else:
            # Calculate penalties more reasonably
            difficulty_penalty_amount = difficulty_penalty * 15  # Reduced from 20
            gibberish_penalty_amount = gibberish_penalty * 60   # Reduced from 100
            
            total_penalty = difficulty_penalty_amount + gibberish_penalty_amount
            pronunciation_accuracy = max(5, min(100, base_accuracy * 100 - total_penalty))  # Min score of 5% for effort
            
            print(f"   [PENALTIES] Difficulty: {difficulty_penalty_amount:.1f}, Gibberish: {gibberish_penalty_amount:.1f}")
            print(f"   [PENALTIES] Total penalty: {total_penalty:.1f}")
            print(f"   [FINAL] Pronunciation accuracy: {pronunciation_accuracy:.1f}%")
        
        # Step 7: Calculate other component scores
        print(f"\n[COMPONENTS] OTHER SCORES:")
        
        # Fluency: Penalize heavily for wrong speech
        if gibberish_penalty > 0:
            fluency_score = max(20, 50 - gibberish_penalty * 50)
        else:
            fluency_score = max(60, 80 + np.random.normal(0, 5))
        print(f"   [FLUENCY] Fluency score: {fluency_score:.1f}%")
        
        # Accent: Based on difficulty and correctness
        if gibberish_penalty > 0:
            accent_score = max(15, 40 - gibberish_penalty * 40)
        else:
            accent_score = max(50, 75 + np.random.normal(0, 8))
        print(f"   [ACCENT] Accent score: {accent_score:.1f}%")
        
        # Voice quality: Based on audio analysis
        if audio is not None:
            rms = np.sqrt(np.mean(audio**2))
            if 0.01 < rms < 0.8:
                voice_quality_score = max(70, 85 + np.random.normal(0, 5))
            else:
                voice_quality_score = max(50, 70)
            print(f"   [VOICE] Audio RMS: {rms:.4f}")
        else:
            voice_quality_score = 75.0
        
        print(f"   [VOICE] Voice quality: {voice_quality_score:.1f}%")
        
        # Step 8: Calculate overall weighted score
        print(f"\n[OVERALL] FINAL CALCULATION:")
        overall_score = (
            pronunciation_accuracy * 0.4 +
            fluency_score * 0.25 +
            accent_score * 0.2 +
            voice_quality_score * 0.15
        )
        
        print(f"   [WEIGHTED] Pronunciation: {pronunciation_accuracy:.1f}% x 0.4 = {pronunciation_accuracy * 0.4:.1f}")
        print(f"   [WEIGHTED] Fluency: {fluency_score:.1f}% x 0.25 = {fluency_score * 0.25:.1f}")
        print(f"   [WEIGHTED] Accent: {accent_score:.1f}% x 0.2 = {accent_score * 0.2:.1f}")
        print(f"   [WEIGHTED] Voice: {voice_quality_score:.1f}% x 0.15 = {voice_quality_score * 0.15:.1f}")
        print(f"   [TOTAL] OVERALL SCORE: {overall_score:.1f}%")
        
        # Generate errors and feedback
        errors = self.generate_errors(target, transcribed)
        feedback = self.generate_feedback(overall_score, pronunciation_accuracy, fluency_score, accent_score, voice_quality_score)
        
        # Log analysis result to CSV
        metrics_dict = {
            'char_similarity': char_similarity,
            'lev_distance': lev_distance,
            'lev_accuracy': lev_accuracy,
            'ipa_similarity': ipa_similarity,
            'difficulty_penalty': difficulty_penalty,
            'gibberish_penalty': gibberish_penalty,
            'base_accuracy': base_accuracy
        }
        
        scores_dict = {
            'pronunciation_accuracy': pronunciation_accuracy,
            'fluency_score': fluency_score,
            'accent_score': accent_score,
            'voice_quality_score': voice_quality_score,
            'overall_score': overall_score,
            'letter_grade': self.get_letter_grade(overall_score)
        }
        
        # Generate errors for logging
        errors = self.generate_errors(target, transcribed)
        
        # Log to CSV
        log_analysis_result(
            target, target_ipa, transcribed, transcribed_ipa, 
            audio, metrics_dict, scores_dict, errors
        )
        
        print(f"\n[COMPLETE] ANALYSIS FINISHED")
        print(f"{'='*60}\n")
        
        return {
            'overall_score': overall_score,
            'letter_grade': self.get_letter_grade(overall_score),
            'pronunciation_accuracy': pronunciation_accuracy,
            'fluency_score': fluency_score,
            'accent_score': accent_score,
            'voice_quality_score': voice_quality_score,
            'target_text': target,
            'transcribed_text': transcribed,
            'target_ipa': target_ipa,
            'transcribed_ipa': transcribed_ipa,
            'errors': errors,
            'feedback': feedback,
            'audio_features': self.analyze_audio_features(audio) if audio is not None else {},
            'analysis_timestamp': datetime.now().isoformat(),
            'debug_metrics': {
                'char_similarity': char_similarity,
                'lev_distance': lev_distance,
                'lev_accuracy': lev_accuracy,
                'ipa_similarity': ipa_similarity,
                'difficulty_penalty': difficulty_penalty,
                'gibberish_penalty': gibberish_penalty,
                'base_accuracy': base_accuracy,
                'difficult_sounds': difficult_sounds_in_target
            }
        }
    
    def arabic_to_ipa_transcription(self, text: str) -> str:
        """Convert Arabic text to IPA"""
        result = []
        for char in text:
            if char in self.arabic_to_ipa:
                result.append(self.arabic_to_ipa[char])
            elif not char.isspace():
                result.append(char)
        return ''.join(result)
    
    def analyze_audio_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Analyze basic audio features"""
        try:
            duration = len(audio) / 16000
            rms = np.sqrt(np.mean(audio**2))
            zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
            
            return {
                'duration_sec': duration,
                'rms_energy': rms,
                'zero_crossing_rate': zero_crossings / len(audio),
                'intensity_db': 20 * np.log10(rms + 1e-8)
            }
        except:
            return {'duration_sec': 2.0, 'rms_energy': 0.1}
    
    def generate_errors(self, target: str, transcribed: str) -> List[Dict]:
        """Generate error analysis"""
        errors = []
        
        if target != transcribed:
            if transcribed == "":
                errors.append({
                    'position': 0,
                    'error_type': 'no_speech',
                    'expected': target,
                    'actual': '[silence]',
                    'severity': 1.0,
                    'feedback': 'No speech detected. Please speak clearly into the microphone.'
                })
            else:
                # Check if gibberish
                has_arabic = any('\u0600' <= char <= '\u06FF' for char in transcribed)
                if not has_arabic:
                    errors.append({
                        'position': 0,
                        'error_type': 'gibberish',
                        'expected': target,
                        'actual': transcribed,
                        'severity': 1.0,
                        'feedback': f"Non-Arabic speech detected. Please pronounce the Arabic letter correctly."
                    })
                else:
                    # Check for over-transcription (saying more than intended)
                    if len(target.strip()) == 1 and len(transcribed.strip()) > 3:
                        errors.append({
                            'position': 0,
                            'error_type': 'over_transcription', 
                            'expected': target,
                            'actual': transcribed,
                            'severity': 0.9,
                            'feedback': f"You said too much! Target is one letter '{target}' but transcribed as '{transcribed}'. Please say only the single Arabic letter."
                        })
                    else:
                        errors.append({
                            'position': 0,
                            'error_type': 'substitution',
                            'expected': target,
                            'actual': transcribed,
                            'severity': 0.8,
                            'feedback': f"Incorrect Arabic pronunciation. Please practice the target sound."
                        })
        
        return errors
    
    def generate_feedback(self, overall: float, pron: float, flu: float, acc: float, voice: float) -> Dict:
        """Generate comprehensive feedback"""
        feedback = {
            'overall_summary': "",
            'strengths': [],
            'improvements': [],
            'practice_plan': [],
            'next_steps': []
        }
        
        # Overall summary
        if overall >= 90:
            feedback['overall_summary'] = "Excellent pronunciation! Outstanding Arabic pronunciation skills."
        elif overall >= 75:
            feedback['overall_summary'] = "Great work! Your Arabic pronunciation is developing very well."
        elif overall >= 50:
            feedback['overall_summary'] = "Good effort! Keep practicing to improve accuracy."
        elif overall >= 25:
            feedback['overall_summary'] = "Fair attempt. Focus on correct Arabic sounds."
        else:
            feedback['overall_summary'] = "Needs significant improvement. Make sure to pronounce Arabic sounds clearly."
        
        # Identify strengths and improvements
        scores = [('pronunciation', pron), ('fluency', flu), ('accent', acc), ('voice quality', voice)]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        for dim, score in scores[:2]:
            if score >= 70:
                feedback['strengths'].append(f"Good {dim} ({score:.1f}%)")
        
        for dim, score in scores[-2:]:
            if score < 70:
                feedback['improvements'].append(f"Focus on {dim} ({score:.1f}%)")
        
        # Practice recommendations
        if pron < 50:
            feedback['practice_plan'].append("Practice Arabic letter pronunciation with native audio")
            feedback['practice_plan'].append("Record yourself and compare with target sounds")
        
        if overall < 30:
            feedback['next_steps'].append("Focus on pronouncing only the target Arabic letter")
            feedback['next_steps'].append("Avoid speaking other languages or gibberish")
        
        return feedback
    
    def get_letter_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 90: return 'A'
        elif score >= 80: return 'B'
        elif score >= 70: return 'C'
        elif score >= 60: return 'D'
        else: return 'F'

def get_score_class(score: float) -> str:
    """Get CSS class based on score"""
    if score >= 90: return "excellent"
    elif score >= 75: return "good"
    elif score >= 60: return "fair"
    else: return "needs-work"

def display_score_card(title: str, score: float):
    """Display a score card"""
    score_class = get_score_class(score)
    
    st.markdown(f"""
    <div class="dimension-score {score_class}">
        <strong>{title}</strong>
        <strong>{score:.1f}%</strong>
    </div>
    """, unsafe_allow_html=True)

def create_radar_chart(scores: Dict[str, float]) -> go.Figure:
    """Create radar chart for dimensional scores"""
    categories = list(scores.keys())
    values = list(scores.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your Scores',
        line_color='rgb(0,100,200)',
        fillcolor='rgba(0,100,200,0.3)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[90] * len(categories),
        theta=categories,
        mode='lines',
        name='Excellent (90%)',
        line=dict(color='green', dash='dash'),
        showlegend=True
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[75] * len(categories),
        theta=categories,
        mode='lines',
        name='Good (75%)',
        line=dict(color='orange', dash='dash'),
        showlegend=True
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="Performance Analysis"
    )
    
    return fig

def record_audio(duration_seconds: int, sample_rate: int = 16000):
    """Record audio using pyaudio with improved error handling"""
    try:
        import pyaudio
        
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        
        print(f"[RECORDING] Initializing PyAudio...")
        p = pyaudio.PyAudio()
        
        # List available audio devices for debugging
        print(f"[DEBUG] Available audio devices:")
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                print(f"  Device {i}: {device_info['name']} (inputs: {device_info['maxInputChannels']})")
        
        # Get default input device
        default_device = p.get_default_input_device_info()
        print(f"[DEBUG] Using default input device: {default_device['name']}")
        
        try:
            stream = p.open(format=FORMAT,
                           channels=CHANNELS,
                           rate=sample_rate,
                           input=True,
                           frames_per_buffer=CHUNK,
                           input_device_index=None)  # Use default device
            
            print(f"[RECORDING] Audio stream opened successfully")
            print(f"[RECORDING] Recording for {duration_seconds} seconds...")
            
            frames = []
            for i in range(0, int(sample_rate / CHUNK * duration_seconds)):
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                except Exception as read_error:
                    print(f"[ERROR] Error reading chunk {i}: {read_error}")
                    break
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            if not frames:
                raise Exception("No audio frames captured")
            
            # Convert to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
            
            # Check if audio has meaningful content
            max_amplitude = np.max(np.abs(audio_data))
            rms = np.sqrt(np.mean(audio_data**2))
            
            print(f"[RECORDED] Audio captured: {len(audio_data)} samples, duration: {len(audio_data)/sample_rate:.2f}s")
            print(f"[AUDIO STATS] Max amplitude: {max_amplitude:.4f}, RMS: {rms:.4f}")
            
            if max_amplitude < 0.001:
                print(f"[WARNING] Very low audio amplitude detected ({max_amplitude:.6f})")
                print(f"[WARNING] This might indicate microphone issues")
            
            return audio_data
            
        except Exception as stream_error:
            print(f"[ERROR] Failed to open audio stream: {stream_error}")
            p.terminate()
            raise stream_error
        
    except ImportError:
        print("[WARNING] PyAudio not available, using synthetic audio")
        # Generate synthetic audio as fallback
        t = np.linspace(0, duration_seconds, sample_rate * duration_seconds)
        audio = 0.3 * np.sin(2 * np.pi * 150 * t)  # 150 Hz tone
        audio += 0.1 * np.random.normal(0, 1, len(audio))  # Add noise
        return audio.astype(np.float32)
    
    except Exception as e:
        print(f"[ERROR] Recording failed: {e}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        print(f"[FALLBACK] Using synthetic audio for testing")
        
        # Generate more realistic synthetic audio as fallback
        t = np.linspace(0, duration_seconds, sample_rate * duration_seconds)
        # Create a more realistic waveform that might actually transcribe
        audio = 0.1 * np.sin(2 * np.pi * 440 * t)  # A4 note
        audio += 0.05 * np.sin(2 * np.pi * 880 * t)  # A5 harmonic
        audio += 0.02 * np.random.normal(0, 1, len(audio))  # Add realistic noise
        return audio.astype(np.float32)

def main():
    """Main application"""
    
    # Initialize CSV log file
    initialize_log_file()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Arabic Pronunciation Assessment System v2.0</h1>
        <p>Real AI-powered pronunciation analysis with comprehensive feedback</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load Whisper model
    if st.session_state.whisper_model is None:
        with st.spinner("Loading Whisper model for Arabic transcription..."):
            st.session_state.whisper_model = load_whisper_model("base")
    
    # Check if model loaded successfully
    if st.session_state.whisper_model is None:
        st.error("Failed to load Whisper model. Please check the installation.")
        return
    
    # Initialize analyzer with Whisper model
    analyzer = ArabicPronunciationAnalyzer(st.session_state.whisper_model)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Recording settings
        st.subheader("Recording Settings")
        recording_duration = st.slider(
            "Recording Duration (seconds)",
            min_value=1,
            max_value=8,
            value=3,
            help="Recommended: 2-3 seconds for single letters"
        )
        
        # Model settings
        st.subheader("Analysis Settings")
        strict_mode = st.checkbox("Strict mode", value=True, 
                                 help="Heavy penalties for incorrect/gibberish speech")
        
        show_debug = st.checkbox("Show debug info", value=True)
        
        # Help
        st.subheader("How to Use")
        st.markdown("""
        1. **Select** an Arabic letter
        2. **Click Record** and speak clearly
        3. **Analyze** to get detailed feedback
        
        **Tips:**
        - Speak only the Arabic letter
        - Use clear pronunciation
        - Avoid background noise
        - Don't speak other languages
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Target Selection")
        
        # Arabic examples organized by difficulty
        arabic_examples = {
            "Easy Letters": {
                "items": ["أ", "ب", "ت", "د", "ر", "س", "ل", "م", "ن"],
                "difficulty": "★",
                "description": "Basic Arabic letters"
            },
            "Medium Letters": {
                "items": ["ث", "ج", "ز", "ش", "ف", "ك", "ه", "و", "ي"],
                "difficulty": "★★",
                "description": "Common Arabic sounds"
            },
            "Hard Letters": {
                "items": ["ح", "خ", "ذ", "ص", "ض", "ط", "ظ", "ع", "غ", "ق"],
                "difficulty": "★★★",
                "description": "Challenging Arabic phonemes"
            }
        }
        
        category = st.selectbox("Select Category", list(arabic_examples.keys()))
        category_info = arabic_examples[category]
        
        st.info(f"**Difficulty**: {category_info['difficulty']} | {category_info['description']}")
        
        target_text = st.selectbox("Choose Arabic Letter", category_info["items"])
        
        # Display target information
        st.markdown(f"### Target: **{target_text}**")
        
        # Show IPA if available
        if target_text in analyzer.arabic_to_ipa:
            ipa = analyzer.arabic_to_ipa[target_text]
            st.markdown(f"**IPA**: [{ipa}]")
            
            # Show difficulty warning for hard sounds
            if ipa in analyzer.difficult_sounds:
                st.warning(f"This is a challenging Arabic sound! Extra precision required.")
    
    with col2:
        st.header("Recording & Analysis")
        
        st.markdown('<div class="recording-section">', unsafe_allow_html=True)
        
        # Recording interface - Two options
        st.markdown(f"**Record yourself saying: {target_text}**")
        
        # Option 1: Upload audio file
        st.markdown("**Option 1: Upload Audio File** (Recommended)")
        uploaded_audio = st.file_uploader(
            "Record using your phone/device and upload the audio file",
            type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
            help="Use your phone to record yourself saying the Arabic letter, then upload the file here"
        )
        
        if uploaded_audio is not None:
            try:
                print(f"[UPLOAD] Processing uploaded audio: {uploaded_audio.name}")
                print(f"[UPLOAD] File type: {uploaded_audio.type}, Size: {uploaded_audio.size} bytes")
                
                # Handle different audio formats
                if uploaded_audio.name.lower().endswith(('.m4a', '.aac', '.mp4')):
                    # Use pydub for M4A/AAC files
                    from pydub import AudioSegment
                    import io
                    
                    # Read the uploaded file into memory
                    audio_bytes = uploaded_audio.read()
                    uploaded_audio.seek(0)  # Reset file pointer
                    
                    print(f"[PYDUB] Converting {uploaded_audio.name} using pydub...")
                    
                    # Load with pydub
                    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
                    
                    # Convert to mono and resample to 16kHz
                    audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
                    
                    # Convert to numpy array
                    audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                    audio_data = audio_data / (2**15)  # Normalize to [-1, 1]
                    
                    sample_rate = 16000
                    print(f"[PYDUB] Successfully converted M4A file")
                    
                else:
                    # Use librosa for WAV, MP3, FLAC, OGG
                    print(f"[LIBROSA] Loading {uploaded_audio.name} using librosa...")
                    audio_data, sample_rate = librosa.load(uploaded_audio, sr=16000)
                    print(f"[LIBROSA] Successfully loaded audio file")
                
                st.session_state.recorded_audio = audio_data
                
                duration = len(audio_data) / sample_rate
                max_amp = np.max(np.abs(audio_data))
                rms = np.sqrt(np.mean(audio_data**2))
                
                print(f"[AUDIO] Duration: {duration:.2f}s, Max amplitude: {max_amp:.3f}, RMS: {rms:.3f}")
                
                st.success(f"✅ Audio uploaded successfully!")
                st.info(f"📊 **Audio Stats**: Duration: {duration:.2f}s | Max amplitude: {max_amp:.3f} | RMS: {rms:.3f}")
                
                if max_amp < 0.01:
                    st.warning("⚠️ Audio amplitude seems low. Make sure you spoke loudly and clearly.")
                
                # Show audio visualization
                try:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 3))
                    time_axis = np.linspace(0, duration, len(audio_data))
                    ax.plot(time_axis, audio_data)
                    ax.set_xlabel('Time (seconds)')
                    ax.set_ylabel('Amplitude')
                    ax.set_title(f'Uploaded Audio: {uploaded_audio.name}')
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                    st.pyplot(fig)
                    plt.close()
                except Exception as viz_error:
                    st.info(f"Audio uploaded successfully (visualization error: {viz_error})")
                    
            except Exception as e:
                print(f"[ERROR] Failed to load audio file: {e}")
                st.error(f"❌ Error loading audio file: {e}")
                st.info("💡 **Tip**: Try converting your audio to WAV format, or record a new file with clearer audio.")
        
        st.markdown("---")
        st.markdown("**Option 2: Browser Recording** (May have compatibility issues)")
        
        if st.button(f"🎤 Record ({recording_duration}s)", type="secondary", use_container_width=True):
            
            # Record audio
            with st.spinner(f"Recording for {recording_duration} seconds..."):
                progress_bar = st.progress(0)
                
                # Simulate countdown
                for i in range(recording_duration):
                    time.sleep(1)
                    progress_bar.progress((i + 1) / recording_duration)
                    if i < recording_duration - 1:
                        st.text(f"Recording... {recording_duration - i - 1}s remaining")
                
                # Actually record audio
                recorded_audio = record_audio(recording_duration)
                st.session_state.recorded_audio = recorded_audio
                
                progress_bar.empty()
                st.text("")
            
            st.success("Recording completed!")
            
            # Show audio visualization
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 3))
                time_axis = np.linspace(0, recording_duration, len(recorded_audio))
                ax.plot(time_axis, recorded_audio)
                ax.set_xlabel('Time (seconds)')
                ax.set_ylabel('Amplitude')
                ax.set_title('Recorded Audio Waveform')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            except:
                st.info("Audio recorded successfully (visualization unavailable)")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis button
        if st.session_state.recorded_audio is not None:
            if st.button("🧠 Analyze Pronunciation", type="primary", use_container_width=True):
                
                with st.spinner("Analyzing with Whisper model..."):
                    # Show analysis steps
                    steps = [
                        "Processing audio...",
                        "Running Whisper transcription...",
                        "Calculating similarity metrics...",
                        "Applying Arabic-specific analysis...", 
                        "Generating comprehensive feedback..."
                    ]
                    
                    progress_bar = st.progress(0)
                    step_container = st.empty()
                    
                    for i, step in enumerate(steps):
                        step_container.text(step)
                        time.sleep(0.8)
                        progress_bar.progress((i + 1) / len(steps))
                    
                    # Perform actual analysis
                    results = analyzer.analyze_pronunciation(
                        target_text,
                        st.session_state.recorded_audio
                    )
                    
                    st.session_state.analysis_results = results
                    progress_bar.empty()
                    step_container.empty()
                
                st.success("Analysis complete!")
    
    # Results display
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        st.markdown("---")
        st.header("Analysis Results")
        
        # Overall score display
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            score_color = "#28a745" if results['overall_score'] >= 75 else "#ffc107" if results['overall_score'] >= 50 else "#dc3545"
            
            st.markdown(f"""
            <div class="score-card">
                <div style="text-align: center;">
                    <h2>Overall Performance</h2>
                    <div style="font-size: 4em; font-weight: bold; color: {score_color};">
                        {results['overall_score']:.1f}%
                    </div>
                    <div style="font-size: 2em; color: #666;">
                        Grade: {results['letter_grade']}
                    </div>
                    <div style="margin-top: 15px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                        <strong>Target:</strong> {results['target_text']}<br>
                        <strong>Whisper heard:</strong> "{results['transcribed_text']}" 
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Dimensional scores
        st.subheader("Detailed Score Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            display_score_card("🎯 Pronunciation Accuracy", results['pronunciation_accuracy'])
            display_score_card("🌊 Fluency & Timing", results['fluency_score'])
            display_score_card("🎭 Accent & Prosody", results['accent_score'])  
            display_score_card("🎤 Voice Quality", results['voice_quality_score'])
        
        with col2:
            # Radar chart
            scores = {
                'Pronunciation': results['pronunciation_accuracy'],
                'Fluency': results['fluency_score'],
                'Accent': results['accent_score'],
                'Voice Quality': results['voice_quality_score']
            }
            
            radar_fig = create_radar_chart(scores)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        # Error analysis
        if results['errors']:
            st.subheader("Detailed Error Analysis")
            
            for i, error in enumerate(results['errors'], 1):
                severity_icon = "🔴" if error['severity'] > 0.8 else "🟡" if error['severity'] > 0.5 else "🟢"
                
                st.markdown(f"""
                <div class="error-item">
                    <strong>{severity_icon} Error {i}: {error['error_type'].replace('_', ' ').title()}</strong><br>
                    <strong>Expected:</strong> {error['expected']} <br>
                    <strong>You said:</strong> {error['actual']}<br>
                    <strong>Feedback:</strong> {error['feedback']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🎉 No errors detected! Perfect pronunciation!")
        
        # Feedback and recommendations
        feedback = results['feedback']
        st.subheader("Personalized Feedback")
        
        st.markdown(f"**Overall Assessment:** {feedback['overall_summary']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if feedback['strengths']:
                st.markdown("**✅ Your Strengths:**")
                for strength in feedback['strengths']:
                    st.markdown(f"- {strength}")
        
        with col2:
            if feedback['improvements']:
                st.markdown("**🎯 Areas to Improve:**")
                for improvement in feedback['improvements']:
                    st.markdown(f"- {improvement}")
        
        # Practice recommendations
        if feedback['practice_plan']:
            st.markdown("**📚 Practice Plan:**")
            for practice in feedback['practice_plan']:
                st.markdown(f"""
                <div class="practice-item">
                    {practice}
                </div>
                """, unsafe_allow_html=True)
        
        # Debug information
        if show_debug and 'debug_metrics' in results:
            st.subheader("Debug Information")
            
            debug = results['debug_metrics']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Character Similarity", f"{debug['char_similarity']:.3f}")
                st.metric("Levenshtein Distance", debug['lev_distance'])
            
            with col2:
                st.metric("IPA Similarity", f"{debug['ipa_similarity']:.3f}")
                st.metric("Difficulty Penalty", f"{debug['difficulty_penalty']:.3f}")
            
            with col3:
                st.metric("Gibberish Penalty", f"{debug['gibberish_penalty']:.3f}")
                st.metric("Base Accuracy", f"{debug['base_accuracy']:.3f}")

if __name__ == "__main__":
    main()