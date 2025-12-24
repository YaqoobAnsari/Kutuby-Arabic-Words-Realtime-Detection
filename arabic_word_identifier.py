import streamlit as st
import pyaudio
import wave
import tempfile
import os
import time
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

st.set_page_config(
    page_title="Arabic Word Identifier",
    page_icon="🎤",
    layout="centered"
)

@st.cache_resource
def load_wav2vec2_model():
    """Load Wav2Vec2 Arabic model"""
    try:
        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
        tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        return model, tokenizer, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, False

class AudioRecorder:
    """Simple audio recorder class"""
    def __init__(self):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.frames = []

    def record_audio(self, duration):
        """Record audio for specified duration"""
        self.frames = []
        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )

            for _ in range(0, int(self.RATE / self.CHUNK * duration)):
                data = stream.read(self.CHUNK)
                self.frames.append(data)

            stream.stop_stream()
            stream.close()
            p.terminate()
            return True
        except Exception as e:
            st.error(f"Recording error: {str(e)}")
            return False

    def save_recording(self, filename):
        """Save recorded audio to file"""
        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            return True
        except Exception as e:
            st.error(f"Save error: {str(e)}")
            return False

def transcribe_audio(model, tokenizer, audio_file):
    """Transcribe audio using Wav2Vec2 model"""
    try:
        # Load audio file
        speech, sample_rate = librosa.load(audio_file, sr=16000)

        # Normalize audio
        if len(speech) > 0 and max(abs(speech)) > 0:
            speech = speech / max(abs(speech))

        # Tokenize and predict
        inputs = tokenizer(speech, sampling_rate=16000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = model(inputs.input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)[0]

        return transcription.strip()

    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return ""

# Initialize components
st.title("🎤 Arabic Word Identifier")
st.markdown("Speak an Arabic word and see what the AI recognizes!")

# Load model
with st.spinner("Loading Arabic speech recognition model..."):
    model, tokenizer, model_loaded = load_wav2vec2_model()

if not model_loaded:
    st.error("❌ Failed to load the model. Please check your internet connection and try again.")
    st.stop()

st.success("✅ Model loaded successfully!")

# Create recorder instance
recorder = AudioRecorder()

# UI Layout
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎛️ Recording Settings")
    duration = st.slider(
        "Recording Duration (seconds)",
        min_value=2,
        max_value=5,
        value=3,
        help="Choose how long to record your voice"
    )

    st.info(f"📝 Recording will last {duration} seconds")

with col2:
    st.subheader("🔴 Record Audio")

    if st.button("🎤 Start Recording", type="primary", use_container_width=True):
        # Recording countdown
        countdown_placeholder = st.empty()
        for i in range(3, 0, -1):
            countdown_placeholder.warning(f"🔴 Recording starts in {i}...")
            time.sleep(1)
        countdown_placeholder.success("🎤 Recording NOW!")

        # Record audio
        with st.spinner(f"Recording for {duration} seconds..."):
            recording_success = recorder.record_audio(duration)

        if recording_success:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                temp_filename = tmp_file.name

            save_success = recorder.save_recording(temp_filename)

            if save_success:
                st.success("✅ Recording completed!")

                # Play recorded audio
                st.audio(temp_filename, format="audio/wav")

                # Transcribe audio
                with st.spinner("🤖 Analyzing your speech..."):
                    transcription = transcribe_audio(model, tokenizer, temp_filename)

                # Display results
                st.markdown("---")
                st.subheader("📝 Recognition Results")

                if transcription:
                    # Main result
                    st.markdown("### 🎯 Recognized Text:")
                    st.markdown(f"**{transcription}**", unsafe_allow_html=True)

                    # Additional info
                    st.info("✨ This is what the Wav2Vec2 Arabic model detected from your speech.")

                else:
                    st.warning("⚠️ No speech detected or recognition failed. Please try again with clearer speech.")

                # Clean up temporary file
                try:
                    time.sleep(0.1)
                    os.unlink(temp_filename)
                except:
                    pass
            else:
                st.error("❌ Failed to save recording. Please try again.")
        else:
            st.error("❌ Recording failed. Please check your microphone and try again.")

# Instructions section
st.markdown("---")
st.subheader("📖 How to Use")

instructions_col1, instructions_col2 = st.columns(2)

with instructions_col1:
    st.markdown("""
    **Steps:**
    1. 🎛️ Set recording duration (2-5 seconds)
    2. 🎤 Click "Start Recording"
    3. 🗣️ Speak your Arabic word clearly
    4. 📝 View the recognized text
    """)

with instructions_col2:
    st.markdown("""
    **Tips for better results:**
    - Speak clearly and at normal pace
    - Use a quiet environment
    - Hold device close to mouth
    - Pronounce words distinctly
    """)

# Technical info
with st.expander("🔧 Technical Information"):
    st.markdown("""
    **Model Information:**
    - **Model**: Wav2Vec2-Large-XLSR-53-Arabic
    - **Parameters**: ~315M
    - **Specialization**: Arabic speech recognition
    - **Sample Rate**: 16kHz
    - **Input Format**: Mono audio, WAV
    """)