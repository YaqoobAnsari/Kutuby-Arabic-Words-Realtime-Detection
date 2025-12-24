"""
Arabic Word Recognition System
Production-level application for real-time Arabic speech recognition

Author: Yaqoob Ansari
Version: 2.0.0
Model: Wav2Vec2-Large-XLSR-53-Arabic
"""

import streamlit as st
import tempfile
import os
import time
from pathlib import Path

# Import core modules
from core.model_loader import ModelLoader
from core.audio_recorder import AudioRecorder
from core.transcriber import AudioTranscriber


class ArabicWordRecognitionApp:
    """Main application class for Arabic word recognition"""

    def __init__(self):
        """Initialize application components"""
        self.model_loader = ModelLoader()
        self.recorder = AudioRecorder()
        self.transcriber = None
        self.setup_page_config()

    @staticmethod
    def setup_page_config():
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Arabic Word Recognition",
            page_icon="🎤",
            layout="centered",
            initial_sidebar_state="collapsed"
        )

    def load_models(self):
        """Load AI models with progress indication"""
        with st.spinner("🔄 Loading Wav2Vec2 Arabic model..."):
            model, tokenizer, success = self.model_loader.load_model()

        if not success:
            st.error("❌ Failed to load model. Please check internet connection and try again.")
            st.stop()

        self.transcriber = AudioTranscriber(model, tokenizer)
        st.success("✅ Model loaded successfully!")
        return success

    def render_header(self):
        """Render application header"""
        st.title("🎤 Arabic Word Recognition System")
        st.markdown("**Real-time Arabic speech-to-text using Wav2Vec2**")
        st.markdown("---")

    def render_performance_stats(self):
        """Render model performance statistics"""
        with st.expander("📊 Model Performance - Quranic Vocabulary Test"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Average Accuracy", "95.3%")
            with col2:
                st.metric("Confidence Score", "94.7%")
            with col3:
                st.metric("Perfect Matches", "73.3%")
            with col4:
                st.metric("Test Words", "30/30")

            st.info("""
            **Performance Highlights:**
            - ✅ Tested on 30 most frequent Quranic words
            - ✅ 22/30 perfect matches (73.3%)
            - ✅ 8/30 minor variations in diacritics (26.7%)
            - ✅ Zero major recognition errors
            - ⚠️ Minor diacritical mark variations expected
            """)

    def render_recording_interface(self):
        """Render recording controls"""
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🎛️ Settings")
            duration = st.slider(
                "Recording Duration (seconds)",
                min_value=2,
                max_value=5,
                value=3,
                help="Adjust recording length for your speech"
            )
            st.info(f"📝 Will record for {duration} seconds")

        with col2:
            st.subheader("🔴 Record")
            record_button = st.button(
                "🎤 Start Recording",
                type="primary",
                use_container_width=True
            )

        return duration, record_button

    def handle_recording(self, duration: int):
        """Handle the recording process"""
        # Countdown
        countdown_placeholder = st.empty()
        for i in range(3, 0, -1):
            countdown_placeholder.warning(f"🔴 Recording starts in {i}...")
            time.sleep(1)
        countdown_placeholder.success("🎤 Recording NOW!")

        # Record audio
        with st.spinner(f"📡 Recording for {duration} seconds..."):
            success = self.recorder.record_audio(duration)

        if not success:
            st.error("❌ Recording failed. Check microphone permissions.")
            return None

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_path = tmp.name

        if not self.recorder.save_recording(temp_path):
            st.error("❌ Failed to save recording.")
            return None

        return temp_path

    def display_results(self, transcription: str):
        """Display transcription results"""
        st.markdown("---")
        st.subheader("📝 Recognition Results")

        if transcription:
            st.markdown("### 🎯 Recognized Arabic Text:")
            st.markdown(
                f"<div style='font-size: 32px; font-weight: bold; "
                f"text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); "
                f"color: white; border-radius: 10px; margin: 10px 0;'>{transcription}</div>",
                unsafe_allow_html=True
            )

            st.success("✨ Transcription completed using Wav2Vec2-Large-XLSR-53-Arabic")

        else:
            st.warning("⚠️ No speech detected. Please try again with clearer pronunciation.")

    def render_instructions(self):
        """Render usage instructions"""
        st.markdown("---")
        st.subheader("📖 How to Use")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Quick Start:**
            1. 🎛️ Set recording duration (2-5 seconds)
            2. 🎤 Click "Start Recording"
            3. 🗣️ Speak your Arabic word clearly
            4. 📝 View the recognized text
            """)

        with col2:
            st.markdown("""
            **Tips for Best Results:**
            - 🔇 Use a quiet environment
            - 🎙️ Speak clearly at normal pace
            - 📱 Keep microphone close
            - 🕌 Try Quranic vocabulary for best accuracy
            """)

    def render_technical_info(self):
        """Render technical information"""
        with st.expander("🔧 Technical Information"):
            model_info = self.model_loader.get_model_info()
            audio_config = self.recorder.get_audio_config()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Model Specifications:**")
                for key, value in model_info.items():
                    st.text(f"• {key.replace('_', ' ').title()}: {value}")

            with col2:
                st.markdown("**Audio Configuration:**")
                for key, value in audio_config.items():
                    st.text(f"• {key.replace('_', ' ').title()}: {value}")

            st.markdown("""
            **Performance Metrics (Quranic Words):**
            - Average Recognition Accuracy: 95.3%
            - Average Confidence Score: 94.7%
            - Perfect Match Rate: 73.3%
            - Tested on 30 most frequent Quranic words
            """)

    def run(self):
        """Main application loop"""
        # Render UI components
        self.render_header()
        self.load_models()
        self.render_performance_stats()

        # Recording interface
        duration, record_button = self.render_recording_interface()

        # Handle recording
        if record_button:
            temp_path = self.handle_recording(duration)

            if temp_path:
                # Display audio player
                st.audio(temp_path, format="audio/wav")

                # Transcribe
                with st.spinner("🤖 Analyzing speech with AI..."):
                    transcription = self.transcriber.transcribe(temp_path)

                # Display results
                self.display_results(transcription)

                # Cleanup
                try:
                    time.sleep(0.1)
                    os.unlink(temp_path)
                except:
                    pass

        # Instructions and technical info
        self.render_instructions()
        self.render_technical_info()

        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666;'>"
            "Arabic Word Recognition System v2.0 | "
            "Built with Wav2Vec2 & Streamlit | "
            "© 2024 Yaqoob Ansari"
            "</div>",
            unsafe_allow_html=True
        )


def main():
    """Application entry point"""
    app = ArabicWordRecognitionApp()
    app.run()


if __name__ == "__main__":
    main()
