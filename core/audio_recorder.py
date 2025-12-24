"""
Audio Recording Module
Handles microphone input and audio file operations
"""

import pyaudio
import wave
from typing import Optional
import streamlit as st


class AudioRecorder:
    """Handles audio recording from microphone"""

    # Audio configuration constants
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    SAMPLE_RATE = 16000

    def __init__(self):
        self.frames = []
        self.pyaudio_instance = None

    def record_audio(self, duration: int) -> bool:
        """
        Record audio for specified duration

        Args:
            duration: Recording duration in seconds

        Returns:
            bool: True if recording successful, False otherwise
        """
        self.frames = []

        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            stream = self.pyaudio_instance.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )

            num_chunks = int(self.SAMPLE_RATE / self.CHUNK * duration)
            for _ in range(num_chunks):
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                self.frames.append(data)

            stream.stop_stream()
            stream.close()
            self.pyaudio_instance.terminate()

            return True

        except Exception as e:
            st.error(f"❌ Recording error: {str(e)}")
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
            return False

    def save_recording(self, filename: str) -> bool:
        """
        Save recorded audio to WAV file

        Args:
            filename: Output file path

        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.FORMAT))
                wf.setframerate(self.SAMPLE_RATE)
                wf.writeframes(b''.join(self.frames))

            return True

        except Exception as e:
            st.error(f"❌ Save error: {str(e)}")
            return False

    def get_audio_config(self) -> dict:
        """Return audio configuration"""
        return {
            "sample_rate": self.SAMPLE_RATE,
            "channels": self.CHANNELS,
            "format": "16-bit PCM",
            "chunk_size": self.CHUNK
        }
