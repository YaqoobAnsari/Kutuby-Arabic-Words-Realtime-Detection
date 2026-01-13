"""
Audio Transcription Module
Handles speech-to-text conversion using Wav2Vec2
"""

import torch
import librosa
from typing import Optional
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer


class AudioTranscriber:
    """Handles audio transcription using Wav2Vec2 model"""

    def __init__(self, model: Wav2Vec2ForCTC, tokenizer: Wav2Vec2Tokenizer):
        """
        Initialize transcriber with model and tokenizer

        Args:
            model: Wav2Vec2 model instance
            tokenizer: Wav2Vec2 tokenizer instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sample_rate = 16000

    def transcribe(self, audio_file: str) -> Optional[str]:
        """
        Transcribe audio file to Arabic text

        Args:
            audio_file: Path to audio file

        Returns:
            Transcribed text or None if failed
        """
        try:
            # Load and normalize audio
            speech, _ = librosa.load(audio_file, sr=self.sample_rate)

            if len(speech) == 0:
                print("⚠️ Empty audio file")
                return None

            # Normalize audio amplitude
            max_amplitude = max(abs(speech))
            if max_amplitude > 0:
                speech = speech / max_amplitude

            # Tokenize audio
            inputs = self.tokenizer(
                speech,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            )

            # Perform inference
            with torch.no_grad():
                logits = self.model(inputs.input_values).logits

            # Decode predictions
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.tokenizer.batch_decode(predicted_ids)[0]

            return transcription.strip()

        except Exception as e:
            print(f"❌ Transcription error: {str(e)}")
            return None

    def transcribe_with_confidence(self, audio_file: str) -> dict:
        """
        Transcribe audio file to Arabic text with confidence score

        Args:
            audio_file: Path to audio file

        Returns:
            Dict with transcription and confidence, or error info
        """
        try:
            # Load and normalize audio
            speech, _ = librosa.load(audio_file, sr=self.sample_rate)

            if len(speech) == 0:
                return {"transcription": None, "confidence": 0.0, "error": "Empty audio file"}

            # Normalize audio amplitude
            max_amplitude = max(abs(speech))
            if max_amplitude > 0:
                speech = speech / max_amplitude

            # Tokenize audio
            inputs = self.tokenizer(
                speech,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            )

            # Perform inference
            with torch.no_grad():
                logits = self.model(inputs.input_values).logits

            # Decode predictions
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.tokenizer.batch_decode(predicted_ids)[0].strip()

            # Calculate confidence
            confidence = self.get_confidence_score(logits)

            return {
                "transcription": transcription,
                "confidence": confidence / 100.0,  # Return as 0-1 scale
                "error": None
            }

        except Exception as e:
            print(f"❌ Transcription error: {str(e)}")
            return {"transcription": None, "confidence": 0.0, "error": str(e)}

    def get_confidence_score(self, logits: torch.Tensor) -> float:
        """
        Calculate confidence score from model logits

        Args:
            logits: Model output logits

        Returns:
            Confidence score (0-100)
        """
        try:
            probabilities = torch.softmax(logits, dim=-1)
            max_probs = torch.max(probabilities, dim=-1).values
            confidence = torch.mean(max_probs).item() * 100
            return round(confidence, 2)
        except:
            return 0.0
