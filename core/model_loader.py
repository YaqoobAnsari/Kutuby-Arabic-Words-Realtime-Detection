"""
Model Loading Module for Arabic Word Recognition
Handles Wav2Vec2 model initialization and caching
"""

from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from typing import Tuple, Optional


class ModelLoader:
    """Handles loading and caching of Wav2Vec2 Arabic model"""

    MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def load_model(self) -> Tuple[Optional[Wav2Vec2ForCTC], Optional[Wav2Vec2Tokenizer], bool]:
        """
        Load Wav2Vec2 Arabic model

        Returns:
            Tuple of (model, tokenizer, success_flag)
        """
        try:
            print(f"Loading model: {self.MODEL_NAME}")
            tokenizer = Wav2Vec2Tokenizer.from_pretrained(self.MODEL_NAME)
            model = Wav2Vec2ForCTC.from_pretrained(self.MODEL_NAME)

            self.model = model
            self.tokenizer = tokenizer
            self.is_loaded = True

            print("✅ Model loaded successfully!")
            return model, tokenizer, True

        except Exception as e:
            print(f"❌ Model loading error: {str(e)}")
            return None, None, False

    def get_model_info(self) -> dict:
        """Return model information"""
        return {
            "name": self.MODEL_NAME,
            "parameters": "~315M",
            "specialization": "Arabic speech recognition",
            "sample_rate": 16000,
            "input_format": "Mono audio, WAV"
        }
