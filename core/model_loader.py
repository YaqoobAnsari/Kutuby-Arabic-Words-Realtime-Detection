"""
Model Loading Module for Arabic Word Recognition
Handles Wav2Vec2 model initialization and caching
"""

import streamlit as st
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from typing import Tuple, Optional


class ModelLoader:
    """Handles loading and caching of Wav2Vec2 Arabic model"""

    MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    @st.cache_resource
    def load_model(_self) -> Tuple[Optional[Wav2Vec2ForCTC], Optional[Wav2Vec2Tokenizer], bool]:
        """
        Load Wav2Vec2 Arabic model with caching

        Returns:
            Tuple of (model, tokenizer, success_flag)
        """
        try:
            tokenizer = Wav2Vec2Tokenizer.from_pretrained(_self.MODEL_NAME)
            model = Wav2Vec2ForCTC.from_pretrained(_self.MODEL_NAME)

            _self.model = model
            _self.tokenizer = tokenizer
            _self.is_loaded = True

            return model, tokenizer, True

        except Exception as e:
            st.error(f"❌ Model loading error: {str(e)}")
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
