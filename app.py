#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI entry point for Hugging Face Spaces
Arabic Word Recognition API - FastAPI Backend
"""

# Import all necessary modules
from __future__ import annotations
import os, io, time, json
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import librosa
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# --------------------------- FastAPI App Setup ---------------------------

app = FastAPI(
    title="Arabic Word Recognition API",
    description="API for transcribing Arabic words from audio using Wav2Vec2-Large-XLSR-53-Arabic",
    version="2.0.0"
)

# Add CORS middleware for Hugging Face Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------- Model Loading ---------------------------

_WORD_MODEL: Optional[Wav2Vec2ForCTC] = None
_WORD_TOKENIZER: Optional[Wav2Vec2Tokenizer] = None

def _load_word_model_once():
    """Load the Arabic word transcription model (Wav2Vec2-Large-XLSR-53-Arabic)"""
    global _WORD_MODEL, _WORD_TOKENIZER
    if _WORD_MODEL is None:
        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
        print(f"Loading model: {model_name}")
        _WORD_TOKENIZER = Wav2Vec2Tokenizer.from_pretrained(model_name)
        _WORD_MODEL = Wav2Vec2ForCTC.from_pretrained(model_name)
        _WORD_MODEL.eval()  # Set to evaluation mode
        print("Model loaded successfully!")
    return _WORD_MODEL, _WORD_TOKENIZER

# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load model when the app starts"""
    _load_word_model_once()

# --------------------------- Health Check Endpoint ---------------------------

@app.get("/")
def root():
    """Root endpoint - health check"""
    return {
        "status": "running",
        "service": "Arabic Word Recognition API",
        "version": "2.0.0",
        "model": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
        "endpoints": {
            "verify": "/verify_word (POST) - Main endpoint",
            "transcribe": "/transcribe_word (POST)",
            "docs": "/docs",
            "health": "/health"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy"}

# --------------------------- Arabic Word Transcription Endpoint ---------------------------

@app.post("/verify_word", response_class=JSONResponse)
async def verify_word(
    audio: UploadFile = File(...),
    target_word: str = Form(...),
    threshold: float = Form(0.6)
):
    """
    Verify if audio matches target Arabic word and exceeds confidence threshold.
    
    Parameters:
    - audio: WAV audio file containing spoken Arabic word
    - target_word: The expected Arabic word (e.g., "اللَّهِ", "مِنَ")
    - threshold: Confidence threshold (0.0 to 1.0, default 0.6 = 60%)
    
    Returns:
    - result: Boolean (True if match AND confidence >= threshold, False otherwise)
    """
    model, tokenizer = _load_word_model_once()
    
    # Read audio file
    content = await audio.read()
    try:
        # Use librosa to load audio
        y, sr = librosa.load(io.BytesIO(content), sr=16000, mono=True)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "result": False,
                "error": f"Could not read audio file. Expected WAV format. Error: {type(e).__name__}: {e}"
            }
        )
    
    if len(y) == 0:
        return JSONResponse(
            status_code=400,
            content={
                "result": False,
                "error": "Empty audio file"
            }
        )
    
    # Validate threshold
    if not 0.0 <= threshold <= 1.0:
        return JSONResponse(
            status_code=400,
            content={
                "result": False,
                "error": "Threshold must be between 0.0 and 1.0"
            }
        )
    
    # Normalize audio amplitude
    max_amplitude = max(abs(y))
    if max_amplitude > 0:
        y = y / max_amplitude
    
    # Perform transcription
    try:
        # Tokenize audio
        inputs = tokenizer(
            y,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Perform inference
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        
        # Decode predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)[0].strip()
        
        # Calculate confidence score
        probabilities = torch.softmax(logits, dim=-1)
        max_probs = torch.max(probabilities, dim=-1).values
        confidence = torch.mean(max_probs).item()  # 0.0 to 1.0
        
        # Check if transcription matches target_word (exact match)
        matches = transcription == target_word.strip()
        
        # Check if confidence exceeds threshold
        exceeds_threshold = confidence >= threshold
        
        # Final result: both conditions must be true
        result = matches and exceeds_threshold
        
        return JSONResponse({
            "result": result
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "result": False,
                "error": f"Transcription failed: {type(e).__name__}: {e}"
            }
        )


@app.post("/transcribe_word", response_class=JSONResponse)
async def transcribe_word(audio: UploadFile = File(...)):
    """
    Endpoint for transcribing Arabic words from audio (speech-to-text).
    
    Parameters:
    - audio: WAV audio file containing spoken Arabic word(s)
    
    Returns:
    - transcription: The recognized Arabic text
    - confidence: Confidence score (if available)
    - latency_ms: Processing time in milliseconds
    """
    model, tokenizer = _load_word_model_once()
    
    # Read audio file
    content = await audio.read()
    try:
        # Use librosa to load audio (same as the Streamlit app)
        y, sr = librosa.load(io.BytesIO(content), sr=16000, mono=True)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Could not read audio file. Expected WAV format. Error: {type(e).__name__}: {e}",
                "transcription": None
            }
        )
    
    if len(y) == 0:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Empty audio file",
                "transcription": None
            }
        )
    
    # Normalize audio amplitude (same as Streamlit app)
    max_amplitude = max(abs(y))
    if max_amplitude > 0:
        y = y / max_amplitude
    
    # Perform transcription
    t0 = time.perf_counter()
    try:
        # Tokenize audio
        inputs = tokenizer(
            y,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Perform inference
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        
        # Decode predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)[0]
        
        # Calculate confidence score
        probabilities = torch.softmax(logits, dim=-1)
        max_probs = torch.max(probabilities, dim=-1).values
        confidence = torch.mean(max_probs).item() * 100
        
        t1 = time.perf_counter()
        
        return JSONResponse({
            "transcription": transcription.strip(),
            "confidence": round(confidence, 2),
            "latency_ms": round((t1 - t0) * 1000.0, 2),
            "model": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Transcription failed: {type(e).__name__}: {e}",
                "transcription": None
            }
        )

# For Hugging Face Spaces, the app is automatically served
# For local development, you can run: uvicorn app:app --host 0.0.0.0 --port 7860
