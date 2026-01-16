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
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from core.arabic_utils import normalize_arabic_text

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

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI"""
    return """
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🎤 Arabic Word Recognition</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 20px;
                padding: 40px;
                max-width: 600px;
                width: 100%;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            h1 {
                text-align: center;
                color: #667eea;
                margin-bottom: 30px;
                font-size: 2em;
            }
            .input-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                color: #333;
                font-weight: 600;
            }
            input[type="text"] {
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 16px;
                transition: border 0.3s;
                text-align: right;
                font-family: 'Arial', sans-serif;
            }
            input[type="text"]:focus {
                outline: none;
                border-color: #667eea;
            }
            .controls {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }
            button {
                flex: 1;
                padding: 15px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
            }
            #recordBtn {
                background: #4CAF50;
                color: white;
            }
            #recordBtn:hover { background: #45a049; }
            #recordBtn:disabled { background: #ccc; cursor: not-allowed; }

            #stopBtn {
                background: #f44336;
                color: white;
            }
            #stopBtn:hover { background: #da190b; }
            #stopBtn:disabled { background: #ccc; cursor: not-allowed; }

            .status {
                text-align: center;
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 20px;
                font-weight: 600;
            }
            .status.recording { background: #ffebee; color: #c62828; }
            .status.processing { background: #fff3e0; color: #e65100; }
            .status.ready { background: #e8f5e9; color: #2e7d32; }

            .results {
                background: #f5f5f5;
                border-radius: 8px;
                padding: 20px;
                margin-top: 20px;
                display: none;
            }
            .results.show { display: block; }

            .result-item {
                margin-bottom: 15px;
                padding-bottom: 15px;
                border-bottom: 1px solid #ddd;
            }
            .result-item:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }

            .result-label {
                font-size: 14px;
                color: #666;
                margin-bottom: 5px;
            }
            .result-value {
                font-size: 20px;
                font-weight: 700;
                color: #333;
            }
            .result-value.arabic {
                font-size: 28px;
                text-align: right;
            }

            .match-badge {
                display: inline-block;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 16px;
                font-weight: 600;
            }
            .match-badge.success { background: #4CAF50; color: white; }
            .match-badge.fail { background: #f44336; color: white; }

            .confidence-bar {
                width: 100%;
                height: 30px;
                background: #e0e0e0;
                border-radius: 15px;
                overflow: hidden;
                margin-top: 8px;
            }
            .confidence-fill {
                height: 100%;
                background: linear-gradient(90deg, #4CAF50, #8BC34A);
                transition: width 0.5s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: 600;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 التعرف على الكلمات العربية</h1>

            <div class="input-group">
                <label for="targetWord">الكلمة المستهدفة (Target Word):</label>
                <input type="text" id="targetWord" placeholder="اكتب الكلمة بالتشكيل (e.g., اللَّهِ)">
            </div>

            <div class="controls">
                <button id="recordBtn">🎤 ابدأ التسجيل (Record)</button>
                <button id="stopBtn" disabled>⏹️ إيقاف (Stop)</button>
            </div>

            <div id="status" class="status ready">جاهز للتسجيل (Ready)</div>

            <div id="results" class="results">
                <div class="result-item">
                    <div class="result-label">النص المنسوخ (Transcription):</div>
                    <div id="transcription" class="result-value arabic">-</div>
                </div>

                <div class="result-item">
                    <div class="result-label">درجة الثقة (Confidence Score):</div>
                    <div class="confidence-bar">
                        <div id="confidenceFill" class="confidence-fill" style="width: 0%">0%</div>
                    </div>
                </div>

                <div class="result-item">
                    <div class="result-label">نتيجة التحقق (Verification):</div>
                    <div id="verification">-</div>
                </div>
            </div>
        </div>

        <script>
            let mediaRecorder;
            let audioChunks = [];
            const recordBtn = document.getElementById('recordBtn');
            const stopBtn = document.getElementById('stopBtn');
            const status = document.getElementById('status');
            const results = document.getElementById('results');
            const targetWordInput = document.getElementById('targetWord');

            recordBtn.onclick = async () => {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];

                    mediaRecorder.ondataavailable = (event) => {
                        audioChunks.push(event.data);
                    };

                    mediaRecorder.onstop = async () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        await processAudio(audioBlob);
                        stream.getTracks().forEach(track => track.stop());
                    };

                    mediaRecorder.start();
                    recordBtn.disabled = true;
                    stopBtn.disabled = false;
                    status.className = 'status recording';
                    status.textContent = '🔴 جاري التسجيل... (Recording...)';
                    results.classList.remove('show');
                } catch (error) {
                    alert('خطأ في الوصول للميكروفون (Microphone access error): ' + error.message);
                }
            };

            stopBtn.onclick = () => {
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    mediaRecorder.stop();
                    recordBtn.disabled = false;
                    stopBtn.disabled = true;
                    status.className = 'status processing';
                    status.textContent = '⏳ جاري المعالجة... (Processing...)';
                }
            };

            async function processAudio(audioBlob) {
                const targetWord = targetWordInput.value.trim();

                if (!targetWord) {
                    alert('الرجاء إدخال الكلمة المستهدفة (Please enter target word)');
                    status.className = 'status ready';
                    status.textContent = 'جاهز للتسجيل (Ready)';
                    return;
                }

                const formData = new FormData();
                formData.append('audio', audioBlob, 'recording.wav');
                formData.append('target_word', targetWord);
                formData.append('threshold', '0.6');

                try {
                    // Call verify_word endpoint
                    const response = await fetch('/verify_word', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error('API request failed');
                    }

                    const data = await response.json();

                    // Also get detailed transcription with confidence
                    const formData2 = new FormData();
                    formData2.append('audio', audioBlob, 'recording.wav');

                    const transcribeResponse = await fetch('/transcribe_word', {
                        method: 'POST',
                        body: formData2
                    });

                    const transcribeData = await transcribeResponse.json();

                    // Display results
                    document.getElementById('transcription').textContent = transcribeData.transcription || 'N/A';

                    const confidence = Math.round((transcribeData.confidence || 0) * 100);
                    document.getElementById('confidenceFill').style.width = confidence + '%';
                    document.getElementById('confidenceFill').textContent = confidence + '%';

                    const verificationDiv = document.getElementById('verification');
                    if (data.result) {
                        verificationDiv.innerHTML = '<span class="match-badge success">✅ تطابق (Match)</span>';
                    } else {
                        verificationDiv.innerHTML = '<span class="match-badge fail">❌ لا تطابق (No Match)</span>';
                    }

                    results.classList.add('show');
                    status.className = 'status ready';
                    status.textContent = '✅ اكتمل! (Complete!)';

                } catch (error) {
                    alert('خطأ: ' + error.message);
                    status.className = 'status ready';
                    status.textContent = 'جاهز للتسجيل (Ready)';
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy"}

# --------------------------- Arabic Word Transcription Endpoint ---------------------------

@app.post("/verify_word", response_class=JSONResponse)
async def verify_word(
    audio: UploadFile = File(...),
    target_word: str = Form(...),
    threshold: float = Form(0.6),
    fuzzy_match: bool = Form(True),
    fuzzy_threshold: float = Form(None)
):
    """
    Verify if audio matches target Arabic word and exceeds confidence threshold.

    Parameters:
    - audio: WAV audio file containing spoken Arabic word
    - target_word: The expected Arabic word (e.g., "اللَّهِ", "مِنَ")
    - threshold: Confidence threshold (0.0 to 1.0, default 0.6 = 60%)
    - fuzzy_match: Enable fuzzy matching for minor variations (default True)
    - fuzzy_threshold: Custom fuzzy threshold (0-100), overrides dynamic threshold (default None = auto)

    Returns:
    - result: Boolean (True if match AND confidence >= threshold, False otherwise)
    - similarity: Fuzzy match similarity score (0-100, only if fuzzy_match=True)
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
        
        # Perform word matching (exact or fuzzy based on fuzzy_match parameter)
        if fuzzy_match:
            # Import fuzzy matching function
            from core.arabic_utils import fuzzy_match_arabic_words

            # Use fuzzy matching with dynamic threshold
            matches, similarity_score = fuzzy_match_arabic_words(
                transcription=transcription,
                target=target_word.strip(),
                custom_threshold=fuzzy_threshold
            )
        else:
            # Exact match (backward compatibility)
            from core.arabic_utils import normalize_arabic_text
            normalized_transcription = normalize_arabic_text(transcription)
            normalized_target = normalize_arabic_text(target_word.strip())
            matches = normalized_transcription == normalized_target
            similarity_score = 100.0 if matches else 0.0

        # Check if confidence exceeds threshold
        exceeds_threshold = confidence >= threshold

        # Final result: both conditions must be true
        result = matches and exceeds_threshold

        # Return result with similarity score
        return JSONResponse({
            "result": result,
            "similarity": round(similarity_score, 2)
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
