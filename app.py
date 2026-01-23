#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI entry point for Hugging Face Spaces
Arabic Word Recognition API - FastAPI Backend
"""

# Import all necessary modules
from __future__ import annotations
import os, io, time, json, logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime

import numpy as np
import librosa
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from core.arabic_utils import normalize_arabic_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
_WORD_PROCESSOR: Optional[Wav2Vec2Processor] = None
_MODEL_DEVICE = None  # Cache device to avoid repeated checks

def _load_word_model_once():
    """Load the Arabic word transcription model (Wav2Vec2-Large-XLSR-53-Arabic)"""
    global _WORD_MODEL, _WORD_PROCESSOR, _MODEL_DEVICE
    if _WORD_MODEL is None:
        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
        logger.info(f"🔄 Loading model: {model_name}")
        start_time = time.time()

        # Load processor and model
        _WORD_PROCESSOR = Wav2Vec2Processor.from_pretrained(model_name)
        _WORD_MODEL = Wav2Vec2ForCTC.from_pretrained(model_name)

        # Check for GPU and move model if available
        _MODEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _WORD_MODEL = _WORD_MODEL.to(_MODEL_DEVICE)
        _WORD_MODEL.eval()  # Set to evaluation mode

        # Enable torch compile for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile') and _MODEL_DEVICE.type == "cuda":
            try:
                logger.info("🚀 Compiling model with torch.compile for faster inference")
                _WORD_MODEL = torch.compile(_WORD_MODEL, mode="reduce-overhead")
            except Exception as e:
                logger.warning(f"⚠️ torch.compile failed, using standard model: {e}")

        load_time = time.time() - start_time
        gpu_info = f" (GPU: {torch.cuda.get_device_name(0)})" if _MODEL_DEVICE.type == "cuda" else ""
        logger.info(f"✅ Model loaded on {_MODEL_DEVICE.type.upper()}{gpu_info} in {load_time:.2f}s")

        # Warm up model with dummy input (eliminates first-request slowness)
        try:
            logger.info("🔥 Warming up model with dummy inference...")
            dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            dummy_inputs = _WORD_PROCESSOR(dummy_audio, sampling_rate=16000, return_tensors="pt", padding=True)
            dummy_inputs = {k: v.to(_MODEL_DEVICE) for k, v in dummy_inputs.items()}

            with torch.no_grad():
                if _MODEL_DEVICE.type == "cuda":
                    with torch.cuda.amp.autocast():
                        _ = _WORD_MODEL(dummy_inputs["input_values"]).logits
                else:
                    _ = _WORD_MODEL(dummy_inputs["input_values"]).logits

            logger.info("✅ Model warmed up and ready")
        except Exception as e:
            logger.warning(f"⚠️ Model warmup failed (non-critical): {e}")

    return _WORD_MODEL, _WORD_PROCESSOR, _MODEL_DEVICE

# Load model on startup for HuggingFace Space (has enough time during build)
@app.on_event("startup")
async def startup_event():
    """Load model when the app starts"""
    logger.info("🚀 Application startup initiated")
    _load_word_model_once()
    logger.info("✅ Application ready to serve requests")

# --------------------------- Audio Loading with Multiple Backends ---------------------------

def load_audio_robust(audio_data: bytes, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio with multiple fallback methods for maximum compatibility.

    Tries in order:
    1. soundfile (fastest, handles WAV natively)
    2. librosa with temp file (handles all formats via FFmpeg)
    3. pydub (alternative decoder)

    Args:
        audio_data: Raw audio bytes
        sr: Target sample rate (default 16000 Hz)

    Returns:
        Tuple of (audio_array, sample_rate)

    Raises:
        Exception: If all methods fail
    """
    import soundfile as sf
    import tempfile
    import os

    errors = []

    # Method 1: Try soundfile directly (fastest for WAV)
    try:
        logger.debug("🔧 Trying soundfile (direct BytesIO)")
        y, original_sr = sf.read(io.BytesIO(audio_data))
        if len(y.shape) > 1:  # Convert stereo to mono
            y = np.mean(y, axis=1)
        # Resample if needed
        if original_sr != sr:
            import librosa
            y = librosa.resample(y, orig_sr=original_sr, target_sr=sr)
        logger.info(f"✅ Audio loaded via soundfile: {len(y)} samples")
        return y, sr
    except Exception as e:
        errors.append(f"soundfile: {type(e).__name__}: {str(e)}")
        logger.debug(f"⚠️ soundfile failed: {e}")

    # Method 2: Try FFmpeg directly (FAST - bypasses slow audioread)
    try:
        logger.debug("🔧 Trying FFmpeg direct decode")
        import subprocess

        # Use FFmpeg to decode directly to WAV in memory
        cmd = [
            'ffmpeg',
            '-i', 'pipe:0',  # Read from stdin
            '-f', 'wav',     # Output format
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', str(sr),  # Sample rate
            '-ac', '1',      # Mono
            'pipe:1'         # Write to stdout
        ]

        result = subprocess.run(
            cmd,
            input=audio_data,
            capture_output=True,
            timeout=5  # 5 second timeout
        )

        if result.returncode == 0:
            # Parse WAV from stdout
            import soundfile as sf
            y, _ = sf.read(io.BytesIO(result.stdout))
            logger.info(f"✅ Audio loaded via FFmpeg: {len(y)} samples")
            return y, sr
        else:
            raise Exception(f"FFmpeg failed: {result.stderr.decode()[:200]}")

    except Exception as e:
        errors.append(f"FFmpeg: {type(e).__name__}: {str(e)}")
        logger.debug(f"⚠️ FFmpeg failed: {e}")

    # Method 3: Try librosa with temp file (slower fallback)
    try:
        logger.debug("🔧 Trying librosa+temp (slower)")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name

        try:
            y, _ = librosa.load(tmp_path, sr=sr, mono=True)
            logger.info(f"✅ Audio loaded via librosa: {len(y)} samples")
            return y, sr
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        errors.append(f"librosa: {type(e).__name__}: {str(e)}")
        logger.debug(f"⚠️ librosa failed: {e}")

    # Method 3: Try pydub (alternative decoder)
    try:
        logger.debug("🔧 Trying pydub (alternative decoder)")
        from pydub import AudioSegment

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name

        try:
            audio = AudioSegment.from_file(tmp_path)
            # Convert to mono
            if audio.channels > 1:
                audio = audio.set_channels(1)
            # Set sample rate
            audio = audio.set_frame_rate(sr)
            # Convert to numpy array
            y = np.array(audio.get_array_of_samples()).astype(np.float32)
            y = y / (2**15)  # Normalize to [-1, 1]
            logger.info(f"✅ Audio loaded via pydub: {len(y)} samples")
            return y, sr
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        errors.append(f"pydub: {type(e).__name__}: {str(e)}")
        logger.debug(f"⚠️ pydub failed: {e}")

    # All methods failed
    error_msg = "All audio loading methods failed:\n" + "\n".join(f"  - {err}" for err in errors)
    logger.error(f"❌ {error_msg}")
    raise Exception(error_msg)

# --------------------------- Health Check Endpoint ---------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Arabic Word Recognition</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', sans-serif;
                background: linear-gradient(135deg, #f5f7fa 0%, #e8eef3 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
                color: #2c3e50;
            }

            .container {
                background: white;
                border-radius: 16px;
                padding: 48px;
                max-width: 650px;
                width: 100%;
                box-shadow: 0 10px 40px rgba(0,0,0,0.08);
            }

            h1 {
                font-size: 28px;
                font-weight: 600;
                color: #1a202c;
                margin-bottom: 8px;
                text-align: center;
            }

            .subtitle {
                text-align: center;
                color: #718096;
                font-size: 14px;
                margin-bottom: 36px;
            }

            .controls {
                display: flex;
                gap: 12px;
                margin-bottom: 24px;
            }

            button {
                flex: 1;
                padding: 16px 24px;
                border: none;
                border-radius: 10px;
                font-size: 15px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
            }

            #recordBtn {
                background: #4a5568;
                color: white;
            }
            #recordBtn:hover:not(:disabled) {
                background: #2d3748;
                transform: translateY(-1px);
            }
            #recordBtn:disabled {
                background: #cbd5e0;
                cursor: not-allowed;
                color: #a0aec0;
            }

            #stopBtn {
                background: #e2e8f0;
                color: #4a5568;
            }
            #stopBtn:hover:not(:disabled) {
                background: #cbd5e0;
                transform: translateY(-1px);
            }
            #stopBtn:disabled {
                background: #f7fafc;
                cursor: not-allowed;
                color: #cbd5e0;
            }

            .status-bar {
                background: #f7fafc;
                border-radius: 10px;
                padding: 16px;
                margin-bottom: 24px;
                text-align: center;
                font-size: 14px;
                font-weight: 500;
                border: 2px solid #e2e8f0;
            }

            .status-bar.recording {
                background: #fef5f5;
                border-color: #fc8181;
                color: #c53030;
            }
            .status-bar.processing {
                background: #fffcf5;
                border-color: #fbd38d;
                color: #c05621;
            }
            .status-bar.ready {
                background: #f0fff4;
                border-color: #9ae6b4;
                color: #2f855a;
            }

            .results {
                background: #f8fafc;
                border-radius: 12px;
                padding: 24px;
                display: none;
            }
            .results.show { display: block; }

            .result-section {
                margin-bottom: 24px;
            }
            .result-section:last-child { margin-bottom: 0; }

            .result-label {
                font-size: 13px;
                font-weight: 600;
                color: #718096;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 8px;
            }

            .result-value {
                font-size: 24px;
                font-weight: 600;
                color: #1a202c;
                word-break: break-word;
            }

            .confidence-container {
                position: relative;
                height: 44px;
                background: #e2e8f0;
                border-radius: 10px;
                overflow: hidden;
            }

            .confidence-fill {
                height: 100%;
                background: linear-gradient(90deg, #4a5568 0%, #718096 100%);
                transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
                display: flex;
                align-items: center;
                padding: 0 16px;
            }

            .confidence-text {
                color: white;
                font-weight: 600;
                font-size: 15px;
            }

            .divider {
                height: 1px;
                background: #e2e8f0;
                margin: 24px 0;
            }

            .model-info {
                text-align: center;
                color: #a0aec0;
                font-size: 12px;
                margin-top: 32px;
                padding-top: 24px;
                border-top: 1px solid #e2e8f0;
            }

            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }

            .recording-indicator {
                display: inline-block;
                width: 8px;
                height: 8px;
                background: #fc8181;
                border-radius: 50%;
                animation: pulse 1.5s ease-in-out infinite;
                margin-right: 8px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 Arabic Word Recognition</h1>
            <div class="subtitle">Real-time speech-to-text transcription</div>

            <div class="controls">
                <button id="recordBtn">
                    <span>⬤</span>
                    <span>Start Recording</span>
                </button>
                <button id="stopBtn" disabled>
                    <span>⬛</span>
                    <span>Stop</span>
                </button>
            </div>

            <div id="statusBar" class="status-bar ready">
                Ready to record
            </div>

            <div id="results" class="results">
                <div class="result-section">
                    <div class="result-label">📝 Transcription</div>
                    <div id="transcription" class="result-value">—</div>
                </div>

                <div class="divider"></div>

                <div class="result-section">
                    <div class="result-label">🎯 Confidence Score</div>
                    <div class="confidence-container">
                        <div id="confidenceFill" class="confidence-fill" style="width: 0%">
                            <span id="confidenceText" class="confidence-text">0%</span>
                        </div>
                    </div>
                </div>

                <div class="divider"></div>

                <div class="result-section">
                    <div class="result-label">⏱️ Processing Time</div>
                    <div id="processingTime" class="result-value" style="font-size: 18px;">—</div>
                </div>
            </div>

            <div class="model-info">
                Powered by Wav2Vec2-Large-XLSR-53-Arabic
            </div>
        </div>

        <script>
            let mediaRecorder;
            let audioChunks = [];
            const recordBtn = document.getElementById('recordBtn');
            const stopBtn = document.getElementById('stopBtn');
            const statusBar = document.getElementById('statusBar');
            const results = document.getElementById('results');

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
                    statusBar.className = 'status-bar recording';
                    statusBar.innerHTML = '<span class="recording-indicator"></span>Recording...';
                    results.classList.remove('show');
                } catch (error) {
                    alert('Microphone access error: ' + error.message);
                }
            };

            stopBtn.onclick = () => {
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    mediaRecorder.stop();
                    recordBtn.disabled = false;
                    stopBtn.disabled = true;
                    statusBar.className = 'status-bar processing';
                    statusBar.textContent = 'Processing audio...';
                }
            };

            async function processAudio(audioBlob) {
                const formData = new FormData();
                formData.append('audio', audioBlob, 'recording.wav');

                try {
                    const response = await fetch('/transcribe_word', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error('API request failed');
                    }

                    const data = await response.json();

                    // Display transcription
                    const transcriptionEl = document.getElementById('transcription');
                    transcriptionEl.textContent = data.transcription || 'No transcription';
                    transcriptionEl.style.color = data.transcription ? '#1a202c' : '#a0aec0';

                    // Display confidence score
                    const confidence = Math.round(data.confidence || 0);
                    document.getElementById('confidenceFill').style.width = confidence + '%';
                    document.getElementById('confidenceText').textContent = confidence + '%';

                    // Display processing time
                    const processingTime = data.total_time_ms || data.latency_ms || 0;
                    document.getElementById('processingTime').textContent = `${Math.round(processingTime)}ms`;

                    // Show results
                    results.classList.add('show');
                    statusBar.className = 'status-bar ready';
                    statusBar.textContent = `✅ Complete in ${Math.round(processingTime)}ms`;

                } catch (error) {
                    alert('Error: ' + error.message);
                    statusBar.className = 'status-bar ready';
                    statusBar.textContent = 'Ready to record';
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/health")
def health():
    """Health check endpoint"""
    logger.info("🏥 Health check requested")
    model_loaded = _WORD_MODEL is not None and _WORD_PROCESSOR is not None
    return {
        "status": "healthy" if model_loaded else "loading",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }

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
    request_start = time.time()
    logger.info(f"🎯 /verify_word called - target: '{target_word}', threshold: {threshold}, fuzzy: {fuzzy_match}")

    model, processor, device = _load_word_model_once()

    # Read audio file
    content = await audio.read()
    logger.info(f"📁 Audio file received: {len(content)} bytes, filename: {audio.filename}")

    # Check if content is empty
    if not content or len(content) == 0:
        logger.error("❌ No audio data received")
        return JSONResponse(
            status_code=400,
            content={
                "result": False,
                "error": "No audio data received"
            }
        )

    # Load audio with robust multi-backend fallback
    try:
        y, sr = load_audio_robust(content, sr=16000)
        logger.info(f"🎵 Audio duration: {len(y)/16000:.2f}s")
    except Exception as e:
        logger.error(f"❌ Audio loading failed: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "result": False,
                "error": f"Could not read audio file. {str(e)}"
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
    inference_start = time.time()
    try:
        # Process audio
        inputs = processor(
            y,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        # Move inputs to model device (cached, no repeated parameter access)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Perform inference with optimizations (mixed precision on GPU)
        if device.type == "cuda":
            with torch.no_grad(), torch.cuda.amp.autocast():
                logits = model(inputs["input_values"]).logits
        else:
            with torch.no_grad():
                logits = model(inputs["input_values"]).logits

        inference_time = (time.time() - inference_start) * 1000
        logger.debug(f"⚡ Inference: {inference_time:.0f}ms on {device.type.upper()}")
        
        # Decode predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0].strip()
        
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

        # Calculate processing time
        processing_time = (time.time() - request_start) * 1000

        logger.info(f"✅ Transcription: '{transcription}' | Confidence: {confidence*100:.1f}% | Similarity: {similarity_score:.1f}% | Match: {matches} | Result: {result} | Time: {processing_time:.0f}ms")

        # Return result with all details
        return JSONResponse({
            "result": result,
            "transcription": transcription,
            "target_word": target_word.strip(),
            "similarity": round(similarity_score, 2),
            "confidence": round(confidence * 100, 2),
            "threshold": threshold * 100,
            "processing_time_ms": round(processing_time, 2)
        })

    except Exception as e:
        logger.error(f"❌ verify_word error: {type(e).__name__}: {e}")
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
    request_start = time.time()
    logger.info(f"🎤 /transcribe_word called - filename: {audio.filename}")

    model, processor, device = _load_word_model_once()

    # Read audio file
    content = await audio.read()
    logger.info(f"📁 Audio file received: {len(content)} bytes")

    # Check if content is empty
    if not content or len(content) == 0:
        logger.error("❌ No audio data received")
        return JSONResponse(
            status_code=400,
            content={
                "error": "No audio data received",
                "transcription": None
            }
        )

    # Load audio with robust multi-backend fallback
    try:
        y, sr = load_audio_robust(content, sr=16000)
        logger.info(f"🎵 Audio duration: {len(y)/16000:.2f}s")
    except Exception as e:
        logger.error(f"❌ Audio loading failed: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Could not read audio file. {str(e)}",
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
        # Process audio
        inputs = processor(
            y,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        # Move inputs to model device (cached, no repeated parameter access)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Perform inference with optimizations (mixed precision on GPU)
        inference_start = time.perf_counter()
        if device.type == "cuda":
            with torch.no_grad(), torch.cuda.amp.autocast():
                logits = model(inputs["input_values"]).logits
        else:
            with torch.no_grad():
                logits = model(inputs["input_values"]).logits

        inference_time = (time.perf_counter() - inference_start) * 1000
        logger.debug(f"⚡ Inference: {inference_time:.0f}ms on {device.type.upper()}")

        # Decode predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        
        # Calculate confidence score
        probabilities = torch.softmax(logits, dim=-1)
        max_probs = torch.max(probabilities, dim=-1).values
        confidence = torch.mean(max_probs).item() * 100
        
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        total_time_ms = (time.time() - request_start) * 1000

        logger.info(f"✅ Transcription: '{transcription.strip()}' | Confidence: {confidence:.1f}% | Latency: {latency_ms:.0f}ms | Total: {total_time_ms:.0f}ms")

        return JSONResponse({
            "transcription": transcription.strip(),
            "confidence": round(confidence, 2),
            "latency_ms": round(latency_ms, 2),
            "total_time_ms": round(total_time_ms, 2),
            "model": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
        })

    except Exception as e:
        logger.error(f"❌ transcribe_word error: {type(e).__name__}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Transcription failed: {type(e).__name__}: {e}",
                "transcription": None
            }
        )

# For Hugging Face Spaces, the app is automatically served
# For local development, you can run: uvicorn app:app --host 0.0.0.0 --port 7860
