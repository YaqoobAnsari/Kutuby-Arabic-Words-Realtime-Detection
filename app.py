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
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from core.inference import get_backend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --------------------------- FastAPI App Setup ---------------------------

app = FastAPI(
    title="Arabic Word Recognition API",
    description="API for verifying Arabic Quranic word pronunciation. Backend selected via MODEL_VARIANT env var.",
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

# --------------------------- Backend Loading ---------------------------

@app.on_event("startup")
async def startup_event():
    """Load the configured inference backend (Tarteel by default, legacy via MODEL_VARIANT=legacy)."""
    logger.info("🚀 Application startup initiated")
    backend = get_backend()
    logger.info(f"✅ Application ready (backend variant={backend.variant})")

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
                Powered by Tarteel Whisper Quranic ASR
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
    """Health check endpoint — also reveals the active backend variant for prod verification."""
    logger.info("🏥 Health check requested")
    try:
        backend = get_backend()
        return {
            "status": "healthy",
            "model_loaded": True,
            "variant": backend.variant,
            "model_name": backend.model_name,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "status": "loading",
            "model_loaded": False,
            "error": f"{type(e).__name__}: {e}",
            "timestamp": datetime.now().isoformat(),
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

    backend = get_backend()

    # Read audio
    content = await audio.read()
    logger.info(f"📁 Audio file received: {len(content)} bytes, filename: {audio.filename}")
    if not content:
        return JSONResponse(status_code=400, content={"result": False, "error": "No audio data received"})

    # Decode with the existing FFmpeg-pipe path
    try:
        y, sr = load_audio_robust(content, sr=16000)
        logger.info(f"🎵 Audio duration: {len(y)/16000:.2f}s")
    except Exception as e:
        logger.error(f"❌ Audio loading failed: {e}")
        return JSONResponse(status_code=400, content={"result": False, "error": f"Could not read audio file. {str(e)}"})

    if len(y) == 0:
        return JSONResponse(status_code=400, content={"result": False, "error": "Empty audio file"})

    if not 0.0 <= threshold <= 1.0:
        return JSONResponse(status_code=400, content={"result": False, "error": "Threshold must be between 0.0 and 1.0"})

    # Silence gate — a constrained-decode backend is FORCED to emit a Quranic
    # word, so silence and noise sneak past the score threshold. Reject inputs
    # with peak amplitude below 0.005 (well below normal speech ~0.1+).
    raw_max_amplitude = float(max(abs(y)))
    if raw_max_amplitude < 0.005:
        logger.info(f"🔇 Silence/near-silence rejected (peak amplitude {raw_max_amplitude:.5f})")
        return JSONResponse({
            "result": False,
            "transcription": "",
            "target_word": target_word.strip(),
            "similarity": 0.0,
            "confidence": 0.0,
            "threshold": round(threshold * 100, 2),
            "processing_time_ms": round((time.time() - request_start) * 1000, 2),
            "latency_ms": 0.0,
            "score": None,
            "top_k_candidates": None,
            "variant": backend.variant,
            "model": backend.model_name,
            "rejection_reason": "audio_silent",
        })

    # Normalize after the silence gate so silence ÷ tiny == garbage doesn't slip past
    y = y / raw_max_amplitude

    # Dispatch to the active backend
    try:
        if backend.variant == "legacy":
            backend_result = backend.verify(
                y, target_word,
                fuzzy_match=fuzzy_match,
                fuzzy_threshold=fuzzy_threshold,
            )
        elif backend.variant == "fastconformer":
            # Closed-set exact-match verification. The client's threshold/fuzzy
            # params are intentionally not honored (no fuzzy matching).
            backend_result = backend.verify(y, target_word)
        else:
            # top_k=1 = greedy beam (fastest); legacy clients still see top_k_candidates as a 1-item list.
            backend_result = backend.verify(y, target_word, top_k=1)

        processing_time_ms = (time.time() - request_start) * 1000

        # Backward-compatible response shape + new additive fields.
        # Honesty fields (additive, non-breaking): make explicit HOW `result` was
        # actually decided. In the tarteel path the client's `threshold` and
        # `fuzzy_threshold` params are NOT used — the gate is exact lexicon match
        # + an internal log-prob score. `threshold` is kept only for back-compat
        # with existing clients so the response shape never shrinks.
        response = {
            "result": backend_result["result"],
            "transcription": backend_result["transcription"],
            "target_word": backend_result["target_word"],
            "similarity": backend_result["similarity"],
            "confidence": backend_result["confidence"],
            # Kept for back-compat; in tarteel mode this is NOT the gate (see decision_* below).
            "threshold": round(threshold * 100, 2),
            "decision_basis": (
                "fuzzy_string_match+confidence" if backend.variant == "legacy"
                else "curriculum_exact_match" if backend.variant == "fastconformer"
                else "exact_lexicon_match+logprob_score"
            ),
            "decision_threshold": backend_result.get("threshold"),
            "threshold_param_applied": (backend.variant == "legacy"),
            "processing_time_ms": round(processing_time_ms, 2),
            "latency_ms": backend_result["latency_ms"],
            "score": backend_result.get("score"),
            "top_k_candidates": backend_result.get("top_k_candidates"),
            "variant": backend.variant,
            "model": backend.model_name,
        }
        return JSONResponse(response)

    except Exception as e:
        logger.error(f"❌ verify_word error: {type(e).__name__}: {e}")
        return JSONResponse(
            status_code=500,
            content={"result": False, "error": f"Verification failed: {type(e).__name__}: {e}"},
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

    backend = get_backend()

    content = await audio.read()
    logger.info(f"📁 Audio file received: {len(content)} bytes")
    if not content:
        return JSONResponse(status_code=400, content={"error": "No audio data received", "transcription": None})

    try:
        y, sr = load_audio_robust(content, sr=16000)
        logger.info(f"🎵 Audio duration: {len(y)/16000:.2f}s")
    except Exception as e:
        logger.error(f"❌ Audio loading failed: {e}")
        return JSONResponse(status_code=400, content={"error": f"Could not read audio file. {e}", "transcription": None})

    if len(y) == 0:
        return JSONResponse(status_code=400, content={"error": "Empty audio file", "transcription": None})

    max_amplitude = max(abs(y))
    if max_amplitude > 0:
        y = y / max_amplitude

    try:
        backend_result = backend.transcribe(y)
        total_time_ms = (time.time() - request_start) * 1000

        return JSONResponse({
            "transcription": backend_result["transcription"],
            "confidence": backend_result["confidence"],
            "latency_ms": backend_result["latency_ms"],
            "total_time_ms": round(total_time_ms, 2),
            "model": backend.model_name,
            "variant": backend.variant,
        })

    except Exception as e:
        logger.error(f"❌ transcribe_word error: {type(e).__name__}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Transcription failed: {type(e).__name__}: {e}", "transcription": None},
        )

# For Hugging Face Spaces, the app is automatically served
# For local development, you can run: uvicorn app:app --host 0.0.0.0 --port 7860
