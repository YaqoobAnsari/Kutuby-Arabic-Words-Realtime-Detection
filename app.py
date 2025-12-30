"""
Arabic Word Recognition API
FastAPI backend for real-time audio processing
"""

import os
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from core.model_loader import ModelLoader
from core.transcriber import AudioTranscriber

# Initialize FastAPI
app = FastAPI(title="Arabic Word Recognition API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
model_loader = ModelLoader()
model, tokenizer, _ = model_loader.load_model()
transcriber = AudioTranscriber(model, tokenizer)

print("✅ Model loaded successfully!")


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main HTML page"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arabic Word Recognition</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
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
            color: #667eea;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }

        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }

        .stats {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
        }

        .stats h3 {
            color: #333;
            margin-bottom: 15px;
        }

        .metrics {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }

        .metric {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }

        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }

        .metric-label {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }

        .recording-section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
        }

        #recordButton {
            background: #667eea;
            color: white;
            border: none;
            padding: 20px 40px;
            font-size: 1.2em;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s;
            margin: 20px 0;
        }

        #recordButton:hover {
            background: #5568d3;
            transform: scale(1.05);
        }

        #recordButton:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: scale(1);
        }

        #recordButton.recording {
            background: #dc3545;
            animation: pulse 1s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        .status {
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            display: none;
        }

        .status.info {
            background: #d1ecf1;
            color: #0c5460;
            display: block;
        }

        .status.success {
            background: #d4edda;
            color: #155724;
        }

        .status.error {
            background: #f8d7da;
            color: #721c24;
        }

        #result {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            font-size: 2em;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
            display: none;
        }

        .instructions {
            background: #e7f3ff;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-top: 30px;
            border-radius: 5px;
        }

        .instructions h3 {
            color: #667eea;
            margin-bottom: 10px;
        }

        .instructions ul {
            margin-left: 20px;
            color: #333;
        }

        .instructions li {
            margin: 8px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Arabic Word Recognition</h1>
        <p class="subtitle">Real-time speech-to-text using Wav2Vec2</p>

        <div class="stats">
            <h3>📊 Model Performance</h3>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">95.3%</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric">
                    <div class="metric-value">94.7%</div>
                    <div class="metric-label">Confidence</div>
                </div>
                <div class="metric">
                    <div class="metric-value">73.3%</div>
                    <div class="metric-label">Perfect Matches</div>
                </div>
                <div class="metric">
                    <div class="metric-value">30/30</div>
                    <div class="metric-label">Quranic Words</div>
                </div>
            </div>
        </div>

        <div class="recording-section">
            <h2>🎙️ Record Your Voice</h2>
            <button id="recordButton">🎤 Click to Record</button>
            <div id="status" class="status info">
                Click the button above to start recording (3 seconds)
            </div>
            <div id="result"></div>
        </div>

        <div class="instructions">
            <h3>📖 How to Use</h3>
            <ul>
                <li>🎤 Click the "Record" button</li>
                <li>🗣️ Speak an Arabic word clearly (you have 3 seconds)</li>
                <li>⏱️ Wait for the recording to finish automatically</li>
                <li>🤖 AI will transcribe your speech instantly</li>
                <li>✨ See the recognized Arabic text below</li>
            </ul>
        </div>
    </div>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        const recordButton = document.getElementById('recordButton');
        const statusDiv = document.getElementById('status');
        const resultDiv = document.getElementById('result');

        recordButton.addEventListener('click', async () => {
            if (!mediaRecorder || mediaRecorder.state === 'inactive') {
                await startRecording();
            }
        });

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    await uploadAudio(audioBlob);
                    stream.getTracks().forEach(track => track.stop());
                };

                mediaRecorder.start();
                recordButton.textContent = '🔴 Recording...';
                recordButton.classList.add('recording');
                recordButton.disabled = true;
                statusDiv.className = 'status info';
                statusDiv.textContent = '🎙️ Recording... Speak now!';
                statusDiv.style.display = 'block';
                resultDiv.style.display = 'none';

                // Auto-stop after 3 seconds
                setTimeout(() => {
                    if (mediaRecorder.state === 'recording') {
                        mediaRecorder.stop();
                        recordButton.textContent = '🎤 Click to Record';
                        recordButton.classList.remove('recording');
                        recordButton.disabled = false;
                    }
                }, 3000);

            } catch (error) {
                statusDiv.className = 'status error';
                statusDiv.textContent = '❌ Error: Could not access microphone. Please allow microphone permissions.';
                statusDiv.style.display = 'block';
                console.error('Error accessing microphone:', error);
            }
        }

        async function uploadAudio(audioBlob) {
            statusDiv.className = 'status info';
            statusDiv.textContent = '🤖 Analyzing speech with AI...';
            statusDiv.style.display = 'block';

            const formData = new FormData();
            formData.append('file', audioBlob, 'recording.wav');

            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    resultDiv.textContent = data.transcription || 'لا يوجد كلام';
                    resultDiv.style.display = 'block';
                    statusDiv.className = 'status success';
                    statusDiv.textContent = '✅ Transcription completed!';
                } else {
                    throw new Error(data.detail || 'Transcription failed');
                }

            } catch (error) {
                statusDiv.className = 'status error';
                statusDiv.textContent = '❌ Error: ' + error.message;
                console.error('Upload error:', error);
            }
        }
    </script>
</body>
</html>
    """


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe uploaded audio file"""
    try:
        # Read audio data
        audio_data = await file.read()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_data)
            temp_path = tmp.name

        # Transcribe
        transcription = transcriber.transcribe(temp_path)

        # Cleanup
        os.unlink(temp_path)

        return JSONResponse({
            "transcription": transcription or "",
            "success": True
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": transcriber is not None}


@app.get("/api/info")
async def api_info():
    """Return available API endpoints"""
    return {
        "endpoints": {
            "GET /": "Web UI for real-time word recognition",
            "POST /transcribe": "Transcribe audio from web UI (file upload)",
            "POST /verify_word": "Verify if audio matches target word (Postman/API)",
            "GET /health": "Health check",
            "GET /api/info": "This endpoint - API information"
        },
        "verify_word_parameters": {
            "audio": "Audio file (WAV format, multipart/form-data)",
            "target_word": "Expected Arabic word (form field)",
            "threshold": "Confidence threshold 0-1 (form field, default: 0.6)"
        },
        "example_curl": "curl -X POST '/verify_word' -F 'audio=@file.wav' -F 'target_word=الله' -F 'threshold=0.6'"
    }


@app.post("/verify_word", response_class=JSONResponse)
async def verify_word(
    audio: UploadFile = File(...),
    target_word: str = Form(...),
    threshold: float = Form(0.6)
):
    """
    Endpoint for verifying if an audio file matches a target Arabic word.

    Parameters:
    - audio: WAV file (1-3 seconds duration)
    - target_word: The expected Arabic word (e.g., "الله", "من", "في")
    - threshold: Confidence threshold (default 0.6 = 60%)

    Returns:
    - result: True if predicted word matches target and confidence >= threshold
    - predicted_word: The transcribed word
    - confidence: The confidence score (0-1)
    - message: Human-readable explanation

    Example usage with curl:
    curl -X POST "http://localhost:7860/verify_word" \
         -F "audio=@recording.wav" \
         -F "target_word=الله" \
         -F "threshold=0.6"
    """
    try:
        # Read audio data
        audio_data = await audio.read()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_data)
            temp_path = tmp.name

        # Transcribe with confidence
        result = transcriber.transcribe_with_confidence(temp_path)

        # Cleanup
        os.unlink(temp_path)

        # Check for errors
        if result["error"]:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Transcription failed: {result['error']}",
                    "result": False
                }
            )

        predicted_word = result["transcription"]
        confidence = result["confidence"]

        # Normalize both words for comparison (remove extra spaces, diacritics comparison)
        target_normalized = target_word.strip()
        predicted_normalized = predicted_word.strip() if predicted_word else ""

        # Check if words match (exact or close match)
        words_match = (
            predicted_normalized == target_normalized or
            predicted_normalized.replace(" ", "") == target_normalized.replace(" ", "")
        )

        # Verify if target word matches prediction AND confidence meets threshold
        verification_result = words_match and confidence >= threshold

        # Generate message
        if verification_result:
            message = f"✓ Success: '{target_word}' detected with {confidence*100:.2f}% confidence (threshold: {threshold*100:.0f}%)"
        elif not words_match:
            message = f"✗ Failed: Expected '{target_word}' but got '{predicted_word}' ({confidence*100:.2f}% confidence)"
        else:
            message = f"✗ Failed: Word matches but confidence {confidence*100:.2f}% is below threshold {threshold*100:.0f}%"

        return JSONResponse({
            "result": verification_result,
            "target_word": target_word,
            "predicted_word": predicted_word,
            "confidence": float(confidence),
            "threshold": float(threshold),
            "message": message
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "result": False
            }
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
