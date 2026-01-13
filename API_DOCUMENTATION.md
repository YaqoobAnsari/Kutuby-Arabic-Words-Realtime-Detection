# FastAPI Backend - Complete Documentation

## ✅ CONFIRMATION: FastAPI is Set Up for Postman Requests

**YES** - Your FastAPI backend is correctly configured to handle POST requests from Postman.

---

## 📍 POST Endpoint Details

### Endpoint: `/transcribe_word`

**Method:** `POST`  
**URL:** `https://yansari-arabic-word-recognition.hf.space/transcribe_word`  
**Content-Type:** `multipart/form-data`

---

## 🔑 POST Request Keys (Parameters)

### Required Parameters:

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `audio` | **File** | WAV audio file containing spoken Arabic word(s) | `audio.wav` |

**Note:** Only ONE parameter is required: `audio` (as a file upload)

---

## 📤 Request Format in Postman

1. **Method:** Select `POST`
2. **URL:** `https://yansari-arabic-word-recognition.hf.space/transcribe_word`
3. **Body Tab:**
   - Select `form-data`
   - Key: `audio`
   - Type: Change from "Text" to **"File"** (dropdown on right)
   - Value: Click "Select Files" and choose your `.wav` file

---

## 📥 Response Payload

### Success Response (200 OK)

```json
{
  "transcription": "اللَّهِ",
  "confidence": 94.7,
  "latency_ms": 123.45,
  "model": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
}
```

**Response Fields:**
- `transcription` (string): The recognized Arabic text
- `confidence` (float): Confidence score as percentage (0-100)
- `latency_ms` (float): Processing time in milliseconds
- `model` (string): Model name used for transcription

### Error Response (400 Bad Request)

```json
{
  "error": "Could not read audio file. Expected WAV format. Error: ...",
  "transcription": null
}
```

### Error Response (500 Internal Server Error)

```json
{
  "error": "Transcription failed: ...",
  "transcription": null
}
```

---

## ✅ Requirements.txt Verification

Your `requirements.txt` includes all necessary dependencies:

```txt
# Core Web Framework
fastapi>=0.104.0          ✅ FastAPI framework
uvicorn[standard]>=0.24.0 ✅ ASGI server
python-multipart>=0.0.6   ✅ File upload support

# Audio Processing
librosa>=0.10.0           ✅ Audio loading
soundfile>=0.12.1         ✅ Audio file support

# Machine Learning
torch>=2.0.0              ✅ PyTorch
transformers>=4.30.0      ✅ Hugging Face transformers
sentencepiece>=0.1.99     ✅ Tokenizer support

# Scientific Computing
numpy>=1.24.0             ✅ Numerical operations
```

**All dependencies are present and correct! ✅**

---

## ✅ Dockerfile Verification

Your `Dockerfile` is correctly configured:

```dockerfile
FROM python:3.10-slim          ✅ Python base image
WORKDIR /app                   ✅ Working directory
# System dependencies          ✅ Audio libraries
COPY requirements.txt .         ✅ Install dependencies
RUN pip install ...            ✅ Install Python packages
EXPOSE 7860                     ✅ Correct port
CMD ["uvicorn", "app:app", ...] ✅ FastAPI startup command
```

**Dockerfile is perfect! ✅**

---

## 🧪 Testing the API

### Using Postman:

1. **Create New Request**
   - Method: `POST`
   - URL: `https://yansari-arabic-word-recognition.hf.space/transcribe_word`

2. **Configure Body**
   - Tab: `Body`
   - Type: `form-data`
   - Key: `audio` (type: File)
   - Value: Select your `.wav` file

3. **Send Request**
   - Click "Send"
   - Wait for response

### Using cURL:

```bash
curl -X POST "https://yansari-arabic-word-recognition.hf.space/transcribe_word" \
  -F "audio=@your_audio_file.wav"
```

### Using Python:

```python
import requests

url = "https://yansari-arabic-word-recognition.hf.space/transcribe_word"
files = {"audio": open("audio.wav", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

---

## 🔍 Additional Endpoints

### Health Check
- **GET** `/health`
- **Response:** `{"status": "healthy"}`

### API Info
- **GET** `/`
- **Response:** API information and available endpoints

### Interactive Docs
- **GET** `/docs` - Swagger UI
- **GET** `/redoc` - ReDoc documentation

---

## ✅ Summary

| Component | Status | Notes |
|-----------|--------|-------|
| FastAPI Setup | ✅ Ready | Correctly configured |
| POST Endpoint | ✅ Ready | `/transcribe_word` |
| Request Keys | ✅ Ready | `audio` (File) |
| Response Format | ✅ Ready | JSON with transcription |
| requirements.txt | ✅ Complete | All dependencies present |
| Dockerfile | ✅ Correct | FastAPI startup configured |

**Everything is set up correctly for Postman requests! 🎉**

