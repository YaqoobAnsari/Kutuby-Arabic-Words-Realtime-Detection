# API Endpoints Documentation

## Arabic Words API
**Base URL**: `https://arabic-words-api-621075448606.europe-west1.run.app`

---

### 1. Health Check
**GET** `/health`

Check if the API is running.

**Response**:
```json
{
  "status": "healthy"
}
```

---

### 2. Transcribe Word
**POST** `/transcribe_word`

Transcribe Arabic audio to text.

**Request**:
- Content-Type: `multipart/form-data`
- Body:
  - `audio` (file): Audio file (WAV format, 16kHz recommended)

**cURL Example**:
```bash
curl -X POST https://arabic-words-api-621075448606.europe-west1.run.app/transcribe_word \
  -F "audio=@your_audio.wav"
```

**Response**:
```json
{
  "transcription": "السلام عليكم",
  "confidence": 87.5,
  "latency_ms": 234.56,
  "model": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
}
```

---

### 3. Verify Word
**POST** `/verify_word`

Verify if spoken audio matches a target Arabic word using fuzzy matching.

**Request**:
- Content-Type: `multipart/form-data`
- Body:
  - `audio` (file): Audio file (WAV format)
  - `target_word` (string): Expected Arabic word
  - `threshold` (float, optional): Similarity threshold (0.0-1.0, default: 0.6)

**cURL Example**:
```bash
curl -X POST https://arabic-words-api-621075448606.europe-west1.run.app/verify_word \
  -F "audio=@audio.wav" \
  -F "target_word=مرحبا" \
  -F "threshold=0.6"
```

**Response**:
```json
{
  "result": true,
  "similarity": 95.32,
  "transcription": "مرحبا",
  "target": "مرحبا",
  "threshold": 0.6
}
```

---

## Arabic Letters API
**Base URL**: `https://arabic-letters-api-621075448606.europe-west1.run.app`

---

### 1. Health Check
**GET** `/health`

Check if the API is running.

**Response**:
```json
{
  "status": "healthy"
}
```

---

### 2. Verify Letter
**POST** `/verify_letter`

Verify if spoken audio matches a target Arabic letter.

**Request**:
- Content-Type: `multipart/form-data`
- Body:
  - `audio` (file): Audio file (WAV format)
  - `target_letter` (string): Expected Arabic letter (single character)

**cURL Example**:
```bash
curl -X POST https://arabic-letters-api-621075448606.europe-west1.run.app/verify_letter \
  -F "audio=@letter_audio.wav" \
  -F "target_letter=ب"
```

**Response**:
```json
{
  "result": true,
  "confidence": 92.3,
  "recognized_letter": "ب",
  "target_letter": "ب"
}
```

---

### 3. Transcribe Letter
**POST** `/transcribe_letter`

Transcribe audio to identify the Arabic letter.

**Request**:
- Content-Type: `multipart/form-data`
- Body:
  - `audio` (file): Audio file (WAV format)

**cURL Example**:
```bash
curl -X POST https://arabic-letters-api-621075448606.europe-west1.run.app/transcribe_letter \
  -F "audio=@letter_audio.wav"
```

**Response**:
```json
{
  "transcription": "ب",
  "confidence": 89.7,
  "latency_ms": 123.45
}
```

---

## Quick Test

### Test Arabic Words API:
```bash
# Health check
curl https://arabic-words-api-621075448606.europe-west1.run.app/health

# Open web UI in browser
open https://arabic-words-api-621075448606.europe-west1.run.app/
```

### Test Arabic Letters API:
```bash
# Health check
curl https://arabic-letters-api-621075448606.europe-west1.run.app/health

# Open web UI in browser
open https://arabic-letters-api-621075448606.europe-west1.run.app/
```

---

## Common Error Responses

### 400 Bad Request
```json
{
  "error": "Could not read audio file. Expected WAV format."
}
```

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "audio"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal server error during processing"
}
```

---

## Notes

- **Audio Format**: WAV format recommended, 16kHz sample rate
- **Max File Size**: 10MB (configurable)
- **Timeout**: 300 seconds
- **Authentication**: None (public endpoints)
- **CORS**: Enabled for all origins
