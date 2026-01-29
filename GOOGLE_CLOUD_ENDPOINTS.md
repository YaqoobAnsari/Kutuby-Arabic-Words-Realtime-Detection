# Google Cloud Run API Endpoints - Postman Testing Guide

## 🎯 Service URLs

### Arabic Words API
**Base URL**: `https://arabic-words-api-d26k2plh4q-ew.a.run.app`
**Status**: ✅ Running on Google Cloud Run (europe-west1)
**Memory**: 5Gi
**CPU**: 2 vCPU

### Arabic Letters API
**Base URL**: `https://arabic-letters-api-d26k2plh4q-ew.a.run.app`
**Status**: ✅ Running on Google Cloud Run (europe-west1)
**Memory**: 2Gi
**CPU**: 2 vCPU

---

## 📋 Arabic Words API Endpoints

### 1. Health Check
**Endpoint**: `GET /health`
**URL**: https://arabic-words-api-d26k2plh4q-ew.a.run.app/health

**Postman Setup**:
```
Method: GET
URL: https://arabic-words-api-d26k2plh4q-ew.a.run.app/health
Headers: None required
Body: None
```

**Expected Response (200 OK)**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-01-23T12:34:56.789012"
}
```

**cURL Test**:
```bash
curl https://arabic-words-api-d26k2plh4q-ew.a.run.app/health
```

---

### 2. Transcribe Word
**Endpoint**: `POST /transcribe_word`
**URL**: https://arabic-words-api-d26k2plh4q-ew.a.run.app/transcribe_word

**Postman Setup**:
```
Method: POST
URL: https://arabic-words-api-d26k2plh4q-ew.a.run.app/transcribe_word
Headers:
  - Content-Type: multipart/form-data (auto-set by Postman)
Body:
  - Type: form-data
  - Key: audio
  - Value: [Select File] - Upload your .wav/.mp3/.webm audio file
```

**Expected Response (200 OK)**:
```json
{
  "transcription": "السلام عليكم",
  "confidence": 87.5,
  "latency_ms": 234.56,
  "total_time_ms": 456.78,
  "model": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
}
```

**cURL Test**:
```bash
curl -X POST https://arabic-words-api-d26k2plh4q-ew.a.run.app/transcribe_word \
  -F "audio=@your_audio.wav"
```

**Response Fields**:
- `transcription`: The recognized Arabic text
- `confidence`: Model confidence score (0-100%)
- `latency_ms`: Inference time only
- `total_time_ms`: End-to-end processing time
- `model`: Model identifier

---

### 3. Verify Word
**Endpoint**: `POST /verify_word`
**URL**: https://arabic-words-api-d26k2plh4q-ew.a.run.app/verify_word

**Postman Setup**:
```
Method: POST
URL: https://arabic-words-api-d26k2plh4q-ew.a.run.app/verify_word
Headers:
  - Content-Type: multipart/form-data (auto-set by Postman)
Body:
  - Type: form-data
  - Key: audio | Value: [Select File]
  - Key: target_word | Value: مرحبا
  - Key: threshold | Value: 0.6
  - Key: fuzzy_match | Value: true
  - Key: fuzzy_threshold | Value: 80 (optional)
```

**Expected Response (200 OK)**:
```json
{
  "result": true,
  "transcription": "مرحبا",
  "target_word": "مرحبا",
  "similarity": 95.32,
  "confidence": 89.45,
  "threshold": 60.0,
  "processing_time_ms": 234.56
}
```

**cURL Test**:
```bash
curl -X POST https://arabic-words-api-d26k2plh4q-ew.a.run.app/verify_word \
  -F "audio=@your_audio.wav" \
  -F "target_word=مرحبا" \
  -F "threshold=0.6" \
  -F "fuzzy_match=true"
```

**Response Fields**:
- `result`: Boolean - true if word matches
- `transcription`: What was actually heard
- `target_word`: What was expected
- `similarity`: Fuzzy match score (0-100%)
- `confidence`: Model confidence (0-100%)
- `threshold`: The threshold used (0-100%)
- `processing_time_ms`: Total processing time

---

## 📋 Arabic Letters API Endpoints

### 1. Health Check
**Endpoint**: `GET /health`
**URL**: https://arabic-letters-api-d26k2plh4q-ew.a.run.app/health

**Postman Setup**:
```
Method: GET
URL: https://arabic-letters-api-d26k2plh4q-ew.a.run.app/health
Headers: None required
Body: None
```

**Expected Response (200 OK)**:
```json
{
  "status": "healthy"
}
```

**cURL Test**:
```bash
curl https://arabic-letters-api-d26k2plh4q-ew.a.run.app/health
```

---

### 2. Verify Letter
**Endpoint**: `POST /verify_letter`
**URL**: https://arabic-letters-api-d26k2plh4q-ew.a.run.app/verify_letter

**Postman Setup**:
```
Method: POST
URL: https://arabic-letters-api-d26k2plh4q-ew.a.run.app/verify_letter
Headers:
  - Content-Type: multipart/form-data (auto-set by Postman)
Body:
  - Type: form-data
  - Key: audio | Value: [Select File]
  - Key: target_letter | Value: ب
```

**Expected Response (200 OK)**:
```json
{
  "result": true,
  "confidence": 92.3,
  "recognized_letter": "ب",
  "target_letter": "ب"
}
```

**cURL Test**:
```bash
curl -X POST https://arabic-letters-api-d26k2plh4q-ew.a.run.app/verify_letter \
  -F "audio=@letter_audio.wav" \
  -F "target_letter=ب"
```

**Response Fields**:
- `result`: Boolean - true if letter matches
- `confidence`: Recognition confidence (0-100%)
- `recognized_letter`: What was detected
- `target_letter`: What was expected

---

### 3. Transcribe Letter
**Endpoint**: `POST /transcribe_letter`
**URL**: https://arabic-letters-api-d26k2plh4q-ew.a.run.app/transcribe_letter

**Postman Setup**:
```
Method: POST
URL: https://arabic-letters-api-d26k2plh4q-ew.a.run.app/transcribe_letter
Headers:
  - Content-Type: multipart/form-data (auto-set by Postman)
Body:
  - Type: form-data
  - Key: audio | Value: [Select File]
```

**Expected Response (200 OK)**:
```json
{
  "transcription": "ب",
  "confidence": 89.7,
  "latency_ms": 123.45
}
```

**cURL Test**:
```bash
curl -X POST https://arabic-letters-api-d26k2plh4q-ew.a.run.app/transcribe_letter \
  -F "audio=@letter_audio.wav"
```

**Response Fields**:
- `transcription`: The recognized letter
- `confidence`: Model confidence (0-100%)
- `latency_ms`: Processing time

---

## 🧪 Quick Postman Collection Test

### Step-by-Step Postman Testing:

#### Test 1: Health Checks
1. Create new request in Postman
2. Set method to `GET`
3. URL: `https://arabic-words-api-d26k2plh4q-ew.a.run.app/health`
4. Click "Send"
5. Should get `{"status":"healthy",...}`

#### Test 2: Transcribe Word (No Audio File Needed)
1. Create new request
2. Set method to `POST`
3. URL: `https://arabic-words-api-d26k2plh4q-ew.a.run.app/transcribe_word`
4. Go to "Body" tab
5. Select "form-data"
6. Add key: `audio`
7. Change type from "Text" to "File"
8. Click "Select Files" and upload an audio file
9. Click "Send"
10. Should get transcription results

#### Test 3: Verify Word
1. Same as Test 2 but use `/verify_word` endpoint
2. Add additional form-data fields:
   - `target_word`: مرحبا (or any Arabic word)
   - `threshold`: 0.6
   - `fuzzy_match`: true
3. Click "Send"
4. Should get match results with similarity score

---

## 🎨 Web UI Access

### Arabic Words API Web Interface
**URL**: https://arabic-words-api-d26k2plh4q-ew.a.run.app/

Open in browser to access the interactive web UI with:
- Microphone recording
- Real-time transcription
- Confidence scores
- Processing time display

---

## 📊 Performance Expectations

### Arabic Words API (Optimized)
- **Cold Start**: 15-20 seconds (first request after idle)
- **Warm Requests**: 3-5 seconds (CPU)
- **Audio Loading**: ~200ms
- **Model Inference**: ~3500ms
- **Total**: ~3.8 seconds average

### Arabic Letters API
- **Cold Start**: 10-15 seconds
- **Warm Requests**: 1-2 seconds
- **Lighter model, faster inference**

---

## 🔍 Troubleshooting

### Error: 503 Service Unavailable
**Cause**: Cold start (model loading)
**Solution**: Wait 20 seconds and retry

### Error: 400 Bad Request
**Cause**: Invalid audio format or missing parameters
**Solution**: Check audio file is valid WAV/MP3/WebM

### Error: 504 Gateway Timeout
**Cause**: Request took >300 seconds
**Solution**: Check audio file size (should be <10MB)

### Slow First Request
**Expected**: First request after idle takes 15-20s (model loading)
**Solution**: This is normal, subsequent requests will be fast

---

## 📝 Postman Collection Export

You can import this JSON into Postman:

```json
{
  "info": {
    "name": "Kutuby Arabic APIs",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Arabic Words API",
      "item": [
        {
          "name": "Health Check",
          "request": {
            "method": "GET",
            "header": [],
            "url": "https://arabic-words-api-d26k2plh4q-ew.a.run.app/health"
          }
        },
        {
          "name": "Transcribe Word",
          "request": {
            "method": "POST",
            "header": [],
            "body": {
              "mode": "formdata",
              "formdata": [
                {
                  "key": "audio",
                  "type": "file",
                  "src": []
                }
              ]
            },
            "url": "https://arabic-words-api-d26k2plh4q-ew.a.run.app/transcribe_word"
          }
        },
        {
          "name": "Verify Word",
          "request": {
            "method": "POST",
            "header": [],
            "body": {
              "mode": "formdata",
              "formdata": [
                {
                  "key": "audio",
                  "type": "file",
                  "src": []
                },
                {
                  "key": "target_word",
                  "value": "مرحبا",
                  "type": "text"
                },
                {
                  "key": "threshold",
                  "value": "0.6",
                  "type": "text"
                },
                {
                  "key": "fuzzy_match",
                  "value": "true",
                  "type": "text"
                }
              ]
            },
            "url": "https://arabic-words-api-d26k2plh4q-ew.a.run.app/verify_word"
          }
        }
      ]
    },
    {
      "name": "Arabic Letters API",
      "item": [
        {
          "name": "Health Check",
          "request": {
            "method": "GET",
            "header": [],
            "url": "https://arabic-letters-api-d26k2plh4q-ew.a.run.app/health"
          }
        },
        {
          "name": "Verify Letter",
          "request": {
            "method": "POST",
            "header": [],
            "body": {
              "mode": "formdata",
              "formdata": [
                {
                  "key": "audio",
                  "type": "file",
                  "src": []
                },
                {
                  "key": "target_letter",
                  "value": "ب",
                  "type": "text"
                }
              ]
            },
            "url": "https://arabic-letters-api-d26k2plh4q-ew.a.run.app/verify_letter"
          }
        },
        {
          "name": "Transcribe Letter",
          "request": {
            "method": "POST",
            "header": [],
            "body": {
              "mode": "formdata",
              "formdata": [
                {
                  "key": "audio",
                  "type": "file",
                  "src": []
                }
              ]
            },
            "url": "https://arabic-letters-api-d26k2plh4q-ew.a.run.app/transcribe_letter"
          }
        }
      ]
    }
  ]
}
```

---

## ✅ Summary

### Arabic Words API
✅ **Base URL**: `https://arabic-words-api-d26k2plh4q-ew.a.run.app`
✅ **Health**: `/health`
✅ **Transcribe**: `POST /transcribe_word`
✅ **Verify**: `POST /verify_word`

### Arabic Letters API
✅ **Base URL**: `https://arabic-letters-api-d26k2plh4q-ew.a.run.app`
✅ **Health**: `/health`
✅ **Transcribe**: `POST /transcribe_letter`
✅ **Verify**: `POST /verify_letter`

**Both services are live and ready for testing!** 🚀
