# Postman Guide for Arabic Word Recognition API

## Hugging Face Space URL

Your Hugging Face Space: https://huggingface.co/spaces/yansari/arabic-word-recognition

## API Endpoints

### Base URL
For Hugging Face Spaces, use one of these formats:
- `https://yansari-arabic-word-recognition.hf.space`
- `https://huggingface.co/spaces/yansari/arabic-word-recognition`

### Available Endpoints

#### 1. Health Check
- **Method**: `GET`
- **URL**: `https://yansari-arabic-word-recognition.hf.space/health`
- **Response**: `{"status": "healthy"}`

#### 2. Root/Info
- **Method**: `GET`
- **URL**: `https://yansari-arabic-word-recognition.hf.space/`
- **Response**: API information and available endpoints

#### 3. Transcribe Word (Main Endpoint)
- **Method**: `POST`
- **URL**: `https://yansari-arabic-word-recognition.hf.space/transcribe_word`
- **Body Type**: `form-data`
- **Parameters**:
  - `audio` (File) - Select your WAV audio file

**Example Response:**
```json
{
  "transcription": "اللَّهِ",
  "confidence": 94.7,
  "latency_ms": 123.45,
  "model": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
}
```

## Postman Setup

### Step 1: Create New Request
1. Open Postman
2. Create a new POST request
3. Set method to `POST`

### Step 2: Set URL
```
https://yansari-arabic-word-recognition.hf.space/transcribe_word
```

### Step 3: Configure Body
1. Go to **Body** tab
2. Select **form-data**
3. Add key: `audio`
4. Change type from "Text" to **"File"** (dropdown on the right)
5. Click **Select Files** and choose your WAV audio file

### Step 4: Send Request
Click **Send** button

## Troubleshooting

### 404 Error
If you get a 404 error, check:
1. **URL Format**: Make sure you're using the correct URL format:
   - ✅ `https://yansari-arabic-word-recognition.hf.space/transcribe_word`
   - ❌ `https://huggingface.co/spaces/yansari/arabic-word-recognition/transcribe_word` (may not work)

2. **Space Status**: Ensure the Space is running (not sleeping)
   - Visit: https://huggingface.co/spaces/yansari/arabic-word-recognition
   - The Space should show "Running" status

3. **Endpoint Path**: Verify the endpoint is `/transcribe_word` (not `/transcribe` or `/api/transcribe_word`)

### 500 Error
If you get a 500 error:
- Check that your audio file is in WAV format
- Ensure the audio file is not corrupted
- Check the Space logs for detailed error messages

### Model Loading
- First request may take longer as the model loads
- Subsequent requests will be faster

## Testing with curl

```bash
curl -X POST "https://yansari-arabic-word-recognition.hf.space/transcribe_word" \
  -F "audio=@your_audio_file.wav"
```

## API Documentation

Once the Space is running, you can access interactive API docs at:
- `https://yansari-arabic-word-recognition.hf.space/docs`
- `https://yansari-arabic-word-recognition.hf.space/redoc`

