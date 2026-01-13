# Verify Word API - Complete Documentation

## ✅ New Endpoint: `/verify_word`

This endpoint verifies if an audio file matches a target Arabic word and exceeds a confidence threshold.

---

## 📍 Endpoint Details

**Method:** `POST`  
**URL:** `https://yansari-arabic-word-recognition.hf.space/verify_word`  
**Content-Type:** `multipart/form-data`

---

## 🔑 POST Request Keys (Parameters)

### Required Parameters:

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `audio` | **File** | WAV audio file containing spoken Arabic word | `audio.wav` |
| `target_word` | **Text** | The expected Arabic word to match | `اللَّهِ` |
| `threshold` | **Text** (float) | Confidence threshold (0.0 to 1.0) | `0.6` |

**Notes:**
- `threshold` is optional, defaults to `0.6` (60%)
- `threshold` must be between 0.0 and 1.0
- `target_word` should be the exact Arabic text you expect

---

## 📤 Request Format in Postman

1. **Method:** Select `POST`
2. **URL:** `https://yansari-arabic-word-recognition.hf.space/verify_word`
3. **Body Tab:**
   - Select `form-data`
   - Add three keys:
     - `audio` (Type: **File**) - Select your `.wav` file
     - `target_word` (Type: **Text**) - Enter Arabic word, e.g., `اللَّهِ`
     - `threshold` (Type: **Text**) - Enter number, e.g., `0.6`

---

## 📥 Response Payload

### Success Response (200 OK)

**Returns a simple boolean result:**

```json
{
  "result": true
}
```

or

```json
{
  "result": false
}
```

**Logic:**
- `result: true` - Transcription matches `target_word` AND confidence >= `threshold`
- `result: false` - Transcription does NOT match `target_word` OR confidence < `threshold`

### Error Response (400 Bad Request)

```json
{
  "result": false,
  "error": "Error message here"
}
```

**Error cases:**
- Invalid audio file format
- Empty audio file
- Threshold out of range (not 0.0-1.0)

### Error Response (500 Internal Server Error)

```json
{
  "result": false,
  "error": "Transcription failed: ..."
}
```

---

## 🧪 Example Requests

### Example 1: Verify "اللَّهِ" with 60% threshold

**Postman Setup:**
- `audio`: `allah_audio.wav` (File)
- `target_word`: `اللَّهِ` (Text)
- `threshold`: `0.6` (Text)

**Expected Response:**
```json
{
  "result": true
}
```
(If audio matches "اللَّهِ" and confidence >= 60%)

### Example 2: Verify "مِنَ" with 80% threshold

**Postman Setup:**
- `audio`: `min_audio.wav` (File)
- `target_word`: `مِنَ` (Text)
- `threshold`: `0.8` (Text)

**Expected Response:**
```json
{
  "result": false
}
```
(If audio doesn't match "مِنَ" OR confidence < 80%)

---

## 🔍 How It Works

1. **Audio Processing:**
   - Loads and normalizes the audio file
   - Converts to 16kHz mono

2. **Transcription:**
   - Uses Wav2Vec2 model to transcribe audio
   - Gets predicted Arabic text

3. **Confidence Calculation:**
   - Calculates confidence score (0.0 to 1.0)
   - Based on model's probability distribution

4. **Verification:**
   - Checks if transcription exactly matches `target_word`
   - Checks if confidence >= `threshold`
   - Returns `true` only if BOTH conditions are met

---

## ✅ Summary

| Component | Details |
|-----------|---------|
| **Endpoint** | `/verify_word` |
| **Method** | `POST` |
| **Required Keys** | `audio` (File), `target_word` (Text), `threshold` (Text, optional) |
| **Response** | `{"result": true/false}` |
| **Logic** | Returns `true` if transcription matches target_word AND confidence >= threshold |

**The API is ready to use! 🎉**

