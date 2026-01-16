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

### Optional Parameters:

| Key | Type | Description | Default | Example |
|-----|------|-------------|---------|---------|
| `threshold` | **Text** (float) | Confidence threshold (0.0 to 1.0) | `0.6` | `0.8` |
| `fuzzy_match` | **Text** (boolean) | Enable fuzzy matching for minor variations | `true` | `false` |
| `fuzzy_threshold` | **Text** (float) | Custom fuzzy threshold (0-100), overrides dynamic threshold | Auto | `85.0` |

**Notes:**
- `threshold` controls audio confidence (0.0-1.0 scale)
- `fuzzy_match` enables smart matching that tolerates 1-2 character differences
- `fuzzy_threshold` allows custom similarity threshold (0-100 scale)
- When `fuzzy_match=true`, dynamic thresholds adapt to word length:
  - 1-2 chars: 95% (very strict)
  - 3-4 chars: 85% (strict)
  - 5-8 chars: 80% (moderate)
  - 9+ chars: 75% (lenient)
- `target_word` is automatically normalized (diacritics, hamza forms)

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

**Returns result with similarity score:**

```json
{
  "result": true,
  "similarity": 100.0
}
```

or

```json
{
  "result": false,
  "similarity": 65.2
}
```

**Fields:**
- `result` (boolean) - True if word matches AND confidence >= threshold, False otherwise
- `similarity` (float) - Fuzzy match similarity score (0-100)
  - 100.0 = perfect match
  - 85-99 = very similar (minor variations)
  - 50-84 = somewhat similar
  - 0-49 = different words

**Logic:**
- `result: true` - Both conditions met:
  1. Fuzzy similarity >= dynamic threshold (or custom threshold)
  2. Audio confidence >= `threshold`
- `result: false` - Either condition failed

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

### Example 1: Exact Match with Fuzzy Matching Enabled (Default)

**Postman Setup:**
- `audio`: `allah_audio.wav` (File)
- `target_word`: `اللَّهِ` (Text)
- `threshold`: `0.6` (Text)
- `fuzzy_match`: `true` (Text) - *optional, default is true*

**Expected Response:**
```json
{
  "result": true,
  "similarity": 100.0
}
```
(Perfect match - audio transcribes to "الله" which matches target after normalization)

### Example 2: Minor Variation with Fuzzy Matching

**Postman Setup:**
- `audio`: `word_audio.wav` (File) - Transcribes to "اللة" (1-char difference)
- `target_word`: `الله` (Text)
- `threshold`: `0.6` (Text)
- `fuzzy_match`: `true` (Text)

**Expected Response:**
```json
{
  "result": true,
  "similarity": 87.5
}
```
(Fuzzy match passes - 87.5% similarity exceeds 85% threshold for 4-char word)

### Example 3: Strict Exact Matching (Fuzzy Disabled)

**Postman Setup:**
- `audio`: `word_audio.wav` (File) - Transcribes to "اللة"
- `target_word`: `الله` (Text)
- `threshold`: `0.6` (Text)
- `fuzzy_match`: `false` (Text)

**Expected Response:**
```json
{
  "result": false,
  "similarity": 0.0
}
```
(Exact match fails - "اللة" != "الله" when fuzzy matching is disabled)

### Example 4: Custom Fuzzy Threshold

**Postman Setup:**
- `audio`: `word_audio.wav` (File)
- `target_word`: `مِنَ` (Text)
- `threshold`: `0.6` (Text)
- `fuzzy_match`: `true` (Text)
- `fuzzy_threshold`: `90.0` (Text) - *Very strict custom threshold*

**Expected Response:**
```json
{
  "result": false,
  "similarity": 87.5
}
```
(Fails custom threshold - 87.5% < 90% even though it would pass default 85%)

---

## 🔍 How It Works

1. **Audio Processing:**
   - Loads and normalizes the audio file
   - Converts to 16kHz mono

2. **Transcription:**
   - Uses Wav2Vec2-Large-XLSR-53-Arabic model to transcribe audio
   - Gets predicted Arabic text

3. **Confidence Calculation:**
   - Calculates confidence score (0.0 to 1.0)
   - Based on model's probability distribution

4. **Text Normalization:**
   - Removes diacritics (tashkeel: َ ِ ُ ْ ّ ً ٌ ٍ)
   - Normalizes hamza forms (أ, إ, آ → ا)
   - Removes tatweel (kashida: ـ)
   - Normalizes whitespace

5. **Fuzzy Matching (if enabled):**
   - Calculates similarity score using RapidFuzz (0-100 scale)
   - Applies dynamic threshold based on word length:
     - Short words (1-2 chars): 95% threshold
     - Medium words (3-4 chars): 85% threshold
     - Long words (5-8 chars): 80% threshold
     - Very long words (9+ chars): 75% threshold
   - Uses both Levenshtein and partial ratio for best match
   - Custom threshold overrides automatic thresholds if provided

6. **Verification:**
   - Checks if fuzzy similarity >= threshold (or exact match if fuzzy disabled)
   - Checks if audio confidence >= `threshold`
   - Returns `true` only if BOTH conditions are met

7. **Response:**
   - Returns `result` (boolean) and `similarity` score (0-100)

---

## ✅ Summary

| Component | Details |
|-----------|---------|
| **Endpoint** | `/verify_word` |
| **Method** | `POST` |
| **Required Keys** | `audio` (File), `target_word` (Text) |
| **Optional Keys** | `threshold` (0.6), `fuzzy_match` (true), `fuzzy_threshold` (auto) |
| **Response** | `{"result": true/false, "similarity": 0-100}` |
| **Matching** | Dynamic fuzzy matching with length-adaptive thresholds (95%/85%/80%/75%) |
| **Normalization** | Automatic diacritics, hamza, and tatweel removal |
| **Backend** | RapidFuzz (preferred) with difflib fallback |

**The API is ready to use! 🎉**

## 🆕 What's New - Fuzzy Matching v2.1

- **Smart Tolerance**: Automatically accepts minor 1-2 character variations
- **Dynamic Thresholds**: Stricter for short words, more lenient for long words
- **Backward Compatible**: Set `fuzzy_match=false` for exact matching
- **Custom Control**: Override thresholds with `fuzzy_threshold` parameter
- **Transparent**: Returns `similarity` score in every response

