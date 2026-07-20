# HANDOVER — Fix the Arabic Word-Pronunciation Model (code-red)

> **You are Claude Code.** This document is your brief, not human prose. Read it fully, build a todo list, then
> execute. The diagnosis below is already proven — **do not re-derive it**; use it. Confirm the plan with the human
> before any production action. Prefer small, verifiable steps.

---

## 0. MISSION
The Kutuby children's app verifies a child's spoken Arabic word against a **known target** from a fixed **191-word**
curriculum via a hosted model. It is in **code-red**: ~4% of real attempts pass. Root cause is proven: the deployed word
model is a **Quran-domain constrained ASR** that structurally can't recognize 36% of the curriculum. A benchmark proved
the fix. **Your job:** implement the fix in the `arabic-words-api` service, **validate it locally against the test set**,
and prepare (do NOT execute) the deploy. Also produce the two smaller server-side matching fixes.

**Deliverable:** a working local `MODEL_VARIANT=fastconformer` backend in the words-API that scores ≥ the acceptance
numbers in §7, plus the matching fixes, plus a short PR-ready summary. Nothing deployed/pushed without explicit human OK.

---

## 1. HARD CONSTRAINTS (do not violate)
- **DO NOT** `git push`, deploy to Cloud Run, rotate keys, or change the live service **without the human explicitly
  approving**. This is a production service under a code-red.
- Work **locally**. Downloading the model from Hugging Face is fine (the model is `CC-BY-4.0`, no token needed).
- Keep the existing `tarteel` and `legacy` backends intact — add `fastconformer` as a **new selectable variant**
  (`MODEL_VARIANT` env), so rollback is trivial.
- The **letters** service (`arabic-letters-api`) is healthy (~93%) and **out of scope** — do not touch it.
- Never commit secrets. The words service needs no API keys.

---

## 2. PROVEN DIAGNOSIS (context — already established, don't reinvestigate)
Benchmark: 3 models, endpoint-faithful, on a controlled 191-word × 6-voice TTS test set (1,146 word clips), run through
a local replica of the real `/verify_word` pipeline. **WORD pass rate (clean audio):**

| Model | exact | fuzzy | out-of-lexicon | latency p50 |
|---|---|---|---|---|
| **Tarteel** `whisper-base-ar-quran` (CURRENT) | 25.1% | 37.0% | 3.4% native | ~3900 ms |
| Legacy `wav2vec2-large-xlsr-53-arabic` | 58.4% | 67.6% | 65.4% | 199 ms |
| **FastConformer** `stt_ar_fastconformer_hybrid_large_pc` | **64.0%** (73.9% w/ punct-strip) | **68.2%** (76.8% w/ punct-strip) | 58.8% | 265 ms |

- **Root cause:** the word model uses **lexicon-constrained decoding over a 14,752-word Quranic lexicon**; **68 of the 191
  curriculum words are not in the Quran** → the decoder can't emit them → 3.4% on those. A **general MSA** model removes the
  constraint (51% of FastConformer's gain over Tarteel is purely those out-of-lexicon words).
- **FastConformer is the chosen replacement** — best under every matching threshold (exact/≥85/≥90/≥95), ~15× faster,
  CC-BY-4.0, CTC (no short-clip hallucination). Independently adversarially reviewed; zero semantic false-accepts.
- Three **endpoint-level issues** were found and are part of this fix (see §5): the fuzzy-threshold scale bug, FastConformer's
  trailing punctuation, and (client-side, separate track) Android narrowband audio.

Full report + raw data live on the human's machine (`Kutuby/test set/MODEL_COMPARISON_REPORT.md`, `*_raw.csv`,
`comparison_summary.json`). Ask the human to share the `test set/` and `asr_bench/` folders — they let you skip
regeneration and validate immediately (§6).

---

## 3. SYSTEM MAP
- **Service to change:** `arabic-words-api` — FastAPI, Cloud Run, GCP project `organic-duality-484219-p5`, region
  `europe-west1`, URL `https://arabic-words-api-d26k2plh4q-ew.a.run.app`. `minScale=1, maxScale=10, concurrency=160`,
  2 vCPU / 4 GiB, `MODEL_VARIANT=tarteel`. Source deploys from `gs://run-sources-organic-duality-484219-p5-europe-west1`
  (or the team's git repo — ask the human for the canonical source location; the deployed source can be pulled from that
  bucket if needed).
- **The `/verify_word` contract (unchanged, do not break):** `POST multipart/form-data`, fields `target_word`
  (vocalized Arabic, e.g. `بَقَرَة`), `audio` (recording.wav), `threshold` (0.6), `fuzzy_match` (true),
  `fuzzy_threshold` (0.85). Response JSON — **the app consumes only `result` (bool)**; keep the same response shape.
- **Key files** (relative to the words-API source root):
  - `app.py` — FastAPI. `load_audio_robust` (line ~59), `verify_word` handler (~561), silence gate (~610), peak-normalize
    (~630), backend dispatch (~634–642).
  - `core/inference.py` — the backends. `InferenceBackend` protocol (~57), `TarteelBackend` (~69), `LegacyBackend` (~287),
    `get_backend()` variant selector (~406–425), `DEFAULT_MODEL_VARIANT="tarteel"` (~40).
  - `core/arabic_utils.py` — `normalize_arabic_text` (~129), `fuzzy_match_arabic_words` (~246), `get_dynamic_threshold`
    (~212), `check_weak_ending_match` (~62).
  - `Dockerfile` (python:3.10-slim, pre-downloads the tarteel model), `requirements.txt`.

---

## 4. THE FIX — implement in this order

### Step A — add a `FastConformerBackend` to `core/inference.py`
Mirror the existing backend interface (`variant`, `model_name`, `transcribe(audio)->dict`, `verify(audio, target_word, …)->dict`).
Load `nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0` and **use the CTC decoder** (fast, no hallucination). Skeleton:

```python
# core/inference.py  — new backend, register alongside Tarteel/Legacy
FASTCONFORMER_MODEL_NAME = "nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0"

class FastConformerBackend:
    variant = "fastconformer"
    model_name = FASTCONFORMER_MODEL_NAME
    def __init__(self):
        import os; os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # torch>=2.6 loads the .nemo
        import nemo.collections.asr as nemo_asr
        self.model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(FASTCONFORMER_MODEL_NAME)
        self.model.change_decoding_strategy(decoder_type="ctc")   # CTC greedy = fastest on CPU
        # warmup on 1s of zeros
        import numpy as np; self.model.transcribe([np.zeros(16000, dtype=np.float32)], batch_size=1, verbose=False)

    def transcribe(self, audio):  # audio: float32 mono 16k numpy
        import time, numpy as np
        t = time.time()
        h = self.model.transcribe([audio.astype(np.float32)], batch_size=1, verbose=False)[0]
        text = (h.text if hasattr(h, "text") else str(h)).strip()
        return {"transcription": text, "confidence": 100.0, "latency_ms": round((time.time()-t)*1000, 2)}

    def verify(self, audio, target_word, top_k=1, fuzzy_match=True, fuzzy_threshold=None):
        from core.arabic_utils import fuzzy_match_arabic_words, normalize_arabic_text
        r = self.transcribe(audio)
        transcription = _strip_trailing_punct(r["transcription"])          # see Step C
        if fuzzy_match:
            matched, similarity = fuzzy_match_arabic_words(transcription, target_word.strip(), custom_threshold=None)  # DYNAMIC — see Step B
        else:
            matched = normalize_arabic_text(transcription) == normalize_arabic_text(target_word.strip())
            similarity = 100.0 if matched else 0.0
        return {"result": bool(matched), "transcription": transcription, "target_word": target_word.strip(),
                "similarity": round(similarity, 2), "confidence": r["confidence"], "score": None,
                "top_k_candidates": None, "threshold": None, "latency_ms": r["latency_ms"]}
```
Register it in `get_backend()` (~line 415): add `elif variant == "fastconformer": _BACKEND = FastConformerBackend()`.

### Step B — fix the `fuzzy_threshold` scale bug (endpoint bug)
The app sends `fuzzy_threshold="0.85"`. The server passes it to `fuzzy_match_arabic_words(custom_threshold=0.85)`, which
compares on a **0–100** scale → `0.85` accepts almost anything (over-permissive, false-accept-prone). **Fix: ignore the
client value and use the dynamic per-length thresholds** (`custom_threshold=None`). Do this in the new backend (above) and,
for consistency, in `app.py verify_word` dispatch (~634–642): route `fastconformer` like `legacy` but **force
`fuzzy_threshold=None`** when calling `verify`. (Do NOT change the request contract; just stop honoring the 0.85.)

### Step C — strip trailing punctuation (the `pc` model appends `.` / `،`)
FastConformer's `pc` variant emits sentence punctuation that is NOT a pronunciation error. Add a small helper and apply it
before matching (used in Step A). Either add a dedicated `_strip_trailing_punct(s)` in `inference.py`, or add the
punctuation chars to `normalize_arabic_text` in `core/arabic_utils.py` (~129). Recommended minimal helper:
```python
import re
def _strip_trailing_punct(s): return re.sub(r"[.,،؟?!…\":;()\-_/]+", " ", s or "").strip()
```
(Adding it inside `normalize_arabic_text` is cleaner and low-risk — tarteel/legacy emit almost no punctuation — but verify
you don't regress their pass rates in §6.)

### Step D — packaging (Docker / deploy prep — do NOT deploy)
NeMo is a heavy dependency. **Two options — recommend the ONNX path for the Cloud Run image:**
- **Option 1 (fastest to a working prototype): NeMo in the image.** Add `nemo_toolkit[asr]` to `requirements.txt`
  (and `torch` CPU). Bigger image (~GBs), slower cold start. Fine for a first validated build; keep `minScale≥1`.
- **Option 2 (recommended for prod): export to ONNX once, run with `sherpa-onnx` (no NeMo/torch in the image).**
  On any machine where NeMo installs, export the CTC branch:
  `sherpa-onnx/scripts/nemo/fast-conformer-hybrid-transducer-ctc/export-onnx-ctc-non-streaming.py --model nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0`
  → `model.int8.onnx` + `tokens.txt`. Commit those; the container installs only `sherpa-onnx` (tiny, arm64+linux CPU
  wheels) and runs `OfflineRecognizer.from_nemo_ctc(...)`. Smaller image, faster CPU, no torch. Update the `Dockerfile`
  pre-download step accordingly (or COPY the .onnx into the image).
Keep the existing `ENV MODEL_VARIANT=tarteel` default; select the new backend with `MODEL_VARIANT=fastconformer` (only flip
it in prod after the human approves).

---

## 5. THE OTHER TWO FIXES (in scope, server-side)
- **Matching:** done via Steps B+C (dynamic fuzzy + punct strip). This recovers "right word, dropped final letter / added a
  space / weak ending" near-misses without false-accepts.
- **(Separate track, client-side — note but don't do here):** Android records compressed narrowband audio
  (`AndroidOutputFormat.DEFAULT` = 3GP/AMR mislabeled `.wav`), which costs **~20–27 pts for every model**. The app should
  record **real 16 kHz PCM WAV** (explicit WAV encoder). File this for whoever owns the RN app
  (`MainAppV1/src/hooks/useVoiceVerifier.tsx` recording config, ~line 234). Not part of the words-API change.

---

## 6. VALIDATION — reproduce the benchmark locally (this is how you know the fix works)
**Method:** run each test clip through a local replica of the `/verify_word` pipeline with the new backend; match the
transcription to the target; compute pass rate. Two ways to get the harness + data:
- **Preferred:** ask the human for the `Kutuby/test set/` folder (1,146 word clips + `manifest.csv`) and the `Kutuby/asr_bench/`
  harness (`common_pipeline.py`, `run_bench.py`, `analyze.py`, `compare.py`). Then a FastConformer run is one command
  (mirror `run_bench.py --model fastconformer`) and `analyze.py` prints exact/fuzzy pass rates.
- **If not shared:** the test set is TTS-generated — regenerate a small held-out set (even 30–50 words × 1 voice is enough
  to sanity-check) with OpenAI `gpt-4o-mini-tts` (mp3, `input=wordWithTashkeel`), or record a few yourself. The harness core
  is: `load_audio_robust(bytes)` (copy from `app.py`) → silence gate `peak<0.005` → `y/=peak` → `backend.transcribe(y)` →
  strip punct → `fuzzy_match_arabic_words(transcription, target, None)`. Word list = the 191 `wordWithTashkeel` strings from
  the app's `src/data/ArabicWords.tsx`.

**Local env (Apple-Silicon mac, py3.12) — the recipe that works (see PITFALLS):**
```bash
python3.12 -m venv .venv_nemo && . .venv_nemo/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU torch FIRST (avoids triton)
pip install "nemo_toolkit[asr]" soundfile librosa rapidfuzz pydub numpy
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

**ACCEPTANCE CRITERIA (on the 191-word test set, clean audio; must beat tarteel's 25.1%):**
- FastConformer word pass rate: **exact ≥ 62%, fuzzy ≥ 66%** (server-faithful, no punct-strip); with the punct-strip:
  **exact ≥ 72%, fuzzy ≥ 75%**. (Reference measured: 64.0/68.2 → 73.9/76.8.)
- **Out-of-lexicon** words (the 68 non-Quranic ones): **≥ 55% exact** (tarteel is 3.4%).
- Model inference latency **< 400 ms** per 1–2 s clip on CPU (CTC decode).
- **False-accept < 6%** (each transcription fuzzy-matched against all 191 targets should hit ~its own only).
- **No regression** on tarteel/legacy variants if you touched `normalize_arabic_text`.
Also verify the `fuzzy_threshold=0.85` no longer inflates results (dynamic thresholds active).

---

## 7. PITFALLS (learned the hard way — avoid these)
- **NeMo on arm64 mac:** install **CPU torch FIRST**, then `nemo_toolkit[asr]` (never `[all]`, never `reinstall.sh`) — else
  it pulls `triton` which has no arm64 wheel and fails. Use **Python 3.12** (3.13 breaks `kaldialign`). Set
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (torch≥2.6 refuses to load the `.nemo` otherwise).
- **FastConformer emits punctuation** (trailing `.`) — must strip before matching or you silently lose ~9 pts (Step C).
- **`fuzzy_threshold=0.85` is on a 0–100 scale** in `fuzzy_match_arabic_words` → passing 0.85 ≈ "accept everything." Use
  `custom_threshold=None` (dynamic) (Step B).
- **mp3 decode:** newer `libsndfile` reads mp3 via `soundfile`; older (the server's) can't → uses the `ffmpeg` fallback.
  Both yield equivalent 16 kHz audio; keep `load_audio_robust`'s fallback chain intact.
- **Silence gate (`peak<0.005`)** rejects ~0.36 s near-empty clips before the model — keep it; it fails equally for all models.
- **NeMo single-clip `transcribe()` has per-call overhead**; under CPU contention (another torch job running) wall-time
  balloons though `model_ms` stays ~270 ms. For a validation loop, run it alone; for prod, keep the model warm (`minScale≥1`)
  and consider batching / ONNX.
- **Don't over-trust the absolute %:** the test set is **adult TTS** — an optimistic ceiling. Real 4–10 yo audio is lower for
  every model. The **ranking** is robust; treat absolute numbers as upper bounds and plan a real-child eval / fine-tune later.

---

## 8. OUT OF SCOPE / SEPARATE TRACKS (mention in your summary, don't implement)
- **Android real-WAV recording** (client RN app) — ~+20 pts, model-independent (see §5).
- **Fine-tune FastConformer on real child audio** already being logged to Supabase bucket `pronunciation-recordings` — the
  durable accuracy ceiling; adult TTS over-estimates kids by ~3–4×.
- **Letters model** — healthy (~93%), untouched.
- **Per-phoneme "goodness of pronunciation"** (IqraEval / wav2vec2-xls-r + MSA phonetiser) — a future upgrade that gives
  per-letter feedback; not needed for the code-red.

---

## 9. REFERENCES
- Model: `nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0` — HF model card; **License CC-BY-4.0** (commercial OK, attribution).
  Class `EncDecHybridRNNTCTCBPEModel`, ~115M, 16 kHz mono, no diacritics, CTC or RNNT decode. `_pcd` variant adds diacritics +
  Quranic acoustics if the 99 Names/classical terms need it.
- Benchmark report + raw CSVs + harness: on the human's machine under `Kutuby/test set/` and `Kutuby/asr_bench/` (ask for them).
- ONNX export: `github.com/k2-fsa/sherpa-onnx` → `scripts/nemo/fast-conformer-hybrid-transducer-ctc/`.

---

## 10. YOUR PROCESS
1. Read this doc; make a todo list. 2. Get the model + test data (download model; ask human for `test set/` + `asr_bench/`).
3. Implement Steps A–C locally behind `MODEL_VARIANT=fastconformer`. 4. Validate against §7 acceptance criteria; iterate.
5. Package (Step D) — prefer ONNX for prod. 6. Write a concise PR summary (what changed, before/after numbers, rollback =
`MODEL_VARIANT=tarteel`). 7. **STOP and ask the human** before pushing/deploying. Report results with the measured numbers.
```
