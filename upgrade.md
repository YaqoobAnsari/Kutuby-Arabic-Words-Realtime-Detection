# Arabic Word Verification — Upgrade Plan

**Status:** Strategic plan (not yet implemented)
**Date:** 2026-05-25
**Owner:** Kutuby Engineering
**Target service:** `arabic-words-realtime` (Cloud Run + HuggingFace Space)

---

## 1. Why this document exists

The current production API (`app.py` + `core/arabic_utils.py`) works, but it works by *covering up* a model problem with a fuzzy-matching band-aid. This document is the durable record of:

- **What the band-aid is** and why it must come off
- **What we are replacing it with** and why that's the principled fix
- **The full landscape of models and decoding strategies** we considered
- **Publicly available datasets** and their honest limitations
- **Cost, latency, and accuracy expectations** at each phase
- **The phased rollout plan** with validation gates

Future-you (or a new engineer) should be able to read this end-to-end and understand both the *what* and the *why* without re-deriving it.

---

## 2. The current state (the band-aid)

### What's deployed today

- **Model:** `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` (~315M params, general MSA, Common Voice trained)
- **Pipeline:** audio → FFmpeg decode → Wav2Vec2 greedy CTC decode → Arabic normalization (tashkeel/hamza/tatweel) → fuzzy string match against target word with dynamic thresholds (98 / 90 / 85 / 80 / 75 by length) → boolean verdict
- **Deployment:** Google Cloud Run europe-west1 (5Gi RAM, 2 vCPU), plus HuggingFace Space mirror
- **End-to-end latency on CPU:** ~3–4s per request (down from ~25s after the FFmpeg pipe optimization)

### What's actually wrong

The system is doing **free-vocabulary transcription** when the real task is **closed-vocabulary verification**. Three concrete failure modes the fuzzy matcher is hiding:

1. **Weak Arabic endings** (`ة`, `ه`, `ا`, `ى`) — the Wav2Vec2 model frequently drops them because they're phonetically weak in casual speech. We compensate by auto-passing weak-ending mismatches at 95% similarity. This is honest *only* in casual MSA — in Tajweed/Tartil recitation, these endings often *are* pronounced and should *not* be hand-waved.
2. **Diacritics dropped on output** — the model emits un-diacritized text, so even a perfect Tajweed recitation never matches a diacritized target. We strip tashkeel from both sides to normalize, which throws away exactly the signal Quranic verification needs.
3. **Off-by-one-character substitutions** — the model swaps phonetically-near letters (e.g., `ك`/`ق`, `ح`/`خ`). We accept these with `fuzz.ratio` thresholds; in a learning app for kids, this is the worst kind of false positive (rewards the wrong pronunciation).

### Why the band-aid was correct then and is wrong now

When the project was a demo, "approximately the right word" was a reasonable bar. As the product targets **kids learning Quranic recitation**, "approximately the right word" is actively harmful — it teaches sloppy pronunciation. The verification semantics need to flip from "looks similar enough" to "is acoustically this word".

---

## 3. The target architecture

**One sentence:** Use a Quranic-domain ASR model with a decoder that is constrained to the 6k-word Quranic lexicon, eliminate fuzzy string matching entirely, and calibrate honest thresholds against a public Quranic audio corpus — with a clear path to closing the kid-voice gap via opt-in recording collection.

### Pipeline (target)

```
audio bytes
    │
    ▼
FFmpeg pipe decode  ────────────────────────  (unchanged, already fast)
    │
    ▼
16kHz mono float32
    │
    ▼
Tarteel whisper-base-ar-quran encoder + decoder
    │   (Quranic-domain acoustic prior; diacritized output)
    ▼
Lexicon-constrained beam decode
    │   (output forced to be one of ~6k Quranic words via
    │    prefix_allowed_tokens_fn over a trie of valid tokens)
    ▼
Top-K Quranic word candidates with log-probabilities
    │
    ▼
Verdict
    ├─ result = (top-1 == target_word) AND (score >= calibrated_threshold)
    ├─ similarity is now an honest log-probability, not fuzz.ratio
    └─ optional: forced-alignment pass returns per-letter scores
                 for diagnostic UI (phase 4)
```

### What's gone

- `core/arabic_utils.fuzzy_match_arabic_words` — replaced
- `check_weak_ending_match` weak-ending shortcut — replaced (the model now handles this correctly because Tarteel is trained on diacritized recitation)
- Length-based dynamic thresholds (98/90/85/80/75) — replaced with measured per-word thresholds
- The legacy Streamlit files (`arabic_word_identifier.py`, `core/audio_recorder.py`, `core/transcriber.py`, `core/model_loader.py`) — cleanup opportunity during the migration

### What's preserved

- `load_audio_robust` — the FFmpeg-pipe path that took us from 25s → 2s. Keep as-is.
- `normalize_arabic_text` — still useful for normalizing the target word string before comparison
- The FastAPI surface (`/health`, `/transcribe_word`, `/verify_word`) — same endpoints, same request shape. Response gains an honest `score` and `top_k_candidates` field.

---

## 4. Model options considered

We surveyed the landscape across three axes: **Quranic-domain fit**, **kid-voice fit**, and **CPU latency**. The recommended model is bold.

| Model | Params | Quranic domain | Kid voices | CPU latency (~1s audio) | Integration | Verdict |
|---|---|---|---|---|---|---|
| `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` *(current)* | 315M | ❌ general MSA | ⚠️ untested | ~3.5s | Already deployed | The status quo we're replacing |
| **`tarteel-ai/whisper-base-ar-quran`** ⭐ | 74M | ✅ Quran-fine-tuned | ⚠️ adult-leaning training data | ~0.7–0.9s | Small (HF model swap) | **Recommended.** Best ratio of domain fit, speed, size. |
| `tarteel-ai/whisper-tiny-ar-quran` | 39M | ✅ same family | ⚠️ same | ~0.4–0.6s | Small | Fallback if `base` is too slow under real load. |
| `openai/whisper-large-v3` | 1.5B | ❌ general | ✅ broad multilingual | ~10–15s on CPU | Small | Too slow on CPU without GPU. Holds the general-Arabic accuracy crown but not domain-tuned. |
| `openai/whisper-medium` | 769M | ❌ general | ✅ | ~5–8s on CPU | Small | Diminishing returns vs. Tarteel for our specific use case. |
| `facebook/wav2vec2-xlsr-53-espeak-cv-ft` | 300M | N/A (IPA output) | ✅ | ~3–4s | Medium (need Quranic G2P) | Strong for phoneme-level verification; pair with rule-based Quranic G2P. Held as a backup architecture if Tarteel under-performs. |
| `facebook/mms-1b-all` + Quranic adapter | 1B + ~2M adapter | ✅ (if we adapter-train) | ✅ (if kid audio in training) | ~4–6s | High (training pipeline) | Adapter fine-tuning is ~10× cheaper than full fine-tune. Bridge option toward a custom model. |
| Fine-tune Tarteel-base ourselves | 74M | ✅✅ (best possible) | ✅✅ (if kid audio included) | ~0.7–0.9s | Very high (training + MLOps) | Phase 5 endpoint after we have collected kid recordings. |

### Why Tarteel whisper-base specifically

- **Acoustic prior is right:** trained on actual Quranic recitation audio, not Common Voice news clips. The model has "heard" Tajweed/Tartil patterns during training.
- **Outputs diacritized text:** matches the diacritization scheme of Quranic target words, eliminating the "strip tashkeel" hack.
- **Smaller than current:** 74M vs. 315M → faster inference, lower RAM, lower Cloud Run cost.
- **Free:** MIT-licensed on HuggingFace, no commercial restriction.
- **Reproducible upstream:** Tarteel AI is an established Quranic-tech non-profit; the model has community traction and isn't an abandoned research artifact.

### Known wrinkle to validate before committing

Tarteel's whisper-ar-quran was trained on *verse-level* audio, not isolated single words. The encoder doesn't care about utterance length, so single-word audio should still produce a single-word transcription — but this needs a 30-minute validation spike before we commit. **Fallback if it under-performs on single words:** use the encoder + a lightweight CTC head, or use the model in verification mode (compute logP of target sequence given audio) rather than free decode.

---

## 5. Decoding / matching strategies considered

Orthogonal to model choice. The recommended strategy is bold.

| Strategy | How it works | Kills fuzzy match? | Per-letter feedback? | Latency overhead | Best when |
|---|---|---|---|---|---|
| Free decode → fuzzy compare *(current)* | Greedy decode, normalize, `fuzz.ratio` threshold | ❌ *is* the fuzzy match | ❌ | None | Quick demo / open vocab |
| Constrained CTC scoring | Compute `logP(target_chars \| audio)` from CTC logits; threshold on score | ✅ | ❌ (one scalar) | None (same forward pass) | CTC models (wav2vec2 family); simplest principled fix |
| **Lexicon-constrained beam decode** ⭐ | Beam search restricted to ~6k Quranic words via `prefix_allowed_tokens_fn`; returns top-K with log-probs | ✅ | Partial (runner-up words) | ~100–200ms beam | **Closed vocabulary — our case.** Best signal-to-noise ratio. |
| Phoneme G2P + alignment | Quranic word → phoneme sequence; phoneme ASR → phonemes; compare sequences | ✅ | ✅ per-phoneme | None | When orthography ≠ pronunciation (Arabic ة, شدة, sukun) |
| Forced alignment + GOP scoring | Align expected phones to audio frames; "goodness-of-pronunciation" per phone | ✅ | ✅✅ best diagnostic | ~20–50ms DP align | Kids learning — diagnostic feedback layer (phase 4) |
| N-best + Quranic LM rescore | Decode top-N with Whisper; rerank against Quranic n-gram LM | Partial (replaces fuzz with LM) | ❌ | LM scoring ~30ms | When the lexicon is large; not needed at 6k words |

### Why lexicon-constrained beam decode

- **Can't produce a non-Quranic word:** the decoder is mechanically incapable of emitting a token sequence that isn't a valid Quranic word. The "wrong word entirely" failure mode disappears.
- **Top-K gives honest disambiguation:** if the audio is genuinely ambiguous between two near-homophone Quranic words, you see both candidates with their scores rather than a fuzz.ratio guess.
- **Calibrate once, applies everywhere:** the score is a real log-probability, so per-word-length calibration is mathematically meaningful (unlike thresholding `fuzz.ratio`, which is a heuristic).
- **Implementation is straightforward:** HuggingFace `transformers` supports `prefix_allowed_tokens_fn` on Whisper. Build a trie from the 6k lexicon at startup, query it during beam search.

### Phase 4 addition — forced alignment for per-letter feedback

After phase 1 ships, layering **forced alignment** on top of the same model's encoder outputs gives per-letter scores. This is the feature that turns the API from "yes/no" into a teaching tool — the UI can highlight which letter in `الْبَقَرَة` the kid mispronounced. Implementation: rule-based Quranic G2P (Quranic Arabic phonology is highly regular), then Viterbi alignment of expected phones to encoder frames. ~20–50ms overhead.

---

## 6. Datasets — what exists, what doesn't

We do **not** have a labelled in-house dataset of correct/incorrect Quranic recitations. The plan acknowledges that and uses public data for calibration.

### Available now (public)

| Dataset | Size | Reciters | Alignment | Kids? | Use for |
|---|---|---|---|---|---|
| `tarteel-ai/everyayah` (HuggingFace) | Hundreds of hours | ~12 professional reciters (Mishary, Sudais, Husary, Husari, etc.) | Verse-level (programmatically splittable to word) | ❌ adult male | **Primary calibration set.** Sets per-word thresholds for adult recitation. |
| `tarteel-ai/tarteel-1.0` | Tens of hours | Crowd-sourced Tarteel app users | Verse-level with quality labels | ❌ mostly adults | Real-world noise robustness; closer to production audio quality. |
| QuranicAudio.com mirror | Thousands of hours | Many professional reciters | Verse-level | ❌ adult male | Reciter-style augmentation if we ever fine-tune. |
| Common Voice Arabic 17+ | ~100+ hours | Open crowd | Sentence-level (not Quranic) | ⚠️ a few kids, not Arabic-specific | Out-of-domain robustness; future kid-voice augmentation pool. |

### What does not exist publicly

- **Kid Arabic Quranic recitation, labelled at word level.** As of knowledge cutoff (Jan 2026), no such public dataset exists. This is the data gap that determines the kid-voice accuracy ceiling.

### The honest implication

- **For adult recitation:** public data is enough. Expect 95–99% verification accuracy after lexicon-constrained decoding + threshold calibration.
- **For kid recitation:** public data alone gets us to roughly 85–93%. Closing the last 5–10% **requires** collecting our own kid recordings. There is no shortcut around this.

### The kid-data collection plan (starts in phase 3)

Add opt-in recording capture to the production app with explicit parental consent:

- Capture audio + target word + user-verdict (was this accepted/rejected) + a "this was actually correct/wrong" override the user can tap if they disagree with the model
- 500–1000 examples in a month is enough to start informing thresholds
- 10–50 hours is enough to fine-tune
- All recordings stored in a Cloud Storage bucket with per-record retention/deletion controls
- This is the **only** path to flawless kid-voice performance, so start it before you need the data

---

## 7. Cost analysis

Cloud Run cost is dominated by *active instance time × instance size*. Tarteel whisper-base is smaller and faster than what's deployed today, so the new stack is **cheaper, not more expensive**.

| Component | Current | After upgrade | Delta |
|---|---|---|---|
| Cloud Run memory tier | 5Gi | 2Gi sufficient | ~50% cheaper per active hour |
| Per-request CPU time | ~3,500ms | ~900ms | ~3.9× less billed compute per call |
| Model artifact storage | 1.2GB | 0.3GB | Negligible either way |
| Lexicon trie in RAM | — | <1MB | Negligible |
| Forced-alignment pass (phase 4) | — | +20–50ms | Marginal |
| GPU? | No | No (CPU latency is fine) | No new spend |
| One-time future fine-tune (phase 5) | — | ~$50–150 on a rented A100 for a day | One-shot, not ongoing |

**Net expectation:** Cloud Run bill drops by ~50% at equivalent traffic. The only cost line items that grow are (a) Cloud Storage for kid recordings (cents/month for the first year), and (b) eventually one-time fine-tune compute.

---

## 8. Latency projections (revised after spike)

Measured on local CPU during the validation spike. The original projection of ~0.7s was optimistic; Whisper-base's actual per-second-of-audio latency is closer to 1:1.3 on a typical CPU. For our use case (single-word audio, ~1s of speech), realistic numbers:

| Stage | Today | Projected (revised) | Notes |
|---|---|---|---|
| Audio decode (FFmpeg pipe) | ~150ms | ~150ms | Unchanged — already optimal |
| Model inference (1s audio) | ~3,500ms | ~1,300–1,800ms | Measured: 1.26s for ~0.5s audio ("الم" verse); scales linearly |
| Lexicon-constrained beam | — | ~100–200ms | 15.8k-word trie, beam width 5–10 |
| Score/threshold | ~5ms | ~5ms | Trivial |
| Forced-alignment (phase 4) | — | +20–50ms | Only when diagnostic mode enabled |
| **Total (single-word, 1s audio)** | **~3.8s** | **~1.5–2.2s** | **~2× faster** |

If 1.5s is still too slow under real load, the fallback is `whisper-tiny-ar-quran` (39M params, ~half the inference time) with a small accuracy hit. We commit to `base` first based on the spike's 100% transcription accuracy.

## 8b. Phase 1 implementation result (2026-05-25)

Phase 1 shipped functionally complete. End-to-end HTTP smoke test against the local FastAPI server passed all cases:

| Test | Latency (real Quranic audio, 2.65s clip on CPU) | Result |
|---|---|---|
| `GET /health` | <5ms | `{"status":"healthy","variant":"tarteel","model_name":"tarteel-ai/whisper-base-ar-quran"}` |
| `POST /transcribe_word` (real audio) | 1.9s | `"transcription":"مَالِكِ ي"` (correctly transcribed first word + start of next) |
| `POST /verify_word` (correct target `مَالِكِ`) | 1.9s | `result:true` `score:-0.0` `top:"مَالِكِ"` |
| `POST /verify_word` (wrong target `بَقَرَة`) | 1.9s | `result:false` `transcription:"مَالِكِ"` (lexicon match check caught it) |
| `POST /verify_word` (silence) | 4ms | `result:false` `rejection_reason:"audio_silent"` (amplitude gate, no inference run) |

For production single-word audio (~1s), expected latency is **~0.7–1.0s** on the same hardware. Cloud Run should match or improve on this.

### Lessons learned during implementation

These took several iterations to discover and are worth flagging for future-you:

1. **Lexicon source must match the model's training-data orthography.** `alquran.cloud/quran-uthmani` uses U+0670 super-alef (e.g., `مَٰلِكِ`), but Tarteel's `whisper-base-ar-quran` was trained on text using regular U+0627 alef (`مَالِكِ`). These tokenize to completely different BPE sequences — the trie was effectively empty of the words the model actually emits. Fix: switched to `alquran.cloud/quran-simple-enhanced`, which uses the same orthography Tarteel was trained on. **Don't change this without re-running the spike.**

2. **`prefix_allowed_tokens_fn` must return the FULL vocabulary during Whisper's decoder prompt prefix steps**, not just the lexicon-allowed tokens. The prompt is 4 tokens (`<|sot|>, <|ar|>, <|transcribe|>, <|notimestamps|>`) and is enforced by Whisper's own `ForceTokensLogitsProcessor`, which runs *before* `PrefixConstrainedLogitsProcessor`. If our fn constrains those positions to Quranic-word tokens, the force gets overridden — the model picks a Quranic word start instead of the language token, breaking everything downstream. Fix: `make_prefix_allowed_tokens_fn` now returns `list(range(vocab_size))` when `len(input_ids) < prompt_token_count`.

3. **Whisper's `generate()` does NOT accept explicit `decoder_input_ids`** — it has custom segmentation logic that crashes (`'NoneType' object has no attribute 'ge'`) if you try. The "pre-populate the prompt" approach used in standard encoder-decoder models doesn't work here.

4. **Greedy decode (`num_beams=1`) with lexicon constraint is sufficient.** Beam=3 takes ~5× longer (8.6s vs 1.9s on a 2.65s clip) but all 3 beams converged to the same answer — the trie is tight enough that beam diversity adds nothing. Greedy doesn't populate `sequences_scores`, so we compute the score manually from per-step logits.

5. **Constrained decode can't reject silence** — the model is mechanically forced to emit *some* Quranic word, and does so with high confidence (~exp(-0.2) ≈ 80%). An amplitude gate in `app.py` (`peak < 0.005` → early reject) is essential. Without it, silence + a "lucky guess" target would falsely verify.

6. **Actual lexicon size is 14,752 entries**, not the ~6k originally estimated. The 6k figure was unique *stems* in some references; raw surface forms (the units Whisper actually emits) are higher. 14.7k is still a tractable constrained-decode lexicon.

### Files created/modified in Phase 1

| File | Status | Purpose |
|---|---|---|
| `scripts/build_lexicon.py` | NEW | One-time fetch from alquran.cloud → `data/quranic_lexicon.json` |
| `scripts/validate_tarteel.py` | NEW | Validation spike using `tarteel-ai/everyayah` (fully automatic) |
| `data/quranic_lexicon.json` | NEW (committed) | 14,752 diacritized Quranic word entries |
| `core/lexicon.py` | NEW | `TokenTrie` + Whisper-aware `prefix_allowed_tokens_fn` factory |
| `core/inference.py` | NEW | `InferenceBackend` protocol + `TarteelBackend` + `LegacyBackend` |
| `app.py` | MODIFIED | Routes inference through `get_backend()`; added silence gate; extended response shape |
| `Dockerfile` | MODIFIED | Copies `data/`; pre-downloads Tarteel weights at build; defaults `MODEL_VARIANT=tarteel` |
| `requirements-dev.txt` | NEW | `datasets>=2.14.0` for one-time scripts only |

### Deployment commands

```bash
# Local docker build + smoke
docker build -t kutuby-words:phase1 .
docker run -p 8080:8080 kutuby-words:phase1
# verify: curl http://localhost:8080/health  → variant=tarteel

# Cloud Run deploy (Tarteel default)
gcloud run deploy arabic-words-api --source . \
    --region europe-west1 --memory 2Gi --cpu 2 \
    --timeout 60 --allow-unauthenticated --no-cpu-throttling

# Rollback to legacy (no redeploy needed — just env var):
gcloud run services update arabic-words-api \
    --region europe-west1 --update-env-vars MODEL_VARIANT=legacy
```

### Known limitations carrying into Phase 2

- `TARTEEL_VERIFY_THRESHOLD` is still a Phase-1 guess (-2.5). Phase 2 calibrates this against `tarteel-ai/everyayah` per word-length bucket.
- Silence gate uses a hard amplitude threshold (0.005); a VAD model would be more principled but adds latency/complexity. Defer to Phase 2 if it becomes an issue.
- The trie has 14.7k × 2 = 29.5k tokenization variants (with-space and without-space). If the model emits a third variant (different surrounding whitespace), it would fall off-trie. Watch for this in production logs.

---

## 8a. Validation spike result (2026-05-25)

Ran `scripts/validate_tarteel.py` against 10 short Quranic verses streamed from `tarteel-ai/everyayah` (reciter: abdulsamad). Result:

- **Transcription match rate: 10/10 (100%)** — every verse transcribed byte-identical to the dataset's ground-truth text after `normalize_arabic_text` comparison
- **Diacritization: working** — Tarteel emits fully-diacritized output that matches Uthmani text shape (only the disjointed-letter verse "الم" lacked diacritics, correctly so)
- **Latency: linear with audio duration** — 1.26s for ~0.5s of audio, 6.73s for ~6s of audio. Production single-word audio (~1s) projects to ~1.5–2s on this hardware.
- **Verification-mode logP: not needed** — free-decode works; verify_logp scores were noted (avg -3.7) but secondary to the perfect free-decode signal.

**Spike verdict: PASS** for the `free-decode + lexicon-constrained beam` path in `TarteelBackend.verify`. The script's automatic FAIL verdict was a gate-logic bug (it compared raw latency to a 1.5s target without accounting for audio duration); the actual data supports proceeding with the planned architecture.

The lexicon also built cleanly in the same session: 6236 verses → 82,456 raw tokens → **15,789 unique normalized words** in `data/quranic_lexicon.json`. Note: this is significantly more than the ~6k figure mentioned earlier (which referenced unique stems, not surface forms). 15,789 surface forms still gives a tractable lexicon for prefix-trie constrained decoding.

---

## 9. Realistic accuracy expectations

A reality check, so nobody expects "flawless" without the data to back it up.

| Scenario | Expected verification accuracy after upgrade | Notes |
|---|---|---|
| Clear adult recitation, Quranic word | 95–99% | What Tarteel + lexicon decode buys us out of the box |
| Adult recitation, noisy environment | 88–95% | Bounded by audio quality, not the model |
| Kid recitation, clear audio, no kid training data | 85–93% | The domain gap that public data cannot close |
| Kid recitation, after collecting ~10–50h and fine-tuning (phase 5) | 93–98% | What kid-data fine-tuning gets us to |
| Acoustic minimal pairs (e.g., `قَالَ` vs. `كَالَ`) | Lower across the board | Some Quranic word pairs are genuinely confusable; we'll surface top-K so the UX can disambiguate |

"Flawless on every Quranic word for every kid in every environment" is not achievable with the data available today. The phased plan exists to get as close as possible, in the right order.

---

## 10. Phased rollout plan

Each phase is independently shippable and unblocks the next.

### Phase 1 — Swap model + lexicon decode + drop fuzzy match  *(week 1)*
- Swap `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` → `tarteel-ai/whisper-base-ar-quran`
- Build the 6k-word Quranic lexicon as a trie loaded at startup
- Implement `prefix_allowed_tokens_fn` over the lexicon for Whisper's beam search
- Replace `fuzzy_match_arabic_words` with: top-K beam decode → exact match against target → log-probability threshold
- Validation spike: confirm Tarteel handles single-word audio cleanly before committing the swap
- Update `/verify_word` response to return `score` (log-prob) and `top_k_candidates`
- Keep `/transcribe_word` backward-compatible (just swap the underlying model)
- **Inputs needed:** the 6k Quranic word list (source TBD — see open questions)
- **Definition of done:** band-aid removed, fuzzy match gone, all existing API consumers still work, latency ≤ 1.5s on Cloud Run

### Phase 2 — Honest threshold calibration  *(3–5 days, parallel with phase 1)*
- Download `tarteel-ai/everyayah`, word-segment the verses, run them through phase-1 pipeline
- Plot score distributions: correct-target vs. wrong-target, by word length
- Set per-bucket thresholds at the F1-optimal point (or recall-favored, given the kids' learning context — better to ask the kid to retry than to falsely accept)
- Bake thresholds into a config file (`core/thresholds.json` or similar) rather than hardcoding
- **Definition of done:** thresholds are measured numbers with provenance, not guesses

### Phase 3 — Kid-recording collection in production  *(2 days dev, then ongoing)*
- Add opt-in audio capture with parental consent flow in the client app (out of scope of this repo, but this repo needs to accept and store the recordings)
- New endpoint: `POST /log_recording` accepting audio + target + verdict + user-correction-flag
- Cloud Storage bucket with per-record metadata (consent timestamp, app version, anonymized session ID)
- Retention/deletion policy: deletion-on-request, automatic purge after N months unless flagged for fine-tuning corpus
- Privacy review (no PII in audio metadata, encryption at rest)
- **Definition of done:** recordings are flowing into a bucket with proper consent and metadata; no model changes yet

### Phase 4 — Forced alignment for per-letter feedback  *(~1 week)*
- Implement rule-based Quranic G2P (Quranic Arabic phonology is regular enough that a few hundred lines of Python handles it)
- Use Tarteel's encoder outputs + Viterbi alignment of expected phones
- New endpoint or response field: `per_letter_scores` returning a list of `{letter, score, was_clear}`
- UI work happens in the client app, not this repo
- **Definition of done:** API can return per-letter feedback; UI team can build the highlighting feature

### Phase 5 — Fine-tune on collected kid data  *(1–2 weeks dev + ~$100 compute, after ~10h kid data accumulated)*
- Take Tarteel whisper-base as base, fine-tune on (Tarteel adult corpus + collected kid corpus) with a curriculum that emphasizes the kid corpus
- Evaluate on held-out kid recordings
- Ship new model version; A/B against the prior model
- **Definition of done:** kid-voice accuracy lifts measurably (target: +5–10 percentage points), no regression on adults

---

## 11. Risks and validation gates

Things that could go wrong and what stops them.

| Risk | Likelihood | Mitigation / gate |
|---|---|---|
| Tarteel whisper-ar-quran handles isolated single words poorly (trained on verses) | Medium | **Phase 1 validation spike** before committing the swap. Fallback: encoder + CTC head, or verification-mode scoring. |
| Lexicon-constrained decoding makes the model "snap" to a Quranic word even when audio is silence/noise | Medium | Phase 1: gate on absolute log-probability threshold, not just relative ranking. Reject if top-1 score is too low. |
| Per-word threshold calibration overfits to Tarteel reciters | Medium | Phase 3 onward: re-calibrate as kid data accumulates. Treat phase 2 thresholds as v1, expect to revise. |
| Kid-voice accuracy stays bad even after phase 1+2 | High (this is *expected*) | Phase 3 is explicitly designed to address this. Communicate the 85–93% kid-voice expectation honestly to product. |
| 6k-word lexicon source has inconsistent diacritization vs. Tarteel's output | Medium | Phase 1: normalize both target and lexicon with the same scheme. May need a small lexicon-cleanup pass. |
| Cloud Run cold-start regressions after model swap | Low | Tarteel base is smaller than current, so cold start should improve. Monitor. |
| Forced alignment is too aggressive (rejects clear pronunciations) | Medium | Phase 4: per-phoneme thresholds also need calibration. Treat scores as advisory not gating. |

---

## 12. Open questions (answer before phase 1 ships)

These need decisions from product/eng before phase 1 implementation begins.

1. **Where does the 6k Quranic word list come from?** Tanzil corpus? An existing in-product database? Hardcoded list? The diacritization scheme of this source must match (or be normalizable to) Tarteel's output format.
2. **Confirm Tarteel whisper-base-ar-quran license terms** for commercial use in a paid kids' app. Likely MIT but verify.
3. **Confirm Cloud Run min-instances strategy** — current is 0 (cold starts on idle). After model swap, do we keep it at 0 or pre-warm during peak hours?
4. **What's the rollback plan?** Keep the current `jonatasgrosman` pipeline behind a feature flag for one release cycle? Or hard cutover?
5. **API contract:** is it acceptable to extend the `/verify_word` response with new fields (`score`, `top_k_candidates`), or is the existing `result + similarity + confidence` shape locked by client app versions in the wild?
6. **Consent flow for phase 3 kid recordings:** product/legal need to draft the parental consent UX before we build the `/log_recording` endpoint.

---

## 13. Out of scope

To keep this plan tight, these are explicitly *not* part of the upgrade:

- Multi-word phrase verification (just single Quranic words for now)
- Multi-reciter style detection (e.g., Hafs vs. Warsh recitation traditions)
- Real-time streaming verification (still batch-per-request)
- The Arabic Letters API (sister service, separate upgrade)
- The legacy Streamlit code in this repo (delete during phase 1 cleanup or leave; tracked separately)

---

## 14. Glossary

- **Tashkeel / diacritics** — short vowel marks in Arabic (fatha, damma, kasra, sukun, shadda, tanween)
- **CTC** — Connectionist Temporal Classification, the loss/decoding scheme Wav2Vec2 uses
- **Lexicon-constrained decoding** — forcing a model's output to come from a predefined word list
- **Forced alignment** — fixing the output sequence and finding which audio frames correspond to which output tokens
- **GOP (Goodness of Pronunciation)** — standard pronunciation-assessment metric, posterior probability of the expected phone at each frame
- **G2P (Grapheme-to-Phoneme)** — converting written word to phoneme sequence
- **Tartil / Tajweed** — formal Quranic recitation rules
- **Hafs / Warsh** — major recitation traditions of the Quran
