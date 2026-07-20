# -*- coding: utf-8 -*-
"""
Model-Agnostic Inference Backends

Phase 1 of the upgrade plan introduces a Whisper-based Quranic-domain backend
(`TarteelBackend`) using lexicon-constrained beam decode in place of the
free-vocabulary Wav2Vec2 + fuzzy-match pipeline (`LegacyBackend`).

Selection is driven by the `MODEL_VARIANT` env var:
- `tarteel` (default): `tarteel-ai/whisper-base-ar-quran` + lexicon-constrained beam
- `legacy`           : `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` + fuzzy match

Both backends expose the same interface so app.py is identical regardless of
which variant is live. Both return the same response shape; Tarteel-only
fields (`score`, `top_k_candidates`) are `None` in the legacy path.

Threshold tuning is parked in env vars; Phase 2 of upgrade.md replaces these
with measured per-word-length values calibrated on `tarteel-ai/everyayah`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import torch

from core.arabic_utils import normalize_arabic_text
from core.lexicon import TokenTrie, build_trie_from_lexicon, make_prefix_allowed_tokens_fn

logger = logging.getLogger(__name__)


# --- Configuration ----------------------------------------------------------

DEFAULT_MODEL_VARIANT = "tarteel"
TARTEEL_MODEL_NAME = "tarteel-ai/whisper-base-ar-quran"
LEGACY_MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
FASTCONFORMER_MODEL_NAME = "nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0"
PHONEME_MDD_MODEL_NAME = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"

DEFAULT_LEXICON_PATH = Path(__file__).resolve().parent.parent / "data" / "quranic_lexicon.json"
DEFAULT_CURRICULUM_PATH = Path(__file__).resolve().parent.parent / "data" / "curriculum_words.json"

# Whisper-Arabic-transcribe decoder prompt is 4 tokens:
#   <|startoftranscript|>, <|ar|>, <|transcribe|>, <|notimestamps|>
WHISPER_PROMPT_TOKEN_COUNT = 4

# Phase 1 initial thresholds (env-overridable). Replaced by calibrated values in Phase 2.
TARTEEL_VERIFY_THRESHOLD = float(os.getenv("TARTEEL_VERIFY_THRESHOLD", "-2.5"))
LEGACY_CONFIDENCE_THRESHOLD = float(os.getenv("LEGACY_CONFIDENCE_THRESHOLD", "0.6"))


# --- Backend protocol -------------------------------------------------------

class InferenceBackend(Protocol):
    variant: str

    def transcribe(self, audio: np.ndarray) -> dict:
        """Free-vocabulary transcription. Returns: transcription, confidence, latency_ms."""

    def verify(self, audio: np.ndarray, target_word: str, top_k: int = 3) -> dict:
        """Verification against a target word. Returns full /verify_word response payload."""


# --- Tarteel backend --------------------------------------------------------

class TarteelBackend:
    """Whisper-Quranic model + lexicon-constrained beam decode."""

    variant = "tarteel"
    model_name = TARTEEL_MODEL_NAME

    def __init__(self, lexicon_path: Path = DEFAULT_LEXICON_PATH):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        logger.info("tarteel: loading %s", TARTEEL_MODEL_NAME)
        load_start = time.time()
        self.processor = WhisperProcessor.from_pretrained(TARTEEL_MODEL_NAME)
        self.model = WhisperForConditionalGeneration.from_pretrained(TARTEEL_MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).eval()
        logger.info("tarteel: model on %s in %.1fs", self.device.type, time.time() - load_start)

        # Build lexicon trie
        logger.info("tarteel: building lexicon trie from %s", lexicon_path)
        trie_start = time.time()
        self.trie: TokenTrie = build_trie_from_lexicon(self.processor.tokenizer, lexicon_path)
        eos_id = self.model.config.eos_token_id or self.processor.tokenizer.eos_token_id
        self._eos_id = eos_id

        # The Whisper decoder emits a prompt prefix (`<|sot|>, <|ar|>, <|transcribe|>,
        # <|notimestamps|>` — 4 tokens) before any transcription. We must not constrain
        # those positions — Whisper's own forced_decoder_ids handles them. Compute the
        # prompt length so our fn can detect "still in prefix, allow everything".
        forced_ids = self.processor.get_decoder_prompt_ids(language="ar", task="transcribe")
        prompt_len = 1 + len(forced_ids)  # +1 for decoder_start_token (<|sot|>)
        self._prompt_len = prompt_len
        vocab_size = self.model.config.vocab_size

        self._prefix_allowed_tokens_fn = make_prefix_allowed_tokens_fn(
            self.trie,
            prompt_token_count=prompt_len,
            eos_token_id=eos_id,
            vocab_size=vocab_size,
        )
        logger.info(
            "tarteel: lexicon trie ready in %.2fs (decoder prompt = %d tokens, vocab = %d)",
            time.time() - trie_start, prompt_len, vocab_size,
        )

        # Warmup with 1 second of silence
        self._warmup()

    def _warmup(self) -> None:
        try:
            dummy = np.zeros(16000, dtype=np.float32)
            t0 = time.time()
            self._generate(dummy, constrained=False, num_beams=1, num_return=1)
            logger.info("tarteel: warmup complete in %.2fs", time.time() - t0)
        except Exception as e:
            logger.warning("tarteel: warmup failed (non-fatal): %s", e)

    def _generate(
        self,
        audio: np.ndarray,
        *,
        constrained: bool,
        num_beams: int,
        num_return: int,
    ):
        """Lower-level inference. Returns a HuggingFace GenerateBeamOutput (or greedy variant)."""
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        # max_new_tokens=10 covers every Quranic word (longest tokenizes to ~6-8 BPE tokens).
        generate_kwargs = dict(
            max_new_tokens=10,
            num_beams=num_beams,
            num_return_sequences=num_return,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

        if constrained:
            generate_kwargs["prefix_allowed_tokens_fn"] = self._prefix_allowed_tokens_fn

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    return self.model.generate(input_features, **generate_kwargs)
            return self.model.generate(input_features, **generate_kwargs)

    def transcribe(self, audio: np.ndarray) -> dict:
        t0 = time.time()
        gen = self._generate(audio, constrained=False, num_beams=1, num_return=1)
        latency_ms = (time.time() - t0) * 1000

        text = self._decode_transcription(gen.sequences)[0].strip()

        # Confidence from per-step max log-probs (matches the legacy backend's 0-100 shape)
        confidence_pct = self._confidence_from_scores(gen.scores)

        logger.info(
            "tarteel.transcribe: '%s' confidence=%.1f%% latency=%.0fms",
            text, confidence_pct, latency_ms,
        )
        return {
            "transcription": text,
            "confidence": round(confidence_pct, 2),
            "latency_ms": round(latency_ms, 2),
        }

    def verify(self, audio: np.ndarray, target_word: str, top_k: int = 1) -> dict:
        t0 = time.time()
        # Beam width = top_k. With lexicon constraint, the search is so narrow
        # that beam>1 almost always converges to the same answer — so default
        # to greedy (top_k=1) for speed. Callers can request top_k=3 if they
        # want to see alternative Quranic-word candidates for diagnostic UX.
        num_beams = max(top_k, 1)
        gen = self._generate(
            audio,
            constrained=True,
            num_beams=num_beams,
            num_return=min(top_k, num_beams),
        )
        latency_ms = (time.time() - t0) * 1000

        decoded = [d.strip() for d in self._decode_transcription(gen.sequences)]

        # Beam search returns `sequences_scores` directly; greedy (num_beams=1) does not —
        # we compute the equivalent from per-step logits in that case.
        if getattr(gen, "sequences_scores", None) is not None:
            seq_scores = gen.sequences_scores.tolist()
        else:
            seq_scores = self._scores_from_step_logits(gen)

        candidates = [
            {"word": w, "score": round(s, 4)}
            for w, s in zip(decoded, seq_scores)
        ]
        top_word = candidates[0]["word"] if candidates else ""
        top_score = candidates[0]["score"] if candidates else float("-inf")

        norm_target = normalize_arabic_text(target_word.strip())
        norm_top = normalize_arabic_text(top_word)
        matches = norm_top == norm_target
        exceeds_threshold = top_score >= TARTEEL_VERIFY_THRESHOLD
        result = matches and exceeds_threshold

        confidence_pct = self._confidence_from_score(top_score)

        logger.info(
            "tarteel.verify: target='%s' top='%s' score=%.3f match=%s pass_threshold=%s result=%s latency=%.0fms",
            target_word, top_word, top_score, matches, exceeds_threshold, result, latency_ms,
        )

        return {
            "result": result,
            "transcription": top_word,
            "target_word": target_word.strip(),
            "score": round(top_score, 4),
            "top_k_candidates": candidates,
            "similarity": 100.0 if matches else 0.0,
            "confidence": round(confidence_pct, 2),
            "threshold": TARTEEL_VERIFY_THRESHOLD,
            "latency_ms": round(latency_ms, 2),
        }

    def _decode_transcription(self, sequences) -> list[str]:
        """Decode generated sequences to text, slicing off Whisper's decoder prompt.

        We can't rely on `skip_special_tokens=True` alone — some transformers
        versions don't register Whisper's language/task tokens as "special",
        so they leak into the decoded string as literal `<|startoftranscript|>`
        prefixes. Slicing off the known prompt length is robust across versions.
        """
        transcription_only = sequences[:, self._prompt_len:]
        return self.processor.batch_decode(transcription_only, skip_special_tokens=True)

    @staticmethod
    def _scores_from_step_logits(gen) -> list[float]:
        """Sum the picked-token log-prob across generation steps, length-normalized.

        Greedy decoding doesn't populate `sequences_scores`, so we reconstruct an
        equivalent from `gen.scores` (per-step logits over vocab) and
        `gen.sequences` (the picked token IDs). This matches what beam search's
        `sequences_scores` reports for num_beams=1.
        """
        if not gen.scores:
            return [float("-inf")] * gen.sequences.shape[0]
        # sequences include the prompt; only the last N tokens correspond to gen.scores
        num_generated = len(gen.scores)
        scores_per_seq: list[float] = []
        for seq_idx in range(gen.sequences.shape[0]):
            generated_tokens = gen.sequences[seq_idx, -num_generated:]
            total_logp = 0.0
            non_eos_count = 0
            for step, tok in enumerate(generated_tokens.tolist()):
                step_logits = gen.scores[step][seq_idx]
                log_probs = torch.log_softmax(step_logits, dim=-1)
                total_logp += float(log_probs[tok].item())
                non_eos_count += 1
            scores_per_seq.append(total_logp / max(non_eos_count, 1))
        return scores_per_seq

    @staticmethod
    def _confidence_from_scores(per_step_scores) -> float:
        """Convert per-step max log-probs into a 0-100 confidence percentage."""
        if not per_step_scores:
            return 0.0
        log_probs = [torch.log_softmax(s, dim=-1).max(dim=-1).values.mean().item() for s in per_step_scores]
        avg = sum(log_probs) / len(log_probs)
        return max(0.0, min(100.0, float(np.exp(avg) * 100)))

    @staticmethod
    def _confidence_from_score(seq_score: float) -> float:
        if seq_score == float("-inf"):
            return 0.0
        return max(0.0, min(100.0, float(np.exp(seq_score) * 100)))


# --- Legacy backend ---------------------------------------------------------

class LegacyBackend:
    """The original Wav2Vec2 + fuzzy-match pipeline, preserved for rollback."""

    variant = "legacy"
    model_name = LEGACY_MODEL_NAME

    def __init__(self):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        logger.info("legacy: loading %s", LEGACY_MODEL_NAME)
        load_start = time.time()
        self.processor = Wav2Vec2Processor.from_pretrained(LEGACY_MODEL_NAME)
        self.model = Wav2Vec2ForCTC.from_pretrained(LEGACY_MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).eval()
        logger.info("legacy: model on %s in %.1fs", self.device.type, time.time() - load_start)

        self._warmup()

    def _warmup(self) -> None:
        try:
            dummy = np.zeros(16000, dtype=np.float32)
            t0 = time.time()
            self._forward(dummy)
            logger.info("legacy: warmup complete in %.2fs", time.time() - t0)
        except Exception as e:
            logger.warning("legacy: warmup failed (non-fatal): %s", e)

    def _forward(self, audio: np.ndarray):
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    logits = self.model(input_values).logits
            else:
                logits = self.model(input_values).logits
        return logits

    def transcribe(self, audio: np.ndarray) -> dict:
        t0 = time.time()
        logits = self._forward(audio)
        latency_ms = (time.time() - t0) * 1000

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0].strip()

        probs = torch.softmax(logits, dim=-1)
        max_probs = torch.max(probs, dim=-1).values
        confidence_pct = float(torch.mean(max_probs).item() * 100)

        logger.info(
            "legacy.transcribe: '%s' confidence=%.1f%% latency=%.0fms",
            transcription, confidence_pct, latency_ms,
        )
        return {
            "transcription": transcription,
            "confidence": round(confidence_pct, 2),
            "latency_ms": round(latency_ms, 2),
        }

    def verify(
        self,
        audio: np.ndarray,
        target_word: str,
        top_k: int = 3,
        fuzzy_match: bool = True,
        fuzzy_threshold: Optional[float] = None,
    ) -> dict:
        from core.arabic_utils import fuzzy_match_arabic_words

        t0 = time.time()
        logits = self._forward(audio)
        latency_ms = (time.time() - t0) * 1000

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0].strip()

        probs = torch.softmax(logits, dim=-1)
        max_probs = torch.max(probs, dim=-1).values
        confidence = float(torch.mean(max_probs).item())  # 0..1

        if fuzzy_match:
            matches, similarity = fuzzy_match_arabic_words(
                transcription=transcription,
                target=target_word.strip(),
                custom_threshold=fuzzy_threshold,
            )
        else:
            norm_t = normalize_arabic_text(transcription)
            norm_w = normalize_arabic_text(target_word.strip())
            matches = norm_t == norm_w
            similarity = 100.0 if matches else 0.0
        exceeds_threshold = confidence >= LEGACY_CONFIDENCE_THRESHOLD
        result = matches and exceeds_threshold

        logger.info(
            "legacy.verify: target='%s' transcription='%s' similarity=%.1f confidence=%.1f%% result=%s latency=%.0fms",
            target_word, transcription, similarity, confidence * 100, result, latency_ms,
        )

        return {
            "result": result,
            "transcription": transcription,
            "target_word": target_word.strip(),
            "score": None,
            "top_k_candidates": None,
            "similarity": round(similarity, 2),
            "confidence": round(confidence * 100, 2),
            "threshold": LEGACY_CONFIDENCE_THRESHOLD * 100,
            "latency_ms": round(latency_ms, 2),
        }


# --- FastConformer backend --------------------------------------------------

# The `_pc` (punctuation+capitalization) FastConformer variant can append
# sentence punctuation to a single word. That is NOT a pronunciation error, so
# strip it before matching. Kept isolated to this backend so tarteel/legacy
# matching behavior is unchanged (no regression risk under code-red).
_TRAILING_PUNCT_RE = re.compile(r"[.,،؛؟?!…\"'“”«»:;()\[\]\-_/\\]+")


def _strip_trailing_punct(s: str) -> str:
    return _TRAILING_PUNCT_RE.sub(" ", s or "").strip()


def _strip_leading_al(token: str) -> str:
    """Drop a leading Arabic definite article 'ال' when a real word remains."""
    return token[2:] if token.startswith("ال") and len(token) > 3 else token


def _curriculum_match(norm_trans: str, norm_target: str) -> bool:
    """Closed-set word match that tolerates common child-speech patterns WITHOUT any
    character-level fuzz, so genuine mispronunciations (e.g. ق->ك) still fail:

      - exact match, OR
      - the target appears as a whole token: covers spamming ('baqara baqara baqara')
        and the target spoken amid extra words, OR
      - a token equals the target after stripping a leading definite article 'ال'.

    Comparison is whole-token exact only; a wrong-but-similar word never matches.
    """
    if not norm_trans or not norm_target:
        return False
    if norm_trans == norm_target:
        return True
    for tok in norm_trans.split():
        if tok == norm_target or _strip_leading_al(tok) == norm_target:
            return True
    return False


def _load_curriculum(path: Path = DEFAULT_CURRICULUM_PATH) -> set[str]:
    """Load the optional curriculum word list as a set of normalized surface forms.

    Used by the fastconformer backend for optional closed-set (curriculum-constrained)
    rescoring — a v2 refinement. Missing/empty file => empty set => pure exact-match
    verification (the v1 behavior). Never raises: a curriculum problem must not take
    the service down.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.info("fastconformer: no curriculum at %s (pure exact-match mode)", path)
        return set()
    except Exception as e:  # malformed JSON, encoding, etc.
        logger.warning("fastconformer: curriculum load failed (%s); pure exact-match mode", e)
        return set()

    norm: set[str] = set()
    for entry in payload.get("words", []):
        arabic = entry.get("arabic") if isinstance(entry, dict) else entry
        if arabic:
            norm.add(normalize_arabic_text(arabic))
    return norm


class FastConformerBackend:
    """NVIDIA FastConformer (general MSA ASR) + greedy CTC decode + closed-set exact match.

    Replaces the lexicon-caged Tarteel Quranic model, which was structurally
    incapable of emitting the curriculum words that are not in the Quran (a
    guaranteed 0% on ~a third of the curriculum). FastConformer is a general
    Arabic model — no lexicon constraint — decoded with greedy CTC (fast on CPU,
    and, unlike autoregressive Whisper, it does not hallucinate a confident word
    from a near-empty clip).

    The verdict is an exact match of the normalized, punctuation-stripped
    transcription against the target word. No fuzzy matching: this is closed-set
    verification with zero sloppy false-accepts. `self.curriculum` is loaded and
    ready for a later closed-set-rescoring refinement (v2) but is not consulted
    by the v1 exact-match verdict.
    """

    variant = "fastconformer"
    model_name = FASTCONFORMER_MODEL_NAME

    def __init__(self, curriculum_path: Path = DEFAULT_CURRICULUM_PATH):
        # torch>=2.6 refuses to load the .nemo checkpoint without this.
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        import nemo.collections.asr as nemo_asr

        logger.info("fastconformer: loading %s", FASTCONFORMER_MODEL_NAME)
        load_start = time.time()
        self.model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(FASTCONFORMER_MODEL_NAME)
        # CTC greedy decode: fastest on CPU and no short-clip hallucination.
        self.model.change_decoding_strategy(decoder_type="ctc")
        self.model.eval()
        logger.info("fastconformer: model ready on %s in %.1fs",
                    next(self.model.parameters()).device.type, time.time() - load_start)

        self.curriculum = _load_curriculum(curriculum_path)
        logger.info("fastconformer: curriculum loaded (%d words, exact-match mode)", len(self.curriculum))

        self._warmup()

    def _warmup(self) -> None:
        try:
            t0 = time.time()
            self._raw_transcribe(np.zeros(16000, dtype=np.float32))
            logger.info("fastconformer: warmup complete in %.2fs", time.time() - t0)
        except Exception as e:
            logger.warning("fastconformer: warmup failed (non-fatal): %s", e)

    def _raw_transcribe(self, audio: np.ndarray) -> str:
        """Greedy CTC transcription of a float32 mono 16k array. Robust across
        NeMo transcribe() signature differences (verbose kw, tuple returns)."""
        arr = audio.astype(np.float32)
        out = None
        last_err: Optional[Exception] = None
        for kwargs in ({"batch_size": 1, "verbose": False}, {"batch_size": 1}, {}):
            try:
                out = self.model.transcribe([arr], **kwargs)
                break
            except TypeError as e:  # signature mismatch across NeMo versions
                last_err = e
        if out is None:
            raise last_err if last_err else RuntimeError("fastconformer: transcribe returned nothing")
        # Hybrid models may return a (best_hyps, all_hyps) tuple; CTC strategy returns a list.
        if isinstance(out, tuple):
            out = out[0]
        h = out[0]
        text = h.text if hasattr(h, "text") else str(h)
        return text.strip()

    def transcribe(self, audio: np.ndarray) -> dict:
        t0 = time.time()
        raw = self._raw_transcribe(audio)
        latency_ms = (time.time() - t0) * 1000
        text = _strip_trailing_punct(raw)
        logger.info("fastconformer.transcribe: '%s' latency=%.0fms", text, latency_ms)
        return {"transcription": text, "confidence": 100.0, "latency_ms": round(latency_ms, 2)}

    def verify(self, audio: np.ndarray, target_word: str, top_k: int = 1, **kwargs) -> dict:
        # `top_k` / **kwargs (fuzzy_match, fuzzy_threshold) are accepted for
        # interface compatibility with app.py's dispatch but intentionally ignored:
        # fastconformer verification is exact-match only.
        t0 = time.time()
        raw = self._raw_transcribe(audio)
        latency_ms = (time.time() - t0) * 1000

        transcription = _strip_trailing_punct(raw)
        norm_trans = normalize_arabic_text(transcription)
        norm_target = normalize_arabic_text(target_word.strip())
        matched = _curriculum_match(norm_trans, norm_target)

        logger.info(
            "fastconformer.verify: target='%s' transcription='%s' match=%s latency=%.0fms",
            target_word, transcription, matched, latency_ms,
        )
        return {
            "result": matched,
            "transcription": transcription,
            "target_word": target_word.strip(),
            "score": None,
            "top_k_candidates": None,
            "similarity": 100.0 if matched else 0.0,
            "confidence": 100.0 if matched else 0.0,
            "threshold": None,
            "latency_ms": round(latency_ms, 2),
        }


# --- Phoneme MDD backend ----------------------------------------------------

class PhonemeMDDBackend:
    """Phoneme recognizer (`facebook/wav2vec2-xlsr-53-espeak-cv-ft`) + phoneme-level
    Mispronunciation Detection & Diagnosis (MDD) and Goodness-of-Pronunciation (GOP).

    Unlike the ASR backends (which return a pass/fail on a whole-word transcription),
    this one returns WHICH phonemes were wrong (substitution / insertion / deletion),
    a 0-100 GOP score, and child-friendly corrective feedback. It compares the
    recognized phoneme string against the canonical (reference) phonemes for the
    target word: the closed curriculum's precomputed `canonical_phonemes` when
    available, else an espeak fallback for off-curriculum words.

    CPU-friendly (~300M params, Wav2Vec2 CTC, Apache-2.0), so it runs on the same
    Cloud Run service as the ASR backends. The pure MDD/GOP logic lives in
    core/phoneme_mdd.py; this class is only the acoustic front-end + response glue.
    """

    variant = "mdd"
    model_name = PHONEME_MDD_MODEL_NAME

    def __init__(self, curriculum_path: Path = DEFAULT_CURRICULUM_PATH):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
        from core.phoneme_mdd import CanonicalIndex, normalize_phoneme_string

        self._normalize = normalize_phoneme_string

        logger.info("mdd: loading %s", PHONEME_MDD_MODEL_NAME)
        load_start = time.time()
        # This 2021 checkpoint ships a Wav2Vec2PhonemeCTCTokenizer whose
        # from_pretrained is broken in current transformers (returns a bool), and
        # the plain Wav2Vec2CTCTokenizer concatenates phones ("baqr") instead of
        # space-separating them. So we load the CTC tokenizer only for its vocab
        # and do manual CTC greedy decoding in _recognize to get clean, space-
        # separated phonemes that match build_canonical_phonemes.py.
        from transformers import Wav2Vec2CTCTokenizer
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(PHONEME_MDD_MODEL_NAME)
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(PHONEME_MDD_MODEL_NAME)
        self._blank_id = self.tokenizer.pad_token_id
        self._word_delim = self.tokenizer.word_delimiter_token
        self.model = Wav2Vec2ForCTC.from_pretrained(PHONEME_MDD_MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).eval()

        self.index = CanonicalIndex.load(curriculum_path)
        logger.info("mdd: ready on %s in %.1fs (%d curriculum words indexed)",
                    self.device.type, time.time() - load_start, len(self.index))

        self._warmup()

    def _warmup(self) -> None:
        try:
            t0 = time.time()
            self._recognize(np.zeros(16000, dtype=np.float32))
            logger.info("mdd: warmup complete in %.2fs", time.time() - t0)
        except Exception as e:
            logger.warning("mdd: warmup failed (non-fatal): %s", e)

    def _recognize(self, audio: np.ndarray) -> str:
        """Greedy CTC decode to a normalized, space-separated phoneme string.

        Manual CTC collapse (drop consecutive repeats, then the blank/pad token)
        so each surviving phoneme becomes its own space-separated token, instead
        of the tokenizer's concatenated 'baqr'."""
        inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.cuda.amp.autocast():
                    logits = self.model(input_values).logits
            else:
                logits = self.model(input_values).logits

        ids = torch.argmax(logits, dim=-1)[0].tolist()
        collapsed: list[int] = []
        prev = None
        for i in ids:
            if i != prev:
                if i != self._blank_id:
                    collapsed.append(i)
                prev = i
        tokens = self.tokenizer.convert_ids_to_tokens(collapsed)
        phonemes = [t for t in tokens if t and t != self._word_delim]
        return self._normalize(" ".join(phonemes))

    def transcribe(self, audio: np.ndarray) -> dict:
        t0 = time.time()
        phonemes = self._recognize(audio)
        latency_ms = (time.time() - t0) * 1000
        logger.info("mdd.transcribe: '%s' latency=%.0fms", phonemes, latency_ms)
        return {"transcription": phonemes, "confidence": 100.0, "latency_ms": round(latency_ms, 2)}

    def _espeak_canonical(self, word: str) -> str:
        """Off-curriculum fallback: G2P the target word at runtime. Requires espeak
        in the image. Returns '' if unavailable so verify can degrade gracefully."""
        try:
            from phonemizer import phonemize
            from phonemizer.separator import Separator

            # phone and word separators must differ; normalize_phoneme_string
            # strips the "|" word markers afterwards.
            ph = phonemize(
                word, language="ar", backend="espeak",
                separator=Separator(phone=" ", word=" | ", syllable=""),
                strip=True, preserve_punctuation=False, with_stress=False,
                language_switch="remove-flags", njobs=1,
            )
            return self._normalize(ph if isinstance(ph, str) else " ".join(ph))
        except Exception as e:
            logger.warning("mdd: espeak fallback failed for %r: %s", word, e)
            return ""

    def verify(self, audio: np.ndarray, target_word: str, top_k: int = 1, **kwargs) -> dict:
        # top_k / **kwargs (fuzzy_match, fuzzy_threshold) accepted for interface
        # parity with app.py's dispatch but not used: MDD is phoneme-diff based.
        from core.phoneme_mdd import diagnose, diagnose_word

        t0 = time.time()
        recognized = self._recognize(audio)
        latency_ms = (time.time() - t0) * 1000

        result = diagnose_word(recognized, target_word, self.index)
        if result is None:
            # not in curriculum: try runtime espeak, else we cannot assess phonemes
            canon = self._espeak_canonical(target_word)
            if canon:
                result = diagnose(recognized, canon)
                result["target_word"] = target_word.strip()

        if result is None:
            logger.info("mdd.verify: no canonical for target='%s' (cannot assess)", target_word)
            return {
                "result": False,
                "transcription": recognized,
                "target_word": target_word.strip(),
                "score": None,
                "top_k_candidates": None,
                "similarity": 0.0,
                "confidence": 0.0,
                "threshold": None,
                "latency_ms": round(latency_ms, 2),
                "recognized_phonemes": recognized,
                "canonical_phonemes": None,
                "gop_score": 0.0,
                "mdd_errors": [],
                "feedback": ["Could not assess this word (no reference pronunciation)."],
            }

        errors = result["errors"]
        feedback = [e["feedback"] for e in errors] or ["That sounds correct."]

        logger.info(
            "mdd.verify: target='%s' heard='%s' correct=%s errors=%d gop=%.1f latency=%.0fms",
            target_word, recognized, result["is_correct"], len(errors),
            result["gop_score"], latency_ms,
        )
        return {
            "result": result["is_correct"],
            "transcription": recognized,
            "target_word": target_word.strip(),
            "score": None,
            "top_k_candidates": None,
            "similarity": result["gop_score"],
            "confidence": result["gop_score"],
            "threshold": None,
            "latency_ms": round(latency_ms, 2),
            # additive MDD/GOP fields (ignored by legacy clients):
            "recognized_phonemes": result["recognized_phonemes"],
            "canonical_phonemes": result["canonical_phonemes"],
            "gop_score": result["gop_score"],
            "mdd_errors": errors,
            "feedback": feedback,
        }


# --- Backend registry -------------------------------------------------------

# Multi-backend cache: one instance per variant, lazily loaded on first request.
# Replaces the single-backend singleton so a council/cascade can hold more than
# one model resident. Backward compatible: get_backend() with no argument still
# returns the MODEL_VARIANT-selected backend.
_BACKENDS: dict[str, InferenceBackend] = {}

_BUILDERS = {
    "tarteel": TarteelBackend,
    "legacy": LegacyBackend,
    "fastconformer": FastConformerBackend,
    "mdd": PhonemeMDDBackend,
}


def get_backend(variant: Optional[str] = None) -> InferenceBackend:
    """Return the cached backend for `variant`, loading it on first request.

    `variant=None` (the default, used by all existing callers) resolves to the
    `MODEL_VARIANT` env var, preserving the original single-backend behavior.
    An explicit variant (e.g. "mdd") loads that backend alongside any others.
    Each backend eagerly loads its model on construction, so the first call for a
    given variant is slow (~10-40s); subsequent calls are O(1).
    """
    if variant is None:
        variant = os.getenv("MODEL_VARIANT", DEFAULT_MODEL_VARIANT)
    variant = variant.strip().lower()

    if variant not in _BACKENDS:
        builder = _BUILDERS.get(variant)
        if builder is None:
            raise ValueError(
                f"Unknown backend variant={variant!r}; expected one of {sorted(_BUILDERS)}"
            )
        _BACKENDS[variant] = builder()
        logger.info("inference: backend ready (variant=%s)", variant)
    return _BACKENDS[variant]
