#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation Spike: Tarteel whisper-base-ar-quran on short Quranic audio (fully automatic)

Purpose
-------
Tarteel's `whisper-base-ar-quran` was trained on verse-level Quranic recitation.
This script verifies that the model also handles **short Quranic audio**
cleanly — the closest public proxy for our production use case (isolated
single-word verification), since no labelled isolated-word Arabic dataset
exists publicly.

It streams a few short verses from the `tarteel-ai/everyayah` HuggingFace
dataset (≤3 words per verse — disjointed-letter verses like ق, ن, يس, طه, plus
short verses like 99:1, 110:1) and runs the model on each in two modes:

    1. Free decode      — what the model produces with no constraints
    2. Verification     — compute logP(target | audio) for the dataset's
                          ground-truth text

If free-decode on short verses lands on the right verse text, Phase 1 proceeds
with free-decode + lexicon-constrained beam in TarteelBackend. If free-decode
is unreliable but verification-mode logP correctly ranks the truth higher than
random Quranic words, we switch TarteelBackend.verify to verification-mode
scoring. Either path is recoverable; this script tells us which.

Usage
-----
    pip install -r requirements-dev.txt    # one-time, installs `datasets`
    python scripts/validate_tarteel.py

Optional flags:
    --num-samples N      How many short-verse clips to test (default: 10)
    --dataset NAME       HF dataset (default: tarteel-ai/everyayah)
    --device cpu|cuda    Inference device (default: cpu)

Public datasets considered
--------------------------
- `tarteel-ai/everyayah`        — verse-level audio + text, used by default
- `tarteel-ai/tarteel-1.0`      — user-submitted recitations with labels
- `MohamedRashad/Quran-Recitations` — alternative reciter recordings
- `mozilla-foundation/common_voice_*_ar` — general Arabic (not Quranic)

If the default dataset fails to load (renamed/gated), pass `--dataset` with
one of the alternatives above.

Spike result
------------
After running, the printed summary should be pasted into `upgrade.md` under a
new "Validation spike result" subsection. The script's exit code is the gate:
    0  PASS    — proceed with free-decode + lexicon-constrained beam
    1  PARTIAL — proceed with verification-mode scoring instead
    2  FAIL    — model unsuitable for our use case; consider whisper-tiny-ar-quran fallback
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path
from typing import Optional

# Force UTF-8 stdout so Arabic text prints correctly on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


MODEL_NAME = "tarteel-ai/whisper-base-ar-quran"
DEFAULT_DATASET = "tarteel-ai/everyayah"
TARGET_LATENCY_SECONDS = 1.5
PASS_TRANSCRIPTION_RATIO = 0.60   # fraction of clips that must match ground truth
PARTIAL_VERIFY_LOGP = -1.5         # verification mode passes if average logP above this


# Arabic diacritic codepoints — used to detect whether output is diacritized
_ARABIC_DIACRITIC_RANGE = set(range(0x064B, 0x065F + 1)) | {0x0670}


def _lazy_imports():
    import numpy as np
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    try:
        from datasets import load_dataset, Audio
    except ImportError as e:
        print("spike: ERROR — `datasets` not installed. Run: pip install -r requirements-dev.txt", file=sys.stderr)
        raise
    return np, torch, WhisperProcessor, WhisperForConditionalGeneration, load_dataset, Audio


def _has_diacritics(text: str) -> bool:
    return any(ord(ch) in _ARABIC_DIACRITIC_RANGE for ch in text)


def _normalize_for_compare(text: str) -> str:
    """Reuse the production normalizer for transcription-vs-target comparison."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.arabic_utils import normalize_arabic_text
    return normalize_arabic_text(text)


def _word_count(text: str) -> int:
    return len([t for t in text.split() if t.strip()])


def _resample_to_16k(audio_array, sampling_rate: int):
    """Resample with librosa if needed."""
    if sampling_rate == 16000:
        return audio_array
    import librosa
    return librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)


def _decode_audio_bytes(raw_bytes: bytes):
    """Decode raw audio bytes to (array, sampling_rate). Tries soundfile, falls back to librosa+temp."""
    import soundfile as sf
    import numpy as np
    try:
        arr, sr = sf.read(io.BytesIO(raw_bytes))
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        return arr.astype(np.float32), sr
    except Exception:
        # Fallback: librosa via temp file (handles formats soundfile can't)
        import tempfile, os, librosa
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as f:
            f.write(raw_bytes)
            tmp_path = f.name
        try:
            arr, sr = librosa.load(tmp_path, sr=None, mono=True)
            return arr.astype(np.float32), sr
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def _free_decode(model, processor, audio_16k, device):
    import torch
    inputs = processor(audio_16k, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    with torch.no_grad():
        generated = model.generate(
            input_features,
            max_new_tokens=40,
            num_beams=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    text = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0].strip()
    score = None
    if hasattr(generated, "scores") and generated.scores:
        per_step_max = [torch.log_softmax(s, dim=-1).max(dim=-1).values.item() for s in generated.scores]
        if per_step_max:
            score = sum(per_step_max) / len(per_step_max)
    return text, score


def _verify_score(model, processor, audio_16k, target_text, device):
    """Compute average per-token log-prob of `target_text` given `audio_16k`."""
    import torch
    inputs = processor(audio_16k, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    target_ids = processor.tokenizer(target_text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model(
            input_features=input_features,
            decoder_input_ids=target_ids[:, :-1],
        )
    logits = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)
    target_tokens = target_ids[:, 1:]
    gathered = log_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)
    return gathered.mean().item()


def _collect_short_verses(load_dataset_fn, Audio_cls, dataset_name: str, num_samples: int, max_words: int = 3):
    """Stream the dataset, keep the first `num_samples` verses with ≤ `max_words` words.

    Uses `decode=False` on the audio column to bypass `torchcodec` and decode raw
    bytes ourselves with soundfile (matches the production audio pipeline).
    """
    print(f"spike: streaming {dataset_name} (filtering to ≤{max_words} words per verse)...")
    ds = load_dataset_fn(dataset_name, split="train", streaming=True)
    # Get raw bytes instead of pre-decoded arrays — sidesteps torchcodec on Windows
    ds = ds.cast_column("audio", Audio_cls(decode=False))

    collected = []
    scanned = 0
    for row in ds:
        scanned += 1
        text = row.get("text") or row.get("transcription") or row.get("sentence") or ""
        if not text:
            continue
        if _word_count(text) > max_words:
            continue
        audio_field = row.get("audio")
        if not audio_field:
            continue
        raw_bytes = audio_field.get("bytes")
        if not raw_bytes:
            continue
        try:
            arr, sr = _decode_audio_bytes(raw_bytes)
        except Exception as e:
            print(f"spike: skip row {scanned} — audio decode failed: {e}")
            continue
        collected.append({
            "text": text.strip(),
            "audio": arr,
            "sampling_rate": sr,
            "reciter": row.get("reciter") or row.get("speaker") or "unknown",
        })
        if len(collected) >= num_samples:
            break
        if scanned > 5000:
            print(f"spike: scanned {scanned} rows without finding enough short verses; stopping early")
            break
    print(f"spike: collected {len(collected)} short verses (scanned {scanned} rows)")
    return collected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-samples", type=int, default=10, help="How many short-verse clips to test")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HuggingFace dataset name")
    parser.add_argument("--max-words", type=int, default=3, help="Maximum words per verse to qualify as short")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)

    np, torch, WhisperProcessor, WhisperForConditionalGeneration, load_dataset, Audio = _lazy_imports()

    # Collect validation samples
    try:
        samples = _collect_short_verses(load_dataset, Audio, args.dataset, args.num_samples, args.max_words)
    except Exception as e:
        print(f"spike: ERROR — failed to load dataset '{args.dataset}': {e}", file=sys.stderr)
        print("spike: try --dataset tarteel-ai/tarteel-1.0 or another alternative", file=sys.stderr)
        return 2
    if not samples:
        print("spike: ERROR — no short verses collected", file=sys.stderr)
        return 2

    # Load model
    print(f"spike: loading {MODEL_NAME} on {args.device}...")
    t_load_start = time.time()
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(args.device)
    model.eval()
    print(f"spike: model loaded in {time.time() - t_load_start:.1f}s")

    rows = []
    for i, sample in enumerate(samples, 1):
        audio_16k = _resample_to_16k(sample["audio"], sample["sampling_rate"])

        t0 = time.time()
        transcription, decode_score = _free_decode(model, processor, audio_16k, args.device)
        latency = time.time() - t0

        verify_logp = _verify_score(model, processor, audio_16k, sample["text"], args.device)

        norm_truth = _normalize_for_compare(sample["text"])
        norm_pred = _normalize_for_compare(transcription)
        match = (norm_truth == norm_pred)

        rows.append({
            "idx": i,
            "truth": sample["text"],
            "transcription": transcription,
            "reciter": sample["reciter"],
            "match": match,
            "diacritized_out": _has_diacritics(transcription),
            "latency_s": round(latency, 2),
            "decode_score": round(decode_score, 3) if decode_score is not None else None,
            "verify_logp": round(verify_logp, 3),
        })

    # Print results
    print()
    print("spike: results")
    print("=" * 100)
    for r in rows:
        marker = "✓" if r["match"] else "✗"
        print(f"  [{r['idx']:>2}] {marker} truth='{r['truth']}'  →  pred='{r['transcription']}'")
        print(f"        reciter={r['reciter']}  latency={r['latency_s']}s  "
              f"decode_score={r['decode_score']}  verify_logp={r['verify_logp']}  "
              f"diacritized_out={r['diacritized_out']}")
    print("=" * 100)

    # Gate evaluation
    total = len(rows)
    match_count = sum(1 for r in rows if r["match"])
    match_ratio = match_count / total
    avg_latency = sum(r["latency_s"] for r in rows) / total
    avg_verify_logp = sum(r["verify_logp"] for r in rows) / total

    print(f"spike: transcription match rate: {match_count}/{total} ({match_ratio*100:.0f}%)")
    print(f"spike: average inference latency: {avg_latency:.2f}s")
    print(f"spike: average verification logP: {avg_verify_logp:.3f}")
    print()

    free_decode_ok = match_ratio >= PASS_TRANSCRIPTION_RATIO
    latency_ok = avg_latency <= TARGET_LATENCY_SECONDS
    verify_mode_ok = avg_verify_logp >= PARTIAL_VERIFY_LOGP

    if free_decode_ok and latency_ok:
        print("spike: PASS — proceed with free-decode + lexicon-constrained beam path in TarteelBackend")
        return 0
    elif verify_mode_ok and latency_ok:
        print("spike: PARTIAL — free decode unreliable; pivot to verification-mode scoring "
              "(compute logP of target sequence) in TarteelBackend.verify")
        return 1
    else:
        print(f"spike: FAIL — model unsuitable. latency_ok={latency_ok}, "
              f"free_decode_ok={free_decode_ok}, verify_mode_ok={verify_mode_ok}. "
              "Consider whisper-tiny-ar-quran or whisper-medium fallback.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
