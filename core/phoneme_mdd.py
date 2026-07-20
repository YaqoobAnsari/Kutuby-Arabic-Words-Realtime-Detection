# -*- coding: utf-8 -*-
"""
Phoneme-level Mispronunciation Detection & Diagnosis (MDD) + Goodness-of-Pronunciation (GOP).

Pure logic. No model and no audio live here. The acoustic phoneme recognition
(running `facebook/wav2vec2-xlsr-53-espeak-cv-ft`) belongs to the backend in
core/inference.py; this module turns a RECOGNIZED phoneme string plus a CANONICAL
phoneme string into:

  - a per-phoneme diff (substitution / insertion / deletion) via global alignment,
  - child-friendly corrective feedback per error,
  - a 0-100 GOP score (from model posteriors when available, else an alignment proxy).

The CANONICAL phonemes are the espeak grapheme-to-phoneme of the diacritized
target word, precomputed for the closed curriculum by
scripts/build_canonical_phonemes.py. Both the canonical and the recognized
strings pass through `normalize_phoneme_string()` so they share ONE inventory
before alignment. That normalizer is the single source of truth for phoneme
cleanup and is imported by the build script too.

Design notes
------------
- No forced alignment: we diff phoneme SYMBOL sequences (Needleman-Wunsch),
  which is symbolic, not acoustic alignment.
- Taa marbuta (ة) is pausal in isolated single-word speech: the correct
  pronunciation of بقرة alone is "baqara", not "baqarat". So a word-final /t/
  derived from ة is marked OPTIONAL and its omission is never an error.
- Emphatics (ص ض ط ظ) carry the espeak superscript wedge (e.g. dˤ). Losing the
  wedge (dˤ -> d) is an emphasis error, detected structurally, not by table.
- Long vowels carry the length mark ː (aː iː uː). Losing it (aː -> a) is a madd
  (shortening) error; adding it is a lengthening error.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

# --- Phoneme normalization (single source of truth) -------------------------

# espeak emits several artifacts that are not distinct Arabic phonemes and must
# be removed so the canonical and recognized inventories line up:
#   - language-switch flags in parentheses: (en) (ar)
#   - syllable-boundary dots inside tokens:  a.  i.ː
#   - combining dental diacritic U+032A:     s̪ t̪  -> s t
#   - stress marks U+02C8 / U+02CC:          ˈ ˌ
#   - primary/secondary tie bars are left intact (they bind affricates like dʒ)
_PAREN_RE = re.compile(r"\([^)]*\)")
_STRESS = "ˈˌ"          # ˈ ˌ
_DENTAL = "̪"                # ̪  (combining bridge below)
_SYLLABLE_DOT = "."
_WORD_SEP = "|"


def normalize_phoneme_string(s: str) -> str:
    """Normalize a raw espeak / recognizer phoneme string to a clean, space-separated
    sequence of comparable phoneme tokens. Idempotent."""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = _PAREN_RE.sub(" ", s)                 # drop (en)/(ar) language flags
    for ch in _STRESS:
        s = s.replace(ch, "")
    s = s.replace(_DENTAL, "")                # dental diacritic -> base consonant
    s = s.replace(_WORD_SEP, " ")             # word separators -> whitespace
    s = s.replace(_SYLLABLE_DOT, "")          # syllable-boundary dots -> nothing
    return " ".join(s.split())


def tokenize(s: str) -> list[str]:
    """Normalize then split into phoneme tokens."""
    return normalize_phoneme_string(s).split()


# Base Arabic phoneme inventory espeak produces for MSA (after normalization).
# Used only to FLAG suspect canonical entries (e.g. non-Arabic leakage), never to
# reject at runtime. A token is Arabic-inventory if, after removing length marks
# and simple gemination, its base is in this set.
_ARABIC_BASE = {
    "ʔ", "b", "t", "θ", "dʒ", "ħ", "χ", "x", "d", "ð", "r", "z", "s", "ʃ",
    "sˤ", "dˤ", "tˤ", "ðˤ", "ʕ", "ɣ", "f", "q", "k", "l", "m", "n", "h", "w", "j",
    "a", "i", "u",
}


def _base_of(tok: str) -> str:
    """Strip length mark and trivial gemination to get a comparable base token."""
    return tok.replace("ː", "").strip()


def is_arabic_inventory(tok: str) -> bool:
    b = _base_of(tok)
    if b in _ARABIC_BASE:
        return True
    # gemination written as a doubled base (e.g. dˤdˤ, tt) or Xː
    for base in _ARABIC_BASE:
        if b == base + base:
            return True
    return False


# --- Arabic phoneme guide (for feedback) ------------------------------------

# IPA(espeak) base -> (Arabic letter, short English name, articulation tip)
PHONEME_GUIDE: dict[str, tuple[str, str, str]] = {
    "ʔ": ("ء", "hamza", "a light stop from the top of the throat"),
    "b": ("ب", "baa", "close your lips then release"),
    "t": ("ت", "taa", "tongue tip on the ridge behind the teeth"),
    "θ": ("ث", "thaa", "tongue tip between the teeth, soft"),
    "dʒ": ("ج", "jeem", "middle of the tongue to the roof"),
    "ħ": ("ح", "Haa", "breathe out hard from the middle of the throat"),
    "χ": ("خ", "khaa", "a raspy sound from the back of the throat"),
    "x": ("خ", "khaa", "a raspy sound from the back of the throat"),
    "d": ("د", "daal", "tongue tip on the ridge behind the teeth"),
    "ð": ("ذ", "dhaal", "tongue tip between the teeth, voiced"),
    "r": ("ر", "raa", "tap the tongue tip on the ridge"),
    "z": ("ز", "zaay", "like س but with your voice on"),
    "s": ("س", "seen", "a thin hiss, tongue near the teeth"),
    "ʃ": ("ش", "sheen", "a wide hush, tongue spread"),
    "sˤ": ("ص", "Saad", "a heavy س, tongue pulled back and down"),
    "dˤ": ("ض", "Daad", "a heavy د, press the tongue side to the molars"),
    "tˤ": ("ط", "Taa", "a heavy ت, tongue full and back"),
    "ðˤ": ("ظ", "DHaa", "a heavy ذ, tongue between teeth and pulled back"),
    "ʕ": ("ع", "ayn", "squeeze from deep in the throat, voiced"),
    "ɣ": ("غ", "ghayn", "a gargle from the back of the throat"),
    "f": ("ف", "faa", "top teeth on the lower lip"),
    "q": ("ق", "qaaf", "from the very back of the tongue, deep"),
    "k": ("ك", "kaaf", "back of the tongue to the soft palate"),
    "l": ("ل", "laam", "tongue tip up, air along the sides"),
    "m": ("م", "meem", "close your lips, hum"),
    "n": ("ن", "noon", "tongue tip up, hum through the nose"),
    "h": ("ه", "haa", "a soft breath from the bottom of the throat"),
    "w": ("و", "waaw", "round your lips"),
    "j": ("ي", "yaa", "spread the tongue high and forward"),
    "a": ("َ", "fatha", "an open 'a'"),
    "i": ("ِ", "kasra", "an 'i' as in 'in'"),
    "u": ("ُ", "damma", "an 'u' as in 'put'"),
    "aː": ("ا", "alif (long a)", "hold the 'aa' longer"),
    "iː": ("ي", "long ee", "hold the 'ee' longer"),
    "uː": ("و", "long oo", "hold the 'oo' longer"),
}


def _letter(tok: str) -> str:
    """Arabic letter for a phoneme token, best effort."""
    info = PHONEME_GUIDE.get(tok) or PHONEME_GUIDE.get(_base_of(tok))
    return info[0] if info else tok


def _has_emphasis(tok: str) -> bool:
    return "ˤ" in tok


def _is_long(tok: str) -> bool:
    return "ː" in tok


# --- Alignment (Needleman-Wunsch) -------------------------------------------

# Op tuple: (kind, expected, got, exp_idx, got_idx)
#   kind in {"match", "sub", "del", "ins"}; None where not applicable.
MATCH, SUB, DEL, INS = "match", "sub", "del", "ins"


def _sub_cost(a: str, b: str) -> int:
    """0 if identical, else 1. Kept < indel-pair (2) so a real one-for-one
    substitution aligns as a single SUB rather than an insert+delete pair."""
    return 0 if a == b else 1


def align(canonical: list[str], recognized: list[str]) -> list[tuple]:
    """Global alignment of canonical (expected) vs recognized (heard).
    Returns a list of ops walking canonical->recognized."""
    n, m = len(canonical), len(recognized)
    # dp cost matrix + backpointers
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c_sub = dp[i - 1][j - 1] + _sub_cost(canonical[i - 1], recognized[j - 1])
            c_del = dp[i - 1][j] + 1           # canonical phoneme missing from speech
            c_ins = dp[i][j - 1] + 1           # extra phoneme in speech
            dp[i][j] = min(c_sub, c_del, c_ins)

    # backtrack (prefer sub/match, then del, then ins on ties for stable output)
    ops: list[tuple] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + _sub_cost(canonical[i - 1], recognized[j - 1]):
            exp, got = canonical[i - 1], recognized[j - 1]
            ops.append((MATCH if exp == got else SUB, exp, got, i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append((DEL, canonical[i - 1], None, i - 1, None))
            i -= 1
        else:
            ops.append((INS, None, recognized[j - 1], None, j - 1))
            j -= 1
    ops.reverse()
    return ops


# --- Error classification + feedback ----------------------------------------

def _feedback_for_op(kind: str, expected: Optional[str], got: Optional[str]) -> str:
    if kind == SUB:
        exp_l, got_l = _letter(expected), _letter(got)
        # emphasis lost/gained on the same base (e.g. sˤ vs s)
        if _base_of(expected).rstrip("ˤ") == _base_of(got).rstrip("ˤ"):
            if _has_emphasis(expected) and not _has_emphasis(got):
                info = PHONEME_GUIDE.get(expected)
                tip = info[2] if info else "make it heavier and pull the tongue back"
                return f"The {exp_l} should be heavy (emphatic): {tip}. You said it light like {got_l}."
            if _has_emphasis(got) and not _has_emphasis(expected):
                return f"The {exp_l} should be light, not heavy. Relax the tongue."
        # length lost/gained (madd) on the same vowel base
        if _base_of(expected).replace("ː", "") == _base_of(got).replace("ː", ""):
            if _is_long(expected) and not _is_long(got):
                return f"Hold the {exp_l} longer (it is a long vowel, madd)."
            if _is_long(got) and not _is_long(expected):
                return f"Make the {exp_l} shorter (it is a short vowel)."
        info = PHONEME_GUIDE.get(expected) or PHONEME_GUIDE.get(_base_of(expected))
        tip = f" Try: {info[2]}." if info else ""
        return f"You said {got_l} where {exp_l} belongs.{tip}"
    if kind == DEL:
        exp_l = _letter(expected)
        info = PHONEME_GUIDE.get(expected) or PHONEME_GUIDE.get(_base_of(expected))
        tip = f" {info[2].capitalize()}." if info else ""
        return f"You missed the {exp_l} sound.{tip}"
    if kind == INS:
        got_l = _letter(got)
        return f"You added an extra {got_l} sound that should not be there."
    return ""


def classify(alignment: list[tuple], optional_exp_idx: Optional[set] = None) -> list[dict]:
    """Turn alignment ops into structured errors, skipping optional (pausal)
    canonical phonemes that were simply omitted."""
    optional_exp_idx = optional_exp_idx or set()
    errors: list[dict] = []
    for kind, expected, got, exp_idx, got_idx in alignment:
        if kind == MATCH:
            continue
        if kind == DEL and exp_idx in optional_exp_idx:
            continue  # pausal taa marbuta etc.: omission is correct
        errors.append({
            "type": kind,
            "expected": expected,
            "got": got,
            "expected_letter": _letter(expected) if expected else None,
            "got_letter": _letter(got) if got else None,
            "position": exp_idx if exp_idx is not None else got_idx,
            "feedback": _feedback_for_op(kind, expected, got),
        })
    return errors


# --- Scoring ----------------------------------------------------------------

def accuracy_score(alignment: list[tuple], optional_exp_idx: Optional[set] = None) -> float:
    """Alignment-based 0-100 proxy score (used when no model posteriors are
    available): fraction of canonical phonemes correctly produced. Optional
    (pausal) phonemes do not count against the score when omitted."""
    optional_exp_idx = optional_exp_idx or set()
    total = 0
    correct = 0
    for kind, expected, got, exp_idx, got_idx in alignment:
        if kind == INS:
            continue  # extra phones handled by errors, not denominator
        if kind == DEL and exp_idx in optional_exp_idx:
            continue
        total += 1
        if kind == MATCH:
            correct += 1
    if total == 0:
        return 0.0
    return round(100.0 * correct / total, 1)


def gop_from_posteriors(posteriors: list[float]) -> float:
    """Goodness-of-Pronunciation from the recognizer's per-canonical-phoneme
    posterior probabilities (0..1 each). Mean posterior mapped to 0-100. The
    backend supplies posteriors aligned to canonical phonemes (M3); this is the
    aggregation contract, unit-tested independently of the model."""
    if not posteriors:
        return 0.0
    vals = [max(0.0, min(1.0, float(p))) for p in posteriors]
    return round(100.0 * sum(vals) / len(vals), 1)


# --- Canonical index (curriculum lookup) ------------------------------------

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_CURRICULUM = _REPO / "data" / "curriculum_words.json"
_TAA_MARBUTA = "ة"  # ة


class CanonicalIndex:
    """Lookup from a target word to its canonical phoneme tokens plus the set of
    optional (pausal) phoneme indices. Keyed by the normalized Arabic surface."""

    def __init__(self, entries: dict[str, dict]):
        self._entries = entries

    @classmethod
    def load(cls, path: Path = _DEFAULT_CURRICULUM):
        from core.arabic_utils import normalize_arabic_text
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        entries: dict[str, dict] = {}
        for w in data.get("words", []):
            arabic = w.get("arabic", "") or ""
            bare = w.get("bare", "") or ""
            phon = tokenize(w.get("canonical_phonemes", "") or "")
            optional = set()
            # pausal taa marbuta: final /t/ from a word whose spelling ends in ة
            arabic_stripped = arabic.rstrip("".join(["ـ", " "]))
            if arabic_stripped.endswith(_TAA_MARBUTA) and phon and phon[-1] == "t":
                optional.add(len(phon) - 1)
            entry = {"arabic": arabic, "phonemes": phon, "optional": optional}
            for key in {normalize_arabic_text(arabic), normalize_arabic_text(bare)}:
                if key:
                    entries[key] = entry
        return cls(entries)

    def get(self, target_word: str) -> Optional[dict]:
        from core.arabic_utils import normalize_arabic_text
        return self._entries.get(normalize_arabic_text((target_word or "").strip()))

    def __len__(self):
        return len(self._entries)


# --- Orchestrators ----------------------------------------------------------

def diagnose(recognized_phonemes: str,
             canonical_phonemes,
             optional_exp_idx: Optional[set] = None,
             posteriors: Optional[list[float]] = None) -> dict:
    """Core MDD/GOP over two phoneme sequences.

    recognized_phonemes: raw string from the recognizer.
    canonical_phonemes:  raw string OR pre-tokenized list (the reference).
    optional_exp_idx:    canonical indices whose omission is not an error (pausal).
    posteriors:          optional per-canonical-phoneme probabilities for GOP.
    """
    heard = tokenize(recognized_phonemes)
    canon = canonical_phonemes if isinstance(canonical_phonemes, list) else tokenize(canonical_phonemes)
    optional_exp_idx = optional_exp_idx or set()

    alignment = align(canon, heard)
    errors = classify(alignment, optional_exp_idx)
    gop = gop_from_posteriors(posteriors) if posteriors else accuracy_score(alignment, optional_exp_idx)
    return {
        "is_correct": len(errors) == 0,
        "canonical_phonemes": " ".join(canon),
        "recognized_phonemes": " ".join(heard),
        "errors": errors,
        "error_count": len(errors),
        "gop_score": gop,
        "gop_basis": "posteriors" if posteriors else "alignment_accuracy",
    }


def diagnose_word(recognized_phonemes: str,
                  target_word: str,
                  index: CanonicalIndex,
                  posteriors: Optional[list[float]] = None) -> Optional[dict]:
    """Diagnose against a curriculum target. Returns None if the word is not in
    the index (caller then does the espeak fallback at runtime)."""
    entry = index.get(target_word)
    if entry is None:
        return None
    result = diagnose(
        recognized_phonemes,
        entry["phonemes"],
        optional_exp_idx=entry["optional"],
        posteriors=posteriors,
    )
    result["target_word"] = target_word
    return result
