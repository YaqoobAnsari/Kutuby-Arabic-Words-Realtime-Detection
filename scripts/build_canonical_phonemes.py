#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build canonical (reference) phoneme strings for the closed curriculum word set.

WHY
---
Mispronunciation Detection & Diagnosis (MDD) compares the phonemes the child
actually produced (from the `facebook/wav2vec2-xlsr-53-espeak-cv-ft` recognizer)
against the *canonical* phonemes the target word SHOULD produce. The canonical
side is derived by grapheme-to-phoneme (G2P) from the already-diacritized
curriculum text using espeak-ng, the SAME phoneme inventory the recognizer
emits. No audio and no human-labeled recordings are needed to build this
reference: the diacritized spelling already encodes the correct pronunciation.

WHAT
----
OFFLINE build step. It does NOT touch the serving path. For every entry in
data/curriculum_words.json it adds, purely additively:
  - canonical_phonemes           : space-separated espeak phones (the reference)
  - canonical_source             : provenance ("espeak-ng:ar")
  - canonical_verified           : False   (a teacher flips this after review)
  - canonical_needs_diacritization : True, only when the source text lacks harakat
It also writes data/canonical_phonemes.review.csv for a teacher to verify/correct
the closed set (766 words), and a top-level `canonical_meta` provenance block.

REPRODUCIBILITY
---------------
Run in the SAME espeak environment as production so the inventory matches the
deployed service. Windows espeak-ng (scoop) can crash in espeak_Initialize, so
the canonical is generated in Linux/Docker (matching the prod base image):

  docker run --rm -v "<repo>:/work" -w /work python:3.10-slim bash -c \
    "apt-get update -qq && apt-get install -y -qq espeak-ng >/dev/null && \
     pip install -q phonemizer && python scripts/build_canonical_phonemes.py"

The script also works locally where a usable libespeak-ng is present (it probes
the scoop path on Windows and falls back to the system library on Linux).

SAFETY
------
Additive only. The serving code (`core/inference.py::_load_curriculum`) reads
only the `arabic`/`bare` fields and ignores unknown keys, so adding these fields
cannot change verification behavior. The file is not redeployed until the
container milestone. A byte-diff against the pre-run snapshot proves additivity.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CURRICULUM = REPO / "data" / "curriculum_words.json"
REVIEW_CSV = REPO / "data" / "canonical_phonemes.review.csv"
LANG = "ar"

# Arabic diacritics: tanwin (064B-064D), short vowels (064E-0650), shadda (0651),
# sukun (0652), plus superscript alef (0670). Used only to flag missing tashkeel.
_HARAKAT = set("ًٌٍَُِّْٰ")


def _load_espeak_library() -> None:
    """Point phonemizer at a usable libespeak-ng.

    On Windows (scoop) the DLL lives under ~/scoop/apps/espeak-ng/current/.
    On Linux/Docker the system package is auto-discovered, so this is a no-op.
    """
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except Exception:
        return
    for pattern in (
        "~/scoop/apps/espeak-ng/current/**/libespeak-ng.dll",
        "~/scoop/apps/espeak-ng/current/libespeak-ng.dll",
    ):
        hits = glob.glob(os.path.expanduser(pattern), recursive=True)
        if hits:
            try:
                EspeakWrapper.set_library(hits[0])
            except Exception:
                pass
            return


def _phonemize(texts: list[str]) -> list[str]:
    from phonemizer import phonemize
    from phonemizer.separator import Separator

    out = phonemize(
        texts,
        language=LANG,
        backend="espeak",
        separator=Separator(phone=" ", word=" | ", syllable=""),
        strip=True,
        preserve_punctuation=False,
        with_stress=False,
        language_switch="remove-flags",  # drop (en)/(ar) voice-switch markers
        njobs=1,
    )
    # phonemize returns a str for a single input, a list for a list.
    if isinstance(out, str):
        return [out]
    return list(out)


def _is_diacritized(text: str) -> bool:
    return any(ch in _HARAKAT for ch in (text or ""))


def main() -> int:
    # The phoneme normalizer is the single source of truth, shared with runtime MDD.
    sys.path.insert(0, str(REPO))
    from core.phoneme_mdd import normalize_phoneme_string, is_arabic_inventory

    if not CURRICULUM.exists():
        print(f"ERROR: {CURRICULUM} not found", file=sys.stderr)
        return 2

    data = json.loads(CURRICULUM.read_text(encoding="utf-8"))
    words = data.get("words", [])
    if not words:
        print("ERROR: no words in curriculum", file=sys.stderr)
        return 2

    texts = [w.get("arabic", "") or "" for w in words]

    _load_espeak_library()
    phones = _phonemize(texts)
    if len(phones) != len(words):
        print(f"ERROR: phonemizer returned {len(phones)} results for {len(words)} words",
              file=sys.stderr)
        return 2

    n_ok = 0
    n_undiacritized = 0
    n_suspect = 0
    for w, ph in zip(words, phones):
        norm = normalize_phoneme_string(ph or "")
        w["canonical_phonemes"] = norm
        w["canonical_source"] = "espeak-ng:ar"
        w["canonical_verified"] = False
        if not _is_diacritized(w.get("arabic", "")):
            w["canonical_needs_diacritization"] = True
            n_undiacritized += 1
        # flag entries whose phones include non-Arabic-inventory tokens (usually
        # a malformed source entry, e.g. English annotation text in `arabic`).
        suspect = [t for t in norm.split() if not is_arabic_inventory(t)]
        if suspect:
            w["canonical_needs_review"] = True
            n_suspect += 1
        if norm:
            n_ok += 1

    data["canonical_meta"] = {
        "source": "espeak-ng g2p over the diacritized `arabic` field",
        "phone_separator": "space",
        "verified": False,
        "note": ("Draft reference phonemes for MDD. Teacher review required: verify/correct "
                 "each row in canonical_phonemes.review.csv, then flip canonical_verified."),
    }

    CURRICULUM.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # utf-8-sig so Excel opens the Arabic correctly for the reviewing teacher.
    with REVIEW_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["bare", "arabic", "translit", "canonical_phonemes",
             "verified(Y/N)", "corrected_phonemes", "notes"]
        )
        for w in words:
            writer.writerow([
                w.get("bare", ""), w.get("arabic", ""), w.get("translit", ""),
                w.get("canonical_phonemes", ""), "", "", "",
            ])

    print(f"OK: {n_ok}/{len(words)} words phonemized "
          f"({n_undiacritized} needs-diacritization, {n_suspect} needs-review)")
    print(f"wrote {CURRICULUM}")
    print(f"wrote {REVIEW_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
