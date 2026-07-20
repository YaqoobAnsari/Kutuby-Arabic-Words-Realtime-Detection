# -*- coding: utf-8 -*-
"""
Tests for the multi-backend registry and the PhonemeMDDBackend response glue.

The registry is exercised with a Dummy backend (no model load). The MDD verify
path is exercised with a MOCKED recognizer (the real wav2vec2 model is not
downloaded here): we build the backend via __new__ and inject a fixed phoneme
string, so the test is deterministic and fast. A separate smoke script loads the
real model.
"""
from __future__ import annotations

import numpy as np
import pytest

import core.inference as inf
from core.phoneme_mdd import CanonicalIndex, normalize_phoneme_string


# --- registry ---------------------------------------------------------------

def test_registry_caches_and_defaults_to_env(monkeypatch):
    class Dummy:
        def __init__(self):
            self.variant = "dummy"
            self.model_name = "dummy-model"

    monkeypatch.setitem(inf._BUILDERS, "dummy", Dummy)
    inf._BACKENDS.pop("dummy", None)

    b1 = inf.get_backend("dummy")
    b2 = inf.get_backend("dummy")
    assert b1 is b2, "same variant must return the cached instance"

    # no-argument call resolves to MODEL_VARIANT (backward-compatible behavior)
    monkeypatch.setenv("MODEL_VARIANT", "dummy")
    inf._BACKENDS.pop("dummy", None)
    assert inf.get_backend().variant == "dummy"

    inf._BACKENDS.pop("dummy", None)


def test_registry_unknown_variant_raises():
    with pytest.raises(ValueError):
        inf.get_backend("no-such-variant-xyz")


def test_all_known_variants_registered():
    assert set(inf._BUILDERS) == {"tarteel", "legacy", "fastconformer", "mdd"}


# --- MDD verify glue (mocked recognizer, real curriculum) -------------------

def _mdd_backend_with_output(recognized: str) -> inf.PhonemeMDDBackend:
    b = object.__new__(inf.PhonemeMDDBackend)
    b._normalize = normalize_phoneme_string
    b.index = CanonicalIndex.load()
    b._recognize = lambda audio: normalize_phoneme_string(recognized)
    return b


_STANDARD_KEYS = ("result", "transcription", "target_word", "score",
                  "top_k_candidates", "similarity", "confidence", "threshold",
                  "latency_ms")
_ADDITIVE_KEYS = ("recognized_phonemes", "canonical_phonemes", "gop_score",
                  "mdd_errors", "feedback")


def test_mdd_verify_correct_pausal_passes():
    idx = CanonicalIndex.load()
    entry = idx.get("بقرة")
    heard = " ".join(entry["phonemes"][:-1])  # correct pausal form (no final t)
    b = _mdd_backend_with_output(heard)

    r = b.verify(np.zeros(16000, dtype=np.float32), "بقرة")
    for k in _STANDARD_KEYS:
        assert k in r, f"missing standard key {k}"
    for k in _ADDITIVE_KEYS:
        assert k in r, f"missing additive key {k}"
    assert r["result"] is True
    assert r["gop_score"] == 100.0
    assert r["mdd_errors"] == []


def test_mdd_verify_wrong_phoneme_gives_feedback():
    idx = CanonicalIndex.load()
    entry = idx.get("بقرة")
    heard = ["k"] + entry["phonemes"][1:-1]  # b -> k at the start
    b = _mdd_backend_with_output(" ".join(heard))

    r = b.verify(np.zeros(16000, dtype=np.float32), "بقرة")
    assert r["result"] is False
    assert r["mdd_errors"], "a wrong phoneme must produce at least one error"
    assert isinstance(r["feedback"], list) and r["feedback"]
    assert 0.0 <= r["gop_score"] < 100.0
