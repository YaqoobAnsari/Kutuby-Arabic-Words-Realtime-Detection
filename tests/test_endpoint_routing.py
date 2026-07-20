# -*- coding: utf-8 -*-
"""
Endpoint routing tests for /verify_word (M4). Uses FastAPI TestClient with MOCKED
backends (no models loaded). The central guarantee: with no `models` field the
response is byte-for-byte the legacy single-backend shape. With `models` set, the
cascade runs and ADDS fields without removing any.
"""
from __future__ import annotations

import io
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient

import app as appmod


# --- fakes ------------------------------------------------------------------

class FakeBackend:
    def __init__(self, variant, result=True, is_mdd=False):
        self.variant = variant
        self.model_name = f"fake/{variant}"
        self._result = result
        self._mdd = is_mdd

    def verify(self, y, target_word, **kwargs):
        r = self._result
        out = {
            "result": r, "transcription": "تجربة", "target_word": target_word.strip(),
            "score": None, "top_k_candidates": None,
            "similarity": 100.0 if r else 0.0, "confidence": 100.0 if r else 0.0,
            "threshold": None, "latency_ms": 1.0,
        }
        if self._mdd:
            out.update({
                "recognized_phonemes": "b a q a r a",
                "canonical_phonemes": "b a q a r a t",
                "gop_score": 100.0 if r else 60.0,
                "mdd_errors": [] if r else [{"type": "sub", "expected": "s", "got": "ʃ",
                                             "feedback": "You said ش where س belongs."}],
                "feedback": ["That sounds correct."] if r else ["You said ش where س belongs."],
            })
        return out

    def transcribe(self, y):
        return {"transcription": "تجربة", "confidence": 100.0, "latency_ms": 1.0}


def _patch_backends(monkeypatch, config, default="fastconformer"):
    def _get(variant=None):
        return config[variant or default]
    monkeypatch.setattr(appmod, "get_backend", _get)


def _wav_bytes(amp=0.5, secs=0.5, sr=16000):
    t = np.arange(int(secs * sr)) / sr
    y = (amp * np.sin(2 * np.pi * 220 * t)).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def _post(client, target="بقرة", models=None):
    data = {"target_word": target}
    if models is not None:
        data["models"] = models
    return client.post("/verify_word", data=data,
                       files={"audio": ("t.wav", _wav_bytes(), "audio/wav")})


LEGACY_KEYS = {
    "result", "transcription", "target_word", "similarity", "confidence",
    "threshold", "decision_basis", "decision_threshold", "threshold_param_applied",
    "processing_time_ms", "latency_ms", "score", "top_k_candidates", "variant", "model",
}
ADDITIVE_KEYS = {"route", "stages", "recognized_phonemes", "canonical_phonemes",
                 "gop_score", "mdd_errors", "feedback"}


# --- the golden backward-compatibility test ---------------------------------

def test_no_models_param_is_byte_identical_legacy_shape(monkeypatch):
    _patch_backends(monkeypatch, {"fastconformer": FakeBackend("fastconformer", result=True)})
    client = TestClient(appmod.app)
    r = _post(client)  # NO models field
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == LEGACY_KEYS, "no-models response must keep exactly the legacy keys"
    assert not (set(body.keys()) & ADDITIVE_KEYS), "no cascade fields when models is absent"
    assert body["result"] is True
    assert body["variant"] == "fastconformer"


def test_models_single_fastconformer_adds_fields_only(monkeypatch):
    _patch_backends(monkeypatch, {"fastconformer": FakeBackend("fastconformer", result=True)})
    client = TestClient(appmod.app)
    r = _post(client, models="fastconformer")
    body = r.json()
    assert LEGACY_KEYS <= set(body.keys())          # nothing removed
    assert ADDITIVE_KEYS <= set(body.keys())        # additive fields present
    assert body["route"] == "fastconformer"
    assert len(body["stages"]) == 1
    assert body["result"] is True


def test_council_passes_on_first_stage_short_circuits(monkeypatch):
    _patch_backends(monkeypatch, {
        "fastconformer": FakeBackend("fastconformer", result=True),
        "mdd": FakeBackend("mdd", result=True, is_mdd=True),
    })
    client = TestClient(appmod.app)
    r = _post(client, models="council")
    body = r.json()
    assert body["result"] is True
    assert body["route"] == "fastconformer"      # mdd never ran (short-circuit)
    assert len(body["stages"]) == 1
    assert body["feedback"] is None


def test_council_falls_through_to_mdd_feedback_on_fail(monkeypatch):
    _patch_backends(monkeypatch, {
        "fastconformer": FakeBackend("fastconformer", result=False),
        "mdd": FakeBackend("mdd", result=False, is_mdd=True),
    })
    client = TestClient(appmod.app)
    r = _post(client, models="council")
    body = r.json()
    assert body["result"] is False
    assert body["route"] == "fastconformer->mdd"
    assert len(body["stages"]) == 2
    assert body["variant"] == "mdd"
    assert body["mdd_errors"], "failed cascade must surface MDD errors"
    assert body["feedback"] and "ش" in body["feedback"][0]


def test_council_fastconformer_fails_mdd_passes(monkeypatch):
    _patch_backends(monkeypatch, {
        "fastconformer": FakeBackend("fastconformer", result=False),
        "mdd": FakeBackend("mdd", result=True, is_mdd=True),
    })
    client = TestClient(appmod.app)
    r = _post(client, models="council")
    body = r.json()
    assert body["result"] is True
    assert body["route"] == "fastconformer->mdd"
    assert body["variant"] == "mdd"


def test_other_endpoints_unchanged(monkeypatch):
    _patch_backends(monkeypatch, {"fastconformer": FakeBackend("fastconformer", result=True)})
    client = TestClient(appmod.app)
    h = client.get("/health")
    assert h.status_code == 200 and h.json()["variant"] == "fastconformer"
    t = client.post("/transcribe_word", files={"audio": ("t.wav", _wav_bytes(), "audio/wav")})
    assert t.status_code == 200 and t.json()["transcription"] == "تجربة"
