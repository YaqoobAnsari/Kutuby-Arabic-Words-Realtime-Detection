# -*- coding: utf-8 -*-
"""
Unit tests for the phoneme MDD/GOP core (core/phoneme_mdd.py).

Pure logic: no audio, no model. Runs on synthetic phoneme strings plus the real
curriculum canonical index. Verifies normalization, alignment, error
classification, pausal taa-marbuta handling, scoring, and feedback.
"""
from __future__ import annotations

import core.phoneme_mdd as m


# --- normalization ----------------------------------------------------------

def test_normalize_strips_dots_dentals_flags_stress():
    # syllable dots, dental diacritic, language flags, stress marks all removed
    raw = "(en) b a. s̪ dˤ a.ː (ar) ˈt |"
    norm = m.normalize_phoneme_string(raw)
    assert "." not in norm
    assert "̪" not in norm          # dental bridge
    assert "(" not in norm and ")" not in norm
    assert "ˈ" not in norm          # primary stress
    assert "|" not in norm
    assert norm.split() == ["b", "a", "s", "dˤ", "aː", "t"]


def test_normalize_idempotent():
    once = m.normalize_phoneme_string("b a. q a r a. t")
    twice = m.normalize_phoneme_string(once)
    assert once == twice == "b a q a r a t"


def test_is_arabic_inventory():
    assert m.is_arabic_inventory("dˤ")
    assert m.is_arabic_inventory("aː")
    assert m.is_arabic_inventory("dˤdˤ")   # gemination
    assert not m.is_arabic_inventory("tʃ")  # English affricate
    assert not m.is_arabic_inventory("eɪ")  # English diphthong


# --- alignment + classification ---------------------------------------------

def test_exact_match_no_errors():
    r = m.diagnose("b a q a r", "b a q a r")
    assert r["is_correct"] is True
    assert r["error_count"] == 0
    assert r["gop_score"] == 100.0


def test_substitution_detected():
    # ʃ said where s belongs (ش for س)
    r = m.diagnose("ʃ a l aː t", "s a l aː t")
    subs = [e for e in r["errors"] if e["type"] == "sub"]
    assert len(subs) == 1
    assert subs[0]["expected"] == "s" and subs[0]["got"] == "ʃ"
    assert r["is_correct"] is False
    assert subs[0]["feedback"]


def test_deletion_detected():
    # missing middle 'a'
    r = m.diagnose("b q a r", "b a q a r")
    dels = [e for e in r["errors"] if e["type"] == "del"]
    assert len(dels) == 1
    assert dels[0]["expected"] == "a"
    assert "miss" in dels[0]["feedback"].lower()


def test_insertion_detected():
    # extra 'n' at the end
    r = m.diagnose("b a q a r n", "b a q a r")
    ins = [e for e in r["errors"] if e["type"] == "ins"]
    assert len(ins) == 1
    assert ins[0]["got"] == "n"
    assert "extra" in ins[0]["feedback"].lower()


# --- emphasis + length feedback ---------------------------------------------

def test_emphasis_loss_feedback():
    # dˤ (ض) said as plain d
    r = m.diagnose("d", "dˤ")
    assert r["error_count"] == 1
    fb = r["errors"][0]["feedback"].lower()
    assert "heavy" in fb or "emphatic" in fb


def test_madd_shortening_feedback():
    # long aː said short a
    r = m.diagnose("a", "aː")
    assert r["error_count"] == 1
    assert "longer" in r["errors"][0]["feedback"].lower()


# --- pausal taa marbuta -----------------------------------------------------

def test_pausal_taa_marbuta_omission_is_correct():
    # canonical ends in optional /t/ (from ة); child omits it -> still correct
    canonical = m.tokenize("b a q a r a t")
    r = m.diagnose("b a q a r a", canonical, optional_exp_idx={len(canonical) - 1})
    assert r["is_correct"] is True
    assert r["error_count"] == 0


def test_pausal_taa_marbuta_spoken_is_correct():
    canonical = m.tokenize("b a q a r a t")
    r = m.diagnose("b a q a r a t", canonical, optional_exp_idx={len(canonical) - 1})
    assert r["is_correct"] is True


# --- scoring ----------------------------------------------------------------

def test_gop_from_posteriors():
    assert m.gop_from_posteriors([1.0, 1.0, 1.0]) == 100.0
    assert m.gop_from_posteriors([0.5, 0.5]) == 50.0
    assert m.gop_from_posteriors([]) == 0.0
    assert m.gop_from_posteriors([2.0, -1.0]) == 50.0  # clamped to [0,1]


def test_accuracy_score_partial():
    # 4 of 5 canonical phonemes correct
    r = m.diagnose("b a q a z", "b a q a r")
    assert r["gop_score"] == 80.0


# --- curriculum index integration (real data, no model) ---------------------

def test_canonical_index_loads_and_diagnoses_real_word():
    idx = m.CanonicalIndex.load()
    assert len(idx) > 700
    entry = idx.get("بقرة")
    assert entry is not None
    assert entry["phonemes"], "بقرة should have canonical phonemes"
    # pausal index recorded (بقرة ends in ة, canonical ends in t)
    assert entry["optional"], "final /t/ from ة should be optional"

    # child says the correct pausal form -> correct
    heard = " ".join(entry["phonemes"][:-1])  # drop the pausal t
    res = m.diagnose_word(heard, "بقرة", idx)
    assert res is not None
    assert res["is_correct"] is True


def test_diagnose_word_off_curriculum_returns_none():
    idx = m.CanonicalIndex.load()
    assert m.diagnose_word("k a l b", "زغربيلوxyz", idx) is None
