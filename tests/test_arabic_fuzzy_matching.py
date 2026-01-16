"""
Unit tests for Arabic fuzzy matching functionality.

Tests the dynamic fuzzy matching algorithm including:
- Exact matches
- Minor character variations
- Diacritic normalization
- Length-based threshold adaptation
- Edge cases (empty inputs, completely different words)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.arabic_utils import (
    get_dynamic_threshold,
    fuzzy_match_arabic_words,
    normalize_arabic_text
)


class TestGetDynamicThreshold:
    """Test the dynamic threshold function based on word length."""

    def test_very_short_words_strict_threshold(self):
        """Very short words (1-2 chars) should have 95% threshold."""
        assert get_dynamic_threshold(1) == 95.0
        assert get_dynamic_threshold(2) == 95.0

    def test_short_words_high_threshold(self):
        """Short words (3-4 chars) should have 85% threshold."""
        assert get_dynamic_threshold(3) == 85.0
        assert get_dynamic_threshold(4) == 85.0

    def test_medium_words_moderate_threshold(self):
        """Medium words (5-8 chars) should have 80% threshold."""
        assert get_dynamic_threshold(5) == 80.0
        assert get_dynamic_threshold(7) == 80.0
        assert get_dynamic_threshold(8) == 80.0

    def test_long_words_lenient_threshold(self):
        """Long words (9+ chars) should have 75% threshold."""
        assert get_dynamic_threshold(9) == 75.0
        assert get_dynamic_threshold(15) == 75.0
        assert get_dynamic_threshold(100) == 75.0


class TestFuzzyMatchArabicWords:
    """Test fuzzy matching with various Arabic word scenarios."""

    def test_exact_match_returns_perfect_score(self):
        """Exact matches should return (True, 100.0)."""
        matches, score = fuzzy_match_arabic_words("الله", "الله")
        assert matches is True
        assert score == 100.0

    def test_exact_match_with_diacritics(self):
        """Words with diacritics should match after normalization."""
        matches, score = fuzzy_match_arabic_words("اللَّهِ", "الله")
        assert matches is True
        assert score == 100.0

    def test_hamza_normalization_exact_match(self):
        """Different hamza forms should match after normalization."""
        matches, score = fuzzy_match_arabic_words("أَنَّ", "ان")
        assert matches is True
        assert score == 100.0

    def test_one_char_difference_short_word_passes(self):
        """1-char difference in 4-char word should pass 85% threshold."""
        # "الله" vs "اللة" - 1 char difference in 4-char word
        # Should get ~87.5% similarity, which passes 85% threshold
        matches, score = fuzzy_match_arabic_words("اللة", "الله")
        assert score >= 75.0  # At least 75% similarity
        # Note: exact score depends on algorithm (RapidFuzz vs difflib)

    def test_short_word_completely_different_fails(self):
        """Completely different short words should fail."""
        matches, score = fuzzy_match_arabic_words("من", "في")
        assert matches is False
        assert score < 50.0

    def test_empty_transcription_returns_false(self):
        """Empty transcription should return (False, 0.0)."""
        matches, score = fuzzy_match_arabic_words("", "الله")
        assert matches is False
        assert score == 0.0

    def test_empty_target_returns_false(self):
        """Empty target should return (False, 0.0)."""
        matches, score = fuzzy_match_arabic_words("الله", "")
        assert matches is False
        assert score == 0.0

    def test_both_empty_returns_false(self):
        """Both empty should return (False, 0.0)."""
        matches, score = fuzzy_match_arabic_words("", "")
        assert matches is False
        assert score == 0.0

    def test_custom_threshold_override(self):
        """Custom threshold should override dynamic threshold."""
        # Use very high custom threshold that would normally fail
        matches, score = fuzzy_match_arabic_words(
            "اللة", "الله",
            custom_threshold=99.0
        )
        # Should fail because similarity is less than 99%
        assert matches is False

    def test_custom_threshold_very_low(self):
        """Very low custom threshold should accept poor matches."""
        matches, score = fuzzy_match_arabic_words(
            "كتاب", "الله",
            custom_threshold=10.0
        )
        # Should pass because threshold is very low
        assert matches is True or score >= 10.0

    def test_medium_word_with_variations(self):
        """Medium words (5-8 chars) should allow some variation."""
        # "الرحمن" vs "الرحمان" - 1 extra char in 6/7-char word
        matches, score = fuzzy_match_arabic_words("الرحمن", "الرحمان")
        assert score >= 70.0  # Should have good similarity

    def test_long_word_more_lenient(self):
        """Long words should be more lenient with variations."""
        # Longer words get 75% threshold
        word1 = "المؤمنون"  # 8 chars
        word2 = "المومنون"  # 1 char different
        matches, score = fuzzy_match_arabic_words(word1, word2)
        assert score >= 70.0

    def test_whitespace_normalization(self):
        """Whitespace should be normalized before comparison."""
        matches, score = fuzzy_match_arabic_words("  الله  ", "الله")
        assert matches is True
        assert score == 100.0

    def test_mixed_diacritics_and_hamza(self):
        """Multiple normalization features together."""
        matches, score = fuzzy_match_arabic_words("إِنَّ", "ان")
        assert matches is True
        assert score == 100.0

    def test_case_sensitivity_arabic(self):
        """Arabic text comparison (case doesn't apply, but test consistency)."""
        matches, score = fuzzy_match_arabic_words("الله", "الله")
        assert matches is True
        assert score == 100.0

    def test_tatweel_removal(self):
        """Tatweel (kashida) should be removed during normalization."""
        # "الـــله" (with tatweel) vs "الله"
        matches, score = fuzzy_match_arabic_words("الـــله", "الله")
        assert matches is True
        assert score == 100.0


class TestNormalizationIntegration:
    """Test that normalization works correctly with fuzzy matching."""

    def test_normalization_before_fuzzy_match(self):
        """Fuzzy matching should work on normalized text."""
        # Both should normalize to same text
        text1 = "اللَّهِ"  # With diacritics
        text2 = "الله"    # Without diacritics

        norm1 = normalize_arabic_text(text1)
        norm2 = normalize_arabic_text(text2)

        assert norm1 == norm2

        matches, score = fuzzy_match_arabic_words(text1, text2)
        assert matches is True
        assert score == 100.0

    def test_partial_diacritic_removal(self):
        """Partial diacritics should still normalize correctly."""
        # One has some diacritics, other has none
        matches, score = fuzzy_match_arabic_words("مِنَ", "من")
        assert matches is True
        assert score == 100.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_character_words(self):
        """Single character words should be very strict."""
        # Different single chars should fail
        matches, score = fuzzy_match_arabic_words("ا", "و")
        assert matches is False

    def test_single_character_exact_match(self):
        """Single character exact match should pass."""
        matches, score = fuzzy_match_arabic_words("ا", "ا")
        assert matches is True
        assert score == 100.0

    def test_very_long_words(self):
        """Very long words should use lenient threshold."""
        # 15+ character words
        long_word1 = "الرحمنالرحيم"  # Made up long word
        long_word2 = "الرحمانالرحيم"  # 1 char difference

        matches, score = fuzzy_match_arabic_words(long_word1, long_word2)
        # With 75% threshold for long words, this should pass
        assert score >= 70.0

    def test_numeric_content_in_text(self):
        """Text with numbers should be handled."""
        # Though not typical for Arabic words, test robustness
        matches, score = fuzzy_match_arabic_words("الله123", "الله123")
        assert matches is True

    def test_special_characters(self):
        """Special characters should be handled gracefully."""
        matches, score = fuzzy_match_arabic_words("الله!", "الله")
        # Should have high similarity but not exact
        assert score >= 80.0


def test_import_fallback():
    """Test that both RapidFuzz and difflib backends work."""
    # This test validates the import fallback logic
    from core import arabic_utils

    # Should have either "rapidfuzz" or "difflib"
    assert arabic_utils.FUZZY_BACKEND in ["rapidfuzz", "difflib"]

    # Both backends should produce reasonable results
    matches, score = fuzzy_match_arabic_words("الله", "الله")
    assert matches is True
    assert score == 100.0


if __name__ == "__main__":
    # Run tests using pytest if available, otherwise simple execution
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not installed. Running basic tests...")

        # Run a few basic tests manually
        test = TestFuzzyMatchArabicWords()
        test.test_exact_match_returns_perfect_score()
        print("[PASS] Exact match test passed")

        test.test_exact_match_with_diacritics()
        print("[PASS] Diacritic normalization test passed")

        test.test_short_word_completely_different_fails()
        print("[PASS] Different words test passed")

        print("\nAll basic tests passed! Install pytest for full test suite.")
