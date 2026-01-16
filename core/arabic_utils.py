"""Arabic Text Normalization Utilities

This module provides utilities for normalizing Arabic text to enable
robust comparison between text with and without diacritics (tashkeel),
hamza variations, and other orthographic differences.
"""

import re
from typing import Optional

# Arabic diacritics (tashkeel) to remove
ARABIC_DIACRITICS = {
    '\u064B',  # ً FATHATAN (tanween fatha)
    '\u064C',  # ٌ DAMMATAN (tanween damma)
    '\u064D',  # ٍ KASRATAN (tanween kasra)
    '\u064E',  # َ FATHA
    '\u064F',  # ُ DAMMA
    '\u0650',  # ِ KASRA
    '\u0651',  # ّ SHADDA
    '\u0652',  # ْ SUKUN
    '\u0653',  # ٓ MADDAH ABOVE
    '\u0654',  # ٔ HAMZA ABOVE
    '\u0655',  # ٕ HAMZA BELOW
    '\u0656',  # ٖ SUBSCRIPT ALEF
    '\u0657',  # ٗ INVERTED DAMMA
    '\u0658',  # ٘ MARK NOON GHUNNA
    '\u0670',  # ٰ SUPERSCRIPT ALEF
}

# Hamza normalization mappings
HAMZA_FORMS = {
    'أ': 'ا',  # U+0623 HAMZA ABOVE → U+0627 ALEF
    'إ': 'ا',  # U+0625 HAMZA BELOW → U+0627 ALEF
    'آ': 'ا',  # U+0622 MADDA ABOVE → U+0627 ALEF
    'ٱ': 'ا',  # U+0671 WASLA → U+0627 ALEF
    'ؤ': 'و',  # U+0624 HAMZA ON WAW → U+0648 WAW
    'ئ': 'ي',  # U+0626 HAMZA ON YA → U+064A YA
}

# Tatweel (text stretching character)
TATWEEL = 'ـ'  # U+0640


def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text for comparison by removing diacritics and normalizing hamza forms.

    This function prepares Arabic text for comparison by:
    1. Removing all diacritical marks (tashkeel)
    2. Normalizing hamza variations to base letter forms
    3. Removing tatweel (kashida/text stretching)
    4. Normalizing whitespace

    Args:
        text: Arabic text to normalize

    Returns:
        Normalized Arabic text ready for comparison

    Examples:
        >>> normalize_arabic_text("اللَّهِ")
        'الله'
        >>> normalize_arabic_text("ذَلِكَ")
        'ذلك'
        >>> normalize_arabic_text("أَنَّ")
        'ان'
    """
    if not text:
        return ""

    normalized = text

    # Step 1: Remove diacritical marks
    for diacritic in ARABIC_DIACRITICS:
        normalized = normalized.replace(diacritic, '')

    # Step 2: Normalize hamza forms
    for hamza_form, base_form in HAMZA_FORMS.items():
        normalized = normalized.replace(hamza_form, base_form)

    # Step 3: Remove tatweel
    normalized = normalized.replace(TATWEEL, '')

    # Step 4: Normalize whitespace
    normalized = normalized.strip()
    normalized = re.sub(r'\s+', ' ', normalized)

    return normalized


def compare_arabic_words(word1: str, word2: str, normalize: bool = True) -> bool:
    """
    Compare two Arabic words for equality, optionally with normalization.

    Args:
        word1: First Arabic word
        word2: Second Arabic word
        normalize: Whether to normalize before comparison (default: True)

    Returns:
        True if words match (after normalization if enabled), False otherwise

    Examples:
        >>> compare_arabic_words("اللَّهِ", "الله")
        True
        >>> compare_arabic_words("أَنَّ", "ان")
        True
        >>> compare_arabic_words("اللَّهِ", "الله", normalize=False)
        False
    """
    if normalize:
        return normalize_arabic_text(word1) == normalize_arabic_text(word2)
    return word1.strip() == word2.strip()


# --------------------------- Fuzzy Matching Utilities ---------------------------

# Import fuzzy matching library with fallback
try:
    from rapidfuzz import fuzz
    FUZZY_BACKEND = "rapidfuzz"
except ImportError:
    import difflib
    FUZZY_BACKEND = "difflib"


def get_dynamic_threshold(word_length: int) -> float:
    """
    Get dynamic fuzzy matching threshold based on word length.

    Shorter words require stricter matching to avoid false positives,
    while longer words can tolerate more character variations.

    Args:
        word_length: Length of the normalized target word

    Returns:
        Threshold value (0-100 scale) for fuzzy matching

    Examples:
        >>> get_dynamic_threshold(2)
        95.0
        >>> get_dynamic_threshold(4)
        85.0
        >>> get_dynamic_threshold(10)
        75.0
    """
    if word_length <= 2:
        return 95.0  # Very strict for short words (1-2 chars)
    elif word_length <= 4:
        return 85.0  # High threshold for short words (3-4 chars)
    elif word_length <= 8:
        return 80.0  # Moderate threshold for medium words (5-8 chars)
    else:
        return 75.0  # More lenient for long words (9+ chars)


def fuzzy_match_arabic_words(
    transcription: str,
    target: str,
    custom_threshold: Optional[float] = None
) -> tuple[bool, float]:
    """
    Fuzzy match two Arabic words with dynamic threshold based on word length.

    Uses RapidFuzz (if available) or difflib for fuzzy string matching.
    Automatically normalizes both inputs before comparison.

    Args:
        transcription: Transcribed Arabic text from audio
        target: Expected Arabic word to match against
        custom_threshold: Optional custom threshold (0-100) to override dynamic threshold

    Returns:
        Tuple of (matches: bool, similarity_score: float)
        - matches: True if similarity >= threshold, False otherwise
        - similarity_score: Similarity percentage (0-100)

    Examples:
        >>> fuzzy_match_arabic_words("الله", "الله")
        (True, 100.0)
        >>> fuzzy_match_arabic_words("اللة", "الله")
        (True, 87.5)  # 1-char difference in 4-char word, passes 85% threshold
        >>> fuzzy_match_arabic_words("من", "في")
        (False, 0.0)  # Completely different short words
    """
    # Normalize both inputs
    norm_trans = normalize_arabic_text(transcription)
    norm_target = normalize_arabic_text(target)

    # Handle empty inputs
    if not norm_trans or not norm_target:
        return (False, 0.0)

    # Fast-path: Check for exact match first
    if norm_trans == norm_target:
        return (True, 100.0)

    # Calculate similarity based on available backend
    if FUZZY_BACKEND == "rapidfuzz":
        # Use RapidFuzz (preferred - faster and more accurate)
        similarity = fuzz.ratio(norm_trans, norm_target)
        # Also try partial_ratio for insertions/deletions
        partial_sim = fuzz.partial_ratio(norm_trans, norm_target)
        # Take maximum of both metrics
        similarity_score = max(similarity, partial_sim)
    else:
        # Fallback to difflib
        similarity = difflib.SequenceMatcher(None, norm_trans, norm_target).ratio()
        similarity_score = similarity * 100.0  # Convert to 0-100 scale

    # Determine threshold
    if custom_threshold is not None:
        threshold = float(custom_threshold)
    else:
        threshold = get_dynamic_threshold(len(norm_target))

    # Check if similarity meets threshold
    matches = similarity_score >= threshold

    return (matches, round(similarity_score, 2))
