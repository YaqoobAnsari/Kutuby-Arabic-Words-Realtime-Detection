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

# Common Arabic endings that are phonetically weak/silent
# The model often struggles with these - they're barely audible
WEAK_ENDINGS = {
    'ة': 'ه',   # Taa marbuta → often sounds like light 'h' or is silent
    'ه': '',    # Final haa → sometimes dropped
    'ا': '',    # Final alef → elongation, sometimes dropped
    'ى': '',    # Alef maqsura → similar to alef
}

# Equivalent final characters (sound similar)
EQUIVALENT_ENDINGS = [
    ('ة', 'ه'),   # Taa marbuta ≈ Haa (both sound like 'h')
    ('ة', ''),    # Taa marbuta ≈ nothing (often silent)
    ('ه', ''),    # Final haa ≈ nothing
    ('ا', 'ى'),   # Alef ≈ Alef maqsura
    ('ي', 'ى'),   # Yaa ≈ Alef maqsura
]


def check_weak_ending_match(transcription: str, target: str) -> tuple[bool, str]:
    """
    Check if transcription matches target when accounting for weak Arabic endings.

    The model often drops or misrecognizes weak endings like:
    - ة (taa marbuta) - often silent or sounds like 'h'
    - ه (final haa) - sometimes dropped
    - ا (final alef) - elongation, sometimes dropped
    - ى (alef maqsura) - similar to alef

    Args:
        transcription: Normalized transcribed text
        target: Normalized target text

    Returns:
        Tuple of (matches: bool, reason: str)
        - matches: True if they match considering weak endings
        - reason: Description of why it matched (for debugging)

    Examples:
        >>> check_weak_ending_match("بقر", "بقرة")
        (True, "target ends with weak ة, transcription matches without it")
        >>> check_weak_ending_match("الله", "الله")
        (True, "exact match")
        >>> check_weak_ending_match("من", "في")
        (False, "no weak ending match")
    """
    # Already exact match
    if transcription == target:
        return (True, "exact match")

    # Case 1: Target has weak ending that transcription is missing
    # E.g., target="بقرة", transcription="بقر"
    for weak_char, replacement in WEAK_ENDINGS.items():
        if target.endswith(weak_char):
            # Check if transcription matches target without the weak ending
            target_without_ending = target[:-1]
            if transcription == target_without_ending:
                return (True, f"target ends with weak {weak_char}, transcription matches without it")

            # Check if transcription matches target with replacement
            if replacement and transcription == target_without_ending + replacement:
                return (True, f"target ends with {weak_char}, transcription has equivalent {replacement}")

    # Case 2: Check equivalent endings
    # E.g., target="مرحبى", transcription="مرحبا"
    for char1, char2 in EQUIVALENT_ENDINGS:
        # Target ends with char1, transcription ends with char2
        if target.endswith(char1) and transcription.endswith(char2):
            if target[:-1] == transcription[:-1]:
                return (True, f"equivalent endings: {char1} ≈ {char2}")
        # Or vice versa
        if char2 and target.endswith(char2) and transcription.endswith(char1):
            if target[:-1] == transcription[:-1]:
                return (True, f"equivalent endings: {char2} ≈ {char1}")

    # Case 3: Transcription has extra weak ending (model hallucinated)
    # E.g., target="بقر", transcription="بقرة" (less common but possible)
    for weak_char in WEAK_ENDINGS.keys():
        if transcription.endswith(weak_char):
            trans_without_ending = transcription[:-1]
            if trans_without_ending == target:
                return (True, f"transcription has extra weak {weak_char}")

    return (False, "no weak ending match")


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
        98.0
        >>> get_dynamic_threshold(4)
        90.0
        >>> get_dynamic_threshold(10)
        80.0
    """
    # INCREASED THRESHOLDS - Previous values were too loose
    if word_length <= 2:
        return 98.0  # Nearly exact for very short words (1-2 chars)
    elif word_length <= 4:
        return 90.0  # Very strict for short words (3-4 chars)
    elif word_length <= 6:
        return 85.0  # High threshold for medium-short words (5-6 chars)
    elif word_length <= 10:
        return 80.0  # Moderate threshold for medium words (7-10 chars)
    else:
        return 75.0  # Slightly lenient for long words (11+ chars)


def fuzzy_match_arabic_words(
    transcription: str,
    target: str,
    custom_threshold: Optional[float] = None
) -> tuple[bool, float]:
    """
    Fuzzy match two Arabic words with dynamic threshold based on word length.

    Uses RapidFuzz (if available) or difflib for fuzzy string matching.
    Automatically normalizes both inputs before comparison.
    Also checks for Arabic-specific weak ending matches (ة, ه, etc.)

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
        >>> fuzzy_match_arabic_words("بقر", "بقرة")
        (True, 95.0)  # Weak ending match - ة is often silent
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

    # Arabic-aware weak ending check
    # This handles cases like بقر vs بقرة where ة is often silent
    weak_match, reason = check_weak_ending_match(norm_trans, norm_target)
    if weak_match:
        # Give high score (95%) for weak ending matches
        # This is intentionally high because the core word matches perfectly
        return (True, 95.0)

    # LENGTH CHECK: Reject if transcription is too short compared to target
    # This prevents "baqr" from matching "baqarah" (missing word endings)
    len_trans = len(norm_trans)
    len_target = len(norm_target)

    if len_target > 0:
        length_ratio = len_trans / len_target
        # Reject if transcription is less than 70% of target length
        # E.g., "بقر" (3) vs "بقرة" (4) = 75% - borderline
        # E.g., "بق" (2) vs "بقرة" (4) = 50% - REJECTED
        if length_ratio < 0.70:
            return (False, round(length_ratio * 50, 2))  # Return low score indicating length mismatch

        # Also reject if transcription is much LONGER than target (wrong word entirely)
        if length_ratio > 1.5:
            return (False, round(50 / length_ratio, 2))

    # Calculate similarity based on available backend
    if FUZZY_BACKEND == "rapidfuzz":
        # Use fuzz.ratio() ONLY - NOT partial_ratio()
        # partial_ratio() returns 100% for substrings which is WRONG for word matching
        # E.g., partial_ratio("baqr", "baqarah") = 100% (INCORRECT!)
        # E.g., fuzz.ratio("baqr", "baqarah") = 60% (CORRECT!)
        similarity_score = fuzz.ratio(norm_trans, norm_target)
    else:
        # Fallback to difflib
        similarity = difflib.SequenceMatcher(None, norm_trans, norm_target).ratio()
        similarity_score = similarity * 100.0  # Convert to 0-100 scale

    # Determine threshold
    if custom_threshold is not None:
        threshold = float(custom_threshold)
    else:
        threshold = get_dynamic_threshold(len_target)

    # Check if similarity meets threshold
    matches = similarity_score >= threshold

    return (matches, round(similarity_score, 2))
