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
