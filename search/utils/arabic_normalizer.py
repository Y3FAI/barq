"""Arabic text normalization for search"""

import re
from typing import List


class ArabicNormalizer:
    """
    Normalize Arabic text for better search matching.

    Handles:
    1. Definite article (ال) removal
    2. Character normalization (different forms of alef, ta, etc.)
    3. Punctuation removal
    4. Diacritics removal (if present)
    5. Extra whitespace cleanup
    """

    # Character normalization mappings
    ALEF_FORMS = ['أ', 'إ', 'آ', 'ٱ']  # Different alef forms
    ALEF_NORMALIZED = 'ا'  # Normalize to basic alef

    YA_FORMS = ['ى']  # Alef maksura
    YA_NORMALIZED = 'ي'  # Normalize to ya

    TA_FORMS = ['ة']  # Ta marbuta
    TA_NORMALIZED = 'ه'  # Normalize to ha

    # Diacritics (tashkeel) - usually not in our data but good to handle
    DIACRITICS = [
        'ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ـ'
    ]

    def __init__(self, remove_stopwords: bool = False):
        """
        Initialize normalizer.

        Args:
            remove_stopwords: Whether to remove common Arabic stopwords
        """
        self.remove_stopwords = remove_stopwords

        # Common Arabic stopwords (optional)
        self.stopwords = {
            'في', 'من', 'إلى', 'على', 'عن', 'هذا', 'هذه', 'ذلك',
            'التي', 'الذي', 'التي', 'أن', 'إن', 'لا', 'ما', 'هل',
            'أو', 'لكن', 'ثم', 'قد', 'لقد', 'كان', 'يكون'
        }

    def normalize(self, text: str) -> str:
        """
        Normalize Arabic text.

        Args:
            text: Input Arabic text

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Lowercase (for mixed content)
        text = text.lower()

        # Remove diacritics
        text = self._remove_diacritics(text)

        # Normalize characters
        text = self._normalize_characters(text)

        # Remove punctuation (but keep Arabic letters)
        text = self._remove_punctuation(text)

        # Remove definite article (ال) - most important!
        text = self._remove_definite_article(text)

        # Clean whitespace
        text = self._clean_whitespace(text)

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Normalize and tokenize Arabic text.

        Args:
            text: Input text

        Returns:
            List of normalized tokens
        """
        # Normalize first
        normalized = self.normalize(text)

        # Split by whitespace
        tokens = normalized.split()

        # Remove stopwords if enabled
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]

        # Remove empty tokens
        tokens = [t for t in tokens if t]

        return tokens

    def _remove_diacritics(self, text: str) -> str:
        """Remove Arabic diacritics (tashkeel)"""
        for diacritic in self.DIACRITICS:
            text = text.replace(diacritic, '')
        return text

    def _normalize_characters(self, text: str) -> str:
        """Normalize different forms of Arabic characters"""
        # Normalize alef forms
        for alef_form in self.ALEF_FORMS:
            text = text.replace(alef_form, self.ALEF_NORMALIZED)

        # Normalize ya forms
        for ya_form in self.YA_FORMS:
            text = text.replace(ya_form, self.YA_NORMALIZED)

        # Normalize ta marbuta
        for ta_form in self.TA_FORMS:
            text = text.replace(ta_form, self.TA_NORMALIZED)

        return text

    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation but keep Arabic letters and numbers"""
        # Keep Arabic letters (0600-06FF), numbers, and spaces
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\s\d]', ' ', text)
        return text

    def _remove_definite_article(self, text: str) -> str:
        """
        Remove Arabic definite article (ال) from the beginning of words.

        This is the KEY normalization for Arabic search!

        Examples:
            الميلاد → ميلاد
            الرخصة → رخصة
            السجل → سجل
        """
        # Remove ال at word boundaries
        # \b matches word boundary
        text = re.sub(r'\bال(\w)', r'\1', text)

        return text

    def _clean_whitespace(self, text: str) -> str:
        """Clean up extra whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Trim
        text = text.strip()
        return text


# Quick test function
def test_normalizer():
    """Test the normalizer"""
    normalizer = ArabicNormalizer()

    test_cases = [
        ("شهادة الميلاد", "شهاده ميلاد"),  # Remove ال, normalize ة
        ("الرخصة القيادة", "رخصه قياده"),  # Remove both ال
        ("إصدار السجل", "اصدار سجل"),      # Normalize إ to ا, remove ال
        ("أنا ذاهب إلى المدرسة", "انا ذاهب الي مدرسه"),  # Multiple normalizations
    ]

    print("Testing Arabic Normalizer:")
    print("="*60)

    for original, expected in test_cases:
        result = normalizer.normalize(original)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{original}'")
        print(f"  → '{result}'")
        if result != expected:
            print(f"  Expected: '{expected}'")
        print()


if __name__ == "__main__":
    test_normalizer()
