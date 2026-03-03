"""Hebrew text cleaning and normalization."""
import re
import unicodedata

class HebrewCleaner:
    """Cleans and normalizes Hebrew text from various sources."""
    NIKUD_PATTERN = re.compile(r"[\u0591-\u05BD\u05BF-\u05C7]")
    TIMESTAMP_PATTERN = re.compile(r"[\[\(]?\d{1,2}:\d{2}(?::\d{2})?[\]\)]?\s*")

    @staticmethod
    def strip_nikud(text: str) -> str:
        return HebrewCleaner.NIKUD_PATTERN.sub("", text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines).strip()

    @staticmethod
    def remove_timestamps(text: str) -> str:
        return HebrewCleaner.TIMESTAMP_PATTERN.sub("", text)

    @staticmethod
    def clean(text: str) -> str:
        text = HebrewCleaner.remove_timestamps(text)
        text = HebrewCleaner.strip_nikud(text)
        text = HebrewCleaner.normalize_whitespace(text)
        return text
