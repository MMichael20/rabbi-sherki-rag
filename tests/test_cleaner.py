import pytest
from src.processing.cleaner import HebrewCleaner

def test_strip_nikud():
    """Should remove Hebrew vowel marks (nikud)."""
    text_with_nikud = "בְּרֵאשִׁית בָּרָא אֱלֹהִים"
    clean = HebrewCleaner.strip_nikud(text_with_nikud)
    assert clean == "בראשית ברא אלהים"

def test_normalize_whitespace():
    """Should collapse multiple spaces and normalize newlines."""
    text = "שורה   ראשונה  \n\n\n  שורה    שנייה"
    clean = HebrewCleaner.normalize_whitespace(text)
    assert clean == "שורה ראשונה\n\nשורה שנייה"

def test_remove_timestamps():
    """Should remove common timestamp patterns from transcripts."""
    text = "[00:15:30] דבר תורה חשוב\n(1:23:45) המשך השיעור"
    clean = HebrewCleaner.remove_timestamps(text)
    assert "00:15:30" not in clean
    assert "1:23:45" not in clean
    assert "דבר תורה חשוב" in clean

def test_full_clean_pipeline():
    """Should apply all cleaning steps in order."""
    raw = "  [00:01:00] בְּרֵאשִׁית   בָּרָא  \n\n\n אֱלֹהִים  "
    clean = HebrewCleaner.clean(raw)
    assert "בראשית ברא" in clean
    assert "אלהים" in clean
    assert "00:01:00" not in clean
