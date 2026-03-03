import pytest
from src.transcription.whisper_transcriber import WhisperTranscriber

def test_vtt_to_text():
    """Should convert VTT subtitle content to clean text."""
    vtt_content = """WEBVTT

00:00:00.000 --> 00:00:05.000
שלום לכולם

00:00:05.000 --> 00:00:10.000
היום נדבר על פרשת השבוע

00:00:10.000 --> 00:00:15.000
שלום לכולם
"""
    text = WhisperTranscriber.vtt_to_text(vtt_content)
    assert "שלום לכולם" in text
    assert "היום נדבר על פרשת השבוע" in text
    assert text.count("שלום לכולם") == 1
    assert "00:00" not in text

def test_clean_transcript():
    """Should clean whitespace and empty lines."""
    raw = "  שורה ראשונה  \n\n\n  שורה שנייה  \n  "
    clean = WhisperTranscriber.clean_transcript(raw)
    assert clean == "שורה ראשונה\nשורה שנייה"
