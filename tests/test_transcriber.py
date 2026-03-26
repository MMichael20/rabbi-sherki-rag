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

def test_vtt_to_segments():
    """Should parse VTT into timestamped segments."""
    vtt_content = """WEBVTT

00:00:01.000 --> 00:00:05.000
שלום לכולם

00:00:05.000 --> 00:00:10.000
היום נדבר על פרשת השבוע

00:00:10.000 --> 00:00:15.000
שלום לכולם
"""
    segments = WhisperTranscriber.vtt_to_segments(vtt_content)
    assert len(segments) == 2  # "שלום לכולם" deduplicated
    assert segments[0] == {"start": 1.0, "text": "שלום לכולם"}
    assert segments[1] == {"start": 5.0, "text": "היום נדבר על פרשת השבוע"}


def test_vtt_to_segments_strips_html_tags():
    """Should strip VTT inline tags like <c> and keep clean text."""
    vtt_content = """WEBVTT

00:00:01.000 --> 00:00:03.000
שלום<00:00:01.360><c> וברכה</c><00:00:01.880><c> קהל</c>

00:00:03.000 --> 00:00:03.010
שלום וברכה קהל

00:00:03.010 --> 00:00:06.000
שלום וברכה קהל
בלימודנו המרתק
"""
    segments = WhisperTranscriber.vtt_to_segments(vtt_content)
    assert segments[0]["text"] == "שלום וברכה קהל"
    assert segments[0]["start"] == 1.0
    assert any("בלימודנו המרתק" in s["text"] for s in segments)


def test_vtt_to_segments_empty():
    """Empty or header-only VTT returns empty list."""
    assert WhisperTranscriber.vtt_to_segments("WEBVTT\n\n") == []
    assert WhisperTranscriber.vtt_to_segments("") == []


def test_clean_transcript():
    """Should clean whitespace and empty lines."""
    raw = "  שורה ראשונה  \n\n\n  שורה שנייה  \n  "
    clean = WhisperTranscriber.clean_transcript(raw)
    assert clean == "שורה ראשונה\nשורה שנייה"
