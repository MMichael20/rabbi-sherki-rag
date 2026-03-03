# tests/test_youtube_scraper.py
import json
import pytest
from unittest.mock import patch, MagicMock
from src.scrapers.youtube import YouTubeScraper


def test_parse_video_info():
    """Should extract relevant fields from yt-dlp info dict."""
    info = {
        "id": "abc123",
        "title": "שיעור בפרשת בראשית",
        "upload_date": "20240115",
        "duration": 3600,
        "webpage_url": "https://www.youtube.com/watch?v=abc123",
        "subtitles": {"he": [{"ext": "vtt", "url": "http://..."}]},
        "automatic_captions": {},
    }
    result = YouTubeScraper.parse_video_info(info)
    assert result["title"] == "שיעור בפרשת בראשית"
    assert result["url"] == "https://www.youtube.com/watch?v=abc123"
    assert result["date"] == "2024-01-15"
    assert result["duration_seconds"] == 3600
    assert result["has_manual_subs"] is True
    assert result["has_auto_subs"] is False


def test_parse_upload_date():
    """Should convert YYYYMMDD to YYYY-MM-DD."""
    assert YouTubeScraper.parse_upload_date("20240115") == "2024-01-15"
    assert YouTubeScraper.parse_upload_date(None) is None
