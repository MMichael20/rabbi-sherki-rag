import pytest
from src.agent.rag_agent import RagAgent

def test_build_context():
    """Should format retrieved chunks into a context string."""
    chunks = [
        {"text": "הרב שרקי אומר שאמונה היא יסוד", "metadata": {"title": "שיעור אמונה", "url": "https://yt.com/1", "date": ""}},
        {"text": "האמונה מחברת את האדם לבורא", "metadata": {"title": "שיעור אמונה", "url": "https://yt.com/1", "date": ""}},
    ]
    context = RagAgent.build_context(chunks)
    assert "הרב שרקי אומר שאמונה היא יסוד" in context
    assert "שיעור אמונה" in context
    assert "https://yt.com/1" in context

def test_build_system_prompt():
    """System prompt should instruct Hebrew response with source citations."""
    prompt = RagAgent.build_system_prompt()
    assert "Hebrew" in prompt or "עברית" in prompt


def test_build_context_with_timestamp_deep_link():
    """Should append &t=Xs to YouTube URLs when start_seconds is present."""
    chunks = [
        {
            "text": "הרב שרקי מדבר על אמונה",
            "metadata": {
                "title": "שיעור אמונה",
                "url": "https://www.youtube.com/watch?v=abc123",
                "date": "2024-01-15",
                "start_seconds": 872,
            },
        },
    ]
    context = RagAgent.build_context(chunks)
    assert "https://www.youtube.com/watch?v=abc123&t=872s" in context
    assert "14:32" in context


def test_build_context_with_timestamp_no_youtube():
    """Non-YouTube URLs should not get &t= appended."""
    chunks = [
        {
            "text": "תוכן מאתר",
            "metadata": {
                "title": "מאמר",
                "url": "https://example.com/article",
                "date": "",
                "start_seconds": 120,
            },
        },
    ]
    context = RagAgent.build_context(chunks)
    assert "&t=" not in context
    assert "https://example.com/article" in context


def test_build_context_without_timestamp():
    """When start_seconds is -1, no timestamp info should appear."""
    chunks = [
        {
            "text": "תוכן ללא חותמת זמן",
            "metadata": {
                "title": "שיעור",
                "url": "https://www.youtube.com/watch?v=xyz",
                "date": "",
                "start_seconds": -1,
            },
        },
    ]
    context = RagAgent.build_context(chunks)
    assert "&t=" not in context
    assert "https://www.youtube.com/watch?v=xyz" in context
