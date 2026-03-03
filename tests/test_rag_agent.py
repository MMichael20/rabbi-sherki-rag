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
