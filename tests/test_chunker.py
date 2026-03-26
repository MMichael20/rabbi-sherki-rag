import pytest
from src.processing.chunker import TextChunker

def test_chunk_short_text():
    """Short text should return a single chunk."""
    text = "טקסט קצר מאוד."
    chunks = TextChunker.chunk(text, max_tokens=500)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text
    assert chunks[0]["start_seconds"] is None

def test_chunk_by_paragraphs():
    """Should split on paragraph boundaries when possible."""
    paragraphs = ["פסקה " + str(i) + ". " + "מילה " * 50 for i in range(10)]
    text = "\n\n".join(paragraphs)
    chunks = TextChunker.chunk(text, max_tokens=200)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk["text"].split()) <= 250

def test_chunk_preserves_all_text():
    """No text should be lost during chunking."""
    paragraphs = ["פסקה " + str(i) + ". " + "תוכן " * 30 for i in range(5)]
    text = "\n\n".join(paragraphs)
    chunks = TextChunker.chunk(text, max_tokens=100, overlap_tokens=0)
    reconstructed = " ".join(c["text"] for c in chunks)
    for i in range(5):
        assert f"פסקה {i}" in reconstructed

def test_chunk_overlap():
    """Chunks should have overlapping text when overlap_tokens > 0."""
    text = " ".join([f"מילה{i}" for i in range(100)])
    chunks = TextChunker.chunk(text, max_tokens=30, overlap_tokens=10)
    if len(chunks) > 1:
        words_0 = set(chunks[0]["text"].split()[-10:])
        words_1 = set(chunks[1]["text"].split()[:10])
        assert len(words_0 & words_1) > 0

def test_chunk_with_segments():
    """Should return chunk dicts with start_seconds when segments provided."""
    segments = [
        {"start": 0.0, "text": "פסקה ראשונה. " + "מילה " * 40},
        {"start": 30.0, "text": "פסקה שנייה. " + "מילה " * 40},
        {"start": 60.0, "text": "פסקה שלישית. " + "מילה " * 40},
        {"start": 90.0, "text": "פסקה רביעית. " + "מילה " * 40},
    ]
    text = "\n\n".join(s["text"] for s in segments)
    result = TextChunker.chunk(text, segments=segments, max_tokens=150, overlap_tokens=0)
    assert len(result) > 1
    assert all(isinstance(r, dict) for r in result)
    assert all("text" in r and "start_seconds" in r for r in result)
    assert result[0]["start_seconds"] == 0
    assert result[-1]["start_seconds"] > result[0]["start_seconds"]

def test_chunk_without_segments_returns_dicts():
    """Without segments, start_seconds should be None."""
    text = "טקסט קצר מאוד."
    result = TextChunker.chunk(text, max_tokens=500)
    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert result[0]["text"] == text
    assert result[0]["start_seconds"] is None
