import pytest
from src.processing.chunker import TextChunker

def test_chunk_short_text():
    """Short text should return a single chunk."""
    text = "טקסט קצר מאוד."
    chunks = TextChunker.chunk(text, max_tokens=500)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_chunk_by_paragraphs():
    """Should split on paragraph boundaries when possible."""
    paragraphs = ["פסקה " + str(i) + ". " + "מילה " * 50 for i in range(10)]
    text = "\n\n".join(paragraphs)
    chunks = TextChunker.chunk(text, max_tokens=200)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 250

def test_chunk_preserves_all_text():
    """No text should be lost during chunking."""
    paragraphs = ["פסקה " + str(i) + ". " + "תוכן " * 30 for i in range(5)]
    text = "\n\n".join(paragraphs)
    chunks = TextChunker.chunk(text, max_tokens=100, overlap_tokens=0)
    reconstructed = " ".join(chunks)
    for i in range(5):
        assert f"פסקה {i}" in reconstructed

def test_chunk_overlap():
    """Chunks should have overlapping text when overlap_tokens > 0."""
    text = " ".join([f"מילה{i}" for i in range(100)])
    chunks = TextChunker.chunk(text, max_tokens=30, overlap_tokens=10)
    if len(chunks) > 1:
        words_0 = set(chunks[0].split()[-10:])
        words_1 = set(chunks[1].split()[:10])
        assert len(words_0 & words_1) > 0
