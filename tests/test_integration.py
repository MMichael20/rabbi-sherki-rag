"""End-to-end test: add text content -> clean -> chunk -> embed -> query."""
import pytest
from src.storage.db import ContentDB
from src.storage.vectorstore import VectorStore
from src.processing.cleaner import HebrewCleaner
from src.processing.chunker import TextChunker


@pytest.fixture
def pipeline(tmp_path):
    db = ContentDB(str(tmp_path / "test.db"))
    vs = VectorStore(persist_dir=str(tmp_path / "chroma"))
    return db, vs


def test_end_to_end_text_to_query(pipeline):
    """Full pipeline: insert text -> chunk -> embed -> query returns relevant result."""
    db, vs = pipeline

    text = HebrewCleaner.clean(
        "הרב אורי שרקי מלמד שהאמונה בה' היא יסוד העבודה הרוחנית. "
        "האמונה אינה עיוורת אלא מבוססת על הכרה שכלית עמוקה. "
        "הרב שרקי מדגיש שהתורה היא דרך החיים של עם ישראל."
    )
    chunks = TextChunker.chunk(text, max_tokens=100)

    vs.add_chunks(
        doc_id="test_article",
        chunks=chunks,
        metadatas=[{"source": "website", "title": "שיעור אמונה"} for _ in chunks],
    )

    results = vs.query("מה הרב שרקי אומר על אמונה?", n_results=3)
    assert len(results) > 0
    assert any("אמונה" in r["text"] for r in results)
