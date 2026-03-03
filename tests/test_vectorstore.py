import pytest
from src.storage.vectorstore import VectorStore

@pytest.fixture
def store(tmp_path):
    return VectorStore(persist_dir=str(tmp_path / "chroma"))

def test_add_and_query(store):
    """Should store chunks and retrieve by similarity."""
    store.add_chunks(
        doc_id="doc1",
        chunks=["הרב שרקי מדבר על אמונה", "שיעור בפרשת בראשית"],
        metadatas=[
            {"source": "youtube", "title": "שיעור אמונה"},
            {"source": "youtube", "title": "שיעור אמונה"},
        ],
    )
    results = store.query("מה הרב אומר על אמונה?", n_results=2)
    assert len(results) > 0
    assert any("אמונה" in r["text"] for r in results)

def test_add_with_metadata_filter(store):
    """Should filter results by metadata."""
    store.add_chunks(
        doc_id="yt1",
        chunks=["שיעור מיוטיוב"],
        metadatas=[{"source": "youtube", "title": "vid"}],
    )
    store.add_chunks(
        doc_id="web1",
        chunks=["מאמר מהאתר"],
        metadatas=[{"source": "website", "title": "art"}],
    )
    results = store.query("שיעור", n_results=5, where={"source": "youtube"})
    assert all(r["metadata"]["source"] == "youtube" for r in results)
