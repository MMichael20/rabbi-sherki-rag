"""ChromaDB vector store for semantic search over content chunks."""
import chromadb
from chromadb.config import Settings

class VectorStore:
    """Manages embeddings and similarity search using ChromaDB."""

    def __init__(self, persist_dir: str = "data/db/chroma",
                 collection_name: str = "sherki_content"):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, doc_id: str, chunks: list[str],
                   metadatas: list[dict] | None = None):
        """Add text chunks with metadata to the vector store."""
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        if metadatas is None:
            metadatas = [{}] * len(chunks)
        self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)

    def query(self, query_text: str, n_results: int = 5,
              where: dict | None = None) -> list[dict]:
        """Search for similar chunks."""
        kwargs = {"query_texts": [query_text], "n_results": n_results}
        if where:
            kwargs["where"] = where
        results = self.collection.query(**kwargs)
        output = []
        for i in range(len(results["documents"][0])):
            output.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
        return output

    def count(self) -> int:
        return self.collection.count()
