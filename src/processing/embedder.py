"""Embedding pipeline: reads transcripts, chunks them, stores in vector DB."""
from pathlib import Path
from src.storage.db import ContentDB
from src.storage.vectorstore import VectorStore
from src.processing.cleaner import HebrewCleaner
from src.processing.chunker import TextChunker

class EmbeddingPipeline:
    """Processes transcribed content into searchable embeddings."""

    def __init__(self, db: ContentDB, vectorstore: VectorStore):
        self.db = db
        self.vectorstore = vectorstore

    def process_item(self, item_id: int):
        """Clean, chunk, and embed a single transcribed item."""
        item = self.db.get_by_id(item_id)
        if not item:
            return
        text_path = item.get("transcript_path") or item.get("raw_path")
        if not text_path:
            return
        text = Path(text_path).read_text(encoding="utf-8")
        text = HebrewCleaner.clean(text)
        chunks = TextChunker.chunk(text)
        if not chunks:
            return
        metadatas = [
            {
                "source": item["source_type"] or "",
                "title": item.get("title") or "",
                "url": item.get("url") or "",
                "date": item.get("content_date") or "",
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]
        self.vectorstore.add_chunks(
            doc_id=str(item_id), chunks=chunks, metadatas=metadatas,
        )
        self.db.update_status(item_id, "embedded")

    def process_all_pending(self, limit: int = 0):
        """Process all transcribed and scraped website content."""
        count = 0
        for item in self.db.get_by_status("transcribed"):
            if 0 < limit <= count:
                break
            print(f"Embedding: {item['title']}")
            self.process_item(item["id"])
            count += 1
        for item in self.db.get_by_status("scraped"):
            if 0 < limit <= count:
                break
            if item["source_type"] == "website":
                print(f"Embedding: {item['title']}")
                self.process_item(item["id"])
                count += 1
