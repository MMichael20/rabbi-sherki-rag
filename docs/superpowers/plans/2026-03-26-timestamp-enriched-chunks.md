# Timestamp-Enriched Chunks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve video timestamps through the transcription/chunking pipeline so the RAG agent can cite specific moments with YouTube deep links.

**Architecture:** VTT and Whisper both produce timestamped segments. The chunker tracks which segment starts each chunk. The embedder stores `start_seconds` in ChromaDB metadata. The RAG agent appends `&t=Xs` to YouTube URLs and displays human-readable timestamps.

**Tech Stack:** Python, faster-whisper, ChromaDB, yt-dlp (VTT format), pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/transcription/whisper_transcriber.py` | Modify | Parse VTT into timestamped segments; unify output from both VTT and Whisper paths |
| `src/processing/chunker.py` | Modify | Accept segments, track start timestamp per chunk |
| `src/processing/embedder.py` | Modify | Load segments JSON, pass to chunker, store `start_seconds` in metadata |
| `src/agent/rag_agent.py` | Modify | Build deep link URLs and display human-readable timestamps |
| `tests/test_transcriber.py` | Modify | Add tests for new `vtt_to_segments` method |
| `tests/test_chunker.py` | Modify | Add tests for segment-aware chunking |
| `tests/test_rag_agent.py` | Modify | Add tests for deep link generation |

---

### Task 1: VTT to timestamped segments

**Files:**
- Modify: `src/transcription/whisper_transcriber.py:32-47`
- Test: `tests/test_transcriber.py`

- [ ] **Step 1: Write failing tests for `vtt_to_segments`**

Add to `tests/test_transcriber.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_transcriber.py -v -k "vtt_to_segments"`
Expected: FAIL — `AttributeError: type object 'WhisperTranscriber' has no attribute 'vtt_to_segments'`

- [ ] **Step 3: Implement `vtt_to_segments`**

In `src/transcription/whisper_transcriber.py`, add this new static method (keep the existing `vtt_to_text` for backward compat):

```python
@staticmethod
def vtt_to_segments(vtt_content: str) -> list[dict]:
    """Parse VTT subtitle content into timestamped segments.

    Returns list of {"start": float_seconds, "text": str} dicts.
    Deduplicates repeated text lines (common in YouTube auto-captions).
    """
    segments = []
    seen_texts = set()
    current_start = None

    for line in vtt_content.split("\n"):
        line = line.strip()
        if not line or line == "WEBVTT" or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        # Parse timestamp line
        if "-->" in line:
            time_part = line.split("-->")[0].strip()
            parts = time_part.replace(",", ".").split(":")
            if len(parts) == 3:
                h, m, s = parts
                current_start = int(h) * 3600 + int(m) * 60 + float(s)
            elif len(parts) == 2:
                m, s = parts
                current_start = int(m) * 60 + float(s)
            continue
        # Text line — strip HTML/VTT tags
        text = re.sub(r"<[^>]+>", "", line).strip()
        text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
        if not text or text in seen_texts:
            continue
        if current_start is not None:
            segments.append({"start": current_start, "text": text})
            seen_texts.add(text)

    return segments
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_transcriber.py -v -k "vtt_to_segments"`
Expected: all 3 tests PASS

- [ ] **Step 5: Update `vtt_to_text` to use `vtt_to_segments`**

Replace the existing `vtt_to_text` method:

```python
@staticmethod
def vtt_to_text(vtt_content: str) -> str:
    """Convert VTT subtitle content to clean deduplicated text."""
    segments = WhisperTranscriber.vtt_to_segments(vtt_content)
    return "\n".join(seg["text"] for seg in segments)
```

- [ ] **Step 6: Run existing `vtt_to_text` tests to verify no regression**

Run: `pytest tests/test_transcriber.py -v`
Expected: all tests PASS (existing + new)

- [ ] **Step 7: Commit**

```bash
git add src/transcription/whisper_transcriber.py tests/test_transcriber.py
git commit -m "feat: add vtt_to_segments to parse VTT with timestamps"
```

---

### Task 2: Update `process_item` to always save segments

**Files:**
- Modify: `src/transcription/whisper_transcriber.py:71-96`

- [ ] **Step 1: Update `process_item` to save segments from both paths**

Replace the `process_item` method:

```python
def process_item(self, item_id: int):
    """Process a single content item."""
    item = self.db.get_by_id(item_id)
    if not item or not item["raw_path"]:
        return
    raw_path = Path(item["raw_path"])
    transcript_path = self.transcripts_dir / f"{raw_path.stem}.txt"
    segments_path = self.transcripts_dir / f"{raw_path.stem}.segments.json"
    try:
        if raw_path.suffix in (".vtt", ".srt"):
            vtt_content = raw_path.read_text(encoding="utf-8")
            segments = self.vtt_to_segments(vtt_content)
            text = "\n".join(seg["text"] for seg in segments)
        elif raw_path.suffix in (".mp3", ".m4a", ".wav", ".opus"):
            text, timed_segments = self.transcribe_audio(str(raw_path))
            segments = [{"start": s["start"], "text": s["text"]} for s in timed_segments]
        else:
            text = raw_path.read_text(encoding="utf-8")
            segments = []
        text = self.clean_transcript(text)
        transcript_path.write_text(text, encoding="utf-8")
        if segments:
            segments_path.write_text(
                json.dumps(segments, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        self.db.set_transcript_path(item_id, str(transcript_path))
        self.db.update_status(item_id, "transcribed")
    except Exception as e:
        self.db.update_status(item_id, "error", str(e))
```

- [ ] **Step 2: Run all transcriber tests**

Run: `pytest tests/test_transcriber.py -v`
Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add src/transcription/whisper_transcriber.py
git commit -m "feat: always save segments JSON from both VTT and Whisper paths"
```

---

### Task 3: Segment-aware chunking

**Files:**
- Modify: `src/processing/chunker.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Write failing tests for segment-aware chunking**

Add to `tests/test_chunker.py`:

```python
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
    # Later chunks should have later timestamps
    assert result[-1]["start_seconds"] > result[0]["start_seconds"]


def test_chunk_without_segments_returns_dicts():
    """Without segments, start_seconds should be None."""
    text = "טקסט קצר מאוד."
    result = TextChunker.chunk(text, max_tokens=500)
    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert result[0]["text"] == text
    assert result[0]["start_seconds"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chunker.py -v -k "segments"`
Expected: FAIL — chunk() returns strings, not dicts

- [ ] **Step 3: Update `TextChunker.chunk` to accept segments and return dicts**

Replace `src/processing/chunker.py`:

```python
"""Text chunking for RAG pipeline."""


class TextChunker:
    """Splits text into chunks suitable for embedding and retrieval."""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate for Hebrew."""
        words = len(text.split())
        return int(words * 1.5)

    @staticmethod
    def _map_segment_starts(text: str, segments: list[dict]) -> dict[int, float]:
        """Map character offsets in the joined text to segment start times.

        Returns a dict of {char_offset: start_seconds} for the first char
        of each segment's text within the full text.
        """
        offset_map = {}
        search_from = 0
        for seg in segments:
            idx = text.find(seg["text"], search_from)
            if idx != -1:
                offset_map[idx] = seg["start"]
                search_from = idx + len(seg["text"])
        return offset_map

    @staticmethod
    def _find_chunk_start(chunk_text: str, full_text: str,
                          offset_map: dict[int, float],
                          search_from: int) -> tuple[float | None, int]:
        """Find the start timestamp for a chunk.

        Returns (start_seconds, char_offset_of_chunk_in_full_text).
        """
        chunk_pos = full_text.find(chunk_text.split("\n\n")[0], search_from)
        if chunk_pos == -1:
            chunk_pos = search_from
        # Find the closest segment start at or before this position
        best_time = None
        for offset, start_time in sorted(offset_map.items()):
            if offset <= chunk_pos:
                best_time = start_time
            else:
                break
        return best_time, chunk_pos

    @staticmethod
    def chunk(text: str, segments: list[dict] | None = None,
              max_tokens: int = 600, overlap_tokens: int = 75) -> list[dict]:
        """Split text into chunks with optional timestamp tracking.

        Args:
            text: The full transcript text.
            segments: Optional list of {"start": float, "text": str} dicts.
            max_tokens: Maximum tokens per chunk.
            overlap_tokens: Token overlap between consecutive chunks.

        Returns:
            List of {"text": str, "start_seconds": int | None} dicts.
        """
        text = text.strip()
        if not text:
            return []

        # Build offset map if segments provided
        offset_map = {}
        if segments:
            offset_map = TextChunker._map_segment_starts(text, segments)

        if TextChunker.estimate_tokens(text) <= max_tokens:
            start = None
            if offset_map:
                start = int(min(offset_map.values()))
            return [{"text": text, "start_seconds": start}]

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        raw_chunks = []
        current_parts = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = TextChunker.estimate_tokens(para)
            if current_tokens + para_tokens > max_tokens and current_parts:
                raw_chunks.append("\n\n".join(current_parts))
                if overlap_tokens > 0:
                    overlap_parts = []
                    overlap_count = 0
                    for p in reversed(current_parts):
                        p_tokens = TextChunker.estimate_tokens(p)
                        if overlap_count + p_tokens > overlap_tokens:
                            break
                        overlap_parts.insert(0, p)
                        overlap_count += p_tokens
                    current_parts = overlap_parts
                    current_tokens = overlap_count
                else:
                    current_parts = []
                    current_tokens = 0

            if para_tokens > max_tokens:
                sentences = [s.strip() for s in para.replace(". ", ".\n").split("\n") if s.strip()]
                for sent in sentences:
                    sent_tokens = TextChunker.estimate_tokens(sent)
                    if current_tokens + sent_tokens > max_tokens and current_parts:
                        raw_chunks.append("\n\n".join(current_parts))
                        current_parts = []
                        current_tokens = 0
                    current_parts.append(sent)
                    current_tokens += sent_tokens
            else:
                current_parts.append(para)
                current_tokens += para_tokens

        if current_parts:
            raw_chunks.append("\n\n".join(current_parts))

        # Assign timestamps to chunks
        if not offset_map:
            return [{"text": c, "start_seconds": None} for c in raw_chunks]

        result = []
        search_from = 0
        for chunk_text in raw_chunks:
            start_time, chunk_pos = TextChunker._find_chunk_start(
                chunk_text, text, offset_map, search_from)
            result.append({
                "text": chunk_text,
                "start_seconds": int(start_time) if start_time is not None else None,
            })
            search_from = chunk_pos
        return result
```

- [ ] **Step 4: Update existing chunker tests for new return format**

Update `tests/test_chunker.py` — existing tests need to extract `["text"]` from dicts:

```python
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
```

- [ ] **Step 5: Run all chunker tests**

Run: `pytest tests/test_chunker.py -v`
Expected: all 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/processing/chunker.py tests/test_chunker.py
git commit -m "feat: segment-aware chunking with start timestamps"
```

---

### Task 4: Embedder loads segments and stores `start_seconds`

**Files:**
- Modify: `src/processing/embedder.py`

- [ ] **Step 1: Update `process_item` to load segments and pass to chunker**

Replace `src/processing/embedder.py`:

```python
"""Embedding pipeline: reads transcripts, chunks them, stores in vector DB."""
import json
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

    @staticmethod
    def _load_segments(transcript_path: str) -> list[dict] | None:
        """Load segments JSON file alongside a transcript, if it exists."""
        tp = Path(transcript_path)
        segments_path = tp.parent / f"{tp.stem}.segments.json"
        if segments_path.exists():
            return json.loads(segments_path.read_text(encoding="utf-8"))
        return None

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
        segments = self._load_segments(text_path) if item.get("transcript_path") else None
        chunks = TextChunker.chunk(text, segments=segments)
        if not chunks:
            return
        metadatas = [
            {
                "source": item["source_type"] or "",
                "title": item.get("title") or "",
                "url": item.get("url") or "",
                "date": item.get("content_date") or "",
                "chunk_index": i,
                "start_seconds": chunk["start_seconds"] if chunk["start_seconds"] is not None else -1,
            }
            for i, chunk in enumerate(chunks)
        ]
        self.vectorstore.add_chunks(
            doc_id=str(item_id),
            chunks=[c["text"] for c in chunks],
            metadatas=metadatas,
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
```

Note: ChromaDB metadata values must be str, int, float, or bool — not None. We use `-1` as sentinel for "no timestamp".

- [ ] **Step 2: Run full test suite to check nothing broke**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add src/processing/embedder.py
git commit -m "feat: store start_seconds in chunk metadata for deep links"
```

---

### Task 5: RAG agent deep links

**Files:**
- Modify: `src/agent/rag_agent.py`
- Test: `tests/test_rag_agent.py`

- [ ] **Step 1: Write failing tests for deep link generation**

Add to `tests/test_rag_agent.py`:

```python
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
    """Non-YouTube URLs should show timestamp text but not append &t=."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rag_agent.py -v -k "timestamp"`
Expected: FAIL — no deep link logic yet

- [ ] **Step 3: Implement deep links in `build_context`**

Replace `build_context` in `src/agent/rag_agent.py`:

```python
@staticmethod
def format_timestamp(seconds: int) -> str:
    """Format seconds into human-readable Hebrew timestamp."""
    if seconds >= 3600:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}"
    m = seconds // 60
    s = seconds % 60
    return f"{m}:{s:02d}"

@staticmethod
def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into context for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        title = meta.get("title", "ללא כותרת")
        url = meta.get("url", "")
        date = meta.get("date", "")
        start_seconds = meta.get("start_seconds", -1)

        header = f"[מקור {i}: {title}"
        if date:
            header += f" | {date}"

        # Add timestamp info
        has_timestamp = isinstance(start_seconds, (int, float)) and start_seconds >= 0
        if has_timestamp:
            header += f" | דקה {RagAgent.format_timestamp(int(start_seconds))}"
        header += "]"

        # Build URL with deep link
        display_url = url
        if has_timestamp and "youtube.com" in url:
            # Strip existing t= param if any
            import re
            display_url = re.sub(r"[&?]t=\d+s?", "", url)
            separator = "&" if "?" in display_url else "?"
            display_url += f"{separator}t={int(start_seconds)}s"

        parts.append(f"{header}\n{display_url}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)
```

- [ ] **Step 4: Run all RAG agent tests**

Run: `pytest tests/test_rag_agent.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/rag_agent.py tests/test_rag_agent.py
git commit -m "feat: RAG agent builds YouTube deep links with timestamps"
```

---

### Task 6: Fix Hebrew subtitle language code (iw/he)

This was already identified and partially fixed earlier in the conversation. Ensure the fix is committed.

**Files:**
- Modify: `src/scrapers/youtube.py` (already modified)

- [ ] **Step 1: Run YouTube scraper tests**

Run: `pytest tests/test_youtube_scraper.py -v`
Expected: all PASS

- [ ] **Step 2: Commit the iw/he fix**

```bash
git add src/scrapers/youtube.py
git commit -m "fix: handle both 'he' and 'iw' Hebrew language codes in YouTube scraper"
```

---

### Task 7: Integration smoke test

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: all tests PASS

- [ ] **Step 2: Manual smoke test with real video**

```bash
python -m yt_dlp --write-auto-subs --sub-lang "he,iw" --sub-format vtt --skip-download -o "data/raw/youtube/test_ts" "https://www.youtube.com/watch?v=lUnTA1g1ZaE"
```

Then in Python:
```python
from src.transcription.whisper_transcriber import WhisperTranscriber
vtt = open("data/raw/youtube/test_ts.iw.vtt", encoding="utf-8").read()
segments = WhisperTranscriber.vtt_to_segments(vtt)
print(f"Segments: {len(segments)}, first: {segments[0]}, last: {segments[-1]}")
```

Expected: segments with start times and Hebrew text.

- [ ] **Step 3: Final commit if any fixes were needed**
