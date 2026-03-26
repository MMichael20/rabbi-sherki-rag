# Timestamp-Enriched Chunks

## Problem

The RAG pipeline discards timestamp information during transcription. When the agent cites a source, it can link to the YouTube video but not to the specific moment. Users must scrub through entire lessons to find the referenced content.

## Solution

Preserve timestamps from both VTT subtitles and Whisper transcription, carry them through the chunking pipeline, store them as chunk metadata in ChromaDB, and build deep links (`&t=Xs`) in the RAG agent's citations.

## Approach

**Approach A: Timestamp metadata only.** Store the start time of each chunk as metadata. The chunk text stays clean (no inline timestamps), preserving embedding quality. The agent builds deep links at query time.

## Design

### 1. Transcription — preserve timestamps

**VTT path (`vtt_to_text`):**
- Currently strips `-->` timing lines and returns flat text.
- Change to parse start times from `HH:MM:SS.mmm --> ...` lines.
- Return a list of segments: `[{"start": float_seconds, "text": str}, ...]`.
- Deduplicate text lines as before (VTT often repeats lines across cues).

**Whisper path (`transcribe_audio`):**
- Already produces `[{"start", "end", "text"}]` segments.
- No change needed to the transcription itself.

**`process_item` changes:**
- Both paths produce the same shape: a list of `{"start": float, "text": str}` dicts.
- Always save segments to `.segments.json` (currently only saved for Whisper path).
- Clean transcript `.txt` stays unchanged (flat text, no timestamps).
- Pass segments forward to the embedding pipeline via a new transcript+segments output.

### 2. Chunking — carry timestamps through

**`TextChunker.chunk` changes:**
- New signature: `chunk(text, segments=None, max_tokens=600, overlap_tokens=75)`.
- When `segments` is provided, track which segment's text starts each chunk.
- Return value changes to `list[dict]` with keys `{"text": str, "start_seconds": int | None}`.
- When `segments` is `None` (backward compat), `start_seconds` is `None`.
- The start time of a chunk is the start time of the first segment that contributes text to that chunk.

### 3. Embedding — store timestamp in metadata

**`EmbeddingPipeline` changes:**
- Read segments JSON alongside transcript text.
- Pass segments to `TextChunker.chunk()`.
- Add `start_seconds` (int) to each chunk's metadata dict.
- No changes to ChromaDB schema or vectorstore code needed — ChromaDB accepts arbitrary metadata keys.

### 4. RAG Agent — deep links in citations

**`build_context` changes:**
- When `start_seconds` exists in chunk metadata and URL contains `youtube.com`:
  - Strip any existing `&t=` or `?t=` param from the URL.
  - Append `&t={start_seconds}s` to the URL.
  - Format human-readable timestamp: `(דקה MM:SS)` or `(שעה H:MM:SS)` for long videos.
- Non-YouTube sources: display timestamp text only, no deep link.

## Files Changed

| File | Change |
|------|--------|
| `src/transcription/whisper_transcriber.py` | `vtt_to_text()` returns segments; `process_item` always saves segments JSON |
| `src/processing/chunker.py` | Accept segments, return `(text, start_seconds)` per chunk |
| `src/processing/embedder.py` | Load segments, pass to chunker, add `start_seconds` to metadata |
| `src/agent/rag_agent.py` | Build YouTube deep link URL + display human-readable timestamp |

## Out of Scope

- Re-processing already embedded videos (user will re-run full pipeline)
- Sentence-level timestamp precision (chunk-level is sufficient)
- Training or fine-tuning any model
- Storage schema changes to SQLite
