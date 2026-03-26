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
        """Map character offsets in the joined text to segment start times."""
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
        """Find the start timestamp for a chunk."""
        chunk_pos = full_text.find(chunk_text.split("\n\n")[0], search_from)
        if chunk_pos == -1:
            chunk_pos = search_from
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

        Returns list of {"text": str, "start_seconds": int | None} dicts.
        """
        text = text.strip()
        if not text:
            return []

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
