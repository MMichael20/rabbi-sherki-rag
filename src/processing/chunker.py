"""Text chunking for RAG pipeline."""


class TextChunker:
    """Splits text into chunks suitable for embedding and retrieval."""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate for Hebrew."""
        words = len(text.split())
        return int(words * 1.5)

    @staticmethod
    def chunk(text: str, max_tokens: int = 600, overlap_tokens: int = 75) -> list[str]:
        """Split text into chunks respecting paragraph boundaries with overlap."""
        text = text.strip()
        if not text:
            return []
        if TextChunker.estimate_tokens(text) <= max_tokens:
            return [text]

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_parts = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = TextChunker.estimate_tokens(para)
            if current_tokens + para_tokens > max_tokens and current_parts:
                chunks.append("\n\n".join(current_parts))
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
                        chunks.append("\n\n".join(current_parts))
                        current_parts = []
                        current_tokens = 0
                    current_parts.append(sent)
                    current_tokens += sent_tokens
            else:
                current_parts.append(para)
                current_tokens += para_tokens

        if current_parts:
            chunks.append("\n\n".join(current_parts))
        return chunks
