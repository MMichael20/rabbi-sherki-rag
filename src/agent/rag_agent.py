"""RAG agent: retrieves relevant content and answers questions via OpenAI."""
from openai import OpenAI
from src.storage.vectorstore import VectorStore


class RagAgent:
    """Answers questions about Rabbi Sherki's teachings using RAG."""

    def __init__(self, vectorstore: VectorStore, model: str = "gpt-4o-mini"):
        self.vectorstore = vectorstore
        self.model = model
        self.client = OpenAI()

    @staticmethod
    def build_system_prompt() -> str:
        return (
            "אתה עוזר מלומד המתמחה בתורתו של הרב אורי שרקי. "
            "ענה על שאלות בעברית בלבד, בהתבסס על הקטעים שמסופקים לך. "
            "ציין את המקורות שלך (שם השיעור וקישור) בתשובה. "
            "אם אין מספיק מידע בקטעים כדי לענות, אמור זאת בכנות. "
            "אל תמציא מידע שלא מופיע בקטעים.\n\n"
            "You are a knowledgeable assistant specializing in the teachings of Rabbi Oury Sherki. "
            "Answer questions in Hebrew only, based on the provided context passages. "
            "Cite your sources (lesson name and link) in the answer. "
            "If there is not enough information in the passages to answer, say so honestly."
        )

    @staticmethod
    def format_timestamp(seconds: int) -> str:
        """Format seconds into human-readable timestamp."""
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
        import re
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

            has_timestamp = isinstance(start_seconds, (int, float)) and start_seconds >= 0
            if has_timestamp:
                header += f" | דקה {RagAgent.format_timestamp(int(start_seconds))}"
            header += "]"

            display_url = url
            if has_timestamp and "youtube.com" in url:
                display_url = re.sub(r"[&?]t=\d+s?", "", url)
                separator = "&" if "?" in display_url else "?"
                display_url += f"{separator}t={int(start_seconds)}s"

            parts.append(f"{header}\n{display_url}\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    def ask(self, question: str, n_results: int = 7) -> str:
        """Ask a question and get an answer based on the knowledge base."""
        results = self.vectorstore.query(question, n_results=n_results)
        if not results:
            return "לא נמצא מידע רלוונטי בבסיס הנתונים."
        context = self.build_context(results)
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": self.build_system_prompt()},
                {
                    "role": "user",
                    "content": f"הקטעים הרלוונטיים:\n\n{context}\n\n---\n\nשאלה: {question}",
                },
            ],
        )
        return response.choices[0].message.content
