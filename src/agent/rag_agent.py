"""RAG agent: retrieves relevant content and answers questions via Claude."""
import anthropic
from src.storage.vectorstore import VectorStore


class RagAgent:
    """Answers questions about Rabbi Sherki's teachings using RAG."""

    def __init__(self, vectorstore: VectorStore, model: str = "claude-sonnet-4-6"):
        self.vectorstore = vectorstore
        self.model = model
        self.client = anthropic.Anthropic()

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
    def build_context(chunks: list[dict]) -> str:
        """Format retrieved chunks into context for the LLM."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            title = meta.get("title", "ללא כותרת")
            url = meta.get("url", "")
            date = meta.get("date", "")
            header = f"[מקור {i}: {title}"
            if date:
                header += f" | {date}"
            header += f"]\n{url}"
            parts.append(f"{header}\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    def ask(self, question: str, n_results: int = 7) -> str:
        """Ask a question and get an answer based on the knowledge base."""
        results = self.vectorstore.query(question, n_results=n_results)
        if not results:
            return "לא נמצא מידע רלוונטי בבסיס הנתונים."
        context = self.build_context(results)
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=self.build_system_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": f"הקטעים הרלוונטיים:\n\n{context}\n\n---\n\nשאלה: {question}",
                }
            ],
        )
        return message.content[0].text
