"""CLI entry point for the Rabbi Sherki RAG pipeline."""
import typer
from pathlib import Path

app = typer.Typer(help="Rabbi Sherki Knowledge Base — Scrape, Transcribe, Embed, Ask")

DB_PATH = "data/db/content.db"
CHROMA_DIR = "data/db/chroma"

def get_db():
    from src.storage.db import ContentDB
    return ContentDB(DB_PATH)

def get_vectorstore():
    from src.storage.vectorstore import VectorStore
    return VectorStore(persist_dir=CHROMA_DIR)

@app.command()
def discover_youtube(channel_url: str):
    """Discover all videos from a YouTube channel or playlist."""
    from src.scrapers.youtube import YouTubeScraper
    db = get_db()
    scraper = YouTubeScraper(db)
    videos = scraper.discover_channel(channel_url)
    typer.echo(f"Discovered {len(videos)} videos.")
    for v in videos[:5]:
        typer.echo(f"  - {v['title']}")
    if len(videos) > 5:
        typer.echo(f"  ... and {len(videos) - 5} more")

@app.command()
def scrape_youtube(limit: int = 0):
    """Download subtitles/audio for all discovered YouTube videos."""
    from src.scrapers.youtube import YouTubeScraper
    db = get_db()
    scraper = YouTubeScraper(db)
    items = db.get_by_status("discovered")
    yt_items = [i for i in items if i["source_type"] == "youtube"]
    if limit > 0:
        yt_items = yt_items[:limit]
    typer.echo(f"Scraping {len(yt_items)} YouTube videos...")
    for item in yt_items:
        typer.echo(f"  Scraping: {item['title']}")
        try:
            scraper.scrape_video(item["id"], item["url"])
        except Exception as e:
            db.update_status(item["id"], "error", str(e))
            typer.echo(f"    Error: {e}")

@app.command()
def scrape_website(url: str, selector: str | None = None):
    """Scrape a single article page."""
    from src.scrapers.websites import WebsiteScraper
    db = get_db()
    scraper = WebsiteScraper(db)
    scraper.scrape_and_save(url, selector)
    typer.echo(f"Scraped: {url}")

@app.command()
def transcribe(model: str = "medium", limit: int = 0):
    """Transcribe all scraped audio/video content using Whisper."""
    from src.transcription.whisper_transcriber import WhisperTranscriber
    db = get_db()
    transcriber = WhisperTranscriber(db, model_name=model)
    transcriber.process_all_pending(limit=limit)
    typer.echo("Transcription complete.")

@app.command()
def embed(limit: int = 0):
    """Chunk and embed all transcribed content into vector store."""
    from src.processing.embedder import EmbeddingPipeline
    db = get_db()
    vs = get_vectorstore()
    pipeline = EmbeddingPipeline(db, vs)
    pipeline.process_all_pending(limit=limit)
    typer.echo(f"Embedding complete. Total chunks in store: {vs.count()}")

@app.command()
def ask(question: str):
    """Ask a question about Rabbi Sherki's teachings."""
    from src.agent.rag_agent import RagAgent
    vs = get_vectorstore()
    agent = RagAgent(vs)
    answer = agent.ask(question)
    typer.echo(f"\n{answer}")

@app.command()
def status():
    """Show pipeline status summary."""
    db = get_db()
    for s in ["discovered", "scraped", "transcribed", "chunked", "embedded", "error"]:
        items = db.get_by_status(s)
        typer.echo(f"  {s}: {len(items)} items")

if __name__ == "__main__":
    app()
