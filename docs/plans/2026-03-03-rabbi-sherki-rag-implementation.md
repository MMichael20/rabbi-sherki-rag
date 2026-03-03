# Rabbi Sherki RAG — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a modular pipeline that scrapes, transcribes, chunks, embeds, and serves Rabbi Oury Sherki's Hebrew content through a RAG-based Q&A agent.

**Architecture:** Decoupled pipeline stages connected via SQLite (metadata/status tracking) and filesystem (raw content). Each stage is independently runnable via CLI. Vector search powered by ChromaDB, Q&A powered by Claude API.

**Tech Stack:** Python 3.11+, yt-dlp, BeautifulSoup, openai-whisper, sentence-transformers (multilingual-e5-large), ChromaDB, SQLite, anthropic SDK, Typer (CLI)

---

### Task 1: Project Scaffolding & Dependencies

**Files:**
- Create: `src/__init__.py`, `src/scrapers/__init__.py`, `src/transcription/__init__.py`, `src/processing/__init__.py`, `src/storage/__init__.py`, `src/agent/__init__.py`
- Create: `requirements.txt`
- Create: `config/sources.yaml`
- Create: `.gitignore`
- Create: `pyproject.toml`

**Step 1: Initialize git repo**

```bash
cd C:/Learnings/rabbi-sherki-rag
git init
```

**Step 2: Create `.gitignore`**

```gitignore
__pycache__/
*.pyc
.env
data/raw/
data/transcripts/
data/db/
*.egg-info/
dist/
build/
.venv/
```

**Step 3: Create `pyproject.toml`**

```toml
[project]
name = "rabbi-sherki-rag"
version = "0.1.0"
requires-python = ">=3.11"

[project.scripts]
sherki = "src.cli:app"
```

**Step 4: Create `requirements.txt`**

```
yt-dlp>=2024.0.0
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=5.0.0
openai-whisper>=20231117
sentence-transformers>=2.5.0
chromadb>=0.4.0
anthropic>=0.40.0
typer>=0.12.0
pyyaml>=6.0.0
```

**Step 5: Create all `__init__.py` files and directory structure**

```bash
mkdir -p src/scrapers src/transcription src/processing src/storage src/agent
mkdir -p data/raw/youtube data/raw/websites data/raw/audio data/transcripts data/db
mkdir -p config tests
touch src/__init__.py src/scrapers/__init__.py src/transcription/__init__.py
touch src/processing/__init__.py src/storage/__init__.py src/agent/__init__.py
```

**Step 6: Create `config/sources.yaml` with placeholder structure**

```yaml
youtube:
  channels: []
    # - url: "https://www.youtube.com/@channel-name"
    #   name: "Channel Display Name"
  playlists: []
    # - url: "https://www.youtube.com/playlist?list=PLxxxxx"
    #   name: "Playlist Name"

websites:
  []
  # - base_url: "https://example.org.il"
  #   name: "Site Name"
  #   article_pattern: "/articles/*"
  #   selector: "div.article-content"

telegram:
  channels: []
    # - username: "@channel_name"
    #   name: "Channel Name"

facebook:
  pages: []
    # - url: "https://facebook.com/pagename"
    #   name: "Page Name"
```

**Step 7: Create virtual environment and install dependencies**

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
```

**Step 8: Commit**

```bash
git add .
git commit -m "chore: project scaffolding with dependencies and directory structure"
```

---

### Task 2: SQLite Database Layer

**Files:**
- Create: `src/storage/db.py`
- Create: `tests/test_db.py`

**Step 1: Write failing tests for DB layer**

```python
# tests/test_db.py
import os
import pytest
from src.storage.db import ContentDB

@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    return ContentDB(str(db_path))

def test_create_tables(db):
    """DB should create content_items table on init."""
    tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [t[0] for t in tables]
    assert "content_items" in table_names

def test_add_content_item(db):
    """Should insert a content item and return its ID."""
    item_id = db.add_item(
        url="https://youtube.com/watch?v=abc123",
        title="שיעור בפרשת השבוע",
        source_type="youtube",
        content_date="2024-01-15",
    )
    assert item_id == 1

def test_get_item_by_url(db):
    """Should retrieve an item by URL."""
    db.add_item(url="https://example.com/article1", title="מאמר", source_type="website")
    item = db.get_by_url("https://example.com/article1")
    assert item is not None
    assert item["title"] == "מאמר"
    assert item["status"] == "discovered"

def test_duplicate_url_skipped(db):
    """Adding same URL twice should not create duplicate."""
    db.add_item(url="https://example.com/a", title="First", source_type="website")
    db.add_item(url="https://example.com/a", title="Second", source_type="website")
    count = db.execute("SELECT COUNT(*) FROM content_items").fetchone()[0]
    assert count == 1

def test_update_status(db):
    """Should update item status."""
    item_id = db.add_item(url="https://example.com/b", title="Test", source_type="youtube")
    db.update_status(item_id, "transcribed")
    item = db.get_by_id(item_id)
    assert item["status"] == "transcribed"

def test_get_items_by_status(db):
    """Should filter items by status."""
    db.add_item(url="https://a.com", title="A", source_type="youtube")
    db.add_item(url="https://b.com", title="B", source_type="youtube")
    id_c = db.add_item(url="https://c.com", title="C", source_type="youtube")
    db.update_status(id_c, "scraped")

    discovered = db.get_by_status("discovered")
    assert len(discovered) == 2
    scraped = db.get_by_status("scraped")
    assert len(scraped) == 1

def test_get_items_by_source_type(db):
    """Should filter items by source type."""
    db.add_item(url="https://yt.com/1", title="Vid", source_type="youtube")
    db.add_item(url="https://site.com/1", title="Art", source_type="website")

    yt_items = db.get_by_source_type("youtube")
    assert len(yt_items) == 1
    assert yt_items[0]["title"] == "Vid"
```

**Step 2: Run tests to verify they fail**

```bash
cd C:/Learnings/rabbi-sherki-rag
python -m pytest tests/test_db.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.storage.db'`

**Step 3: Implement `src/storage/db.py`**

```python
"""SQLite database for tracking content items through the pipeline."""

import sqlite3
from pathlib import Path


class ContentDB:
    """Manages content item metadata and pipeline status tracking."""

    VALID_STATUSES = ("discovered", "scraped", "transcribed", "chunked", "embedded", "error")

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS content_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                title TEXT,
                source_type TEXT NOT NULL,
                content_date TEXT,
                status TEXT NOT NULL DEFAULT 'discovered',
                raw_path TEXT,
                transcript_path TEXT,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def execute(self, sql, params=()):
        return self.conn.execute(sql, params)

    def add_item(self, url: str, title: str, source_type: str, content_date: str | None = None) -> int:
        try:
            cursor = self.conn.execute(
                "INSERT INTO content_items (url, title, source_type, content_date) VALUES (?, ?, ?, ?)",
                (url, title, source_type, content_date),
            )
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Duplicate URL — return existing ID
            row = self.get_by_url(url)
            return row["id"]

    def get_by_url(self, url: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM content_items WHERE url = ?", (url,)).fetchone()
        return dict(row) if row else None

    def get_by_id(self, item_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM content_items WHERE id = ?", (item_id,)).fetchone()
        return dict(row) if row else None

    def update_status(self, item_id: int, status: str, error_message: str | None = None):
        self.conn.execute(
            "UPDATE content_items SET status = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, error_message, item_id),
        )
        self.conn.commit()

    def set_raw_path(self, item_id: int, path: str):
        self.conn.execute(
            "UPDATE content_items SET raw_path = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (path, item_id),
        )
        self.conn.commit()

    def set_transcript_path(self, item_id: int, path: str):
        self.conn.execute(
            "UPDATE content_items SET transcript_path = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (path, item_id),
        )
        self.conn.commit()

    def get_by_status(self, status: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM content_items WHERE status = ?", (status,)).fetchall()
        return [dict(r) for r in rows]

    def get_by_source_type(self, source_type: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM content_items WHERE source_type = ?", (source_type,)).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_db.py -v
```

Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/storage/db.py tests/test_db.py
git commit -m "feat: SQLite database layer for content tracking"
```

---

### Task 3: YouTube Scraper — Metadata & Subtitles

**Files:**
- Create: `src/scrapers/youtube.py`
- Create: `tests/test_youtube_scraper.py`

**Step 1: Write tests for YouTube scraper helpers**

```python
# tests/test_youtube_scraper.py
import json
import pytest
from unittest.mock import patch, MagicMock
from src.scrapers.youtube import YouTubeScraper


def test_parse_video_info():
    """Should extract relevant fields from yt-dlp info dict."""
    info = {
        "id": "abc123",
        "title": "שיעור בפרשת בראשית",
        "upload_date": "20240115",
        "duration": 3600,
        "webpage_url": "https://www.youtube.com/watch?v=abc123",
        "subtitles": {"he": [{"ext": "vtt", "url": "http://..."}]},
        "automatic_captions": {},
    }
    result = YouTubeScraper.parse_video_info(info)
    assert result["title"] == "שיעור בפרשת בראשית"
    assert result["url"] == "https://www.youtube.com/watch?v=abc123"
    assert result["date"] == "2024-01-15"
    assert result["duration_seconds"] == 3600
    assert result["has_manual_subs"] is True
    assert result["has_auto_subs"] is False


def test_parse_upload_date():
    """Should convert YYYYMMDD to YYYY-MM-DD."""
    assert YouTubeScraper.parse_upload_date("20240115") == "2024-01-15"
    assert YouTubeScraper.parse_upload_date(None) is None
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_youtube_scraper.py -v
```

**Step 3: Implement `src/scrapers/youtube.py`**

```python
"""YouTube scraper using yt-dlp for metadata, subtitles, and audio download."""

import subprocess
import json
import os
from pathlib import Path
from src.storage.db import ContentDB


class YouTubeScraper:
    """Scrapes YouTube channels/playlists for Rabbi Sherki content."""

    def __init__(self, db: ContentDB, raw_dir: str = "data/raw/youtube"):
        self.db = db
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def parse_upload_date(date_str: str | None) -> str | None:
        if not date_str or len(date_str) != 8:
            return None
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    @staticmethod
    def parse_video_info(info: dict) -> dict:
        subs = info.get("subtitles") or {}
        auto_caps = info.get("automatic_captions") or {}
        return {
            "video_id": info.get("id"),
            "title": info.get("title"),
            "url": info.get("webpage_url"),
            "date": YouTubeScraper.parse_upload_date(info.get("upload_date")),
            "duration_seconds": info.get("duration"),
            "has_manual_subs": "he" in subs,
            "has_auto_subs": "he" in auto_caps,
        }

    def discover_channel(self, channel_url: str) -> list[dict]:
        """Fetch metadata for all videos in a channel. Returns list of parsed video info."""
        cmd = [
            "yt-dlp",
            "--flat-playlist",
            "--dump-json",
            "--no-download",
            channel_url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        videos = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            info = json.loads(line)
            # flat-playlist gives minimal info; store URL for detailed fetch later
            url = info.get("url") or f"https://www.youtube.com/watch?v={info.get('id')}"
            title = info.get("title", "Unknown")
            item_id = self.db.add_item(
                url=url,
                title=title,
                source_type="youtube",
            )
            videos.append({"id": item_id, "url": url, "title": title})
        return videos

    def fetch_video_details(self, video_url: str) -> dict:
        """Fetch full metadata for a single video including subtitle availability."""
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-lang", "he",
            video_url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        info = json.loads(result.stdout)
        return self.parse_video_info(info)

    def download_subtitles(self, video_url: str, video_id: str) -> str | None:
        """Download Hebrew subtitles (manual or auto). Returns path to subtitle file or None."""
        out_path = self.raw_dir / video_id
        cmd = [
            "yt-dlp",
            "--write-subs",
            "--write-auto-subs",
            "--sub-lang", "he",
            "--skip-download",
            "--sub-format", "vtt",
            "-o", str(out_path),
            video_url,
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        # yt-dlp creates files like {video_id}.he.vtt
        for ext in [".he.vtt", ".he.srt"]:
            path = Path(str(out_path) + ext)
            if path.exists():
                return str(path)
        return None

    def download_audio(self, video_url: str, video_id: str) -> str:
        """Download audio only for transcription. Returns path to audio file."""
        out_template = str(self.raw_dir / f"{video_id}.%(ext)s")
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "5",  # medium quality, smaller file
            "-o", out_template,
            video_url,
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        return str(self.raw_dir / f"{video_id}.mp3")

    def scrape_video(self, item_id: int, video_url: str):
        """Full scrape pipeline for a single video: subtitles first, audio as fallback."""
        details = self.fetch_video_details(video_url)
        video_id = details["video_id"]

        # Try subtitles first (instant, free)
        sub_path = self.download_subtitles(video_url, video_id)
        if sub_path:
            self.db.set_raw_path(item_id, sub_path)
            self.db.update_status(item_id, "scraped")
            return

        # No subtitles — download audio for Whisper
        audio_path = self.download_audio(video_url, video_id)
        self.db.set_raw_path(item_id, audio_path)
        self.db.update_status(item_id, "scraped")
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_youtube_scraper.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/scrapers/youtube.py tests/test_youtube_scraper.py
git commit -m "feat: YouTube scraper with subtitle and audio download"
```

---

### Task 4: Website Scraper

**Files:**
- Create: `src/scrapers/websites.py`
- Create: `tests/test_website_scraper.py`

**Step 1: Write tests**

```python
# tests/test_website_scraper.py
import pytest
from src.scrapers.websites import WebsiteScraper


def test_clean_html_to_text():
    """Should extract clean text from HTML, stripping tags and scripts."""
    html = """
    <html><body>
    <script>var x = 1;</script>
    <nav>Menu stuff</nav>
    <div class="article-content">
        <h1>שיעור בנושא אמונה</h1>
        <p>זהו שיעור חשוב מאוד.</p>
        <p>נושא השיעור הוא אמונה.</p>
    </div>
    <footer>Copyright</footer>
    </body></html>
    """
    text = WebsiteScraper.extract_text(html, selector="div.article-content")
    assert "שיעור בנושא אמונה" in text
    assert "זהו שיעור חשוב מאוד" in text
    assert "var x = 1" not in text
    assert "Menu stuff" not in text


def test_extract_text_fallback_to_body():
    """If selector not found, should fall back to body text."""
    html = "<html><body><p>תוכן המאמר</p></body></html>"
    text = WebsiteScraper.extract_text(html, selector="div.missing")
    assert "תוכן המאמר" in text
```

**Step 2: Run tests to verify failure**

```bash
python -m pytest tests/test_website_scraper.py -v
```

**Step 3: Implement `src/scrapers/websites.py`**

```python
"""Website scraper for article pages."""

import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from src.storage.db import ContentDB


class WebsiteScraper:
    """Scrapes article pages from configured websites."""

    def __init__(self, db: ContentDB, raw_dir: str = "data/raw/websites"):
        self.db = db
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; SherKiBot/1.0; educational research)"
        })

    @staticmethod
    def extract_text(html: str, selector: str | None = None) -> str:
        soup = BeautifulSoup(html, "lxml")
        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        if selector:
            element = soup.select_one(selector)
            if element:
                return element.get_text(separator="\n", strip=True)

        # Fallback to body
        body = soup.find("body")
        if body:
            return body.get_text(separator="\n", strip=True)
        return soup.get_text(separator="\n", strip=True)

    def scrape_page(self, url: str, selector: str | None = None) -> dict:
        """Fetch and extract text from a single page. Returns {title, text, url}."""
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        response.encoding = response.apparent_encoding  # Handle Hebrew encoding

        soup = BeautifulSoup(response.text, "lxml")
        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else url

        text = self.extract_text(response.text, selector)

        return {"title": title_text, "text": text, "url": url}

    def scrape_and_save(self, url: str, selector: str | None = None, delay: float = 1.0):
        """Scrape a page, save text to file, register in DB."""
        existing = self.db.get_by_url(url)
        if existing and existing["status"] != "discovered":
            return  # Already processed

        try:
            result = self.scrape_page(url, selector)
            # Save text to file
            safe_name = url.replace("https://", "").replace("http://", "").replace("/", "_")[:100]
            file_path = self.raw_dir / f"{safe_name}.txt"
            file_path.write_text(result["text"], encoding="utf-8")

            item_id = self.db.add_item(
                url=url, title=result["title"], source_type="website"
            )
            self.db.set_raw_path(item_id, str(file_path))
            self.db.update_status(item_id, "scraped")
        except Exception as e:
            item_id = self.db.add_item(url=url, title=url, source_type="website")
            self.db.update_status(item_id, "error", str(e))

        time.sleep(delay)  # Be polite
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_website_scraper.py -v
```

**Step 5: Commit**

```bash
git add src/scrapers/websites.py tests/test_website_scraper.py
git commit -m "feat: website scraper with HTML text extraction"
```

---

### Task 5: Whisper Transcription Module

**Files:**
- Create: `src/transcription/whisper_transcriber.py`
- Create: `tests/test_transcriber.py`

**Step 1: Write tests (unit tests that mock Whisper)**

```python
# tests/test_transcriber.py
import pytest
from unittest.mock import patch, MagicMock
from src.transcription.whisper_transcriber import WhisperTranscriber


def test_vtt_to_text():
    """Should convert VTT subtitle content to clean text."""
    vtt_content = """WEBVTT

00:00:00.000 --> 00:00:05.000
שלום לכולם

00:00:05.000 --> 00:00:10.000
היום נדבר על פרשת השבוע

00:00:10.000 --> 00:00:15.000
שלום לכולם
"""
    text = WhisperTranscriber.vtt_to_text(vtt_content)
    assert "שלום לכולם" in text
    assert "היום נדבר על פרשת השבוע" in text
    # Should deduplicate consecutive identical lines
    assert text.count("שלום לכולם") == 1
    # Should not contain timestamps
    assert "00:00" not in text


def test_clean_transcript():
    """Should clean whitespace and empty lines."""
    raw = "  שורה ראשונה  \n\n\n  שורה שנייה  \n  "
    clean = WhisperTranscriber.clean_transcript(raw)
    assert clean == "שורה ראשונה\nשורה שנייה"
```

**Step 2: Run to verify failure**

```bash
python -m pytest tests/test_transcriber.py -v
```

**Step 3: Implement `src/transcription/whisper_transcriber.py`**

```python
"""Whisper-based audio transcription for Hebrew content."""

import re
from pathlib import Path
from src.storage.db import ContentDB


class WhisperTranscriber:
    """Transcribes audio files to Hebrew text using OpenAI Whisper."""

    def __init__(self, db: ContentDB, model_name: str = "medium",
                 transcripts_dir: str = "data/transcripts"):
        self.db = db
        self.model_name = model_name
        self.transcripts_dir = Path(transcripts_dir)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    @property
    def model(self):
        """Lazy-load the Whisper model (heavy, only load when needed)."""
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self.model_name)
        return self._model

    @staticmethod
    def vtt_to_text(vtt_content: str) -> str:
        """Convert VTT subtitle content to clean deduplicated text."""
        lines = []
        prev_line = ""
        for line in vtt_content.split("\n"):
            line = line.strip()
            # Skip VTT header, timestamps, and empty lines
            if not line or line == "WEBVTT" or "-->" in line:
                continue
            # Skip HTML-style tags
            line = re.sub(r"<[^>]+>", "", line)
            if not line:
                continue
            # Deduplicate consecutive identical lines
            if line != prev_line:
                lines.append(line)
                prev_line = line
        return "\n".join(lines)

    @staticmethod
    def clean_transcript(text: str) -> str:
        """Clean whitespace and normalize text."""
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        return "\n".join(lines)

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe an audio file to Hebrew text."""
        result = self.model.transcribe(
            audio_path,
            language="he",
            task="transcribe",
        )
        return self.clean_transcript(result["text"])

    def process_item(self, item_id: int):
        """Process a single content item: convert VTT or transcribe audio."""
        item = self.db.get_by_id(item_id)
        if not item or not item["raw_path"]:
            return

        raw_path = Path(item["raw_path"])
        transcript_path = self.transcripts_dir / f"{raw_path.stem}.txt"

        try:
            if raw_path.suffix in (".vtt", ".srt"):
                # Already have subtitles — just convert to text
                vtt_content = raw_path.read_text(encoding="utf-8")
                text = self.vtt_to_text(vtt_content)
            elif raw_path.suffix in (".mp3", ".m4a", ".wav", ".opus"):
                # Need Whisper transcription
                text = self.transcribe_audio(str(raw_path))
            else:
                # Already text (website scrape)
                text = raw_path.read_text(encoding="utf-8")

            text = self.clean_transcript(text)
            transcript_path.write_text(text, encoding="utf-8")
            self.db.set_transcript_path(item_id, str(transcript_path))
            self.db.update_status(item_id, "transcribed")
        except Exception as e:
            self.db.update_status(item_id, "error", str(e))

    def process_all_pending(self):
        """Process all items with 'scraped' status."""
        items = self.db.get_by_status("scraped")
        for item in items:
            print(f"Processing: {item['title']}")
            self.process_item(item["id"])
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_transcriber.py -v
```

**Step 5: Commit**

```bash
git add src/transcription/whisper_transcriber.py tests/test_transcriber.py
git commit -m "feat: Whisper transcription module with VTT parsing"
```

---

### Task 6: Text Cleaner & Hebrew Normalization

**Files:**
- Create: `src/processing/cleaner.py`
- Create: `tests/test_cleaner.py`

**Step 1: Write tests**

```python
# tests/test_cleaner.py
import pytest
from src.processing.cleaner import HebrewCleaner


def test_strip_nikud():
    """Should remove Hebrew vowel marks (nikud)."""
    text_with_nikud = "בְּרֵאשִׁית בָּרָא אֱלֹהִים"
    clean = HebrewCleaner.strip_nikud(text_with_nikud)
    assert clean == "בראשית ברא אלהים"


def test_normalize_whitespace():
    """Should collapse multiple spaces and normalize newlines."""
    text = "שורה   ראשונה  \n\n\n  שורה    שנייה"
    clean = HebrewCleaner.normalize_whitespace(text)
    assert clean == "שורה ראשונה\n\nשורה שנייה"


def test_remove_timestamps():
    """Should remove common timestamp patterns from transcripts."""
    text = "[00:15:30] דבר תורה חשוב\n(1:23:45) המשך השיעור"
    clean = HebrewCleaner.remove_timestamps(text)
    assert "00:15:30" not in clean
    assert "1:23:45" not in clean
    assert "דבר תורה חשוב" in clean


def test_full_clean_pipeline():
    """Should apply all cleaning steps in order."""
    raw = "  [00:01:00] בְּרֵאשִׁית   בָּרָא  \n\n\n אֱלֹהִים  "
    clean = HebrewCleaner.clean(raw)
    assert "בראשית ברא" in clean
    assert "אלהים" in clean
    assert "00:01:00" not in clean
```

**Step 2: Run to verify failure**

```bash
python -m pytest tests/test_cleaner.py -v
```

**Step 3: Implement `src/processing/cleaner.py`**

```python
"""Hebrew text cleaning and normalization."""

import re
import unicodedata


class HebrewCleaner:
    """Cleans and normalizes Hebrew text from various sources."""

    # Unicode range for Hebrew nikud (vowel marks)
    NIKUD_PATTERN = re.compile(r"[\u0591-\u05BD\u05BF-\u05C7]")
    # Common timestamp patterns in transcripts
    TIMESTAMP_PATTERN = re.compile(r"[\[\(]?\d{1,2}:\d{2}(?::\d{2})?[\]\)]?\s*")

    @staticmethod
    def strip_nikud(text: str) -> str:
        return HebrewCleaner.NIKUD_PATTERN.sub("", text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        # Collapse multiple spaces to one
        text = re.sub(r"[^\S\n]+", " ", text)
        # Collapse 3+ newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip leading/trailing whitespace per line
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines).strip()

    @staticmethod
    def remove_timestamps(text: str) -> str:
        return HebrewCleaner.TIMESTAMP_PATTERN.sub("", text)

    @staticmethod
    def clean(text: str) -> str:
        text = HebrewCleaner.remove_timestamps(text)
        text = HebrewCleaner.strip_nikud(text)
        text = HebrewCleaner.normalize_whitespace(text)
        return text
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_cleaner.py -v
```

**Step 5: Commit**

```bash
git add src/processing/cleaner.py tests/test_cleaner.py
git commit -m "feat: Hebrew text cleaner with nikud stripping"
```

---

### Task 7: Text Chunker

**Files:**
- Create: `src/processing/chunker.py`
- Create: `tests/test_chunker.py`

**Step 1: Write tests**

```python
# tests/test_chunker.py
import pytest
from src.processing.chunker import TextChunker


def test_chunk_short_text():
    """Short text should return a single chunk."""
    text = "טקסט קצר מאוד."
    chunks = TextChunker.chunk(text, max_tokens=500)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_by_paragraphs():
    """Should split on paragraph boundaries when possible."""
    paragraphs = ["פסקה " + str(i) + ". " + "מילה " * 50 for i in range(10)]
    text = "\n\n".join(paragraphs)
    chunks = TextChunker.chunk(text, max_tokens=200)
    assert len(chunks) > 1
    # Each chunk should be within token limit (roughly)
    for chunk in chunks:
        assert len(chunk.split()) <= 250  # generous margin for Hebrew tokenization


def test_chunk_preserves_all_text():
    """No text should be lost during chunking (ignoring overlap)."""
    paragraphs = ["פסקה " + str(i) + ". " + "תוכן " * 30 for i in range(5)]
    text = "\n\n".join(paragraphs)
    chunks = TextChunker.chunk(text, max_tokens=100, overlap_tokens=0)
    reconstructed = " ".join(chunks)
    for p in paragraphs:
        # Each paragraph's unique identifier should appear in some chunk
        assert f"פסקה {paragraphs.index(p)}" in reconstructed or f"פסקה {paragraphs.index(p)}" in " ".join(chunks)


def test_chunk_overlap():
    """Chunks should have overlapping text when overlap_tokens > 0."""
    text = " ".join([f"מילה{i}" for i in range(100)])
    chunks = TextChunker.chunk(text, max_tokens=30, overlap_tokens=10)
    # Check that consecutive chunks share some words
    if len(chunks) > 1:
        words_0 = set(chunks[0].split()[-10:])
        words_1 = set(chunks[1].split()[:10])
        assert len(words_0 & words_1) > 0
```

**Step 2: Run to verify failure**

```bash
python -m pytest tests/test_chunker.py -v
```

**Step 3: Implement `src/processing/chunker.py`**

```python
"""Text chunking for RAG pipeline. Splits on natural boundaries with overlap."""


class TextChunker:
    """Splits text into chunks suitable for embedding and retrieval."""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate for Hebrew. Hebrew words ≈ 1.5 tokens on average."""
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

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        current_parts = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = TextChunker.estimate_tokens(para)

            if current_tokens + para_tokens > max_tokens and current_parts:
                # Flush current chunk
                chunks.append("\n\n".join(current_parts))

                # Calculate overlap: keep last paragraphs up to overlap_tokens
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

            # If a single paragraph exceeds max_tokens, split it by sentences
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

        # Flush remaining
        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_chunker.py -v
```

**Step 5: Commit**

```bash
git add src/processing/chunker.py tests/test_chunker.py
git commit -m "feat: text chunker with paragraph-aware splitting and overlap"
```

---

### Task 8: Embedding & Vector Store

**Files:**
- Create: `src/processing/embedder.py`
- Create: `src/storage/vectorstore.py`
- Create: `tests/test_vectorstore.py`

**Step 1: Write tests**

```python
# tests/test_vectorstore.py
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
```

**Step 2: Run to verify failure**

```bash
python -m pytest tests/test_vectorstore.py -v
```

**Step 3: Implement `src/storage/vectorstore.py`**

```python
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

        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas,
        )

    def query(self, query_text: str, n_results: int = 5,
              where: dict | None = None) -> list[dict]:
        """Search for similar chunks. Returns list of {text, metadata, distance}."""
        kwargs = {
            "query_texts": [query_text],
            "n_results": n_results,
        }
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
```

**Step 4: Implement `src/processing/embedder.py`**

```python
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

        # Read transcript or raw text
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
                "source": item["source_type"],
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "date": item.get("content_date", ""),
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        self.vectorstore.add_chunks(
            doc_id=str(item_id),
            chunks=chunks,
            metadatas=metadatas,
        )
        self.db.update_status(item_id, "embedded")

    def process_all_pending(self):
        """Process all items with 'transcribed' or 'scraped' (text) status."""
        # Process transcribed audio/video
        for item in self.db.get_by_status("transcribed"):
            print(f"Embedding: {item['title']}")
            self.process_item(item["id"])

        # Process website text that was already scraped (no transcription needed)
        for item in self.db.get_by_status("scraped"):
            if item["source_type"] == "website":
                print(f"Embedding: {item['title']}")
                self.process_item(item["id"])
```

**Step 5: Run tests**

```bash
python -m pytest tests/test_vectorstore.py -v
```

Note: ChromaDB uses its own default embedding function (all-MiniLM-L6-v2). We'll swap to multilingual-e5 in a follow-up task once the pipeline works end-to-end. Getting it working first > perfect embeddings.

**Step 6: Commit**

```bash
git add src/processing/embedder.py src/storage/vectorstore.py tests/test_vectorstore.py
git commit -m "feat: vector store and embedding pipeline"
```

---

### Task 9: RAG Agent (Claude-powered Q&A)

**Files:**
- Create: `src/agent/rag_agent.py`
- Create: `tests/test_rag_agent.py`

**Step 1: Write tests**

```python
# tests/test_rag_agent.py
import pytest
from src.agent.rag_agent import RagAgent


def test_build_context():
    """Should format retrieved chunks into a context string."""
    chunks = [
        {"text": "הרב שרקי אומר שאמונה היא יסוד", "metadata": {"title": "שיעור אמונה", "url": "https://yt.com/1"}},
        {"text": "האמונה מחברת את האדם לבורא", "metadata": {"title": "שיעור אמונה", "url": "https://yt.com/1"}},
    ]
    context = RagAgent.build_context(chunks)
    assert "הרב שרקי אומר שאמונה היא יסוד" in context
    assert "שיעור אמונה" in context
    assert "https://yt.com/1" in context


def test_build_system_prompt():
    """System prompt should instruct Hebrew response with source citations."""
    prompt = RagAgent.build_system_prompt()
    assert "Hebrew" in prompt or "עברית" in prompt
```

**Step 2: Run to verify failure**

```bash
python -m pytest tests/test_rag_agent.py -v
```

**Step 3: Implement `src/agent/rag_agent.py`**

```python
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
        # Retrieve relevant chunks
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
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_rag_agent.py -v
```

**Step 5: Commit**

```bash
git add src/agent/rag_agent.py tests/test_rag_agent.py
git commit -m "feat: RAG agent with Claude-powered Hebrew Q&A"
```

---

### Task 10: CLI Interface

**Files:**
- Create: `src/cli.py`

**Step 1: Implement `src/cli.py`**

```python
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
def scrape_youtube():
    """Download subtitles/audio for all discovered YouTube videos."""
    from src.scrapers.youtube import YouTubeScraper
    db = get_db()
    scraper = YouTubeScraper(db)
    items = db.get_by_status("discovered")
    yt_items = [i for i in items if i["source_type"] == "youtube"]
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
def transcribe(model: str = "medium"):
    """Transcribe all scraped audio/video content using Whisper."""
    from src.transcription.whisper_transcriber import WhisperTranscriber
    db = get_db()
    transcriber = WhisperTranscriber(db, model_name=model)
    transcriber.process_all_pending()
    typer.echo("Transcription complete.")


@app.command()
def embed():
    """Chunk and embed all transcribed content into vector store."""
    from src.processing.embedder import EmbeddingPipeline
    db = get_db()
    vs = get_vectorstore()
    pipeline = EmbeddingPipeline(db, vs)
    pipeline.process_all_pending()
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
    for s in ["discovered", "scraped", "transcribed", "embedded", "error"]:
        items = db.get_by_status(s)
        typer.echo(f"  {s}: {len(items)} items")


if __name__ == "__main__":
    app()
```

**Step 2: Test CLI manually**

```bash
cd C:/Learnings/rabbi-sherki-rag
python -m src.cli --help
python -m src.cli status
```

**Step 3: Commit**

```bash
git add src/cli.py
git commit -m "feat: CLI interface for all pipeline stages"
```

---

### Task 11: Swap to Multilingual Embeddings

**Files:**
- Modify: `src/storage/vectorstore.py`

**Step 1: Update VectorStore to use multilingual-e5-large**

Modify the `__init__` of `VectorStore` to use a custom embedding function:

```python
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

class VectorStore:
    def __init__(self, persist_dir="data/db/chroma", collection_name="sherki_content",
                 embedding_model="intfloat/multilingual-e5-large"):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn,
        )
```

**Step 2: Run existing tests**

```bash
python -m pytest tests/test_vectorstore.py -v
```

Note: First run will download the ~2.2GB multilingual-e5-large model. Subsequent runs use cache.

**Step 3: Commit**

```bash
git add src/storage/vectorstore.py
git commit -m "feat: swap to multilingual-e5-large for Hebrew embeddings"
```

---

### Task 12: End-to-End Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""End-to-end test: add text content → clean → chunk → embed → query."""

import pytest
from src.storage.db import ContentDB
from src.storage.vectorstore import VectorStore
from src.processing.cleaner import HebrewCleaner
from src.processing.chunker import TextChunker


@pytest.fixture
def pipeline(tmp_path):
    db = ContentDB(str(tmp_path / "test.db"))
    vs = VectorStore(persist_dir=str(tmp_path / "chroma"))
    return db, vs


def test_end_to_end_text_to_query(pipeline):
    """Full pipeline: insert text → chunk → embed → query returns relevant result."""
    db, vs = pipeline

    # Simulate a scraped article
    text = HebrewCleaner.clean(
        "הרב אורי שרקי מלמד שהאמונה בה' היא יסוד העבודה הרוחנית. "
        "האמונה אינה עיוורת אלא מבוססת על הכרה שכלית עמוקה. "
        "הרב שרקי מדגיש שהתורה היא דרך החיים של עם ישראל."
    )
    chunks = TextChunker.chunk(text, max_tokens=100)

    vs.add_chunks(
        doc_id="test_article",
        chunks=chunks,
        metadatas=[{"source": "website", "title": "שיעור אמונה"} for _ in chunks],
    )

    # Query should find relevant content
    results = vs.query("מה הרב שרקי אומר על אמונה?", n_results=3)
    assert len(results) > 0
    assert any("אמונה" in r["text"] for r in results)
```

**Step 2: Run integration test**

```bash
python -m pytest tests/test_integration.py -v
```

**Step 3: Run full test suite**

```bash
python -m pytest tests/ -v
```

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end integration test for text-to-query pipeline"
```

---

## Execution Order Summary

| Task | Component | Depends On | ~Time |
|------|-----------|-----------|-------|
| 1 | Project scaffolding | — | 5 min |
| 2 | SQLite DB layer | Task 1 | 10 min |
| 3 | YouTube scraper | Task 2 | 15 min |
| 4 | Website scraper | Task 2 | 10 min |
| 5 | Whisper transcription | Task 2 | 10 min |
| 6 | Hebrew cleaner | Task 1 | 5 min |
| 7 | Text chunker | Task 1 | 10 min |
| 8 | Embedder + vector store | Tasks 6, 7 | 15 min |
| 9 | RAG agent | Task 8 | 10 min |
| 10 | CLI interface | Tasks 3-5, 8-9 | 10 min |
| 11 | Multilingual embeddings | Task 8 | 5 min |
| 12 | Integration test | All | 10 min |

Tasks 3, 4, 5, 6, 7 can run in parallel (they only depend on Task 2).
