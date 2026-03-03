# Rabbi Oury Sherki Knowledge Base — Design Document

**Date:** 2026-03-03
**Goal:** Build a comprehensive RAG-based Q&A agent over all of Rabbi Oury Sherki's publicly available content (articles, video lessons, audio recordings, social media posts) in Hebrew.

## Content Sources

| Source Type | Examples | Scraping Method |
|-------------|----------|----------------|
| YouTube | Channel(s) TBD, guest appearances | `yt-dlp` for metadata + subtitles/audio |
| Websites | machon-meir.org.il, yeshiva.org.il, personal sites | `requests` + `BeautifulSoup` / `Scrapy` |
| Audio/Podcasts | TBD | Direct download + Whisper transcription |
| Social Media | Facebook, Telegram | Telegram API; Facebook manual export or Playwright |

## Architecture Overview

```
[Scrapers] → [Raw Content Store] → [Transcription] → [Text Processing] → [Vector DB] → [RAG Agent]
```

Each stage is independent and can run on different machines or at different times.

## Phase 1: Scraping & Downloading

### YouTube Module
1. Use `yt-dlp` to fetch channel/playlist metadata (title, date, duration, URL)
2. Download existing Hebrew subtitles/auto-captions (free, instant)
3. For videos without captions: download audio only (m4a/mp3)
4. Store metadata in SQLite database for tracking

### Website Module
1. Crawl known article pages
2. Extract article text, title, date, author attribution
3. Handle pagination and archive pages
4. Respect robots.txt and add delays between requests

### Audio/Podcast Module
1. Download audio files from identified platforms
2. Queue for transcription (same as YouTube audio)

### Social Media Module
1. Telegram: Use Telethon (Python Telegram client) to pull channel messages
2. Facebook: Manual data export or Playwright-based scraping (last resort due to anti-bot)

## Phase 2: Transcription

- **Model:** Whisper `medium` on T500 laptop (4GB VRAM), upgradable to `large-v3` on RTX 1080 Ti (11GB VRAM) later
- **Language:** Force Hebrew (`--language he`) for better accuracy
- **Output:** Raw text + timestamps (SRT format for reference)
- **Priority:** Process videos without existing subtitles first
- **Deduplication:** Track processed files in SQLite to avoid re-transcribing

## Phase 3: Text Processing

### Cleaning
- Strip HTML artifacts, timestamps, formatting noise
- Normalize Hebrew text (handle nikud/vowels if present, normalize final forms)
- Remove duplicate content (same shiur appearing on multiple platforms)

### Chunking Strategy
- Chunk by natural boundaries: paragraphs for articles, topic segments for transcriptions
- Target chunk size: ~500-800 tokens (balances context and retrieval precision)
- Include overlap of ~50-100 tokens between chunks
- Preserve metadata: source URL, date, title, content type

### Embedding
- **Model:** `intfloat/multilingual-e5-large` (runs locally, strong Hebrew support)
- Generates 1024-dim vectors
- Can run on CPU (slower) or GPU

## Phase 4: Storage

### SQLite — Content Metadata
- Track all discovered content (URL, title, date, source, status)
- Track scraping/transcription/embedding status per item
- Enable resumable pipeline (restart without re-doing work)

### ChromaDB — Vector Store
- Store embeddings + text chunks + metadata
- Local, file-based, no server needed
- Supports metadata filtering (by date, source type, etc.)

## Phase 5: RAG Agent

### Query Flow
1. User asks question in Hebrew
2. Question is embedded using same multilingual-e5 model
3. Top-k similar chunks retrieved from ChromaDB (k=5-10)
4. Chunks + question sent to Claude API as context
5. Claude generates answer in Hebrew, citing sources

### Prompt Design
- System prompt instructs Claude to answer based only on provided context
- Include source metadata so Claude can cite "from shiur X on date Y"
- Handle cases where no relevant content is found

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Language | Python 3.11+ | Best scraping/ML ecosystem |
| YouTube | yt-dlp | Industry standard, actively maintained |
| Web scraping | requests + BeautifulSoup | Simple, sufficient for article sites |
| Transcription | openai-whisper (local) | Free, good Hebrew, runs on GPU |
| Embeddings | sentence-transformers + multilingual-e5 | Free, local, strong multilingual |
| Vector DB | ChromaDB | Simple, local, Python-native |
| Metadata DB | SQLite | Zero config, file-based |
| LLM | Claude API (Anthropic) | Excellent Hebrew, best reasoning |
| CLI interface | Click or Typer | Clean CLI for running pipeline stages |

## Hebrew-Specific Considerations

1. **Morphology:** Hebrew prefixes (ב, ל, מ, ה, ש, כ) attach to words. Embedding-based search handles this naturally (unlike keyword search)
2. **Nikud:** Most web content is unvocalized. Normalize by stripping nikud if present
3. **RTL:** Text storage is fine (UTF-8). Only matters for display
4. **Whisper Hebrew:** `medium` model is decent; `large-v3` is significantly better. Force `--language he`
5. **Claude Hebrew:** Handles Hebrew natively, including answering questions and understanding context

## Known Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Facebook anti-scraping | Can't get FB content | Manual export, or skip FB initially |
| Whisper medium Hebrew errors | Lower quality transcripts | Review samples; upgrade to large-v3 on 1080 Ti later |
| Rate limiting (YouTube) | IP blocked | Throttle requests, add random delays |
| Content deduplication | Same content indexed multiple times | Hash-based dedup + metadata matching |
| Large transcription time on T500 | Days of GPU time | Prioritize captioned videos first; defer uncaptioned to 1080 Ti |
| Website structure changes | Scraper breaks | Modular scrapers with per-site config; monitor for failures |

## Cost Estimate

| Item | Cost |
|------|------|
| Whisper (local) | Free |
| Embeddings (local) | Free |
| ChromaDB | Free |
| Claude API (Q&A phase) | ~$0.01-0.05 per question |
| Total infrastructure | Free (excluding Claude API for Q&A) |

## Project Structure

```
rabbi-sherki-rag/
├── src/
│   ├── scrapers/
│   │   ├── youtube.py
│   │   ├── websites.py
│   │   ├── telegram.py
│   │   └── facebook.py
│   ├── transcription/
│   │   └── whisper_transcriber.py
│   ├── processing/
│   │   ├── cleaner.py
│   │   ├── chunker.py
│   │   └── embedder.py
│   ├── storage/
│   │   ├── db.py          # SQLite operations
│   │   └── vectorstore.py # ChromaDB operations
│   ├── agent/
│   │   └── rag_agent.py   # Claude-powered Q&A
│   └── cli.py             # CLI entry point
├── data/
│   ├── raw/               # Downloaded audio/text
│   ├── transcripts/       # Whisper output
│   └── db/                # SQLite + ChromaDB files
├── config/
│   └── sources.yaml       # URLs, channels, site configs
├── docs/
│   └── plans/
├── requirements.txt
└── README.md
```
