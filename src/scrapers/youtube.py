"""YouTube scraper using yt-dlp for metadata, subtitles, and audio download."""
import subprocess
import sys
import json
import os
from pathlib import Path
from src.storage.db import ContentDB

# Use python -m yt_dlp so it works even when yt-dlp isn't on PATH
_YT_DLP = [sys.executable, "-m", "yt_dlp"]


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

    # YouTube uses the legacy "iw" code for Hebrew; yt-dlp may also
    # normalise it to "he" depending on version.  Check both.
    _HEBREW_CODES = ("he", "iw")

    @staticmethod
    def _has_hebrew(mapping: dict) -> bool:
        return any(code in mapping for code in YouTubeScraper._HEBREW_CODES)

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
            "has_manual_subs": YouTubeScraper._has_hebrew(subs),
            "has_auto_subs": YouTubeScraper._has_hebrew(auto_caps),
        }

    def discover_channel(self, channel_url: str) -> list[dict]:
        """Fetch metadata for all videos in a channel."""
        cmd = [
            *_YT_DLP, "--flat-playlist", "--dump-json", "--no-download", channel_url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        videos = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            info = json.loads(line)
            url = info.get("url") or f"https://www.youtube.com/watch?v={info.get('id')}"
            title = info.get("title", "Unknown")
            item_id = self.db.add_item(url=url, title=title, source_type="youtube")
            videos.append({"id": item_id, "url": url, "title": title})
        return videos

    def fetch_video_details(self, video_url: str) -> dict:
        """Fetch full metadata for a single video."""
        cmd = [
            *_YT_DLP, "--dump-json", "--no-download",
            "--write-subs", "--write-auto-subs", "--sub-lang", "he,iw", video_url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        info = json.loads(result.stdout)
        return self.parse_video_info(info)

    def download_subtitles(self, video_url: str, video_id: str) -> str | None:
        """Download Hebrew subtitles. Returns path or None."""
        out_path = self.raw_dir / video_id
        cmd = [
            *_YT_DLP, "--write-subs", "--write-auto-subs", "--sub-lang", "he,iw",
            "--skip-download", "--sub-format", "vtt", "-o", str(out_path), video_url,
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        # yt-dlp may save with either language code depending on the source
        for ext in [".he.vtt", ".iw.vtt", ".he.srt", ".iw.srt"]:
            path = Path(str(out_path) + ext)
            if path.exists():
                return str(path)
        return None

    def download_audio(self, video_url: str, video_id: str) -> str:
        """Download audio only for transcription."""
        out_template = str(self.raw_dir / f"{video_id}.%(ext)s")
        cmd = [
            *_YT_DLP, "--extract-audio", "--audio-format", "mp3",
            "--audio-quality", "5", "-o", out_template, video_url,
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        return str(self.raw_dir / f"{video_id}.mp3")

    def scrape_video(self, item_id: int, video_url: str):
        """Full scrape for a single video: subtitles first, audio as fallback."""
        details = self.fetch_video_details(video_url)
        video_id = details["video_id"]
        sub_path = self.download_subtitles(video_url, video_id)
        if sub_path:
            self.db.set_raw_path(item_id, sub_path)
            self.db.update_status(item_id, "scraped")
            return
        audio_path = self.download_audio(video_url, video_id)
        self.db.set_raw_path(item_id, audio_path)
        self.db.update_status(item_id, "scraped")
