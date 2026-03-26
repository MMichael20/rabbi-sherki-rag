"""Whisper-based audio transcription for Hebrew content."""
import json
import re
from pathlib import Path
from src.storage.db import ContentDB


class WhisperTranscriber:
    """Transcribes audio files to Hebrew text using faster-whisper."""

    def __init__(self, db: ContentDB, model_name: str = "ivrit-ai/faster-whisper-v2-d4",
                 transcripts_dir: str = "data/transcripts"):
        self.db = db
        self.model_name = model_name
        self.transcripts_dir = Path(transcripts_dir)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    @property
    def model(self):
        """Lazy-load the faster-whisper model."""
        if self._model is None:
            from faster_whisper import WhisperModel
            import torch
            if torch.cuda.is_available():
                device, compute = "cuda", "float16"
            else:
                device, compute = "cpu", "int8"
            self._model = WhisperModel(self.model_name, device=device, compute_type=compute)
        return self._model

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
            text = re.sub(r"<[^>]+>", "", line).strip()
            text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
            if not text or text in seen_texts:
                continue
            if current_start is not None:
                segments.append({"start": current_start, "text": text})
                seen_texts.add(text)

        return segments

    @staticmethod
    def vtt_to_text(vtt_content: str) -> str:
        """Convert VTT subtitle content to clean deduplicated text."""
        segments = WhisperTranscriber.vtt_to_segments(vtt_content)
        return "\n".join(seg["text"] for seg in segments)

    @staticmethod
    def clean_transcript(text: str) -> str:
        """Clean whitespace and normalize text."""
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        return "\n".join(lines)

    def transcribe_audio(self, audio_path: str) -> tuple[str, list[dict]]:
        """Transcribe an audio file to Hebrew text. Returns (text, timestamped_segments)."""
        segments, info = self.model.transcribe(audio_path, language="he")
        timed_segments = []
        texts = []
        for segment in segments:
            timed_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip(),
            })
            texts.append(segment.text)
        text = " ".join(texts)
        return self.clean_transcript(text), timed_segments

    def process_item(self, item_id: int):
        """Process a single content item."""
        item = self.db.get_by_id(item_id)
        if not item or not item["raw_path"]:
            return
        raw_path = Path(item["raw_path"])
        transcript_path = self.transcripts_dir / f"{raw_path.stem}.txt"
        timestamps_path = self.transcripts_dir / f"{raw_path.stem}.json"
        try:
            if raw_path.suffix in (".vtt", ".srt"):
                vtt_content = raw_path.read_text(encoding="utf-8")
                text = self.vtt_to_text(vtt_content)
            elif raw_path.suffix in (".mp3", ".m4a", ".wav", ".opus"):
                text, timed_segments = self.transcribe_audio(str(raw_path))
                timestamps_path.write_text(
                    json.dumps(timed_segments, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            else:
                text = raw_path.read_text(encoding="utf-8")
            text = self.clean_transcript(text)
            transcript_path.write_text(text, encoding="utf-8")
            self.db.set_transcript_path(item_id, str(transcript_path))
            self.db.update_status(item_id, "transcribed")
        except Exception as e:
            self.db.update_status(item_id, "error", str(e))

    def process_all_pending(self, limit: int = 0):
        """Process all items with 'scraped' status."""
        items = self.db.get_by_status("scraped")
        if limit > 0:
            items = items[:limit]
        for item in items:
            print(f"Processing: {item['title']}")
            self.process_item(item["id"])
