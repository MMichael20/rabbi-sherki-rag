"""Whisper-based audio transcription for Hebrew content."""
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
    def vtt_to_text(vtt_content: str) -> str:
        """Convert VTT subtitle content to clean deduplicated text."""
        lines = []
        seen = set()
        for line in vtt_content.split("\n"):
            line = line.strip()
            if not line or line == "WEBVTT" or "-->" in line:
                continue
            line = re.sub(r"<[^>]+>", "", line)
            if not line:
                continue
            if line not in seen:
                lines.append(line)
                seen.add(line)
        return "\n".join(lines)

    @staticmethod
    def clean_transcript(text: str) -> str:
        """Clean whitespace and normalize text."""
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        return "\n".join(lines)

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe an audio file to Hebrew text."""
        segments, info = self.model.transcribe(audio_path, language="he")
        text = " ".join(segment.text for segment in segments)
        return self.clean_transcript(text)

    def process_item(self, item_id: int):
        """Process a single content item."""
        item = self.db.get_by_id(item_id)
        if not item or not item["raw_path"]:
            return
        raw_path = Path(item["raw_path"])
        transcript_path = self.transcripts_dir / f"{raw_path.stem}.txt"
        try:
            if raw_path.suffix in (".vtt", ".srt"):
                vtt_content = raw_path.read_text(encoding="utf-8")
                text = self.vtt_to_text(vtt_content)
            elif raw_path.suffix in (".mp3", ".m4a", ".wav", ".opus"):
                text = self.transcribe_audio(str(raw_path))
            else:
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
