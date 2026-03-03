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
