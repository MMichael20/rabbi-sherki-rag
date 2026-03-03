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
