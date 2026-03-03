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
