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
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        if selector:
            element = soup.select_one(selector)
            if element:
                return element.get_text(separator="\n", strip=True)
        body = soup.find("body")
        if body:
            return body.get_text(separator="\n", strip=True)
        return soup.get_text(separator="\n", strip=True)

    def scrape_page(self, url: str, selector: str | None = None) -> dict:
        """Fetch and extract text from a single page."""
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, "lxml")
        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else url
        text = self.extract_text(response.text, selector)
        return {"title": title_text, "text": text, "url": url}

    def scrape_and_save(self, url: str, selector: str | None = None, delay: float = 1.0):
        """Scrape a page, save text to file, register in DB."""
        existing = self.db.get_by_url(url)
        if existing and existing["status"] != "discovered":
            return
        try:
            result = self.scrape_page(url, selector)
            safe_name = url.replace("https://", "").replace("http://", "").replace("/", "_")[:100]
            file_path = self.raw_dir / f"{safe_name}.txt"
            file_path.write_text(result["text"], encoding="utf-8")
            item_id = self.db.add_item(url=url, title=result["title"], source_type="website")
            self.db.set_raw_path(item_id, str(file_path))
            self.db.update_status(item_id, "scraped")
        except Exception as e:
            item_id = self.db.add_item(url=url, title=url, source_type="website")
            self.db.update_status(item_id, "error", str(e))
        time.sleep(delay)
