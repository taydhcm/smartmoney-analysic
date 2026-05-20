"""
news/hose_announcements.py
Lấy thông báo chính thức từ HOSE & HNX (cổ tức, phát hành thêm, ĐHCĐ, BCTC).
Nguồn: RSS feeds của sàn hoặc HTTP scrape trang công bố thông tin.
"""

from __future__ import annotations
from typing import Any

import feedparser
import httpx
from bs4 import BeautifulSoup

from utils.cache import ttl_cache
from utils.logger import get_logger

log = get_logger(__name__)

# HOSE công bố thông tin (RSS/XML)
HOSE_DISCLOSURE_RSS = "https://www.hsx.vn/Modules/CMS/Web/ArticleList?lang=vi&CatId=1"
HNX_DISCLOSURE_RSS  = "https://hnx.vn/vi-vn/tin-tuc-su-kien.html"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# Từ khóa phân loại sự kiện
EVENT_KEYWORDS = {
    "dividend":   ["cổ tức", "chia cổ tức", "tạm ứng cổ tức"],
    "issuance":   ["phát hành thêm", "chào bán", "phát hành cổ phiếu"],
    "agm":        ["đại hội", "ĐHCĐ", "đại hội cổ đông"],
    "earnings":   ["BCTC", "báo cáo tài chính", "kết quả kinh doanh"],
    "buyback":    ["mua lại cổ phiếu quỹ", "mua cổ phiếu quỹ"],
}


def _classify_event(title: str) -> str:
    title_lower = title.lower()
    for event_type, keywords in EVENT_KEYWORDS.items():
        if any(kw in title_lower for kw in keywords):
            return event_type
    return "other"


@ttl_cache(ttl=600)
def fetch_hose_announcements(limit: int = 30) -> list[dict]:
    """
    Lấy thông báo từ HOSE. Thử RSS trước, fallback sang HTTP scrape.
    """
    try:
        feed = feedparser.parse(HOSE_DISCLOSURE_RSS)
        if feed.entries:
            return [
                {
                    "source":     "HOSE",
                    "title":      e.get("title", ""),
                    "url":        e.get("link", ""),
                    "published":  e.get("published", ""),
                    "event_type": _classify_event(e.get("title", "")),
                    "ticker":     _extract_ticker(e.get("title", "")),
                }
                for e in feed.entries[:limit]
            ]
    except Exception as exc:
        log.warning("HOSE RSS lỗi: %s – thử HTTP scrape", exc)

    # Fallback: HTTP scrape
    try:
        with httpx.Client(headers=HEADERS, timeout=10) as client:
            resp = client.get(HOSE_DISCLOSURE_RSS)
            resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        items = soup.select("item")[:limit]
        return [
            {
                "source":     "HOSE",
                "title":      item.find("title").get_text(strip=True) if item.find("title") else "",
                "url":        item.find("link").get_text(strip=True) if item.find("link") else "",
                "published":  item.find("pubDate").get_text(strip=True) if item.find("pubDate") else "",
                "event_type": _classify_event(item.find("title").get_text() if item.find("title") else ""),
                "ticker":     "",
            }
            for item in items
        ]
    except Exception as exc:
        log.warning("HOSE scrape lỗi: %s", exc)
        return []


def _extract_ticker(text: str) -> str:
    """Trích xuất mã cổ phiếu từ tiêu đề thông báo (dạng 'Công ty ABC (VIC):...')."""
    import re
    match = re.search(r"\(([A-Z]{2,4})\)", text)
    return match.group(1) if match else ""


@ttl_cache(ttl=600)
def get_announcements_by_ticker(ticker: str) -> list[dict]:
    """Lọc thông báo theo mã cổ phiếu."""
    all_ann = fetch_hose_announcements(limit=100)
    return [a for a in all_ann if a.get("ticker", "").upper() == ticker.upper()]
