"""
news/cafef_scraper.py
Lấy tin tức từ CafeF qua RSS feeds (ưu tiên) hoặc HTTP scraping fallback.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any

import feedparser
import httpx
from bs4 import BeautifulSoup

from utils.cache import ttl_cache
from utils.logger import get_logger

log = get_logger(__name__)

# ── RSS Feeds CafeF ───────────────────────────────────────────────────────────
CAFEF_RSS_FEEDS = {
    "thi_truong":    "https://cafef.vn/thi-truong-chung-khoan.rss",
    "doanh_nghiep":  "https://cafef.vn/doanh-nghiep.rss",
    "vi_mo":         "https://cafef.vn/vi-mo-dau-tu.rss",
    "bat_dong_san":  "https://cafef.vn/bat-dong-san.rss",
    "ngan_hang":     "https://cafef.vn/ngan-hang.rss",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _parse_rss_entry(entry: Any) -> dict:
    return {
        "source":    "cafef",
        "title":     entry.get("title", ""),
        "summary":   entry.get("summary", "")[:300],
        "url":       entry.get("link", ""),
        "published": entry.get("published", ""),
        "tags":      [t.get("term", "") for t in entry.get("tags", [])],
    }


@ttl_cache(ttl=300)  # 5 phút
def fetch_cafef_rss(category: str = "thi_truong", limit: int = 20) -> list[dict]:
    """Lấy tin mới từ RSS feed CafeF."""
    url = CAFEF_RSS_FEEDS.get(category, CAFEF_RSS_FEEDS["thi_truong"])
    try:
        feed = feedparser.parse(url)
        return [_parse_rss_entry(e) for e in feed.entries[:limit]]
    except Exception as exc:
        log.warning("cafef RSS (%s) lỗi: %s", category, exc)
        return []


@ttl_cache(ttl=300)
def search_cafef_news(ticker: str, limit: int = 10) -> list[dict]:
    """
    Tìm tin liên quan đến ticker trên CafeF.
    Strategy: lấy tất cả RSS → filter theo ticker trong title/summary.
    """
    all_news: list[dict] = []
    for cat in CAFEF_RSS_FEEDS:
        all_news.extend(fetch_cafef_rss(cat, limit=50))

    ticker_upper = ticker.upper()
    matched = [
        n for n in all_news
        if ticker_upper in n["title"].upper() or ticker_upper in n["summary"].upper()
    ]
    return matched[:limit]


def fetch_all_cafef_news(limit_per_feed: int = 15) -> list[dict]:
    """Lấy tin từ tất cả feeds."""
    result: list[dict] = []
    for cat in CAFEF_RSS_FEEDS:
        result.extend(fetch_cafef_rss(cat, limit_per_feed))
    return result
