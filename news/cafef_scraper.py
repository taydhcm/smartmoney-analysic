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
def _parse_published(pub_str: str) -> "datetime | None":
    """Parse RSS 'published' string thành datetime (timezone-aware → naive UTC+7)."""
    if not pub_str:
        return None
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(pub_str)
        return dt.replace(tzinfo=None)          # drop tz, treat as local
    except Exception:
        pass
    try:
        from dateutil import parser as dp        # type: ignore
        return dp.parse(pub_str, ignoretz=True)
    except Exception:
        return None


@ttl_cache(ttl=300)
def search_cafef_news(ticker: str, limit: int = 50, days: int = 7) -> list[dict]:
    """
    Tìm tin liên quan đến ticker trên CafeF trong `days` ngày gần nhất.
    Strategy: lấy tất cả RSS → filter theo ticker trong title/summary → filter theo ngày.
    """
    all_news: list[dict] = []
    for cat in CAFEF_RSS_FEEDS:
        all_news.extend(fetch_cafef_rss(cat, limit=100))

    ticker_upper = ticker.upper()
    cutoff = datetime.utcnow() - timedelta(days=days)

    matched = []
    for n in all_news:
        # Lọc theo ticker
        if ticker_upper not in n["title"].upper() and ticker_upper not in n["summary"].upper():
            continue
        # Lọc theo ngày nếu có thể parse
        pub_dt = _parse_published(n.get("published", ""))
        if pub_dt is not None and pub_dt < cutoff:
            continue        # bài cũ hơn `days` ngày → bỏ qua
        matched.append(n)

    return matched[:limit]


def fetch_all_cafef_news(limit_per_feed: int = 15) -> list[dict]:
    """Lấy tin từ tất cả feeds."""
    result: list[dict] = []
    for cat in CAFEF_RSS_FEEDS:
        result.extend(fetch_cafef_rss(cat, limit_per_feed))
    return result
