"""
news/vietstock_scraper.py
Lấy tin tức từ Vietstock qua RSS feeds.
"""

from __future__ import annotations
from typing import Any

import feedparser

from utils.cache import ttl_cache
from utils.logger import get_logger

log = get_logger(__name__)

VIETSTOCK_RSS_FEEDS = {
    "chung_khoan":  "https://vietstock.vn/830/chung-khoan.rss",
    "doanh_nghiep": "https://vietstock.vn/827/doanh-nghiep.rss",
    "tai_chinh":    "https://vietstock.vn/828/tai-chinh.rss",
    "kinh_te":      "https://vietstock.vn/829/kinh-te.rss",
}


def _parse_entry(entry: Any) -> dict:
    return {
        "source":    "vietstock",
        "title":     entry.get("title", ""),
        "summary":   entry.get("summary", "")[:300],
        "url":       entry.get("link", ""),
        "published": entry.get("published", ""),
        "tags":      [],
    }


@ttl_cache(ttl=300)
def fetch_vietstock_rss(category: str = "chung_khoan", limit: int = 20) -> list[dict]:
    url = VIETSTOCK_RSS_FEEDS.get(category, VIETSTOCK_RSS_FEEDS["chung_khoan"])
    try:
        feed = feedparser.parse(url)
        return [_parse_entry(e) for e in feed.entries[:limit]]
    except Exception as exc:
        log.warning("Vietstock RSS (%s) lỗi: %s", category, exc)
        return []


@ttl_cache(ttl=300)
def search_vietstock_news(ticker: str, limit: int = 10) -> list[dict]:
    all_news: list[dict] = []
    for cat in VIETSTOCK_RSS_FEEDS:
        all_news.extend(fetch_vietstock_rss(cat, limit=40))

    ticker_upper = ticker.upper()
    matched = [
        n for n in all_news
        if ticker_upper in n["title"].upper() or ticker_upper in n["summary"].upper()
    ]
    return matched[:limit]
