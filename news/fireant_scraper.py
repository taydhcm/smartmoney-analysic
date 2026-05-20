"""
news/fireant_scraper.py
Lấy dữ liệu từ Fireant.vn qua public API (ưu tiên) hoặc HTTP scrape.
Fireant là mạng xã hội chứng khoán VN với dữ liệu sentiment tốt.
"""

from __future__ import annotations
import time
import random

import httpx

from utils.cache import ttl_cache
from utils.logger import get_logger

log = get_logger(__name__)

FIREANT_API_BASE = "https://restv2.fireant.vn"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://fireant.vn/",
}


@ttl_cache(ttl=300)
def get_fireant_posts(ticker: str, limit: int = 20) -> list[dict]:
    """
    Lấy bài đăng về ticker từ Fireant API.
    Endpoint: GET /posts?symbol=VIC&type=1&offset=0&limit=20
    """
    try:
        url = f"{FIREANT_API_BASE}/posts"
        params = {"symbol": ticker.upper(), "type": 1, "offset": 0, "limit": limit}

        with httpx.Client(headers=HEADERS, timeout=10) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        posts = data if isinstance(data, list) else data.get("posts", [])
        results = []
        for p in posts:
            results.append({
                "source":    "fireant",
                "title":     p.get("content", "")[:100],
                "snippet":   p.get("content", "")[:300],
                "url":       f"https://fireant.vn/post/{p.get('postID', '')}",
                "published": p.get("datePublished", ""),
                "likes":     p.get("totalLikes", 0),
                "comments":  p.get("totalComments", 0),
                "weight":    min((p.get("totalLikes", 0) + p.get("totalComments", 0)) / 10, 5.0),
            })

        time.sleep(random.uniform(0.5, 1.0))
        return results

    except Exception as exc:
        log.warning("Fireant API (%s) lỗi: %s", ticker, exc)
        return []


@ttl_cache(ttl=300)
def get_fireant_market_feed(limit: int = 30) -> list[dict]:
    """Lấy feed thị trường tổng quát (không theo ticker)."""
    try:
        url = f"{FIREANT_API_BASE}/posts"
        params = {"type": 0, "offset": 0, "limit": limit}

        with httpx.Client(headers=HEADERS, timeout=10) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        posts = data if isinstance(data, list) else data.get("posts", [])
        return [
            {
                "source":    "fireant",
                "title":     p.get("content", "")[:100],
                "snippet":   p.get("content", "")[:300],
                "url":       f"https://fireant.vn/post/{p.get('postID', '')}",
                "published": p.get("datePublished", ""),
                "likes":     p.get("totalLikes", 0),
                "weight":    min(p.get("totalLikes", 0) / 10, 5.0),
            }
            for p in posts
        ]
    except Exception as exc:
        log.warning("Fireant market feed lỗi: %s", exc)
        return []
