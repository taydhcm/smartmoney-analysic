"""
news/f319_scraper.py
Lấy bài đăng từ diễn đàn chứng khoán f319.com theo ticker.
Sử dụng HTTP scraping với rate limiting.
"""

from __future__ import annotations
import time
import random
import re

import httpx
from bs4 import BeautifulSoup

from utils.cache import ttl_cache
from utils.logger import get_logger

log = get_logger(__name__)

BASE_URL   = "https://f319.com"
SEARCH_URL = f"{BASE_URL}/search/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
}


@ttl_cache(ttl=600)
def search_f319(ticker: str, limit: int = 10) -> list[dict]:
    """
    Tìm bài đăng về ticker trên f319.com.
    Trả về: [{source, title, url, snippet, views, published}]
    """
    results: list[dict] = []
    try:
        params = {"q": ticker, "o": "date"}
        with httpx.Client(headers=HEADERS, timeout=15, follow_redirects=True) as client:
            resp = client.get(SEARCH_URL, params=params)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")
        threads = soup.select(".structItem--thread")[:limit]

        for t in threads:
            title_el = t.select_one(".structItem-title a")
            meta_el  = t.select_one(".structItem-minor")
            if not title_el:
                continue

            title   = title_el.get_text(strip=True)
            url     = BASE_URL + title_el.get("href", "")
            snippet = meta_el.get_text(strip=True)[:200] if meta_el else ""

            # Lấy view count nếu có
            views_el = t.select_one(".structItem-cell--meta dl:last-child dd")
            views = int(re.sub(r"\D", "", views_el.get_text()) or "0") if views_el else 0

            results.append({
                "source":    "f319",
                "title":     title,
                "url":       url,
                "snippet":   snippet,
                "views":     views,
                "published": "",
                "weight":    min(views / 1000, 5.0),  # interaction weight cap 5
            })

        # Polite delay
        time.sleep(random.uniform(1.0, 2.5))

    except Exception as exc:
        log.warning("f319 search(%s) lỗi: %s", ticker, exc)

    return results
