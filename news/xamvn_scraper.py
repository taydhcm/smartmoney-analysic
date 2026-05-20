"""
news/xamvn_scraper.py
Lấy bài đăng từ diễn đàn XamVN.com.
XamVN là diễn đàn chứng khoán cộng đồng tiếng Việt.
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

BASE_URL   = "https://xamvn.com"
SEARCH_URL = f"{BASE_URL}/search"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9",
}


@ttl_cache(ttl=600)
def search_xamvn(ticker: str, limit: int = 10) -> list[dict]:
    """
    Tìm bài đăng về ticker trên XamVN.
    Trả về: [{source, title, url, snippet, views, published}]
    """
    results: list[dict] = []
    try:
        params = {"q": ticker, "t": "post"}
        with httpx.Client(headers=HEADERS, timeout=15, follow_redirects=True) as client:
            resp = client.get(SEARCH_URL, params=params)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")

        # XamVN dùng XenForo-like structure – selector cần verify sau
        threads = soup.select(".structItem, .searchResult")[:limit]

        for t in threads:
            title_el = t.select_one("h3 a, .title a, .searchResult-title a")
            if not title_el:
                continue

            title   = title_el.get_text(strip=True)
            url     = title_el.get("href", "")
            if url and not url.startswith("http"):
                url = BASE_URL + url

            snippet_el = t.select_one(".searchResult-body, .preview")
            snippet = snippet_el.get_text(strip=True)[:200] if snippet_el else ""

            # Views / replies
            meta_text = t.get_text()
            views_match = re.search(r"(\d[\d,.]+)\s*(view|lượt)", meta_text, re.IGNORECASE)
            views = int(re.sub(r"\D", "", views_match.group(1))) if views_match else 0

            results.append({
                "source":    "xamvn",
                "title":     title,
                "url":       url,
                "snippet":   snippet,
                "views":     views,
                "published": "",
                "weight":    min(views / 500, 5.0),
            })

        time.sleep(random.uniform(1.0, 2.5))

    except Exception as exc:
        log.warning("XamVN search(%s) lỗi: %s", ticker, exc)

    return results
