"""
news/fireant_scraper.py
Lấy dữ liệu từ Fireant.vn qua public API (ưu tiên) hoặc HTTP scrape.
Fireant là mạng xã hội chứng khoán VN với dữ liệu sentiment tốt.

Fireant API v2 (restv2.fireant.vn) yêu cầu Bearer token để gọi /posts.
Cách lấy token:
  1. Đăng nhập fireant.vn trên Chrome
  2. Mở DevTools → Network → chọn bất kỳ request nào tới restv2.fireant.vn
  3. Copy giá trị header "Authorization: Bearer <token>"
  4. Thêm vào .env: FIREANT_TOKEN=<token>

Khi không có token: buzz_count = 0 (graceful fallback, log cấp DEBUG).
"""

from __future__ import annotations
import time
import random

import httpx

from utils.cache import ttl_cache
from utils.logger import get_logger

log = get_logger(__name__)

FIREANT_API_BASE = "https://restv2.fireant.vn"
_BASE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://fireant.vn/",
}


def _build_headers() -> dict:
    """Thêm Bearer token nếu FIREANT_TOKEN được cấu hình trong .env."""
    try:
        from config.settings import _get_secret
        token = _get_secret("FIREANT_TOKEN", "")
        if token:
            return {**_BASE_HEADERS, "Authorization": f"Bearer {token}"}
    except Exception:
        pass
    return dict(_BASE_HEADERS)


# Flag để warn 1 lần duy nhất khi không có token
_warned_no_token: bool = False


@ttl_cache(ttl=300)
def get_fireant_posts(ticker: str, limit: int = 20) -> list[dict]:
    """
    Lấy bài đăng về ticker từ Fireant API.
    Endpoint: GET /posts?symbol=VIC&type=1&offset=0&limit=20
    Yêu cầu FIREANT_TOKEN trong .env để vượt qua xác thực 401.
    """
    global _warned_no_token
    try:
        headers = _build_headers()
        _has_token = "Authorization" in headers

        url = f"{FIREANT_API_BASE}/posts"
        params = {"symbol": ticker.upper(), "type": 1, "offset": 0, "limit": limit}

        with httpx.Client(headers=headers, timeout=10) as client:
            resp = client.get(url, params=params)

        # 401 = API yêu cầu token
        if resp.status_code == 401:
            if _has_token:
                # Có token nhưng vẫn 401 = token hết hạn/sai
                log.warning(
                    "Fireant API 401 dù có token [%s] — kiểm tra lại FIREANT_TOKEN trong .env",
                    ticker,
                )
            else:
                # Không có token — warn 1 lần, sau đó im lặng
                if not _warned_no_token:
                    log.info(
                        "Fireant API cần xác thực (401). "
                        "Thêm FIREANT_TOKEN vào .env để lấy buzz data. "
                        "Hiện tại: buzz=0 (graceful fallback)."
                    )
                    _warned_no_token = True
                else:
                    log.debug("Fireant 401 [%s] — không có FIREANT_TOKEN", ticker)
            return []

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
        headers = _build_headers()
        url = f"{FIREANT_API_BASE}/posts"
        params = {"type": 0, "offset": 0, "limit": limit}

        with httpx.Client(headers=headers, timeout=10) as client:
            resp = client.get(url, params=params)

        if resp.status_code == 401:
            log.debug("Fireant market feed 401 — không có FIREANT_TOKEN")
            return []
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
