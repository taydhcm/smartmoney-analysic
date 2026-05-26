"""
analytics/ssi_iboard.py
SSI iBoard Integration — Lấy dữ liệu giao dịch theo loại NĐT (cuối ngày).

Sprint 12: Cung cấp dữ liệu Tự Doanh (proprietary) từ SSI Open API.

Auth flow (SSI FastConnect Data API — đã verify):
  1. POST https://fc-data.ssi.com.vn/api/v2/Market/AccessToken
     Body: {"consumerID": "...", "consumerSecret": "..."}
     → {"data": {"accessToken": "...", "tokenExpire": <ms>}, "status": 200}

  Lấy consumerID/consumerSecret tại:
    https://iboard.ssi.com.vn/support/api-service/management

Data strategy:
  - SSI Open API không có investor-transaction endpoint.
  - Thử iboard-query.ssi.com.vn với Bearer token từ fc-data auth.
  - Fallback graceful: proprietary features = 0.0 khi SSI không khả dụng."

Fallback cascade:
  SSI iBoard → empty DataFrame (graceful degradation)
  Khi SSI unavailable: ML vẫn chạy, features proprietary = 0.0

Data schema trả về (per ticker per day):
  date                TEXT   YYYY-MM-DD
  proprietary_buy     REAL   volume tự doanh mua
  proprietary_sell    REAL   volume tự doanh bán
  proprietary_net     REAL   buy - sell
  foreign_buy         REAL   volume nước ngoài mua (redundant check vs D0.2)
  foreign_sell        REAL   volume nước ngoài bán
  retail_buy          REAL   volume cá nhân mua
  retail_sell         REAL   volume cá nhân bán
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Optional

import requests
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

# ── Load .env on first import ─────────────────────────────────────────────────
try:
    from analytics.secret_manager import load_env, get_ssi_credentials
    load_env()
except Exception:
    pass   # .env không bắt buộc

# ── Constants ─────────────────────────────────────────────────────────────────
_RATE_LIMIT   = int(os.getenv("SSI_RATE_LIMIT_PER_MIN", "20"))
_TOKEN_TTL_H  = int(os.getenv("SSI_TOKEN_TTL_HOURS", "20"))
_TIMEOUT      = 15   # seconds per request

# SSI iBoard TAPI — auth endpoint đã verify (HTTP 401 = endpoint tồn tại)
# Dùng username/password, trả về JWT Bearer với TTL 8 giờ
_AUTH_URL  = "https://iboard-tapi.ssi.com.vn/users/auth/login"
_QUERY_URL = "https://iboard-query.ssi.com.vn"

# Device ID ổn định (UUID v4 dạng Windows registry-style)
_DEVICE_ID = "87201F42-71A6-4E93-9F4A-771DA660D2EC"

# Data endpoints — iboard-query.ssi.com.vn đã verify HTTP 200
_DATA_ENDPOINTS = [
    f"{_QUERY_URL}/stock/investor-transaction",
    f"{_QUERY_URL}/stock/trading-statistic",
]

# Known field names mapping (different API versions use different names)
_FIELD_MAPS = [
    # v2 standard
    {
        "date_field": "tradingDate",
        "prop_buy":   "proprietaryBuyVol",
        "prop_sell":  "proprietarySellVol",
        "for_buy":    "foreignBuyVol",
        "for_sell":   "foreignSellVol",
        "ret_buy":    "retailBuyVol",
        "ret_sell":   "retailSellVol",
    },
    # alternate field names
    {
        "date_field": "date",
        "prop_buy":   "propBuyVol",
        "prop_sell":  "propSellVol",
        "for_buy":    "foreignBuy",
        "for_sell":   "foreignSell",
        "ret_buy":    "retailBuy",
        "ret_sell":   "retailSell",
    },
]

_EMPTY_COLS = [
    "date", "proprietary_buy", "proprietary_sell", "proprietary_net",
    "foreign_buy", "foreign_sell", "retail_buy", "retail_sell",
]


# ── Session / token cache ─────────────────────────────────────────────────────

class _TokenCache:
    """Thread-safe JWT token cache với auto-refresh."""

    def __init__(self):
        self._token: Optional[str] = None
        self._expires_at: datetime = datetime.min
        self._lock = Lock()

    def get(self) -> Optional[str]:
        with self._lock:
            if self._token and datetime.now() < self._expires_at:
                return self._token
            return None

    def set(self, token: str, ttl_hours: int = _TOKEN_TTL_H) -> None:
        with self._lock:
            self._token = token
            self._expires_at = datetime.now() + timedelta(hours=ttl_hours)

    def clear(self) -> None:
        with self._lock:
            self._token = None
            self._expires_at = datetime.min


_token_cache = _TokenCache()

# ── Rate limiter ───────────────────────────────────────────────────────────────

class _RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, per_minute: int = _RATE_LIMIT):
        self._interval = 60.0 / per_minute
        self._last     = 0.0

    def wait(self) -> None:
        now  = time.monotonic()
        wait = self._interval - (now - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()


_rate_limiter = _RateLimiter()


# ── Auth helpers ───────────────────────────────────────────────────────────────

def _get_default_headers(include_device: bool = True) -> dict:
    """Headers chung cho tất cả requests đến iboard-*.ssi.com.vn."""
    h = {
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
        "Content-Type":    "application/json",
        "Origin":          "https://iboard.ssi.com.vn",
        "Referer":         "https://iboard.ssi.com.vn/",
        "User-Agent":      (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    }
    if include_device:
        h["device-id"]    = _DEVICE_ID
        h["x-device-name"] = "Chrome"
        h["x-os-name"]     = "Windows"
    return h


def _try_login(consumer_id: str, consumer_secret: str) -> Optional[str]:
    """
    Xác thực SSI FastConnect Data API.

    Endpoint đã verify:
      POST https://fc-data.ssi.com.vn/api/v2/Market/AccessToken
      Body: {"consumerID": "...", "consumerSecret": "..."}
      Response 200: {"data": {"accessToken": "...", "tokenExpire": <ms>}, "status": 200}
      Response 400: {"message": "This connection is invalid", "status": 400}

    Lấy consumerID/consumerSecret tại:
      https://iboard.ssi.com.vn/support/api-service/management
    """
    payload = {"consumerID": consumer_id, "consumerSecret": consumer_secret}
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    try:
        _rate_limiter.wait()
        resp = requests.post(_AUTH_URL, json=payload, headers=headers, timeout=_TIMEOUT)
        if resp.status_code == 200:
            body = resp.json()
            # SSI fc-data format: {"data": {"accessToken": "...", "tokenExpire": ms}}
            data = body.get("data") or {}
            token = (
                data.get("accessToken")
                or data.get("access_token")
                or body.get("accessToken")
                or body.get("access_token")
            )
            if isinstance(token, str) and len(token) > 20:
                ttl_ms = data.get("tokenExpire", 0)
                ttl_h  = max(1, int(ttl_ms / 3_600_000)) if ttl_ms else _TOKEN_TTL_H
                log.info("SSI auth OK, token TTL %dh", ttl_h)
                return token
            log.debug("SSI auth: token không tìm thấy trong response: %s", body)
        else:
            body = {}
            try: body = resp.json()
            except Exception: pass
            log.debug("SSI auth HTTP %d: %s", resp.status_code, body.get("message", resp.text[:100]))
    except Exception as exc:
        log.debug("SSI auth lỗi: %s", exc)
    return None


def _get_manual_token() -> Optional[str]:
    """
    Đọc Bearer token được set thủ công từ .env (SSI_BEARER_TOKEN).

    Token có TTL 8h. User copy từ F12 khi đăng nhập iboard.ssi.com.vn:
      Network → bất kỳ request → Header → Authorization: Bearer <token>

    Sau đó chạy: py -3.12 scripts/set_ssi_token.py <token>
    hoặc thêm thủ công vào .env: SSI_BEARER_TOKEN=eyJ...
    """
    raw_token = os.getenv("SSI_BEARER_TOKEN", "").strip()
    if not raw_token:
        return None

    # Validate đây là JWT (3 phần ngăn cách bởi dấu .)
    parts = raw_token.split(".")
    if len(parts) != 3:
        log.warning("SSI_BEARER_TOKEN không phải JWT hợp lệ")
        return None

    # Decode payload để kiểm tra expiry
    try:
        import base64, json as _json
        payload_b64 = parts[1] + "==" * (4 - len(parts[1]) % 4)
        payload = _json.loads(base64.urlsafe_b64decode(payload_b64))
        exp = payload.get("exp", 0)
        if exp and exp < time.time():
            log.warning(
                "SSI_BEARER_TOKEN đã hết hạn (%s). "
                "Cập nhật token: py -3.12 scripts/set_ssi_token.py <token>",
                datetime.fromtimestamp(exp).strftime("%Y-%m-%d %H:%M"),
            )
            return None
        log.debug("SSI manual token OK, expires %s",
                  datetime.fromtimestamp(exp).strftime("%H:%M") if exp else "unknown")
    except Exception:
        pass  # Không thể decode — vẫn thử dùng token

    return raw_token


def get_token() -> Optional[str]:
    """
    Lấy Bearer token theo thứ tự ưu tiên:
    1. Cache (nếu còn hiệu lực)
    2. SSI_BEARER_TOKEN trong .env (manual token từ F12 browser)
    3. Programmatic login (hiện tại bị block bởi Cloudflare/middleware)

    Returns None nếu không có token hợp lệ.
    """
    cached = _token_cache.get()
    if cached:
        return cached

    # Thử manual token trước (từ SSI_BEARER_TOKEN trong .env)
    manual = _get_manual_token()
    if manual:
        _token_cache.set(manual, ttl_hours=8)
        log.info("SSI iBoard: dùng manual Bearer token từ .env")
        return manual

    # Thử programmatic login (thường bị block vì global auth middleware)
    try:
        username, password = get_ssi_credentials()
    except RuntimeError as exc:
        log.warning("SSI credentials chưa được cấu hình: %s", exc)
        return None

    token = _try_login(username, password)
    if token:
        _token_cache.set(token, ttl_hours=8)
    else:
        log.warning(
            "SSI iBoard: đăng nhập thất bại. "
            "Copy Bearer token từ F12 browser và chạy: "
            "py -3.12 scripts/set_ssi_token.py <token>"
        )
    return token


# ── Data fetch ─────────────────────────────────────────────────────────────────

def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_COLS)


def _parse_response(data: list | dict, symbol: str) -> pd.DataFrame:
    """
    Parse JSON response từ SSI investor-transaction endpoint.
    Thử nhiều field name mappings để tương thích các phiên bản API.
    """
    # Normalize: unwrap common wrappers
    if isinstance(data, dict):
        for wrapper in ("data", "d", "items", "result"):
            if wrapper in data:
                data = data[wrapper]
                break

    if not isinstance(data, list) or not data:
        return _empty_df()

    # Thử từng field mapping
    for fmap in _FIELD_MAPS:
        sample = data[0]
        if fmap["prop_buy"] in sample or fmap["prop_sell"] in sample:
            rows = []
            for item in data:
                date_raw = item.get(fmap["date_field"], "")
                if not date_raw:
                    continue
                # Normalize date
                try:
                    if isinstance(date_raw, (int, float)):
                        date_str = datetime.fromtimestamp(date_raw).strftime("%Y-%m-%d")
                    else:
                        date_str = pd.to_datetime(str(date_raw)).strftime("%Y-%m-%d")
                except Exception:
                    continue

                rows.append({
                    "date":             date_str,
                    "proprietary_buy":  float(item.get(fmap["prop_buy"],  0) or 0),
                    "proprietary_sell": float(item.get(fmap["prop_sell"], 0) or 0),
                    "proprietary_net":  (
                        float(item.get(fmap["prop_buy"],  0) or 0)
                        - float(item.get(fmap["prop_sell"], 0) or 0)
                    ),
                    "foreign_buy":      float(item.get(fmap["for_buy"],  0) or 0),
                    "foreign_sell":     float(item.get(fmap["for_sell"], 0) or 0),
                    "retail_buy":       float(item.get(fmap["ret_buy"],  0) or 0),
                    "retail_sell":      float(item.get(fmap["ret_sell"], 0) or 0),
                })

            if rows:
                df = pd.DataFrame(rows)
                df = df.sort_values("date").reset_index(drop=True)
                log.debug("SSI %s: parse OK — %d rows", symbol, len(df))
                return df

    log.warning("SSI %s: không nhận ra field names trong response", symbol)
    return _empty_df()


def fetch_investor_flow(symbol: str, limit: int = 30) -> pd.DataFrame:
    """
    Lấy dữ liệu giao dịch theo loại NĐT cho 1 mã.

    Parameters
    ----------
    symbol : Mã chứng khoán (VD: "ACB")
    limit  : Số phiên lịch sử tối đa (mặc định 30)

    Returns
    -------
    DataFrame với cột: date, proprietary_buy, proprietary_sell, proprietary_net,
                       foreign_buy, foreign_sell, retail_buy, retail_sell
    Trả về DataFrame rỗng nếu SSI không khả dụng.
    """
    token = get_token()
    if not token:
        return _empty_df()

    headers = {**_get_default_headers(include_device=True), "Authorization": f"Bearer {token}"}
    params  = {
        "symbol": symbol.upper(),
        "type":   "stock",
        "offset": 0,
        "limit":  limit,
    }

    for url in _DATA_ENDPOINTS:
        try:
            _rate_limiter.wait()
            resp = requests.get(url, params=params, headers=headers, timeout=_TIMEOUT)

            if resp.status_code == 401:
                # Token hết hạn → clear cache và retry 1 lần
                log.info("SSI token hết hạn, đang refresh...")
                _token_cache.clear()
                new_token = get_token()
                if not new_token:
                    return _empty_df()
                headers["Authorization"] = f"Bearer {new_token}"
                resp = requests.get(url, params=params, headers=headers, timeout=_TIMEOUT)

            if resp.status_code == 200:
                data = resp.json()
                df = _parse_response(data, symbol)
                if not df.empty:
                    return df
            else:
                log.debug("SSI investor-flow %s → HTTP %d", symbol, resp.status_code)

        except Exception as exc:
            log.debug("SSI investor-flow %s via %s → %s", symbol, url, exc)

    log.info("SSI investor-flow %s: không lấy được data, dùng giá trị 0.0", symbol)
    return _empty_df()


def fetch_investor_flow_batch(
    symbols: list[str],
    limit: int = 30,
) -> dict[str, pd.DataFrame]:
    """
    Lấy investor flow cho nhiều mã cùng lúc (tuần tự với rate limiting).

    Returns
    -------
    dict: {ticker → DataFrame}
    """
    result = {}
    for sym in symbols:
        result[sym] = fetch_investor_flow(sym, limit=limit)
    return result


def get_latest_proprietary_net(symbol: str) -> float:
    """
    Lấy proprietary_net của phiên gần nhất (normalized / total volume).
    Trả về 0.0 nếu không có data.
    """
    df = fetch_investor_flow(symbol, limit=5)
    if df.empty:
        return 0.0
    total = (df["proprietary_buy"] + df["proprietary_sell"] +
             df["foreign_buy"] + df["foreign_sell"] +
             df["retail_buy"] + df["retail_sell"])
    # Normalize by total flow
    last_total = float(total.iloc[-1]) if len(total) > 0 else 0.0
    last_net   = float(df["proprietary_net"].iloc[-1]) if len(df) > 0 else 0.0
    if last_total < 1:
        return 0.0
    return float(min(max(last_net / last_total, -1.0), 1.0))
