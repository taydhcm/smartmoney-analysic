"""
analytics/ssi_iboard.py
SSI iBoard — Lấy dữ liệu Tự Doanh qua FiinMarket API.

Data source (đã verify HTTP 200, không cần auth):
  GET https://fiin-market.ssi.com.vn/MoneyFlow/GetProprietaryV2
  Params: language=vi, ComGroupCode=VNINDEX|VN30, time=<unix_ms>

Response cung cấp dữ liệu theo 4 khoảng thời gian trong 1 lần gọi:
  today      → phiên gần nhất hoàn tất (T nếu sau 15h, T-1 nếu đang phiên)
  oneWeek    → 5 phiên
  oneMonth   → ~20 phiên
  yearToDate → từ đầu năm

Mỗi time-range trả về danh sách ticker với:
  totalBuyTradeVolume   → khối lượng tự doanh mua
  totalSellTradeVolume  → khối lượng tự doanh bán
  totalNetBuyTradeVolume → net (dương = mua ròng, âm = bán ròng)

Fallback cascade:
  FiinMarket API → empty DataFrame (graceful degradation)
  Khi không khả dụng: ML vẫn chạy, features proprietary = 0.0
"""

from __future__ import annotations

import time
from datetime import datetime
from threading import Lock
from typing import Optional

import requests
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

# ── FiinMarket API constants ───────────────────────────────────────────────────
_FIIN_URL = "https://fiin-market.ssi.com.vn/MoneyFlow/GetProprietaryV2"
_FIIN_HEADERS = {
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate, br",
    "Content-Type": "application/json",
    "Origin": "https://iboard.ssi.com.vn",
    "Referer": "https://iboard.ssi.com.vn/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/148.0.0.0 Safari/537.36"
    ),
    "x-fiin-key": "KEY",
    "x-fiin-seed": "SEED",
    "x-fiin-user-id": "ID",
    "x-fiin-user-token": "0,212,108,244",
}

_TIMEOUT = 15               # seconds per request
_CACHE_TTL = 300            # 5 phút — refresh cache sau mỗi 5 phút

# ── Output schema ─────────────────────────────────────────────────────────────
_EMPTY_COLS = [
    "date", "proprietary_buy", "proprietary_sell", "proprietary_net",
    "foreign_buy", "foreign_sell", "retail_buy", "retail_sell",
]


# ── Batch cache (per ComGroupCode) ────────────────────────────────────────────
# {com_group -> (fetched_at_ts, {ticker -> {date, proprietary_buy, sell, net}})}
_batch_cache: dict[str, tuple[float, dict[str, dict]]] = {}
_cache_lock = Lock()


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_COLS)


def _parse_time_range_entries(entries: list[dict]) -> dict[str, dict]:
    """
    Chuyển list ticker entries từ FiinMarket (buy hoặc sell array)
    thành dict: {ticker -> {date, proprietary_buy, sell, net}}.
    Bỏ qua ticker đã có trong dict (tránh ghi đè bởi array thứ hai).
    """
    result: dict[str, dict] = {}
    for entry in entries:
        ticker = (entry.get("ticker") or "").strip().upper()
        if not ticker or ticker in result:
            continue
        # fromDate: "2026-05-25T00:00:00" → "2026-05-25"
        from_date = (entry.get("fromDate") or "")[:10]
        result[ticker] = {
            "date":             from_date,
            "proprietary_buy":  float(entry.get("totalBuyTradeVolume")    or 0),
            "proprietary_sell": float(entry.get("totalSellTradeVolume")   or 0),
            "proprietary_net":  float(entry.get("totalNetBuyTradeVolume") or 0),
        }
    return result


def _fetch_batch_from_api(com_group: str) -> dict[str, dict]:
    """
    Gọi FiinMarket API và trả về dict {ticker -> data} cho phiên gần nhất.
    Raises exception nếu HTTP không phải 200.
    """
    params = {
        "language":     "vi",
        "ComGroupCode": com_group,
        "time":         int(time.time() * 1000),
    }
    resp = requests.get(
        _FIIN_URL, params=params, headers=_FIIN_HEADERS, timeout=_TIMEOUT
    )
    if resp.status_code != 200:
        raise RuntimeError(f"FiinMarket HTTP {resp.status_code}")

    body  = resp.json()
    items = body.get("items") or []
    if not items:
        raise RuntimeError("FiinMarket: response items rỗng")

    item  = items[0]
    today = item.get("today") or {}

    from_date = (today.get("fromDate") or "")[:10]
    log.info(
        "FiinMarket %s: ngày=%s (today.fromDate)",
        com_group, from_date,
    )

    # Merge buy + sell arrays; buy đã đủ nhưng sell có thể có thêm ticker
    result: dict[str, dict] = {}
    for arr in (today.get("buy") or [], today.get("sell") or []):
        result.update(_parse_time_range_entries(arr))

    log.info("FiinMarket %s: %d tickers loaded", com_group, len(result))
    return result


def get_proprietary_batch(com_group: str = "VNINDEX") -> dict[str, dict]:
    """
    Lấy dữ liệu tự doanh theo đợt cho toàn bộ nhóm CP từ FiinMarket.

    Kết quả được cache trong _CACHE_TTL giây (5 phút mặc định).

    Parameters
    ----------
    com_group : "VNINDEX" (toàn HOSE), "VN30", "HNX", v.v.

    Returns
    -------
    dict: {ticker (upper) -> {date, proprietary_buy, proprietary_sell, proprietary_net}}
    Trả về dict rỗng nếu API không khả dụng.
    """
    with _cache_lock:
        if com_group in _batch_cache:
            ts, data = _batch_cache[com_group]
            if time.time() - ts < _CACHE_TTL:
                return data

    try:
        data = _fetch_batch_from_api(com_group)
    except Exception as exc:
        log.warning("FiinMarket batch fetch lỗi (%s): %s — dùng cache cũ nếu có", com_group, exc)
        with _cache_lock:
            if com_group in _batch_cache:
                return _batch_cache[com_group][1]
        return {}

    with _cache_lock:
        _batch_cache[com_group] = (time.time(), data)
    return data


def fetch_investor_flow(symbol: str, limit: int = 30) -> pd.DataFrame:
    """
    Lấy dữ liệu tự doanh cho 1 mã từ FiinMarket.

    Parameters
    ----------
    symbol : Mã CK (e.g. "ACB")
    limit  : Tham số tương thích ngược — không còn dùng (FiinMarket chỉ trả phiên gần nhất)

    Returns
    -------
    DataFrame 1 row với cột: date, proprietary_buy, proprietary_sell, proprietary_net,
                             foreign_buy, foreign_sell, retail_buy, retail_sell
    Trả về DataFrame rỗng nếu không có data.
    """
    sym = symbol.upper()

    # Thử VNINDEX (toàn sàn) trước, sau đó VN30 nếu không tìm thấy
    for group in ("VNINDEX", "VN30"):
        batch = get_proprietary_batch(group)
        if sym in batch:
            row = batch[sym]
            df = pd.DataFrame([{
                "date":             row["date"],
                "proprietary_buy":  row["proprietary_buy"],
                "proprietary_sell": row["proprietary_sell"],
                "proprietary_net":  row["proprietary_net"],
                "foreign_buy":      0.0,   # không có trong FiinMarket proprietary API
                "foreign_sell":     0.0,
                "retail_buy":       0.0,
                "retail_sell":      0.0,
            }])
            return df

    log.info("FiinMarket: không có data cho %s", sym)
    return _empty_df()


def fetch_investor_flow_batch(
    symbols: list[str],
    limit: int = 30,
) -> dict[str, pd.DataFrame]:
    """
    Lấy investor flow cho nhiều mã từ FiinMarket batch API.

    Chỉ cần 1-2 HTTP call: VNINDEX (toàn sàn) + VN30 fallback cho tickers không tìm thấy.

    Returns
    -------
    dict: {ticker → DataFrame}
    """
    # Pre-warm cache: VNINDEX trước (top net buyers/sellers)
    batch_vni = get_proprietary_batch("VNINDEX")
    # VN30 luôn trả về đủ 30 tickers (dù tự doanh ít)
    batch_vn30 = get_proprietary_batch("VN30")

    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        key = sym.upper()
        row = batch_vni.get(key) or batch_vn30.get(key)
        if row:
            result[sym] = pd.DataFrame([{
                "date":             row["date"],
                "proprietary_buy":  row["proprietary_buy"],
                "proprietary_sell": row["proprietary_sell"],
                "proprietary_net":  row["proprietary_net"],
                "foreign_buy":      0.0,
                "foreign_sell":     0.0,
                "retail_buy":       0.0,
                "retail_sell":      0.0,
            }])
        else:
            result[sym] = _empty_df()
    return result


def get_latest_proprietary_net(symbol: str) -> float:
    """
    Lấy proprietary_net của phiên gần nhất.
    Trả về 0.0 nếu không có data.
    """
    df = fetch_investor_flow(symbol)
    if df.empty:
        return 0.0
    last_net = float(df["proprietary_net"].iloc[-1])
    last_buy = float(df["proprietary_buy"].iloc[-1])
    last_sell = float(df["proprietary_sell"].iloc[-1])
    total_flow = last_buy + last_sell
    if total_flow < 1:
        return 0.0
    # Normalized: net / total_flow → [-1, 1]
    return float(min(max(last_net / total_flow, -1.0), 1.0))


# ── Legacy compatibility (unused but may be imported externally) ───────────────

def get_token() -> Optional[str]:
    """Tương thích ngược — FiinMarket không cần token."""
    return None


# ── Legacy compatibility stubs (duy trì API cho test suite / external imports) ──

class _TokenCache:
    """Tương thích ngược — token cache in-memory."""
    def __init__(self) -> None:
        self._token: Optional[str] = None

    def get(self) -> Optional[str]:
        return self._token

    def set(self, token: str, ttl_hours: float = 24) -> None:
        self._token = token

    def clear(self) -> None:
        self._token = None


_token_cache = _TokenCache()


class _RateLimiter:
    """Tương thích ngược — rate limiter stub."""
    def __init__(self, per_minute: int = 60):
        self._min_interval: float = 60.0 / max(per_minute, 1)
        self._last_call: float = 0.0

    def wait(self) -> None:
        now = time.time()
        if self._last_call > 0:
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()


def _parse_response(data, ticker: str) -> pd.DataFrame:
    """
    Tương thích ngược — parse dữ liệu SSI iBoard format cũ thành DataFrame.
    Hỗ trợ list of dicts hoặc {'data': [...]} wrapper.
    """
    if isinstance(data, dict):
        data = data.get("data") or []
    if not data:
        return _empty_df()

    rows = []
    for item in data:
        date_val = (
            item.get("tradingDate") or
            item.get("date") or
            item.get("fromDate") or ""
        )
        if isinstance(date_val, (int, float)) and date_val > 1_000_000_000:
            from datetime import datetime as _dt
            date_val = _dt.fromtimestamp(date_val).strftime("%Y-%m-%d")
        elif isinstance(date_val, str):
            date_val = date_val[:10]

        prop_buy = float(
            item.get("proprietaryBuyVol") or
            item.get("propBuyVol") or
            item.get("proprietary_buy") or 0
        )
        prop_sell = float(
            item.get("proprietarySellVol") or
            item.get("propSellVol") or
            item.get("proprietary_sell") or 0
        )
        for_buy = float(
            item.get("foreignBuyVol") or
            item.get("foreignBuy") or
            item.get("foreign_buy") or 0
        )
        for_sell = float(
            item.get("foreignSellVol") or
            item.get("foreignSell") or
            item.get("foreign_sell") or 0
        )
        ret_buy = float(
            item.get("retailBuyVol") or
            item.get("retailBuy") or
            item.get("retail_buy") or 0
        )
        ret_sell = float(
            item.get("retailSellVol") or
            item.get("retailSell") or
            item.get("retail_sell") or 0
        )
        rows.append({
            "date":             date_val,
            "proprietary_buy":  prop_buy,
            "proprietary_sell": prop_sell,
            "proprietary_net":  prop_buy - prop_sell,
            "foreign_buy":      for_buy,
            "foreign_sell":     for_sell,
            "retail_buy":       ret_buy,
            "retail_sell":      ret_sell,
        })

    if not rows:
        return _empty_df()
    return pd.DataFrame(rows)
