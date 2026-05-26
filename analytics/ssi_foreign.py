"""
analytics/ssi_foreign.py
SSI iBoard — Lấy dữ liệu Khối Ngoại (Foreign Flow) qua FiinMarket API.

Data source (đã verify HTTP 200, không cần auth):
  GET https://fiin-market.ssi.com.vn/MoneyFlow/GetForeign
  Params: language=vi, ComGroupCode=VNINDEX|VN30, time=<unix_ms>

Response cung cấp dữ liệu theo 4 khoảng thời gian trong 1 lần gọi:
  today      → phiên gần nhất hoàn tất
  oneWeek    → 5 phiên
  oneMonth   → ~20 phiên
  yearToDate → từ đầu năm

Mỗi time-range trả về:
  Aggregate: foreignBuyValue, foreignSellValue, foreignNetBuyValue, foreignNetSellValue
  Per-ticker (trong arrays buy/sell/netBuy/netSell):
    ticker, foreignBuyValue, foreignSellValue,
    foreignNetBuyValue (trong netBuy[]) hoặc foreignNetSellValue (trong netSell[]),
    fromDate, toDate

Thêm series[]: dữ liệu tích lũy theo phút 09:15–14:45 (dùng cho intraday chart).

Fallback cascade:
  FiinMarket API → empty result (graceful degradation)
  Khi không khả dụng: ML vẫn chạy, features foreign = 0.0
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
_FIIN_URL = "https://fiin-market.ssi.com.vn/MoneyFlow/GetForeign"
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
    "date", "foreign_buy", "foreign_sell", "foreign_net",
]


# ── Batch cache (per ComGroupCode) ────────────────────────────────────────────
# {com_group -> (fetched_at_ts, {ticker -> {date, foreign_buy, foreign_sell, foreign_net}})}
_batch_cache: dict[str, tuple[float, dict[str, dict]]] = {}
_cache_lock = Lock()


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_COLS)


def _parse_per_ticker_entries(
    entries: list[dict],
    net_field: str,
) -> dict[str, dict]:
    """
    Chuyển list ticker entries từ FiinMarket foreign array
    thành dict: {ticker -> {date, foreign_buy, foreign_sell, foreign_net}}.

    Parameters
    ----------
    entries   : list[dict] từ today.netBuy hoặc today.netSell
    net_field : Tên field net trong mỗi entry:
                "foreignNetBuyValue"  (cho netBuy array  → net dương)
                "foreignNetSellValue" (cho netSell array → net âm)
    """
    result: dict[str, dict] = {}
    for entry in entries:
        ticker = (entry.get("ticker") or "").strip().upper()
        if not ticker or ticker in result:
            continue

        from_date = (entry.get("fromDate") or "")[:10]
        buy_val  = float(entry.get("foreignBuyValue")  or 0)
        sell_val = float(entry.get("foreignSellValue") or 0)

        # Ưu tiên net field riêng; fallback tính từ buy - sell
        raw_net  = entry.get(net_field)
        if raw_net is not None:
            net_val = float(raw_net)
            if net_field == "foreignNetSellValue":
                net_val = -abs(net_val)   # sell array → net âm
        else:
            net_val = buy_val - sell_val

        result[ticker] = {
            "date":         from_date,
            "foreign_buy":  buy_val,
            "foreign_sell": sell_val,
            "foreign_net":  net_val,
        }
    return result


def _fetch_batch_from_api(com_group: str) -> dict[str, dict]:
    """
    Gọi FiinMarket GetForeign API và trả về dict {ticker -> data} cho phiên gần nhất.
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
        raise RuntimeError(f"FiinMarket GetForeign HTTP {resp.status_code}")

    body  = resp.json()
    items = body.get("items") or []
    if not items:
        raise RuntimeError("FiinMarket GetForeign: response items rỗng")

    item  = items[0]
    today = item.get("today") or {}

    from_date = (today.get("fromDate") or "")[:10]
    log.info(
        "FiinMarket GetForeign %s: ngày=%s (today.fromDate)",
        com_group, from_date,
    )

    # Ưu tiên netBuy[] + netSell[] vì cho thấy net flow rõ nhất
    # Sau đó bổ sung từ buy[] + sell[] cho các ticker còn thiếu
    result: dict[str, dict] = {}

    net_buy_entries  = today.get("netBuy")  or []
    net_sell_entries = today.get("netSell") or []
    buy_entries      = today.get("buy")     or []
    sell_entries     = today.get("sell")    or []

    # netBuy → net dương (mua ròng)
    result.update(_parse_per_ticker_entries(net_buy_entries,  "foreignNetBuyValue"))
    # netSell → net âm (bán ròng) — bổ sung ticker không có trong netBuy
    for ticker, data in _parse_per_ticker_entries(net_sell_entries, "foreignNetSellValue").items():
        if ticker not in result:
            result[ticker] = data

    # buy/sell arrays → bổ sung ticker còn thiếu
    for ticker, data in _parse_per_ticker_entries(buy_entries,  "foreignNetBuyValue").items():
        if ticker not in result:
            result[ticker] = data
    for ticker, data in _parse_per_ticker_entries(sell_entries, "foreignNetSellValue").items():
        if ticker not in result:
            result[ticker] = data

    log.info("FiinMarket GetForeign %s: %d tickers loaded", com_group, len(result))
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def get_foreign_batch(com_group: str = "VNINDEX") -> dict[str, dict]:
    """
    Lấy dữ liệu khối ngoại theo đợt cho toàn bộ nhóm CP từ FiinMarket.

    Kết quả được cache trong _CACHE_TTL giây (5 phút mặc định).

    Parameters
    ----------
    com_group : "VNINDEX" (toàn HOSE), "VN30", "HNX", v.v.

    Returns
    -------
    dict: {ticker (upper) -> {date, foreign_buy, foreign_sell, foreign_net}}
    Đơn vị: VND (giá trị, không phải khối lượng cổ phiếu).
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
        log.warning(
            "FiinMarket GetForeign batch lỗi (%s): %s — dùng cache cũ nếu có",
            com_group, exc,
        )
        with _cache_lock:
            if com_group in _batch_cache:
                return _batch_cache[com_group][1]
        return {}

    with _cache_lock:
        _batch_cache[com_group] = (time.time(), data)
    return data


def fetch_foreign_flow(symbol: str) -> pd.DataFrame:
    """
    Lấy dữ liệu khối ngoại cho 1 mã từ FiinMarket.

    Thử VNINDEX (toàn sàn) trước, sau đó VN30 nếu không tìm thấy.

    Returns
    -------
    DataFrame 1 row với cột: date, foreign_buy, foreign_sell, foreign_net
    Trả về DataFrame rỗng nếu không có data.
    """
    sym = symbol.upper()
    for group in ("VNINDEX", "VN30"):
        batch = get_foreign_batch(group)
        if sym in batch:
            row = batch[sym]
            return pd.DataFrame([{
                "date":         row["date"],
                "foreign_buy":  row["foreign_buy"],
                "foreign_sell": row["foreign_sell"],
                "foreign_net":  row["foreign_net"],
            }])

    log.info("FiinMarket GetForeign: không có data cho %s", sym)
    return _empty_df()


def fetch_foreign_flow_batch(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """
    Lấy dữ liệu khối ngoại cho nhiều mã từ FiinMarket batch API.

    Chỉ cần 1-2 HTTP call: VNINDEX (toàn sàn) + VN30 fallback cho tickers không tìm thấy.

    Returns
    -------
    dict: {ticker → DataFrame}
    """
    batch_vni  = get_foreign_batch("VNINDEX")
    batch_vn30 = get_foreign_batch("VN30")

    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        key = sym.upper()
        row = batch_vni.get(key) or batch_vn30.get(key)
        if row:
            result[sym] = pd.DataFrame([{
                "date":         row["date"],
                "foreign_buy":  row["foreign_buy"],
                "foreign_sell": row["foreign_sell"],
                "foreign_net":  row["foreign_net"],
            }])
        else:
            result[sym] = _empty_df()
    return result


def get_foreign_intraday_series(com_group: str = "VNINDEX") -> pd.DataFrame:
    """
    Lấy dữ liệu khối ngoại theo phút (tích lũy) từ FiinMarket.

    Dữ liệu từ 09:15 đến 14:45, không có khoảng 11:30–13:00 (nghỉ trưa).

    Returns
    -------
    DataFrame với cột: trading_date, buy_cum, sell_cum, net_cum
    Trả về DataFrame rỗng nếu không khả dụng.
    """
    params = {
        "language":     "vi",
        "ComGroupCode": com_group,
        "time":         int(time.time() * 1000),
    }
    try:
        resp = requests.get(
            _FIIN_URL, params=params, headers=_FIIN_HEADERS, timeout=_TIMEOUT
        )
        if resp.status_code != 200:
            log.warning("FiinMarket GetForeign series HTTP %s", resp.status_code)
            return pd.DataFrame()

        body  = resp.json()
        items = body.get("items") or []
        if not items:
            return pd.DataFrame()

        series = items[0].get("series") or []
        if not series:
            return pd.DataFrame()

        rows = []
        for pt in series:
            buy  = float(pt.get("foreignBuyValue")  or 0)
            sell = float(pt.get("foreignSellValue") or 0)
            dt_str = pt.get("tradingDate") or ""
            rows.append({
                "trading_date": dt_str,
                "buy_cum":      buy,
                "sell_cum":     sell,
                "net_cum":      buy - sell,
            })

        df = pd.DataFrame(rows)
        df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")
        df = df.dropna(subset=["trading_date"]).sort_values("trading_date").reset_index(drop=True)
        return df

    except Exception as exc:
        log.warning("FiinMarket GetForeign series lỗi: %s", exc)
        return pd.DataFrame()


def get_latest_foreign_net_pct(symbol: str) -> float:
    """
    Lấy foreign_net_pct của phiên gần nhất (net / total flow → [-1, 1]).
    Trả về 0.0 nếu không có data.
    """
    df = fetch_foreign_flow(symbol)
    if df.empty:
        return 0.0
    buy  = float(df["foreign_buy"].iloc[-1])
    sell = float(df["foreign_sell"].iloc[-1])
    net  = float(df["foreign_net"].iloc[-1])
    total = buy + sell
    if total < 1:
        return 0.0
    return float(min(max(net / total, -1.0), 1.0))
