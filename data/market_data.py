"""
data/market_data.py
Lấy dữ liệu thị trường tổng quan: OHLCV, index, breadth.
Sử dụng vnstock3 (TCBS / SSI source).
"""

from __future__ import annotations
from datetime import date, timedelta

import pandas as pd

from config.constants import PERIOD_DAYS, INDICES
from utils.cache import ttl_cache
from utils.logger import get_logger

log = get_logger(__name__)


def _date_range(period: str) -> tuple[str, str]:
    days = PERIOD_DAYS.get(period, 22)
    end = date.today()
    start = end - timedelta(days=days * 2)  # buffer cho ngày nghỉ
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


@ttl_cache()
def get_ohlcv(ticker: str, period: str = "1m") -> pd.DataFrame:
    """
    OHLCV cho 1 mã. Trả về DataFrame với cột:
    [time, open, high, low, close, volume]
    """
    try:
        from vnstock import stock_historical_data  # type: ignore
        start, end = _date_range(period)
        df = stock_historical_data(ticker, start, end, "1D", "stock")
        df = df.rename(columns={"time": "date"})
        df["date"] = pd.to_datetime(df["date"])
        return df.tail(PERIOD_DAYS[period])
    except Exception as exc:
        log.warning("get_ohlcv(%s) lỗi: %s – trả về DataFrame rỗng", ticker, exc)
        return pd.DataFrame()


@ttl_cache()
def get_multiple_ohlcv(tickers: list[str], period: str = "1m") -> dict[str, pd.DataFrame]:
    """OHLCV cho nhiều mã cùng lúc."""
    return {t: get_ohlcv(t, period) for t in tickers}


@ttl_cache()
def get_index_data(index_code: str = "VNINDEX", period: str = "1m") -> pd.DataFrame:
    """
    Dữ liệu chỉ số (VNINDEX, HNX30, UPCOM…).
    """
    try:
        from vnstock import stock_historical_data  # type: ignore
        start, end = _date_range(period)
        df = stock_historical_data(index_code, start, end, "1D", "index")
        df["date"] = pd.to_datetime(df["time"])
        return df.tail(PERIOD_DAYS[period])
    except Exception as exc:
        log.warning("get_index_data(%s) lỗi: %s", index_code, exc)
        return pd.DataFrame()


@ttl_cache(ttl=60)  # breadth cập nhật nhanh hơn
def get_market_breadth() -> dict:
    """
    Số cổ phiếu tăng / giảm / đứng tham chiếu trên HOSE hôm nay.
    Trả về dict: {advance, decline, unchanged, ceiling, floor}
    """
    try:
        from vnstock import market_top_movers  # type: ignore
        # vnstock cung cấp danh sách toàn thị trường intraday
        df = market_top_movers(floor="HOSE", top=999)
        advance   = int((df["change_pct"] > 0).sum())
        decline   = int((df["change_pct"] < 0).sum())
        unchanged = int((df["change_pct"] == 0).sum())
        ceiling   = int((df.get("ceiling_hit", pd.Series(dtype=bool))).sum())
        floor_hit = int((df.get("floor_hit",   pd.Series(dtype=bool))).sum())
        return {
            "advance": advance,
            "decline": decline,
            "unchanged": unchanged,
            "ceiling": ceiling,
            "floor": floor_hit,
            "total": advance + decline + unchanged,
        }
    except Exception as exc:
        log.warning("get_market_breadth lỗi: %s", exc)
        return {"advance": 0, "decline": 0, "unchanged": 0,
                "ceiling": 0, "floor": 0, "total": 0}


@ttl_cache()
def get_all_tickers(exchange: str = "HOSE") -> list[str]:
    """Danh sách tất cả mã trên sàn."""
    try:
        from vnstock import listing_companies  # type: ignore
        df = listing_companies(floor=exchange)
        return df["ticker"].dropna().tolist()
    except Exception as exc:
        log.warning("get_all_tickers(%s) lỗi: %s", exchange, exc)
        return []
