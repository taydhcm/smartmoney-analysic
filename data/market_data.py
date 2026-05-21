"""
data/market_data.py
Lấy dữ liệu thị trường tổng quan: OHLCV, index, breadth.
Sử dụng vnstock 4.x API (Quote, Trading, Listing).
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


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa cột date từ vnstock 4.x output."""
    if df.empty:
        return df
    if "time" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"time": "date"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


@ttl_cache()
def get_ohlcv(ticker: str, period: str = "1m") -> pd.DataFrame:
    """
    OHLCV cho 1 mã. Trả về DataFrame với cột:
    [date, open, high, low, close, volume]
    Rate limiter chỉ được gọi khi cache miss (tức là sẽ gọi VCI API thực sự).
    """
    # Thờ́t nếu đang dùng free tier (đã check cache miss vì ta trong body @ttl_cache)
    from utils.rate_limiter import vnstock_limiter  # lazy import để tránh circular
    vnstock_limiter.acquire()
    try:
        from vnstock.api.quote import Quote  # type: ignore
        start, end = _date_range(period)
        q = Quote(symbol=ticker, source="VCI")
        df = q.history(start=start, end=end, interval="1D")
        df = _normalize_ohlcv(df)
        return df.tail(PERIOD_DAYS[period]).reset_index(drop=True)
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
    Dùng VCI source với vnstock 4.x Quote API.
    """
    from utils.rate_limiter import vnstock_limiter
    vnstock_limiter.acquire()   # chỉ gọi khi cache miss
    from vnstock.api.quote import Quote  # type: ignore
    start, end = _date_range(period)
    for source in ("VCI", "KBS"):
        try:
            q = Quote(symbol=index_code, source=source)
            df = q.history(start=start, end=end, interval="1D")
            if not df.empty:
                df = _normalize_ohlcv(df)
                return df.tail(PERIOD_DAYS[period]).reset_index(drop=True)
        except Exception as exc:
            log.debug("get_index_data(%s) source=%s lỗi: %s", index_code, source, exc)
    log.warning("get_index_data(%s) lỗi: không lấy được dữ liệu", index_code)
    return pd.DataFrame()


@ttl_cache(ttl=60)  # breadth cập nhật nhanh hơn
def get_market_breadth() -> dict:
    """
    Số cổ phiếu tăng / giảm / đứng tham chiếu.
    Dùng KBS price_board với VN30 làm proxy.
    Trả về dict: {advance, decline, unchanged, ceiling, floor, total}
    """
    try:
        from vnstock.api.trading import Trading  # type: ignore
        from config.constants import VN30_TICKERS

        board = Trading(source="KBS").price_board(symbols_list=VN30_TICKERS)
        if board.empty:
            return {"advance": 0, "decline": 0, "unchanged": 0, "ceiling": 0, "floor": 0, "total": 0}

        # KBS price_board trả về cột phẳng: price_change, close_price, ceiling_price, floor_price
        change_col = "price_change"
        if change_col not in board.columns:
            # Thử tính từ close - reference
            price_col = next((c for c in board.columns if "close" in c.lower()), None)
            ref_col   = next((c for c in board.columns if "reference" in c.lower() or c == "re"), None)
            if price_col and ref_col:
                board = board.copy()
                board["_chg"] = pd.to_numeric(board[price_col], errors="coerce") - \
                                 pd.to_numeric(board[ref_col], errors="coerce")
                change_col = "_chg"
            else:
                return {"advance": 0, "decline": 0, "unchanged": 0, "ceiling": 0, "floor": 0, "total": len(board)}

        changes   = pd.to_numeric(board[change_col], errors="coerce").fillna(0)
        advance   = int((changes > 0).sum())
        decline   = int((changes < 0).sum())
        unchanged = int((changes == 0).sum())

        # Ceiling / floor hits
        price_col   = next((c for c in board.columns if c in ("close_price", "match_price", "close")), None)
        ceiling_col = next((c for c in board.columns if "ceiling" in c.lower()), None)
        floor_col   = next((c for c in board.columns if "floor" in c.lower() and "price" not in c.lower()), None)
        # KBS columns: ceiling_price, floor_price
        if not ceiling_col:
            ceiling_col = next((c for c in board.columns if c == "ceiling_price"), None)
        if not floor_col:
            floor_col = next((c for c in board.columns if c == "floor_price"), None)

        ceiling_hits = floor_hits = 0
        if price_col and ceiling_col:
            mp = pd.to_numeric(board[price_col], errors="coerce")
            cp = pd.to_numeric(board[ceiling_col], errors="coerce")
            ceiling_hits = int((mp >= cp * 0.999).sum())  # 0.1% tolerance
        if price_col and floor_col:
            mp = pd.to_numeric(board[price_col], errors="coerce")
            fp = pd.to_numeric(board[floor_col], errors="coerce")
            floor_hits = int((mp <= fp * 1.001).sum())

        return {
            "advance":   advance,
            "decline":   decline,
            "unchanged": unchanged,
            "ceiling":   ceiling_hits,
            "floor":     floor_hits,
            "total":     advance + decline + unchanged,
        }
    except Exception as exc:
        log.warning("get_market_breadth lỗi: %s", exc)
        return {"advance": 0, "decline": 0, "unchanged": 0, "ceiling": 0, "floor": 0, "total": 0}


@ttl_cache()
def get_all_tickers(exchange: str = "HOSE") -> list[str]:
    """Danh sách tất cả mã trên sàn (dùng vnstock 4.x Listing API)."""
    try:
        from vnstock.api.listing import Listing  # type: ignore
        df = Listing(source="KBS").symbols_by_exchange()
        # Lọc theo sàn
        exch_col = next((c for c in df.columns if "exchange" in c.lower()), None)
        if exch_col:
            df = df[df[exch_col].str.upper() == exchange.upper()]
        # Lấy cột symbol
        sym_col = next(
            (c for c in df.columns if c.lower() in ("symbol", "ticker", "code")),
            df.columns[0]
        )
        return df[sym_col].dropna().tolist()
    except Exception as exc:
        log.warning("get_all_tickers(%s) lỗi: %s", exchange, exc)
        return []

