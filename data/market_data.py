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
    """
    try:
        from vnstock import Vnstock  # type: ignore
        start, end = _date_range(period)
        stock = Vnstock().stock(symbol=ticker, source="TCBS")
        df = stock.quote.history(start=start, end=end, interval="1D")
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
    Thử TCBS trước, fallback sang VCI.
    """
    from vnstock import Vnstock  # type: ignore
    start, end = _date_range(period)
    for source in ("TCBS", "VCI"):
        try:
            stock = Vnstock().stock(symbol=index_code, source=source)
            df = stock.quote.history(start=start, end=end, interval="1D")
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
    Dùng VCI price board với VN30 làm proxy (thay thế market_top_movers đã bị xóa).
    Trả về dict: {advance, decline, unchanged, ceiling, floor, total}
    """
    try:
        from vnstock import Vnstock  # type: ignore
        from config.constants import VN30_TICKERS

        board = Vnstock().stock(source="VCI").trading.price_board(symbols=VN30_TICKERS)
        if board.empty:
            return {"advance": 0, "decline": 0, "unchanged": 0, "ceiling": 0, "floor": 0, "total": 0}

        # Tìm cột thay đổi giá (ưu tiên theo tên phổ biến)
        change_col = None
        for candidate in ("priceChange", "price_change", "change", "changePct", "change_pct"):
            if candidate in board.columns:
                change_col = candidate
                break

        if change_col is None:
            # Tính từ match_price và ref_price
            price_col = next((c for c in board.columns if "match" in c.lower() and "price" in c.lower()), None)
            ref_col   = next((c for c in board.columns if "ref" in c.lower()), None)
            if price_col and ref_col:
                board["_chg"] = pd.to_numeric(board[price_col], errors="coerce") - pd.to_numeric(board[ref_col], errors="coerce")
                change_col = "_chg"

        if change_col is None:
            return {"advance": 0, "decline": 0, "unchanged": 0, "ceiling": 0, "floor": 0, "total": len(board)}

        changes   = pd.to_numeric(board[change_col], errors="coerce").fillna(0)
        advance   = int((changes > 0).sum())
        decline   = int((changes < 0).sum())
        unchanged = int((changes == 0).sum())

        # Ceiling / floor hits
        price_col   = next((c for c in board.columns if "match" in c.lower() and "price" in c.lower()), None)
        ceiling_col = next((c for c in board.columns if "ceil" in c.lower()), None)
        floor_col   = next((c for c in board.columns if "floor" in c.lower() and "price" in c.lower()), None)
        ceiling_hits = floor_hits = 0
        if price_col and ceiling_col:
            mp = pd.to_numeric(board[price_col], errors="coerce")
            cp = pd.to_numeric(board[ceiling_col], errors="coerce")
            ceiling_hits = int((mp >= cp).sum())
        if price_col and floor_col:
            mp = pd.to_numeric(board[price_col], errors="coerce")
            fp = pd.to_numeric(board[floor_col], errors="coerce")
            floor_hits = int((mp <= fp).sum())

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
        from vnstock import Vnstock  # type: ignore
        df = Vnstock().stock(source="VCI").listing.all_symbols()
        # Lọc theo sàn nếu có cột exchange
        exch_col = next((c for c in df.columns if "exchange" in c.lower()), None)
        if exch_col:
            df = df[df[exch_col].str.upper() == exchange.upper()]
        # Lấy cột symbol/ticker
        sym_col = next(
            (c for c in df.columns if c.lower() in ("symbol", "ticker", "code")),
            df.columns[0]
        )
        return df[sym_col].dropna().tolist()
    except Exception as exc:
        log.warning("get_all_tickers(%s) lỗi: %s", exchange, exc)
        return []
