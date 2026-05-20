"""
data/foreign_flow.py
Phân tích dòng tiền khối ngoại (foreign investor flows).
Dữ liệu từ TCBS API thông qua vnstock3.

Cột quan trọng:
    - foreignBuyVolume / foreignSellVolume  (khối lượng)
    - foreignBuyValue  / foreignSellValue   (giá trị tỷ đồng)
    - foreignNetVolume / foreignNetValue    (net mua–bán)
    - currentForeignRoom                   (room còn lại, %)
"""

from __future__ import annotations

import pandas as pd

from config.constants import PERIOD_DAYS, FOREIGN_ROOM_ALERT_PCT
from utils.cache import ttl_cache
from utils.logger import get_logger

log = get_logger(__name__)


@ttl_cache()
def get_foreign_flow(ticker: str, period: str = "1m") -> pd.DataFrame:
    """
    Dữ liệu khối ngoại theo ngày cho 1 mã.
    Dùng vnstock 4.x: Vnstock().stock(symbol, source='TCBS').quote.history()
    Trả về DataFrame với các cột:
        date, buy_vol, sell_vol, net_vol, buy_val, sell_val, net_val
    """
    try:
        from vnstock import Vnstock  # type: ignore
        from datetime import date as dt, timedelta

        days = PERIOD_DAYS.get(period, 22)
        end   = dt.today().strftime("%Y-%m-%d")
        start = (dt.today() - timedelta(days=days * 2)).strftime("%Y-%m-%d")

        stock = Vnstock().stock(symbol=ticker, source="TCBS")
        df = stock.quote.history(start=start, end=end, interval="1D")

        if df.empty:
            return pd.DataFrame()

        # Chuẩn hóa cột date
        if "time" in df.columns:
            df["date"] = pd.to_datetime(df["time"])
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # TCBS trả về camelCase → snake_case
        col_map = {
            "foreignBuyVolume":  "buy_vol",
            "foreignSellVolume": "sell_vol",
            "foreignBuyValue":   "buy_val",
            "foreignSellValue":  "sell_val",
        }
        df = df.rename(columns=col_map)

        if "buy_vol" in df.columns and "sell_vol" in df.columns:
            df["net_vol"] = df["buy_vol"] - df["sell_vol"]
        if "buy_val" in df.columns and "sell_val" in df.columns:
            df["net_val"] = df["buy_val"] - df["sell_val"]

        cols = ["date", "buy_vol", "sell_vol", "net_vol", "buy_val", "sell_val", "net_val"]
        available = [c for c in cols if c in df.columns]
        return df[available].tail(days).reset_index(drop=True)

    except Exception as exc:
        log.warning("get_foreign_flow(%s) lỗi: %s", ticker, exc)
        return pd.DataFrame()


@ttl_cache()
def get_foreign_room(ticker: str) -> dict:
    """
    Lấy room nước ngoài còn lại của 1 mã.
    Dùng vnstock 4.x: stock.company.overview() trả về DataFrame.
    Trả về: {ticker, max_room_pct, used_pct, remaining_pct, alert}
    """
    try:
        from vnstock import Vnstock  # type: ignore

        stock = Vnstock().stock(symbol=ticker, source="TCBS")
        overview = stock.company.overview()

        if overview is None or (hasattr(overview, "empty") and overview.empty):
            raise ValueError("company.overview() trả về rỗng")

        # overview là DataFrame, lấy hàng đầu tiên
        row = overview.iloc[0] if hasattr(overview, "iloc") else overview

        # foreignPercent có thể là 0-1 (tỷ lệ) hoặc 0-100 (phần trăm)
        fp = float(row.get("foreignPercent", 0) or 0)
        used_pct = fp * 100 if fp <= 1 else fp

        mfp = float(row.get("maxForeignPercent", 49) or 49)
        max_pct = mfp * 100 if mfp <= 1 else mfp

        remaining = max_pct - used_pct
        return {
            "ticker":        ticker,
            "max_room_pct":  round(max_pct, 2),
            "used_pct":      round(used_pct, 2),
            "remaining_pct": round(remaining, 2),
            "alert":         remaining < FOREIGN_ROOM_ALERT_PCT,
        }
    except Exception as exc:
        log.warning("get_foreign_room(%s) lỗi: %s", ticker, exc)
        return {"ticker": ticker, "remaining_pct": None, "alert": False}


@ttl_cache()
def get_top_foreign_net(exchange: str = "HOSE", top_n: int = 10) -> dict[str, pd.DataFrame]:
    """
    Top mã có khối ngoại mua ròng và bán ròng nhiều nhất hôm nay.
    Dùng VCI price board với VN30 làm proxy.
    Trả về: {"buy": DataFrame, "sell": DataFrame}
    """
    try:
        from vnstock import Vnstock  # type: ignore
        from config.constants import VN30_TICKERS

        board = Vnstock().stock(source="VCI").trading.price_board(symbols=VN30_TICKERS)
        if board.empty:
            return {"buy": pd.DataFrame(), "sell": pd.DataFrame()}

        # Tìm cột ticker
        ticker_col = next(
            (c for c in board.columns if c.lower() in ("listing_symbol", "symbol", "ticker", "code")),
            board.columns[0]
        )

        # Tìm cột foreign buy/sell volume
        fbuy_col  = next((c for c in board.columns if "foreign" in c.lower() and "buy" in c.lower() and "vol" in c.lower()), None)
        fsell_col = next((c for c in board.columns if "foreign" in c.lower() and "sell" in c.lower() and "vol" in c.lower()), None)

        if not fbuy_col or not fsell_col:
            log.warning("get_top_foreign_net: price_board không có cột foreign flow. Columns: %s", list(board.columns))
            return {"buy": pd.DataFrame(), "sell": pd.DataFrame()}

        board = board.copy()
        board["net_vol"] = pd.to_numeric(board[fbuy_col], errors="coerce") - pd.to_numeric(board[fsell_col], errors="coerce")
        board["ticker"] = board[ticker_col]

        df = board[["ticker", "net_vol"]].dropna(subset=["net_vol"])
        return {
            "buy":  df.nlargest(top_n, "net_vol").reset_index(drop=True),
            "sell": df.nsmallest(top_n, "net_vol").reset_index(drop=True),
        }
    except Exception as exc:
        log.warning("get_top_foreign_net lỗi: %s", exc)
        return {"buy": pd.DataFrame(), "sell": pd.DataFrame()}


def summarize_foreign_flow(ticker: str, period: str = "1m") -> str:
    """Tóm tắt dòng tiền ngoại cho 1 mã (dùng bởi LangGraph tools)."""
    df = get_foreign_flow(ticker, period)
    room = get_foreign_room(ticker)

    if df.empty:
        return f"Không có dữ liệu khối ngoại cho {ticker}."

    net_total = df["net_val"].sum() if "net_val" in df.columns else 0
    trend = "MUA RÒNG" if net_total > 0 else "BÁN RÒNG"
    room_str = (
        f"Room còn lại: {room['remaining_pct']}%"
        + (" ⚠️ GẦN ĐẦY" if room.get("alert") else "")
        if room.get("remaining_pct") is not None else "N/A"
    )

    return (
        f"[{ticker}] Khối ngoại {period}: {trend} | "
        f"Net value: {net_total/1e9:.1f} tỷ | {room_str}"
    )
