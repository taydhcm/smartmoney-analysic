"""
data/proprietary_trading.py
Phân tích dòng tiền tự doanh (proprietary trading của công ty chứng khoán).
Tự doanh là một trong những chỉ báo smart money quan trọng trên thị trường VN.
"""

from __future__ import annotations

import pandas as pd

from config.constants import PERIOD_DAYS
from utils.cache import ttl_cache
from utils.logger import get_logger

log = get_logger(__name__)


@ttl_cache()
def get_tu_doan_flow(ticker: str, period: str = "1m") -> pd.DataFrame:
    """
    Dữ liệu tự doanh cho 1 mã theo ngày.
    Dùng vnstock 4.x: Vnstock().stock(symbol, source='TCBS').quote.history()
    Trả về: date, buy_vol, sell_vol, net_vol, buy_val, sell_val, net_val
    """
    try:
        from vnstock import Vnstock  # type: ignore
        from datetime import date as dt, timedelta

        days  = PERIOD_DAYS.get(period, 22)
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

        # TCBS camelCase → snake_case cho tự doanh
        col_map = {
            "propBuyVolume":  "buy_vol",
            "propSellVolume": "sell_vol",
            "propBuyValue":   "buy_val",
            "propSellValue":  "sell_val",
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
        log.warning("get_tu_doan_flow(%s) lỗi: %s", ticker, exc)
        return pd.DataFrame()


@ttl_cache()
def get_top_tu_doan_net(exchange: str = "HOSE", top_n: int = 10) -> dict[str, pd.DataFrame]:
    """
    Top mã tự doanh mua ròng / bán ròng nhiều nhất.
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

        # Tìm cột prop/tu-doan net value
        prop_col = next(
            (c for c in board.columns if "prop" in c.lower() and "net" in c.lower()),
            None
        )
        # Fallback: tính từ buy - sell
        if prop_col is None:
            pbuy  = next((c for c in board.columns if "prop" in c.lower() and "buy" in c.lower()), None)
            psell = next((c for c in board.columns if "prop" in c.lower() and "sell" in c.lower()), None)
            if pbuy and psell:
                board = board.copy()
                board["_prop_net"] = pd.to_numeric(board[pbuy], errors="coerce") - pd.to_numeric(board[psell], errors="coerce")
                prop_col = "_prop_net"

        if prop_col is None:
            log.warning("get_top_tu_doan_net: price_board không có cột tự doanh. Columns: %s", list(board.columns))
            return {"buy": pd.DataFrame(), "sell": pd.DataFrame()}

        board = board.copy() if "_prop_net" not in board.columns else board
        board["net_val"] = pd.to_numeric(board[prop_col], errors="coerce")
        board["ticker"] = board[ticker_col]

        df = board[["ticker", "net_val"]].dropna(subset=["net_val"])
        return {
            "buy":  df.nlargest(top_n, "net_val").reset_index(drop=True),
            "sell": df.nsmallest(top_n, "net_val").reset_index(drop=True),
        }
    except Exception as exc:
        log.warning("get_top_tu_doan_net lỗi: %s", exc)
        return {"buy": pd.DataFrame(), "sell": pd.DataFrame()}


def summarize_tu_doan(ticker: str, period: str = "1m") -> str:
    """Tóm tắt dòng tiền tự doanh (dùng bởi LangGraph tools)."""
    df = get_tu_doan_flow(ticker, period)
    if df.empty:
        return f"Không có dữ liệu tự doanh cho {ticker}."
    net_total = df["net_val"].sum() if "net_val" in df.columns else 0
    trend = "MUA RÒNG" if net_total > 0 else "BÁN RÒNG"
    return f"[{ticker}] Tự doanh {period}: {trend} | Net value: {net_total/1e9:.1f} tỷ"
