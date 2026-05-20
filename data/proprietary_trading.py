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
    Trả về: date, buy_vol, sell_vol, net_vol, buy_val, sell_val, net_val
    """
    try:
        from vnstock import stock_historical_data  # type: ignore
        from datetime import date, timedelta

        days = PERIOD_DAYS.get(period, 22)
        end   = date.today().strftime("%Y-%m-%d")
        start = (date.today() - timedelta(days=days * 2)).strftime("%Y-%m-%d")

        df = stock_historical_data(ticker, start, end, "1D", "stock")

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

        df["date"] = pd.to_datetime(df.get("time", df.index))
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
    Trả về: {"buy": DataFrame, "sell": DataFrame}
    """
    try:
        from vnstock import market_top_movers  # type: ignore
        df = market_top_movers(floor=exchange, top=200)

        prop_col = next((c for c in df.columns if "prop" in c.lower() and "net" in c.lower()), None)
        if prop_col is None:
            log.warning("market_top_movers không có cột tự doanh net")
            return {"buy": pd.DataFrame(), "sell": pd.DataFrame()}

        df = df[["ticker", prop_col]].copy()
        df.columns = ["ticker", "net_val"]
        df = df.dropna(subset=["net_val"])

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
