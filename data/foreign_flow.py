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
    Trả về DataFrame với các cột:
        date, buy_vol, sell_vol, net_vol, buy_val, sell_val, net_val
    """
    try:
        from vnstock import stock_intraday_data  # type: ignore  # noqa: F401
        # TCBS cung cấp foreign flow trong historical data
        from vnstock import stock_historical_data  # type: ignore
        from datetime import date, timedelta

        days = PERIOD_DAYS.get(period, 22)
        end = date.today().strftime("%Y-%m-%d")
        start = (date.today() - timedelta(days=days * 2)).strftime("%Y-%m-%d")

        df = stock_historical_data(ticker, start, end, "1D", "stock")

        # Chuẩn hóa tên cột (TCBS trả về camelCase)
        col_map = {
            "foreignBuyVolume": "buy_vol",
            "foreignSellVolume": "sell_vol",
            "foreignBuyValue":  "buy_val",
            "foreignSellValue": "sell_val",
        }
        df = df.rename(columns=col_map)

        # Tính net nếu chưa có
        if "buy_vol" in df.columns and "sell_vol" in df.columns:
            df["net_vol"] = df["buy_vol"] - df["sell_vol"]
        if "buy_val" in df.columns and "sell_val" in df.columns:
            df["net_val"] = df["buy_val"] - df["sell_val"]

        df["date"] = pd.to_datetime(df.get("time", df.index))
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
    Trả về: {ticker, max_room_pct, used_pct, remaining_pct, alert}
    """
    try:
        from vnstock import company_overview  # type: ignore
        info = company_overview(ticker)
        # TCBS trả về foreignPercent (% sở hữu nước ngoài hiện tại)
        used_pct = float(info.get("foreignPercent", 0)) * 100
        max_pct  = float(info.get("maxForeignPercent", 49)) * 100
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
    Trả về: {"buy": DataFrame, "sell": DataFrame}
    """
    try:
        from vnstock import market_top_movers  # type: ignore
        df = market_top_movers(floor=exchange, top=200)

        # Lọc cột foreign nếu có
        if "foreignNetValue" not in df.columns:
            log.warning("market_top_movers không có cột foreignNetValue")
            return {"buy": pd.DataFrame(), "sell": pd.DataFrame()}

        df = df[["ticker", "foreignNetValue", "foreignBuyValue", "foreignSellValue"]].copy()
        df.columns = ["ticker", "net_val", "buy_val", "sell_val"]
        df = df.dropna(subset=["net_val"])

        top_buy  = df.nlargest(top_n, "net_val")
        top_sell = df.nsmallest(top_n, "net_val")
        return {"buy": top_buy.reset_index(drop=True),
                "sell": top_sell.reset_index(drop=True)}

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
