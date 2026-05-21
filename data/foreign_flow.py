"""
data/foreign_flow.py
Phân tích dòng tiền khối ngoại (foreign investor flows).

Kiến trúc Provider:
  - VNDirect FINFO API  (ưu tiên 1): lịch sử N ngày, miễn phí, không cần key
  - KBS Snapshot        (fallback 2): hôm nay + tích lũy lên đĩa tự động
  - SSI Fast Connect    (tương lai) : khi có credentials từ iBoard SSI

Để chuyển provider: set biến môi trường FLOW_PROVIDER=ssi/kbs/vndirect/auto
Mặc định: "auto" (VNDirect → KBS fallback)
"""

from __future__ import annotations

import os
import pandas as pd

from config.constants import PERIOD_DAYS, FOREIGN_ROOM_ALERT_PCT, VN30_TICKERS
from utils.cache import ttl_cache
from utils.logger import get_logger
from data.providers import get_provider

log = get_logger(__name__)

# Lấy provider theo cấu hình (mặc định "auto")
_PROVIDER_NAME = os.getenv("FLOW_PROVIDER", "auto")
_provider = get_provider(_PROVIDER_NAME)
log.info("foreign_flow: dùng provider [%s]", _provider.name)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

@ttl_cache()
def get_foreign_flow(ticker: str, period: str = "1w") -> pd.DataFrame:
    """
    Lịch sử dòng tiền khối ngoại cho 1 mã.

    Args:
        ticker: Mã chứng khoán (VD: "VIC")
        period: "1w" | "2w" | "1m" | "3m" (dùng PERIOD_DAYS để convert)

    Returns:
        DataFrame: date, buy_vol, sell_vol, net_vol, net_val
        Sắp xếp tăng dần theo date.
    """
    days = PERIOD_DAYS.get(period, 7)
    try:
        df = _provider.get_foreign_flow(ticker, days=days)
        return df
    except Exception as exc:
        log.warning("get_foreign_flow(%s, %s) lỗi: %s", ticker, period, exc)
        return pd.DataFrame()


@ttl_cache()
def get_foreign_room(ticker: str) -> dict:
    """
    Lấy room nước ngoài còn lại của 1 mã.
    Luôn dùng KBS (real-time, không cần lịch sử).

    Returns:
        {ticker, max_room_pct, used_pct, remaining_pct, alert}
    """
    try:
        from vnstock.api.trading import Trading  # type: ignore
        board = Trading(source="KBS").price_board(symbols_list=[ticker])
        if board.empty:
            raise ValueError("price_board trống")

        row = board.iloc[0]
        fp = float(pd.to_numeric(row.get("foreign_ownership_ratio", 0), errors="coerce") or 0)
        used_pct = fp if fp > 1 else fp * 100

        max_pct   = 49.0
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
    Top mã có khối ngoại mua ròng / bán ròng nhiều nhất hôm nay.

    Returns:
        {"buy": DataFrame, "sell": DataFrame}
        Mỗi DataFrame: ticker, net_vol, net_val
    """
    try:
        return _provider.get_top_foreign_net(VN30_TICKERS, top_n=top_n)
    except Exception as exc:
        log.warning("get_top_foreign_net lỗi: %s", exc)
        empty = pd.DataFrame(columns=["ticker", "net_vol", "net_val"])
        return {"buy": empty, "sell": empty}

def summarize_foreign_flow(ticker: str, period: str = "1w") -> str:
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
        f"Net val: {net_total/1e9:.1f} tỷ ({len(df)} ngày) | {room_str}"
    )
