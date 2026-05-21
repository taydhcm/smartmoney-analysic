"""
data/proprietary_trading.py
Phân tích dòng tiền tự doanh (proprietary trading của công ty chứng khoán).

Nguồn dữ liệu:
  - VNDirect FINFO API: thử lấy field propBuyVolume/propSellVolume (không chắc có)
  - SSI Fast Connect  : có khi kích hoạt credentials (TODO)
  - KBS/VCI           : không có endpoint này

Hiện tại: dùng provider layer — VNDirect → empty nếu không có field.
"""

from __future__ import annotations

import os
import pandas as pd

from config.constants import PERIOD_DAYS
from utils.cache import ttl_cache
from utils.logger import get_logger
from data.providers import get_provider

log = get_logger(__name__)

_provider = get_provider(os.getenv("FLOW_PROVIDER", "auto"))


@ttl_cache()
def get_tu_doan_flow(ticker: str, period: str = "1w") -> pd.DataFrame:
    """
    Dữ liệu tự doanh cho 1 mã theo ngày.

    Nguồn: VNDirect FINFO (nếu có field propBuyVolume), SSI FC (khi kích hoạt).
    Trả về DataFrame rỗng nếu không có dữ liệu.

    Schema: date, buy_vol, sell_vol, net_vol, net_val
    """
    days = PERIOD_DAYS.get(period, 7)
    try:
        df = _provider.get_prop_trading(ticker, days=days)
        if df is not None and not df.empty:
            return df
    except Exception as exc:
        log.info("get_tu_doan_flow(%s): %s", ticker, exc)

    log.info("get_tu_doan_flow(%s): không có dữ liệu tự doanh từ provider [%s]",
             ticker, _provider.name)
    return pd.DataFrame()


@ttl_cache()
def get_top_tu_doan_net(exchange: str = "HOSE", top_n: int = 10) -> dict[str, pd.DataFrame]:
    """
    Top mã tự doanh mua ròng / bán ròng nhiều nhất.
    Trả về dict với DataFrame rỗng cho đến khi có nguồn dữ liệu.

    TODO: Implement khi có SSI Fast Connect credentials.
    """
    return {"buy": pd.DataFrame(), "sell": pd.DataFrame()}


def summarize_tu_doan(ticker: str, period: str = "1w") -> str:
    """Tóm tắt dòng tiền tự doanh cho 1 mã (dùng bởi LangGraph tools)."""
    df = get_tu_doan_flow(ticker, period)
    if df.empty:
        return (
            f"[{ticker}] Tự doanh: Chưa có dữ liệu. "
            f"Sẽ khả dụng khi kích hoạt SSI Fast Connect."
        )
    net_total = df["net_val"].sum() if "net_val" in df.columns else 0
    trend = "MUA RÒNG" if net_total > 0 else "BÁN RÒNG"
    return (
        f"[{ticker}] Tự doanh {period}: {trend} | "
        f"Net val: {net_total/1e9:.1f} tỷ ({len(df)} ngày)"
    )
