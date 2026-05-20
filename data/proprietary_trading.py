"""
data/proprietary_trading.py
Phân tích dòng tiền tự doanh (proprietary trading của công ty chứng khoán).

Ghi chú: vnstock 4.x (KBS/VCI) không cung cấp dữ liệu tự doanh theo mã.
TCBS đã bị xóa khỏi vnstock. Các hàm này trả về DataFrame rỗng.
"""

from __future__ import annotations

import pandas as pd

from config.constants import PERIOD_DAYS
from utils.cache import ttl_cache
from utils.logger import get_logger

log = get_logger(__name__)

_NO_DATA_MSG = (
    "Dữ liệu tự doanh không khả dụng trong vnstock 4.x "
    "(TCBS đã bị xóa, KBS/VCI không cung cấp endpoint này)."
)


@ttl_cache()
def get_tu_doan_flow(ticker: str, period: str = "1m") -> pd.DataFrame:
    """
    Dữ liệu tự doanh cho 1 mã theo ngày.
    Hiện không khả dụng trong vnstock 4.x.
    Trả về DataFrame rỗng.
    """
    log.info("get_tu_doan_flow(%s): %s", ticker, _NO_DATA_MSG)
    return pd.DataFrame()


@ttl_cache()
def get_top_tu_doan_net(exchange: str = "HOSE", top_n: int = 10) -> dict[str, pd.DataFrame]:
    """
    Top mã tự doanh mua ròng / bán ròng nhiều nhất.
    Hiện không khả dụng trong vnstock 4.x.
    Trả về dict với DataFrame rỗng.
    """
    log.info("get_top_tu_doan_net: %s", _NO_DATA_MSG)
    return {"buy": pd.DataFrame(), "sell": pd.DataFrame()}


def summarize_tu_doan(ticker: str, period: str = "1m") -> str:
    """Tóm tắt dòng tiền tự doanh cho 1 mã (dùng bởi LangGraph tools)."""
    return f"[{ticker}] Tự doanh: Không có dữ liệu (vnstock 4.x không hỗ trợ)."
