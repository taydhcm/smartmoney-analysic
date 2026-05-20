"""
data/block_deals.py
Phát hiện giao dịch thỏa thuận lớn (block deals) và giao dịch nội bộ.
Block deals thường là tín hiệu smart money mạnh nhất trên TTCK VN.
"""

from __future__ import annotations

import pandas as pd

from utils.cache import ttl_cache
from utils.logger import get_logger

log = get_logger(__name__)

# Ngưỡng xem là block deal (tỷ đồng)
BLOCK_DEAL_THRESHOLD_BILLION = 10.0


@ttl_cache()
def get_block_deals(exchange: str = "HOSE", days_back: int = 5) -> pd.DataFrame:
    """
    Lấy danh sách giao dịch thỏa thuận lớn gần đây.
    Trả về: date, ticker, volume, value_billion, price, buyer_type
    """
    try:
        from vnstock import market_top_movers  # type: ignore  # placeholder
        # Thực tế: TCBS có endpoint giao dịch thỏa thuận
        # vnstock >= 3.x: dùng stock_intraday_data với type='deal'
        # Đây là skeleton – implement khi confirm API endpoint chính xác
        log.info("get_block_deals: cần xác nhận endpoint TCBS giao dịch thỏa thuận")
        return pd.DataFrame(columns=["date", "ticker", "volume", "value_billion", "price"])
    except Exception as exc:
        log.warning("get_block_deals lỗi: %s", exc)
        return pd.DataFrame()


@ttl_cache()
def get_insider_transactions(ticker: str) -> pd.DataFrame:
    """
    Giao dịch cổ phiếu của cổ đông nội bộ (ban lãnh đạo, cổ đông lớn).
    Nguồn: HOSE/HNX công bố thông tin.
    Trả về: date, person, role, action, volume, price
    """
    try:
        # TODO: scrape từ hsx.vn disclosure page hoặc SSI API
        log.info("get_insider_transactions(%s): cần implement HOSE disclosure scraper", ticker)
        return pd.DataFrame(columns=["date", "person", "role", "action", "volume", "price"])
    except Exception as exc:
        log.warning("get_insider_transactions(%s) lỗi: %s", ticker, exc)
        return pd.DataFrame()


def flag_block_deal_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lọc block deals vượt ngưỡng và đánh dấu là tín hiệu.
    """
    if df.empty or "value_billion" not in df.columns:
        return df
    return df[df["value_billion"] >= BLOCK_DEAL_THRESHOLD_BILLION].copy()
