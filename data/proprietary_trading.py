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
import re
import pandas as pd

from config.constants import PERIOD_DAYS
from utils.cache import ttl_cache
from utils.logger import get_logger
from data.providers import get_provider

log = get_logger(__name__)

_provider = get_provider(os.getenv("FLOW_PROVIDER", "auto"))

# Regex: chỉ giữ cổ phiếu cơ sở (2-5 chữ cái), loại bỏ chứng quyền, phái sinh, ETF có số
_EQUITY_RE = re.compile(r"^[A-Z]{2,5}$")


@ttl_cache()
def get_tu_doan_flow(ticker: str, period: str = "1w") -> pd.DataFrame:
    """
    Dữ liệu tự doanh cho 1 mã theo ngày.

    Nguồn (ưu tiên):
      1. VNDirect FINFO (nếu có field propBuyVolume)
      2. SSI FiinMarket iBoard (batch toàn sàn, dữ liệu phiên hiện tại)
    Trả về DataFrame rỗng nếu không có dữ liệu.

    Schema: date, buy_vol, sell_vol, net_vol, net_val
    Ghi chú: khi dùng SSI fallback, net_val=0 (chưa có giá quy đổi);
              net_vol đơn vị cổ phiếu (shares).
    """
    days = PERIOD_DAYS.get(period, 7)
    try:
        df = _provider.get_prop_trading(ticker, days=days)
        if df is not None and not df.empty:
            return df
    except Exception as exc:
        log.info("get_tu_doan_flow(%s): %s", ticker, exc)

    # Fallback: SSI FiinMarket iBoard (batch toàn sàn)
    try:
        from analytics.ssi_iboard import get_proprietary_batch
        batch = get_proprietary_batch("VNINDEX")
        if batch and ticker in batch:
            entry = batch[ticker]
            net_vol  = entry.get("proprietary_net", 0)
            buy_vol  = entry.get("proprietary_buy", 0)
            sell_vol = entry.get("proprietary_sell", 0)
            date_str = entry.get("date", "")
            if net_vol != 0:
                log.info("get_tu_doan_flow(%s): SSI iBoard fallback, net_vol=%s", ticker, net_vol)
                return pd.DataFrame([{
                    "date":     date_str,
                    "buy_vol":  buy_vol,
                    "sell_vol": sell_vol,
                    "net_vol":  net_vol,
                    "net_val":  0,   # chưa có giá quy đổi sang VND
                }])
    except Exception as exc:
        log.info("get_tu_doan_flow(%s) SSI fallback lỗi: %s", ticker, exc)

    log.info("get_tu_doan_flow(%s): không có dữ liệu tự doanh", ticker)
    return pd.DataFrame()


@ttl_cache()
def get_top_tu_doan_net(exchange: str = "HOSE", top_n: int = 15) -> dict:
    """
    Top mã tự doanh mua ròng / bán ròng nhiều nhất.

    Nguồn: FiinMarket GetProprietaryV2 (SSI iBoard), batch toàn sàn.
    Đơn vị: khối lượng cổ phiếu (shares volume).
    Chỉ bao gồm cổ phiếu cơ sở (loại bỏ chứng quyền, phái sinh).

    Returns:
        {"buy": DataFrame, "sell": DataFrame, "date": str}
        Mỗi DataFrame có cột: ticker, net_vol, date
    """
    empty = {"buy": pd.DataFrame(columns=["ticker", "net_vol"]),
             "sell": pd.DataFrame(columns=["ticker", "net_vol"]),
             "date": "N/A"}
    try:
        from analytics.ssi_iboard import get_proprietary_batch
        com_group = "VNINDEX" if exchange in ("HOSE", "VNINDEX") else exchange
        batch = get_proprietary_batch(com_group)
        if batch:
            # Lấy ngày dữ liệu từ ticker đầu tiên hợp lệ
            batch_date = "N/A"
            for d in batch.values():
                d_date = d.get("date", "")
                if d_date and d_date >= "2020-01-01":
                    batch_date = d_date
                    break

            rows = [
                {
                    "ticker":  t,
                    "net_vol": d["proprietary_net"],
                    "date":    d.get("date", ""),
                }
                for t, d in batch.items()
                # Chỉ giữ cổ phiếu cơ sở (không có chữ số trong ticker)
                if _EQUITY_RE.match(t) and d.get("proprietary_net", 0) != 0
            ]
            if rows:
                all_df  = pd.DataFrame(rows)
                buy_df  = all_df[all_df["net_vol"] > 0].nlargest(top_n, "net_vol").reset_index(drop=True)
                sell_df = all_df[all_df["net_vol"] < 0].nsmallest(top_n, "net_vol").reset_index(drop=True)
                return {"buy": buy_df, "sell": sell_df, "date": batch_date}
    except Exception as exc:
        log.warning("get_top_tu_doan_net FiinMarket lỗi: %s", exc)

    return empty


def summarize_tu_doan(ticker: str, period: str = "1w") -> str:
    """Tóm tắt dòng tiền tự doanh cho 1 mã (dùng bởi LangGraph tools)."""
    df = get_tu_doan_flow(ticker, period)
    if df.empty:
        return (
            f"[{ticker}] Tự doanh: Chưa có dữ liệu. "
            f"Sẽ khả dụng khi kích hoạt SSI Fast Connect."
        )
    net_vol  = df["net_vol"].sum()  if "net_vol"  in df.columns else 0
    net_val  = df["net_val"].sum()  if "net_val"  in df.columns else 0
    date_str = df["date"].iloc[-1]  if "date"     in df.columns else ""

    if net_val != 0:
        # Có số liệu VND đầy đủ
        trend = "MUA RÒNG" if net_val > 0 else "BÁN RÒNG"
        return (
            f"[{ticker}] Tự doanh {period}: {trend} | "
            f"Net val: {net_val/1e9:.1f} tỷ ({len(df)} ngày)"
        )
    else:
        # SSI iBoard fallback: chỉ có khối lượng
        trend = "MUA RÒNG" if net_vol > 0 else "BÁN RÒNG"
        return (
            f"[{ticker}] Tự doanh (phiên {date_str}): {trend} | "
            f"Net vol: {net_vol/1000:+.0f} nghìn CP (nguồn: SSI iBoard)"
        )
