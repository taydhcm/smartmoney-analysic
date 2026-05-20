"""
data/volume_analysis.py
Phân tích khối lượng: OBV, MFI, VWAP, Volume Profile, Relative Volume.
Tất cả chỉ báo tính từ OHLCV local – không cần thêm API call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.constants import VOLUME_SURGE_RATIO
from utils.logger import get_logger

log = get_logger(__name__)


# ── OBV – On-Balance Volume ────────────────────────────────────────────────────
def calc_obv(df: pd.DataFrame) -> pd.Series:
    """df cần có: close, volume"""
    direction = np.sign(df["close"].diff()).fillna(0)
    obv = (direction * df["volume"]).cumsum()
    return obv.rename("obv")


# ── MFI – Money Flow Index (14-period) ────────────────────────────────────────
def calc_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """df cần có: high, low, close, volume"""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]

    pos_mf = mf.where(tp > tp.shift(1), 0)
    neg_mf = mf.where(tp < tp.shift(1), 0)

    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()

    mfr = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mfr))
    return mfi.rename("mfi")


# ── VWAP – Volume Weighted Average Price ──────────────────────────────────────
def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP cộng dồn từ đầu chuỗi dữ liệu."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (tp * df["volume"]).cumsum()
    cum_vol    = df["volume"].cumsum()
    return (cum_tp_vol / cum_vol).rename("vwap")


# ── Relative Volume ────────────────────────────────────────────────────────────
def calc_relative_volume(df: pd.DataFrame, avg_period: int = 20) -> pd.Series:
    """Volume hôm nay so với trung bình {avg_period} phiên."""
    avg = df["volume"].rolling(avg_period).mean()
    return (df["volume"] / avg).rename("rel_vol")


# ── Volume Profile (price levels) ─────────────────────────────────────────────
def calc_volume_profile(df: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    """
    Gom volume theo dải giá. Hữu ích để tìm Point of Control (PoC).
    Trả về DataFrame: price_level, volume, pct
    """
    price_min = df["low"].min()
    price_max = df["high"].max()
    bin_edges = np.linspace(price_min, price_max, bins + 1)

    records = []
    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (df["close"] >= lo) & (df["close"] < hi)
        vol  = df.loc[mask, "volume"].sum()
        records.append({"price_level": round((lo + hi) / 2, 0), "volume": vol})

    prof = pd.DataFrame(records)
    total = prof["volume"].sum()
    prof["pct"] = (prof["volume"] / total * 100).round(2) if total > 0 else 0
    return prof.sort_values("price_level").reset_index(drop=True)


# ── Volume Surge Detection ─────────────────────────────────────────────────────
def detect_volume_surge(df: pd.DataFrame, avg_period: int = 20) -> pd.DataFrame:
    """
    Đánh dấu các phiên có volume đột biến (> VOLUME_SURGE_RATIO × TB).
    Thêm cột: rel_vol, is_surge, direction
    """
    df = df.copy()
    df["rel_vol"]  = calc_relative_volume(df, avg_period)
    df["is_surge"] = df["rel_vol"] >= VOLUME_SURGE_RATIO
    df["direction"] = np.where(df["close"] >= df["open"], "up", "down")
    return df


# ── Tổng hợp tất cả chỉ báo ───────────────────────────────────────────────────
def enrich_with_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm OBV, MFI, VWAP, rel_vol, is_surge vào DataFrame OHLCV.
    """
    if df.empty or len(df) < 5:
        return df
    df = df.copy()
    df["obv"]     = calc_obv(df)
    df["mfi"]     = calc_mfi(df)
    df["vwap"]    = calc_vwap(df)
    df["rel_vol"] = calc_relative_volume(df)
    df["is_surge"] = df["rel_vol"] >= VOLUME_SURGE_RATIO
    return df


def summarize_volume(ticker: str, df: pd.DataFrame) -> str:
    """Tóm tắt chỉ báo volume thành chuỗi cho LangGraph agent."""
    if df.empty:
        return f"Không có dữ liệu volume cho {ticker}."

    df = enrich_with_volume_indicators(df)
    last = df.iloc[-1]
    surges = int(df["is_surge"].sum()) if "is_surge" in df.columns else 0

    mfi_val  = round(last.get("mfi", 0), 1)
    mfi_note = "QUÁ MUA" if mfi_val > 80 else ("QUÁ BÁN" if mfi_val < 20 else "trung tính")

    return (
        f"[{ticker}] OBV xu hướng {'tăng' if df['obv'].iloc[-1] > df['obv'].iloc[0] else 'giảm'} | "
        f"MFI={mfi_val} ({mfi_note}) | "
        f"RelVol={round(last.get('rel_vol', 1), 2)}x | "
        f"Phiên đột biến volume: {surges}/{len(df)}"
    )
