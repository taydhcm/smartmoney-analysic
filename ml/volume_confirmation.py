"""
ml/volume_confirmation.py
S4 Volume Confirmation Engine.

Xác nhận tín hiệu ML bằng volume pattern:
  - vol_surge : volume hôm nay / MA20 (ratio)
  - vol_trend : MA5/MA20 slope — volume đang build-up?
  - vol_quality: % volume từ up-days trong 10 phiên gần nhất
  - obv_score : OBV 5-day momentum [-1, +1]
  - volume_score: composite [-1, +1]
  - is_confirmed: True khi volume_score >= VOL_CONFIRM_THRESHOLD (0.15)

Không look-ahead: tất cả tính từ T trở về trước.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

from utils.logger import get_logger

log = get_logger(__name__)

# ── Ngưỡng xác nhận ──────────────────────────────────────────────────────────
VOL_CONFIRM_THRESHOLD = 0.15   # volume_score >= này → is_confirmed = True


@dataclass
class VolumeConfirmation:
    """Kết quả S4 volume analysis cho 1 cổ phiếu."""
    ticker:       str
    vol_surge:    float   # current_vol / MA20 (1.5 = 50% trên average)
    vol_trend:    float   # slope MA5/MA20 ratio (+ = volume đang tăng dần)
    vol_quality:  float   # fraction volume từ up-days [0, 1]
    obv_score:    float   # OBV 5-day momentum [-1, +1]
    volume_score: float   # Composite score [-1, +1]
    is_confirmed: bool    # volume_score >= VOL_CONFIRM_THRESHOLD
    label:        str     # "Strong" / "Moderate" / "Weak" / "Bearish"

    def as_dict(self) -> dict:
        return asdict(self)


def compute_volume_confirmation(
    df: pd.DataFrame,
    ticker: str = "",
    window: int = 20,
    quality_window: int = 10,
) -> VolumeConfirmation | None:
    """
    Tính volume confirmation cho 1 ticker từ OHLCV.

    Parameters
    ----------
    df             : OHLCV (bắt buộc: date, open, high, low, close, volume).
    ticker         : Mã CK (dùng cho log + VolumeConfirmation.ticker).
    window         : Cửa sổ MA volume (mặc định 20).
    quality_window : Cửa sổ tính volume quality (mặc định 10 phiên).

    Returns
    -------
    VolumeConfirmation hoặc None nếu không đủ dữ liệu.
    """
    if df is None or df.empty or len(df) < window + 2:
        return None

    df = df.copy()
    # Sort by date — handle both DatetimeIndex and a "date" column
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    else:
        df = df.sort_index().reset_index(drop=True)
    close  = df["close"].astype(float)
    volume = df["volume"].astype(float)

    # ── Vol Surge: current vol vs MA20 ────────────────────────────────────────
    vol_ma20    = volume.rolling(window, min_periods=window // 2).mean()
    current_vol = float(volume.iloc[-1])
    ma20_val    = float(vol_ma20.iloc[-1])
    vol_surge   = current_vol / (ma20_val + 1e-9)

    # ── Vol Trend: slope của ratio MA5/MA20 (8 phiên gần nhất) ───────────────
    vol_ma5      = volume.rolling(5, min_periods=3).mean()
    ratio_series = (vol_ma5 / (vol_ma20 + 1e-9)).iloc[-10:]
    if len(ratio_series) >= 4:
        x    = np.arange(len(ratio_series), dtype=float)
        y    = ratio_series.values.astype(float)
        mask = ~np.isnan(y)
        vol_trend = float(np.polyfit(x[mask], y[mask], 1)[0]) if mask.sum() >= 3 else 0.0
    else:
        vol_trend = 0.0

    # ── Vol Quality: % volume đến từ up-days trong quality_window phiên ──────
    recent    = df.iloc[-quality_window:].copy()
    up_mask   = recent["close"] > recent["open"]
    up_vol    = float(recent.loc[up_mask, "volume"].sum())
    total_vol = float(recent["volume"].sum())
    vol_quality = up_vol / (total_vol + 1e-9)

    # ── OBV Score: OBV 5-day momentum ────────────────────────────────────────
    direction = np.sign(close.diff().fillna(0))
    obv       = (direction * volume).cumsum()
    obv_5d    = obv.pct_change(5).iloc[-1]
    obv_score = float(np.clip(obv_5d / 0.20, -1.0, 1.0)) if not np.isnan(obv_5d) else 0.0

    # ── Composite volume_score ────────────────────────────────────────────────
    # vol_surge: 2× avg → +1, 0.5× → -0.5
    surge_comp   = float(np.clip((vol_surge - 1.0) / 1.0, -1.0, 1.0))
    # vol_trend: + = accumulating volume
    trend_comp   = float(np.clip(vol_trend / 0.10, -1.0, 1.0))
    # vol_quality: > 0.6 → good, < 0.4 → bad, normalized around 0.5
    quality_comp = float(np.clip((vol_quality - 0.50) * 4.0, -1.0, 1.0))

    volume_score = float(np.clip(
        0.35 * surge_comp
        + 0.20 * trend_comp
        + 0.25 * quality_comp
        + 0.20 * obv_score,
        -1.0, 1.0,
    ))

    is_confirmed = volume_score >= VOL_CONFIRM_THRESHOLD

    label = (
        "Strong"   if volume_score >= 0.50
        else "Moderate" if volume_score >= VOL_CONFIRM_THRESHOLD
        else "Weak"     if volume_score >= -0.15
        else "Bearish"
    )

    return VolumeConfirmation(
        ticker=ticker,
        vol_surge=round(vol_surge, 3),
        vol_trend=round(vol_trend, 4),
        vol_quality=round(vol_quality, 3),
        obv_score=round(obv_score, 4),
        volume_score=round(volume_score, 4),
        is_confirmed=is_confirmed,
        label=label,
    )
