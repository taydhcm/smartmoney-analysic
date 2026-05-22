"""
ml/entry_timing.py
S5 Entry Timing Engine.

Tính entry zone, stop-loss, target price từ OHLCV:
  - Support: swing low trong lookback phiên (fractal low)
  - Resistance: swing high trong lookback phiên
  - Entry zone: [current * 0.997, current * 1.003] — không chase giá
  - SL price: max(support − ATR × multiplier,  current × 0.95)
  - Target: entry_mid × (1 + target_pct)
  - R:R = target_gain_pct / sl_loss_pct

D3.3 SL Filter:
  - Reject nếu sl_pct > MAX_SL_PCT (7%) — SL quá rộng, risk không kiểm soát được
  - Reject nếu rr_ratio < MIN_RR (1.0) — R:R không đủ hấp dẫn
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

from utils.logger import get_logger

log = get_logger(__name__)

# ── Tham số D3.3 Filter ───────────────────────────────────────────────────────
MAX_SL_PCT  = 0.07   # SL tối đa 7% từ entry
MIN_RR      = 0.80   # Risk-reward tối thiểu 0.8 (với P≥0.65 vẫn dương EV)
TARGET_PCT  = 0.05   # M2 target +5%

# ATR buffer bên dưới support để đặt SL
ATR_SL_MULT = 0.50


@dataclass
class EntryZone:
    """Entry zone + SL + target cho 1 cổ phiếu."""
    entry_low:    float   # Giá thấp nhất nên mua (không mua quá cao hơn)
    entry_high:   float   # Giá cao nhất trong entry zone
    sl_price:     float   # Giá đặt stop-loss
    target_price: float   # Giá mục tiêu (+5%)
    sl_pct:       float   # % loss từ entry_mid đến sl (0.05 = 5%)
    rr_ratio:     float   # Risk:Reward (target_gain / sl_loss)
    support:      float   # Swing low gần nhất trong lookback
    resistance:   float   # Swing high gần nhất trong lookback
    atr:          float   # ATR(14) tính bằng điểm giá
    in_entry_zone: bool   # Giá hiện tại <= entry_high?

    def is_valid(self) -> bool:
        """D3.3 SL Filter: reject nếu SL quá rộng hoặc R:R không đủ."""
        return self.sl_pct <= MAX_SL_PCT and self.rr_ratio >= MIN_RR

    def as_dict(self) -> dict:
        d = asdict(self)
        d["is_valid"] = self.is_valid()
        return d


def compute_entry_zone(
    df: pd.DataFrame,
    lookback: int = 15,
    atr_window: int = 14,
    atr_multiplier: float = ATR_SL_MULT,
    target_pct: float = TARGET_PCT,
) -> EntryZone | None:
    """
    Tính entry zone, SL, target cho 1 ticker từ OHLCV.

    Parameters
    ----------
    df             : OHLCV (bắt buộc: date, open, high, low, close).
    lookback       : Số phiên lookback để xác định support/resistance.
    atr_window     : Cửa sổ ATR.
    atr_multiplier : Hệ số nhân ATR cho buffer bên dưới support → SL.
    target_pct     : Mục tiêu lợi nhuận (+5%).

    Returns
    -------
    EntryZone hoặc None nếu không đủ dữ liệu / giá không hợp lệ.
    """
    if df is None or df.empty or len(df) < max(lookback, atr_window + 1):
        return None

    df = df.copy().sort_values("date").reset_index(drop=True)
    close = df["close"]
    current_price = float(close.iloc[-1])

    if current_price <= 0:
        return None

    # ── ATR(14) ────────────────────────────────────────────────────────────────
    prev_close = close.shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.rolling(atr_window, min_periods=max(atr_window // 2, 3)).mean()
    atr_14 = float(atr_series.iloc[-1])
    if np.isnan(atr_14) or atr_14 <= 0:
        # fallback: 1% của giá hiện tại
        atr_14 = current_price * 0.01

    # ── Support / Resistance trong lookback phiên ──────────────────────────────
    recent = df.iloc[-lookback:].copy()
    support    = float(recent["low"].min())
    resistance = float(recent["high"].max())

    # Sanity check
    if support <= 0 or support >= current_price * 1.10:
        support = current_price * 0.95   # fallback: -5% từ giá hiện tại

    # ── Entry zone ─────────────────────────────────────────────────────────────
    # ±0.3% quanh giá hiện tại — đủ tight để không chase
    entry_low  = round(current_price * 0.997, 2)
    entry_high = round(current_price * 1.003, 2)
    entry_mid  = (entry_low + entry_high) / 2

    # ── SL price ───────────────────────────────────────────────────────────────
    # SL = support − ATR×multiplier  (tránh bị sweep false breakout)
    # Cap: không được thấp hơn −8% từ entry (tránh SL phi lý)
    sl_raw   = support - atr_14 * atr_multiplier
    sl_floor = entry_mid * (1 - MAX_SL_PCT - 0.01)   # 1% buffer dưới max_sl
    sl_price = round(max(sl_raw, sl_floor), 2)

    # ── Target ─────────────────────────────────────────────────────────────────
    target_price = round(entry_mid * (1 + target_pct), 2)

    # ── Risk metrics ───────────────────────────────────────────────────────────
    sl_pct   = (entry_mid - sl_price) / entry_mid if entry_mid > 0 else MAX_SL_PCT
    rr_ratio = target_pct / sl_pct if sl_pct > 1e-6 else 0.0

    return EntryZone(
        entry_low=entry_low,
        entry_high=entry_high,
        sl_price=sl_price,
        target_price=target_price,
        sl_pct=round(sl_pct, 4),
        rr_ratio=round(rr_ratio, 2),
        support=round(support, 2),
        resistance=round(resistance, 2),
        atr=round(atr_14, 2),
        in_entry_zone=(current_price <= entry_high),
    )
