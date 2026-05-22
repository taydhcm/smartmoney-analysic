"""
ml/regime.py  —  S3: Market Regime Engine
==========================================
5-state machine xác định trạng thái thị trường từ VN30 OHLCV.

States (từ bullish → bearish):
    BULL       — uptrend rõ, ADX mạnh, DI+ > DI-
    BULL_WEAK  — uptrend nhưng momentum yếu hoặc breadth kém
    NEUTRAL    — không rõ xu hướng, thị trường đi ngang
    BEAR_WEAK  — bắt đầu downtrend, cẩn thận
    BEAR       — downtrend mạnh, TẮT hoàn toàn long signal

Threshold P_alpha và max position theo từng state:
    BULL       → P_min=0.65, max_pos=3
    BULL_WEAK  → P_min=0.68, max_pos=2
    NEUTRAL    → P_min=0.72, max_pos=1
    BEAR_WEAK  → P_min=0.80, max_pos=1  (chỉ pick cực kỳ chọn lọc)
    BEAR       → DISABLED (P_min=1.01 → luôn reject)

Confirmation rule:
    - Flip BULL→BEAR yêu cầu `confirmation_days` phiên liên tiếp bearish signal
    - Tránh whipsaw khi thị trường volatile
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


# ── Regime states ─────────────────────────────────────────────────────────────

class RegimeState(str, Enum):
    BULL      = "BULL"
    BULL_WEAK = "BULL_WEAK"
    NEUTRAL   = "NEUTRAL"
    BEAR_WEAK = "BEAR_WEAK"
    BEAR      = "BEAR"


# Mapping state → emoji cho UI
REGIME_EMOJI: dict[RegimeState, str] = {
    RegimeState.BULL:      "🟢",
    RegimeState.BULL_WEAK: "🟡",
    RegimeState.NEUTRAL:   "⚪",
    RegimeState.BEAR_WEAK: "🟠",
    RegimeState.BEAR:      "🔴",
}

REGIME_LABEL_VI: dict[RegimeState, str] = {
    RegimeState.BULL:      "Bull mạnh",
    RegimeState.BULL_WEAK: "Bull yếu",
    RegimeState.NEUTRAL:   "Trung tính",
    RegimeState.BEAR_WEAK: "Bear yếu",
    RegimeState.BEAR:      "Bear mạnh",
}

# Threshold P_alpha và max vị thế mở đồng thời theo regime
REGIME_THRESHOLDS: dict[RegimeState, dict] = {
    RegimeState.BULL:      {"min_prob": 0.65, "max_positions": 3},
    RegimeState.BULL_WEAK: {"min_prob": 0.68, "max_positions": 2},
    RegimeState.NEUTRAL:   {"min_prob": 0.72, "max_positions": 1},
    RegimeState.BEAR_WEAK: {"min_prob": 0.80, "max_positions": 1},
    RegimeState.BEAR:      {"min_prob": 1.01, "max_positions": 0},   # disable long
}


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class RegimeInfo:
    """Kết quả đầy đủ từ Market Regime Engine."""
    state:          RegimeState
    vn30_vs_sma20:  float   # VN30/SMA20 - 1  (dương = trên SMA, âm = dưới)
    adx:            float   # ADX(14) của VN30  [0, 100]
    di_plus:        float   # +DI(14)
    di_minus:       float   # -DI(14)
    breadth_ratio:  float   # % cổ phiếu tăng [0,1]; -1 = chưa có data D0.3
    min_prob:       float   # Threshold P tối thiểu cho regime này
    max_positions:  int     # Max số lệnh mở đồng thời
    reason:         str     # Lý do xác định regime (debug)
    confirmed_days: int = field(default=1)  # Số phiên liên tiếp signal này

    @property
    def is_tradeable(self) -> bool:
        """False khi BEAR — không vào lệnh long."""
        return self.state != RegimeState.BEAR

    @property
    def emoji(self) -> str:
        return REGIME_EMOJI[self.state]

    @property
    def label_vi(self) -> str:
        return REGIME_LABEL_VI[self.state]

    def as_dict(self) -> dict:
        return {
            "regime":          self.state.value,
            "regime_label":    self.label_vi,
            "vn30_vs_sma20":   round(self.vn30_vs_sma20 * 100, 2),   # %
            "adx":             round(self.adx, 1),
            "di_plus":         round(self.di_plus, 1),
            "di_minus":        round(self.di_minus, 1),
            "breadth_ratio":   round(self.breadth_ratio, 3) if self.breadth_ratio >= 0 else None,
            "min_prob":        self.min_prob,
            "max_positions":   self.max_positions,
            "is_tradeable":    self.is_tradeable,
            "confirmed_days":  self.confirmed_days,
            "reason":          self.reason,
        }


# ── ADX / DI indicator ────────────────────────────────────────────────────────

def _compute_adx_di(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Tính ADX(window), +DI(window), -DI(window) từ OHLCV.
    Dùng Wilder's smoothing (EMA với alpha=1/window).

    Returns: DataFrame[adx, di_plus, di_minus] cùng index với df.
    """
    high       = df["high"]
    low        = df["low"]
    close      = df["close"]
    prev_close = close.shift(1)
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)

    # True Range
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Directional Movement raw
    dm_plus_raw  = (high - prev_high).clip(lower=0)
    dm_minus_raw = (prev_low - low).clip(lower=0)

    # Khi cả 2 dương, chỉ giữ cái lớn hơn
    both_pos    = (dm_plus_raw > 0) & (dm_minus_raw > 0)
    dm_plus_raw  = dm_plus_raw.where(~both_pos | (dm_plus_raw >= dm_minus_raw), 0.0)
    dm_minus_raw = dm_minus_raw.where(~both_pos | (dm_minus_raw > dm_plus_raw),  0.0)

    alpha = 1.0 / window
    atr_s  = tr.ewm(alpha=alpha, min_periods=window // 2, adjust=False).mean()
    dmp_s  = dm_plus_raw.ewm(alpha=alpha, min_periods=window // 2, adjust=False).mean()
    dmm_s  = dm_minus_raw.ewm(alpha=alpha, min_periods=window // 2, adjust=False).mean()

    di_plus  = 100.0 * dmp_s / (atr_s + 1e-9)
    di_minus = 100.0 * dmm_s / (atr_s + 1e-9)

    dx  = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
    adx = dx.ewm(alpha=alpha, min_periods=window // 2, adjust=False).mean()

    return pd.DataFrame(
        {"adx": adx.values, "di_plus": di_plus.values, "di_minus": di_minus.values},
        index=df.index,
    )


# ── Raw signal scoring per row ────────────────────────────────────────────────

def _raw_regime_score(
    vn30_vs_sma20: float,
    adx: float,
    di_plus: float,
    di_minus: float,
    breadth_ratio: float,
) -> int:
    """
    Tính raw regime score từ –4 (very bearish) đến +4 (very bullish).
    Dùng nội bộ để xác định state.

    Điểm số:
      vn30_vs_sma20:  +2 / +1 / 0 / -1 / -2
      ADX trend:      +1 (trending up) / -1 (trending down) / 0 (no trend)
      breadth:        +1 (broad) / -1 (narrow) / 0 (no data)
    """
    score = 0

    # ── VN30 position vs SMA20 ────────────────────────────────────────────────
    if vn30_vs_sma20 > 0.03:       # > SMA20 rõ ràng (>3%)
        score += 2
    elif vn30_vs_sma20 > 0.005:    # > SMA20 nhẹ (0.5%–3%)
        score += 1
    elif vn30_vs_sma20 < -0.03:    # < SMA20 rõ ràng
        score -= 2
    elif vn30_vs_sma20 < -0.005:   # < SMA20 nhẹ
        score -= 1
    # else: ±0.5% = neutral, score += 0

    # ── ADX + DI direction ────────────────────────────────────────────────────
    if adx > 20:
        if di_plus > di_minus:
            score += 1   # trending UP
        else:
            score -= 1   # trending DOWN
    # ADX < 20: weak trend → no directional score

    # ── Market breadth (optional, D0.3) ──────────────────────────────────────
    if breadth_ratio >= 0:          # có data
        if breadth_ratio > 0.60:
            score += 1
        elif breadth_ratio < 0.40:
            score -= 1

    return score


def _score_to_state(score: int) -> RegimeState:
    """Convert raw score → RegimeState."""
    if score >= 3:
        return RegimeState.BULL
    elif score >= 1:
        return RegimeState.BULL_WEAK
    elif score >= -1:
        return RegimeState.NEUTRAL
    elif score >= -2:
        return RegimeState.BEAR_WEAK
    else:
        return RegimeState.BEAR


# ── Public API ────────────────────────────────────────────────────────────────

def get_market_regime(
    vn30_df: pd.DataFrame,
    breadth_ratio: float = -1.0,
    confirmation_days: int = 3,
) -> RegimeInfo:
    """
    Xác định Market Regime từ VN30 OHLCV.

    Parameters
    ----------
    vn30_df          : OHLCV của VN30 (cột date, open, high, low, close, volume).
                       Cần ít nhất 30 phiên để ADX ổn định.
    breadth_ratio    : % cổ phiếu tăng giá [0, 1].
                       Truyền -1 nếu D0.3 Market Breadth Feed chưa có data.
    confirmation_days: Số phiên liên tiếp bearish signal để flip sang BEAR.
                       Chống whipsaw trong thị trường volatile.

    Returns
    -------
    RegimeInfo với state đầy đủ context.
    """
    if vn30_df is None or vn30_df.empty:
        log.warning("VN30 data rỗng – fallback NEUTRAL")
        return _fallback_regime("VN30 data rỗng – không xác định được regime")

    df = vn30_df.copy().sort_values("date").reset_index(drop=True)
    if len(df) < 20:
        log.warning("VN30 chỉ có %d phiên – cần ít nhất 20 để tính regime", len(df))
        return _fallback_regime(f"Thiếu data: chỉ có {len(df)} phiên")

    close = df["close"]

    # ── Tính chỉ báo ────────────────────────────────────────────────────────
    sma20     = close.rolling(20, min_periods=10).mean()
    adx_df    = _compute_adx_di(df, window=14)

    # Lấy các giá trị của N phiên cuối để tính confirmed signal
    n_check = min(confirmation_days, len(df))
    tail_idx = slice(-n_check, None)

    vs_sma20_tail  = ((close / (sma20 + 1e-9)) - 1).iloc[tail_idx].values
    adx_tail       = adx_df["adx"].iloc[tail_idx].values
    di_plus_tail   = adx_df["di_plus"].iloc[tail_idx].values
    di_minus_tail  = adx_df["di_minus"].iloc[tail_idx].values

    # ── Tính score cho từng phiên trong confirmation window ─────────────────
    scores = [
        _raw_regime_score(
            float(vs_sma20_tail[k]),
            float(adx_tail[k]),
            float(di_plus_tail[k]),
            float(di_minus_tail[k]),
            breadth_ratio,
        )
        for k in range(n_check)
    ]

    # State của phiên CUỐI (hôm nay / hôm qua)
    current_state = _score_to_state(scores[-1])

    # Xác nhận: nếu flip từ BULL → BEAR/BEAR_WEAK cần confirmation_days phiên
    # Để đơn giản: nếu trung bình score < ngưỡng thì mới flip hoàn toàn
    avg_score     = float(np.mean(scores))
    avg_state     = _score_to_state(int(round(avg_score)))

    # Anti-whipsaw: chỉ flip khi avg_state cũng nguy hiểm
    # Nếu current = BEAR nhưng avg = NEUTRAL → xuống BEAR_WEAK thôi
    final_state = current_state
    if current_state == RegimeState.BEAR and avg_state not in (RegimeState.BEAR, RegimeState.BEAR_WEAK):
        final_state = RegimeState.BEAR_WEAK
        reason_suffix = " (confirmed_days chưa đủ → giảm xuống BEAR_WEAK)"
    else:
        reason_suffix = ""

    # Số phiên liên tiếp cùng state
    confirmed = 1
    for k in range(len(scores) - 2, -1, -1):
        if _score_to_state(scores[k]) == final_state:
            confirmed += 1
        else:
            break

    # ── Tạo lý do mô tả ──────────────────────────────────────────────────────
    last_vs  = float(vs_sma20_tail[-1])
    last_adx = float(adx_tail[-1])
    last_dip = float(di_plus_tail[-1])
    last_dim = float(di_minus_tail[-1])

    reason_parts = [
        f"VN30 {'trên' if last_vs > 0 else 'dưới'} SMA20 {abs(last_vs)*100:.1f}%",
        f"ADX={last_adx:.1f}",
        f"DI+={last_dip:.1f} DI-={last_dim:.1f}",
    ]
    if breadth_ratio >= 0:
        reason_parts.append(f"Breadth={breadth_ratio*100:.0f}%")
    reason = " · ".join(reason_parts) + reason_suffix

    thresholds = REGIME_THRESHOLDS[final_state]

    log.info(
        "Market Regime: %s (score=%d avg=%.1f confirmed=%d) | %s",
        final_state.value, scores[-1], avg_score, confirmed, reason,
    )

    return RegimeInfo(
        state=final_state,
        vn30_vs_sma20=last_vs,
        adx=last_adx,
        di_plus=last_dip,
        di_minus=last_dim,
        breadth_ratio=breadth_ratio,
        min_prob=thresholds["min_prob"],
        max_positions=thresholds["max_positions"],
        reason=reason,
        confirmed_days=confirmed,
    )


def _fallback_regime(reason: str) -> RegimeInfo:
    """Fallback an toàn khi không đủ data — trả về NEUTRAL."""
    thresholds = REGIME_THRESHOLDS[RegimeState.NEUTRAL]
    return RegimeInfo(
        state=RegimeState.NEUTRAL,
        vn30_vs_sma20=0.0,
        adx=0.0,
        di_plus=0.0,
        di_minus=0.0,
        breadth_ratio=-1.0,
        min_prob=thresholds["min_prob"],
        max_positions=thresholds["max_positions"],
        reason=reason,
        confirmed_days=0,
    )


# ── Historical regime series (dùng để build feature cho ML) ──────────────────

def compute_regime_series(
    vn30_df: pd.DataFrame,
    confirmation_days: int = 3,
) -> pd.DataFrame:
    """
    Tính chuỗi regime theo thời gian — dùng trong M1 Feature Builder.

    Trả về DataFrame[date, regime_score, adx_vn30, vn30_di_diff] cùng index date với vn30_df.
    KHÔNG có look-ahead: tại ngày T, dùng dữ liệu đến T.

    regime_score: -4 → +4 (raw score trước khi map sang state)
    adx_vn30:     ADX(14) của VN30, normalized [0, 1]
    vn30_di_diff: (DI+ - DI-) / 100 — dương = bullish trend
    """
    if vn30_df is None or vn30_df.empty:
        return pd.DataFrame()

    df = vn30_df.copy().sort_values("date").reset_index(drop=True)
    close = df["close"]

    sma20  = close.rolling(20, min_periods=10).mean()
    vs_sma = (close / (sma20 + 1e-9)) - 1
    adx_df = _compute_adx_di(df, window=14)

    scores = pd.Series(
        [
            _raw_regime_score(
                float(vs_sma.iloc[i]),
                float(adx_df["adx"].iloc[i]),
                float(adx_df["di_plus"].iloc[i]),
                float(adx_df["di_minus"].iloc[i]),
                -1.0,   # breadth không có trong historical training
            )
            for i in range(len(df))
        ],
        index=df.index,
    )

    result = pd.DataFrame({
        "date":          df["date"],
        "regime_score":  scores.clip(-4, 4).astype(float),
        "adx_vn30":      (adx_df["adx"] / 100.0).clip(0, 1),    # normalize to [0,1]
        "vn30_di_diff":  ((adx_df["di_plus"] - adx_df["di_minus"]) / 100.0).clip(-1, 1),
    })

    return result
