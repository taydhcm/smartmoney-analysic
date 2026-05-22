"""
analytics/wyckoff.py  –  S1 Wyckoff VSA Engine v2.0  (Sprint 5)
================================================================
Comprehensive Wyckoff / Volume Spread Analysis từ OHLCV thuần túy.

Outputs WyckoffResult với 9 trường:
  phase            – 'phase_b' | 'phase_c' | 'phase_d' | 'distribution' | 'none'
  phase_duration   – số bar ước tính phase đang hiệu lực
  spring_quality   – chất lượng Spring [0, 1]
  lps_detected     – Last Point of Support bool
  effort_vs_result – nỗ lực / kết quả [-1, +1]
  no_supply_count  – số bar no-supply trong _NS_WINDOW phiên gần nhất
  stopping_volume  – Selling Climax / stopping volume bool
  volume_profile   – 'contracting' | 'expanding' | 'climax' | 'neutral'
  wyckoff_score    – tổng hợp [0, 1]

Không look-ahead bias: mọi tính toán chỉ dùng data[0:t+1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
_MIN_BARS        = 10    # tối thiểu để chạy phân tích
_VOL_WINDOW      = 20    # cửa sổ rolling volume average
_ATR_WINDOW      = 14    # cửa sổ ATR
_NS_WINDOW       = 10    # no-supply detection: N bar gần nhất
_SPRING_LOOKBACK = 5     # bar sau spring tìm recovery
_VOL_CLIMAX_THR  = 2.5   # vol > 2.5× avg = climax / stopping vol
_NARROW_RANGE    = 0.70  # bar range < 70% ATR = narrow
_LOW_VOL         = 0.70  # bar vol < 70% avg = low volume
_TR_FRACTION     = 2     # dùng 1/2 đầu window để xác định Trading Range


# ── Data class ─────────────────────────────────────────────────────────────────

@dataclass
class WyckoffResult:
    """Kết quả phân tích Wyckoff/VSA đầy đủ cho 1 cửa sổ OHLCV."""

    phase:            str    # Wyckoff phase
    phase_duration:   int    # bars đang ở phase hiện tại (ước tính)
    spring_quality:   float  # chất lượng Spring [0, 1]
    lps_detected:     bool   # Last Point of Support hiện diện
    effort_vs_result: float  # effort (vol) vs result (price) [-1, +1]
    no_supply_count:  int    # số bar no-supply trong _NS_WINDOW phiên gần nhất
    stopping_volume:  bool   # Stopping Volume / Selling Climax
    volume_profile:   str    # đặc điểm volume tổng thể
    wyckoff_score:    float  # tổng hợp [0, 1]

    def as_dict(self) -> dict:
        return {
            "phase":            self.phase,
            "phase_duration":   self.phase_duration,
            "spring_quality":   round(self.spring_quality, 4),
            "lps_detected":     self.lps_detected,
            "effort_vs_result": round(self.effort_vs_result, 4),
            "no_supply_count":  self.no_supply_count,
            "stopping_volume":  self.stopping_volume,
            "volume_profile":   self.volume_profile,
            "wyckoff_score":    round(self.wyckoff_score, 4),
        }


def _default_result(phase: str = "none") -> WyckoffResult:
    return WyckoffResult(
        phase=phase, phase_duration=0, spring_quality=0.0,
        lps_detected=False, effort_vs_result=0.0, no_supply_count=0,
        stopping_volume=False, volume_profile="neutral", wyckoff_score=0.3,
    )


# ── Sub-signal helpers ─────────────────────────────────────────────────────────

def _calc_spring_quality(
    close:   pd.Series,
    low:     pd.Series,
    volume:  pd.Series,
    tr_low:  float,
    vol_avg: float,
) -> float:
    """
    Spring = giá phá briefly xuống dưới TR low rồi hồi phục.
    Chất lượng Spring [0, 1]:
      1.0 = Spring hoàn hảo: penetration nhỏ, vol thấp, recovery nhanh.
      0.0 = không có Spring hoặc Spring kém.
    """
    n = len(close)
    if n < 5 or tr_low <= 0:
        return 0.0

    # Tìm Spring trong 1/3 cuối cửa sổ
    lookback = max(3, n // 3)
    recent_low = low.iloc[-lookback:]
    spring_idx = int(recent_low.idxmin())    # label (= vị trí vì reset_index)
    spring_low = float(low.iloc[spring_idx])

    # Spring phải đi xuống dưới TR low
    if spring_low >= tr_low * 0.9985:
        return 0.0

    # Volume tại bar spring (thấp = tốt: weak supply)
    spring_vol  = float(volume.iloc[spring_idx])
    vol_ratio   = spring_vol / (vol_avg + 1e-9)

    # Độ sâu penetration
    penetration = max(0.0, (tr_low - spring_low) / (tr_low + 1e-9))

    # Recovery: close > tr_low trong vòng _SPRING_LOOKBACK bar
    after = close.iloc[spring_idx + 1:]
    if after.empty:
        return 0.0
    recovered = after > tr_low
    if not recovered.any():
        return 0.0

    recovery_label = int(recovered.idxmax())            # label của bar đầu tiên recover
    recovery_speed = recovery_label - spring_idx        # số bar để recover

    if recovery_speed > _SPRING_LOOKBACK:
        return 0.0

    # Ba thành phần:
    q_pen   = max(0.0, 1.0 - penetration * 15.0)          # 0 nếu > ~6.7% below TR
    q_vol   = max(0.0, 1.0 - (vol_ratio - 0.3) / 1.7)    # 0 nếu vol > ~2× avg
    q_spd   = max(0.0, 1.0 - (recovery_speed - 1) / max(1, _SPRING_LOOKBACK - 1))

    quality = q_pen * 0.40 + q_vol * 0.35 + q_spd * 0.25
    return float(np.clip(quality, 0.0, 1.0))


def _detect_lps(
    close:   pd.Series,
    low:     pd.Series,
    volume:  pd.Series,
    vol_avg: float,
) -> bool:
    """
    Last Point of Support: pullback trên volume thấp, giá giữ vùng hỗ trợ.
    Kiểm tra 5 bar gần nhất.
    """
    n = len(close)
    if n < 6 or vol_avg < 1:
        return False

    support = float(low.quantile(0.20))   # mức hỗ trợ bảo thủ

    rc = close.tail(5)
    rl = low.tail(5)
    rv = volume.tail(5)

    # Giá phải giữ trên mức hỗ trợ
    if rl.min() < support * 0.990:
        return False

    # Cần ít nhất 1 bar giảm (pullback)
    if (rc.diff() < 0).sum() < 1:
        return False

    # Volume pullback phải dưới 80% avg
    if rv.mean() > vol_avg * 0.80:
        return False

    # Bar cuối đóng cửa ở nửa trên range gần nhất
    rng = rc.max() - rc.min()
    if rng < 1e-9:
        return False
    last_pos = (float(rc.iloc[-1]) - float(rc.min())) / rng
    return last_pos >= 0.35


def _calc_effort_vs_result(
    df:      pd.DataFrame,
    vol_avg: float,
    atr_avg: float,
    window:  int = 5,
) -> float:
    """
    Effort (volume) so với Result (biên độ giá):
      Dương  = demand mạnh (giá di chuyển nhiều / volume vừa phải).
      Âm     = supply hấp thụ (volume lớn, giá đứng yên).
    Trả về [-1, +1].
    """
    if len(df) < 3 or vol_avg < 1 or atr_avg < 1e-9:
        return 0.0

    evr_vals: list[float] = []
    for _, row in df.tail(window).iterrows():
        effort    = float(row["volume"]) / (vol_avg + 1e-9)
        result    = (float(row["high"]) - float(row["low"])) / (atr_avg + 1e-9)
        ratio     = result / (effort + 1e-9)
        direction = 1.0 if float(row["close"]) >= float(row["open"]) else -1.0
        evr_vals.append(direction * ratio)

    if not evr_vals:
        return 0.0
    return float(np.tanh(np.mean(evr_vals) * 0.30))


def _count_no_supply_bars(
    df:      pd.DataFrame,
    vol_avg: float,
    atr_avg: float,
) -> int:
    """
    No-supply bar: range hẹp + volume thấp + đóng cửa nửa trên.
    Đếm trong _NS_WINDOW bar gần nhất.
    """
    if len(df) < 3 or vol_avg < 1 or atr_avg < 1e-9:
        return 0

    count = 0
    for _, row in df.tail(_NS_WINDOW).iterrows():
        bar_range = float(row["high"]) - float(row["low"])
        is_narrow = bar_range < atr_avg * _NARROW_RANGE
        is_low_v  = float(row["volume"]) < vol_avg * _LOW_VOL
        upper_cls = (float(row["close"]) - float(row["low"])) / (bar_range + 1e-9) > 0.50
        if is_narrow and is_low_v and upper_cls:
            count += 1
    return count


def _detect_stopping_volume(df: pd.DataFrame, vol_avg: float) -> bool:
    """
    Stopping Volume / Selling Climax:
    Volume cực lớn gần vùng đáy, nhưng đóng cửa nửa trên thanh nến.
    Kiểm tra 5 bar gần nhất.
    """
    if len(df) < 5 or vol_avg < 1:
        return False

    low_threshold = float(df["low"].quantile(0.20))
    for _, row in df.tail(5).iterrows():
        bar_range  = float(row["high"]) - float(row["low"])
        climax_vol = float(row["volume"]) > vol_avg * _VOL_CLIMAX_THR
        upper_cls  = (float(row["close"]) - float(row["low"])) / (bar_range + 1e-9) > 0.40
        near_low   = float(row["low"]) <= low_threshold * 1.02
        if climax_vol and upper_cls and near_low:
            return True
    return False


def _classify_volume_profile(
    volume:  pd.Series,
    close:   pd.Series,
    vol_avg: float,
) -> str:
    """Phân loại đặc điểm volume: contracting / expanding / climax / neutral."""
    n = len(volume)
    if n < 5 or vol_avg < 1:
        return "neutral"

    if volume.tail(5).max() > vol_avg * 3.0:
        return "climax"

    x        = np.arange(n, dtype=float)
    vol_norm = volume.values / (vol_avg + 1e-9)
    slope    = float(np.polyfit(x, vol_norm, 1)[0])
    px_slope = float(np.polyfit(x, close.values, 1)[0])

    if slope < -0.02:
        return "contracting"
    if slope > 0.02 and px_slope > 0:
        return "expanding"
    return "neutral"


def _estimate_phase_duration(
    phase: str,
    close: pd.Series,
    low:   pd.Series,
    high:  pd.Series,
) -> int:
    """Ước tính số bar đang ở phase hiện tại (walk backwards)."""
    n = len(close)
    if phase == "none" or n < 3:
        return 0

    if phase == "phase_d":
        count = 0
        for i in range(n - 1, 0, -1):
            if float(close.iloc[i]) > float(close.iloc[i - 1]):
                count += 1
            else:
                break
        return max(1, count)

    if phase == "phase_c":
        return 1

    if phase in ("phase_b", "distribution"):
        tr_h = float(high.max())
        tr_l = float(low.min())
        tol  = (tr_h - tr_l) * 0.10
        count = 0
        for i in range(n - 1, -1, -1):
            c = float(close.iloc[i])
            if (tr_l - tol) <= c <= (tr_h + tol):
                count += 1
            else:
                break
        return max(1, count)

    return 1


def _phase_base_score(phase: str) -> float:
    return {
        "phase_d":      0.90,
        "phase_c":      0.75,
        "phase_b":      0.55,
        "none":         0.30,
        "distribution": 0.10,
    }.get(phase, 0.30)


# ── Main detection function ────────────────────────────────────────────────────

def detect_wyckoff(df: pd.DataFrame) -> WyckoffResult:
    """
    Phân tích Wyckoff / VSA toàn diện cho một cửa sổ OHLCV.

    Parameters
    ----------
    df : DataFrame có cột [open, high, low, close, volume].
         Cần ≥ 10 bar, sắp xếp tăng dần theo thời gian.

    Returns
    -------
    WyckoffResult (không bao giờ None).
    """
    if df is None or df.empty or len(df) < _MIN_BARS:
        return _default_result()

    df = df.copy().reset_index(drop=True)
    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    open_  = df["open"].astype(float)
    volume = df["volume"].astype(float)
    n      = len(df)

    # Rolling averages
    w_vol = min(_VOL_WINDOW, n)
    vol_avg = float(volume.rolling(w_vol, min_periods=3).mean().iloc[-1])
    if vol_avg < 1:
        vol_avg = float(volume.mean()) + 1e-9

    tr_vals = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_avg = float(tr_vals.rolling(min(_ATR_WINDOW, n), min_periods=3).mean().iloc[-1])
    if atr_avg < 1e-9:
        atr_avg = float(tr_vals.mean()) + 1e-9

    # Trading Range: dùng nửa đầu cửa sổ để tránh Spring contamination
    tr_n    = max(5, n // _TR_FRACTION)
    tr_high = float(high.iloc[:tr_n].max())
    tr_low  = float(low.iloc[:tr_n].min())
    tr_mean = float(close.iloc[:tr_n].mean()) + 1e-9
    tr_width_pct = (tr_high - tr_low) / tr_mean

    # Slope của price và volume
    x           = np.arange(n, dtype=float)
    price_slope = float(np.polyfit(x, close.values, 1)[0])
    vol_slope   = float(np.polyfit(x, volume.values, 1)[0])

    last_close  = float(close.iloc[-1])
    near_low    = last_close < tr_low  * 1.06
    near_high   = last_close > tr_high * 0.94

    # Volume contraction / expansion (nửa sau so với toàn window)
    recent_vol_mean  = float(volume.tail(max(5, n // 4)).mean())
    vol_contract     = recent_vol_mean < vol_avg * 0.75
    vol_expand       = recent_vol_mean > vol_avg * 1.25

    # ── Sub-signals ───────────────────────────────────────────────────────────
    spring_q  = _calc_spring_quality(close, low, volume, tr_low, vol_avg)
    lps       = _detect_lps(close, low, volume, vol_avg)
    evr       = _calc_effort_vs_result(df, vol_avg, atr_avg, window=5)
    no_supply = _count_no_supply_bars(df, vol_avg, atr_avg)
    stop_vol  = _detect_stopping_volume(df, vol_avg)
    vol_prof  = _classify_volume_profile(volume, close, vol_avg)

    # ── Phase classification ───────────────────────────────────────────────────
    # Phase D: uptrend, volume mở rộng, effort/result dương
    if price_slope > 0 and vol_expand and near_high and evr > 0:
        phase = "phase_d"

    # Phase C: Spring quality đủ cao → vừa có Spring
    elif spring_q > 0.15:
        phase = "phase_c"

    # Phase D fallback: uptrend rõ + volume profile đang mở rộng
    elif price_slope > 0 and vol_prof == "expanding" and evr > 0:
        phase = "phase_d"

    # Phase B: range hẹp, volume thu hẹp, không trending
    elif tr_width_pct < 0.18 and vol_contract and abs(price_slope) < tr_mean * 0.001:
        phase = "phase_b"

    # Distribution: gần đỉnh, volume tăng nhưng giá không tăng tiếp
    elif near_high and (vol_expand or vol_slope > 0) and price_slope <= 0:
        phase = "distribution"

    # Phase B rộng hơn: volume co lại + nhiều no-supply bar
    elif tr_width_pct < 0.25 and vol_contract and no_supply >= 2:
        phase = "phase_b"

    else:
        phase = "none"

    phase_dur = _estimate_phase_duration(phase, close, low, high)

    # ── Composite Wyckoff Score ───────────────────────────────────────────────
    # Trọng số: phase 0.35 | spring 0.20 | lps 0.15 | evr 0.15 | supply 0.10 | stop_vol 0.05
    evr_norm    = (evr + 1.0) / 2.0                            # [0, 1]
    supply_norm = min(1.0, no_supply / max(1, _NS_WINDOW * 0.4))
    stop_bonus  = 0.05 if stop_vol else 0.0

    wyckoff_score = (
        _phase_base_score(phase) * 0.35
        + spring_q               * 0.20
        + float(lps)             * 0.15
        + evr_norm               * 0.15
        + supply_norm            * 0.10
        + stop_bonus
    )
    wyckoff_score = float(np.clip(wyckoff_score, 0.0, 1.0))

    return WyckoffResult(
        phase=phase,
        phase_duration=phase_dur,
        spring_quality=spring_q,
        lps_detected=lps,
        effort_vs_result=evr,
        no_supply_count=no_supply,
        stopping_volume=stop_vol,
        volume_profile=vol_prof,
        wyckoff_score=wyckoff_score,
    )
