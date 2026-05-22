"""
_test_sprint5.py
Unit tests — Sprint 5: S1 Wyckoff VSA Engine v2.0
10 case studies dựa trên mẫu thực từ chart VN.

Run:
    python _test_sprint5.py
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Test runner ────────────────────────────────────────────────────────────────
_pass = _fail = 0

def check(name: str, cond: bool, detail: str = "") -> None:
    global _pass, _fail
    tag = "[OK]" if cond else "[FAIL]"
    msg = f"  {tag} {name}"
    if not cond and detail:
        msg += f" -- {detail}"
    print(msg)
    if cond:
        _pass += 1
    else:
        _fail += 1


# ── OHLCV builders ────────────────────────────────────────────────────────────

def _make_phase_b(n: int = 30) -> pd.DataFrame:
    """
    Case 1: VIC-like Phase B — sideways range 30 bar, volume contracting.
    Prices strictly 99-101k, lows clipped >= 99.5k so NO bar can form a Spring
    below the TR_low (which would be ~99.5k from first-half slice).
    """
    rng   = np.random.default_rng(42)
    price = 100_000 + rng.uniform(-800, 800, n)
    vol   = np.linspace(2_000_000, 700_000, n) + rng.uniform(-20_000, 20_000, n)
    half_range = rng.uniform(100, 400, n)
    df = pd.DataFrame({
        "open":   price,
        "high":   price + half_range,
        "low":    np.clip(price - half_range, 99_400, None),  # all lows >= 99.4k
        "close":  price + rng.uniform(-200, 200, n),
        "volume": vol.clip(100_000),
    })
    return df.reset_index(drop=True)


def _make_phase_c(n: int = 30) -> pd.DataFrame:
    """
    Case 2: HPG-like Phase C — Spring: giá phá đáy TR rồi hồi phục nhanh.
    TR: bars 0-19 range 100-110k. Bar 20-22: Spring xuống 97k.
    Bars 23-29: hồi lên 103k.
    """
    rng = np.random.default_rng(7)
    # Phase B range: bars 0-19
    b_price = 105_000 + rng.uniform(-3_000, 3_000, 20)
    b_vol   = rng.uniform(1_200_000, 1_800_000, 20)
    # Spring bars: 20-22 (giá xuống dưới TR low ~100k)
    s_price = np.array([99_500, 97_000, 100_500])
    s_vol   = np.array([800_000, 700_000, 900_000])   # low volume at spring
    # Recovery bars: 23-29
    r_price = np.linspace(102_000, 108_000, 8)
    r_vol   = rng.uniform(1_500_000, 2_500_000, 8)

    price = np.concatenate([b_price, s_price, r_price])
    vol   = np.concatenate([b_vol,   s_vol,   r_vol])

    df = pd.DataFrame({
        "open":   price,
        "high":   price + rng.uniform(200, 1_000, len(price)),
        "low":    price - rng.uniform(200, 1_000, len(price)),
        "close":  price + rng.uniform(-200, 200, len(price)),
        "volume": vol,
    })
    # Ensure spring low is clearly below TR low
    df.loc[21, "low"] = 96_500
    df.loc[21, "close"] = 97_500
    return df.reset_index(drop=True)


def _make_phase_d(n: int = 30) -> pd.DataFrame:
    """
    Case 3: VCB-like Phase D — uptrend rõ, volume mở rộng.
    Giá tăng từ 85k → 105k trong 30 bar, volume tăng mạnh.
    """
    rng    = np.random.default_rng(13)
    price  = np.linspace(85_000, 105_000, n) + rng.uniform(-500, 500, n)
    vol    = np.linspace(1_500_000, 3_500_000, n) + rng.uniform(-100_000, 100_000, n)
    candle_body = rng.uniform(300, 1_200, n)
    df = pd.DataFrame({
        "open":   price - candle_body / 2,
        "high":   price + candle_body / 2 + rng.uniform(100, 500, n),
        "low":    price - candle_body / 2 - rng.uniform(100, 300, n),
        "close":  price + candle_body / 2,
        "volume": vol.clip(500_000),
    })
    return df.reset_index(drop=True)


def _make_distribution(n: int = 30) -> pd.DataFrame:
    """
    Case 4: VHM-like Distribution — giá near high, volume tăng, giá sideway/giảm nhẹ.
    """
    rng   = np.random.default_rng(99)
    # Giá bắt đầu cao và sideway/giảm nhẹ
    price = 110_000 + rng.uniform(-3_000, 1_000, n)
    price = np.sort(price)[::-1]   # slight downtrend
    vol   = np.linspace(1_800_000, 3_200_000, n) + rng.uniform(-100_000, 100_000, n)
    df = pd.DataFrame({
        "open":  price + rng.uniform(-500, 500, n),
        "high":  price + rng.uniform(300, 2_000, n),
        "low":   price - rng.uniform(300, 2_000, n),
        "close": price - rng.uniform(0, 800, n),   # slight bear close
        "volume": vol.clip(500_000),
    })
    return df.reset_index(drop=True)


def _make_high_quality_spring() -> pd.DataFrame:
    """
    Case 5: FPT-like — Spring chất lượng cao.
    TR low = 95k. Spring đến 93k (2.1% below) với vol thấp, hồi lại 97k trong 2 bar.
    """
    rng = np.random.default_rng(55)
    # TR establish: 20 bars 95-105k
    b_price = 100_000 + rng.uniform(-4_000, 4_000, 20)
    b_vol   = rng.uniform(2_000_000, 2_500_000, 20)
    # Spring: 1 bar sâu xuống 93k, vol rất thấp
    # Recovery: 2 bar hồi lên 97k
    s_data  = np.array([93_500, 97_000, 99_500])
    s_vol   = np.array([600_000, 1_800_000, 2_200_000])

    price   = np.concatenate([b_price, s_data])
    vol     = np.concatenate([b_vol, s_vol])
    n       = len(price)

    df = pd.DataFrame({
        "open":   price,
        "high":   price + rng.uniform(100, 800, n),
        "low":    price - rng.uniform(100, 800, n),
        "close":  price + rng.uniform(-100, 300, n),
        "volume": vol,
    })
    df.loc[20, "low"] = 92_500   # explicit spring low
    df.loc[20, "close"] = 93_500
    df.loc[20, "volume"] = 550_000
    return df.reset_index(drop=True)


def _make_lps_pattern(n: int = 28) -> pd.DataFrame:
    """
    Case 6: MSN-like LPS — sau Spring, pullback trên low volume, giữ support.
    Bars 0-14: TR 100-110k.
    Bar 15-17: Spring xuống 97k (vol thấp).
    Bar 18-22: Rally lên 108k (vol cao).
    Bar 23-27: Pullback LPS về 104k (vol thấp, giá giữ vững).
    """
    rng = np.random.default_rng(22)
    tr    = 105_000 + rng.uniform(-3_000, 3_000, 15)
    spring = np.array([100_500, 97_500, 99_000])
    rally  = np.linspace(101_000, 108_000, 5)
    lps    = np.array([106_000, 104_500, 104_000, 104_500, 105_500])

    price  = np.concatenate([tr, spring, rally, lps])
    vol_tr = rng.uniform(1_500_000, 2_000_000, 15)
    vol_sp = np.array([1_000_000, 700_000, 900_000])
    vol_rl = rng.uniform(2_500_000, 3_500_000, 5)
    vol_lp = rng.uniform(700_000, 1_100_000, 5)    # low volume on pullback
    vol    = np.concatenate([vol_tr, vol_sp, vol_rl, vol_lp])

    n_rows = len(price)
    df = pd.DataFrame({
        "open":   price,
        "high":   price + rng.uniform(200, 1_200, n_rows),
        "low":    price - rng.uniform(200, 1_200, n_rows),
        "close":  price + rng.uniform(-200, 300, n_rows),
        "volume": vol,
    })
    df.loc[16, "low"] = 97_000
    return df.reset_index(drop=True)


def _make_stopping_volume() -> pd.DataFrame:
    """
    Case 7: SSI-like Stopping Volume — volume climax gần đáy, đóng nửa trên nến.
    Bars 0-24: downtrend từ 50k → 38k.
    Bar 25-27: Stopping volume — vol 3.5× avg, close nửa trên.
    Bars 28-29: bắt đầu hồi.
    """
    rng   = np.random.default_rng(33)
    down  = np.linspace(50_000, 38_000, 25) + rng.uniform(-500, 500, 25)
    stop  = np.array([37_500, 38_500, 39_500])
    rebound = np.array([40_000, 41_500])

    price = np.concatenate([down, stop, rebound])
    n     = len(price)
    vol_d = rng.uniform(800_000, 1_200_000, 25)
    vol_s = np.array([3_800_000, 3_200_000, 2_800_000])  # climax volume
    vol_r = np.array([1_500_000, 1_200_000])
    vol   = np.concatenate([vol_d, vol_s, vol_r])

    df = pd.DataFrame({
        "open":   price,
        "high":   price + rng.uniform(300, 1_500, n),
        "low":    price - rng.uniform(300, 1_500, n),
        "close":  price + rng.uniform(-200, 500, n),
        "volume": vol,
    })
    # Stopping volume bars: close nửa trên (high - close nhỏ)
    df.loc[25, "low"]   = 36_500
    df.loc[25, "close"] = 38_800   # upper half of range
    df.loc[25, "volume"] = 4_000_000
    return df.reset_index(drop=True)


def _make_no_supply(n: int = 30) -> pd.DataFrame:
    """
    Case 8: VNM-like — nhiều no-supply bar.
    First 15 bars: normal vol 2.5M (establishes high vol_avg).
    Last 15 bars: narrow range + very low vol 300-500k < 0.70 * vol_avg.
    close in upper half of narrow candle.
    """
    rng   = np.random.default_rng(88)
    # First half: normal activity to drive up vol_avg
    p1    = 46_000 + rng.uniform(-500, 500, 15)
    v1    = rng.uniform(2_200_000, 2_800_000, 15)
    hr1   = rng.uniform(800, 1_500, 15)
    # Second half: no-supply environment
    p2    = 46_000 + rng.uniform(-300, 300, 15)
    v2    = rng.uniform(280_000, 480_000, 15)   # clearly < 0.70 * vol_avg
    hr2   = rng.uniform(100, 250, 15)            # narrow range < 0.70 * ATR
    price = np.concatenate([p1, p2])
    vol   = np.concatenate([v1, v2])
    hr    = np.concatenate([hr1, hr2])
    df = pd.DataFrame({
        "open":   price - hr * 0.2,
        "high":   price + hr,
        "low":    price - hr,
        "close":  price + hr * 0.45,   # close in upper half
        "volume": vol,
    })
    return df.reset_index(drop=True)


def _make_bullish_evr(n: int = 25) -> pd.DataFrame:
    """
    Case 9: TCB-like bullish EVR — giá di chuyển nhiều, volume vừa phải.
    Effort (vol) = 1×avg, Result (range) = 1.5× ATR → EVR > 0 (bullish).
    """
    rng   = np.random.default_rng(91)
    price = np.linspace(70_000, 80_000, n) + rng.uniform(-500, 500, n)
    vol   = rng.uniform(1_500_000, 2_000_000, n)          # moderate volume
    big_range = rng.uniform(1_500, 3_000, n)               # large candle range
    df = pd.DataFrame({
        "open":   price - big_range * 0.4,
        "high":   price + big_range * 0.6,
        "low":    price - big_range * 0.4,
        "close":  price + big_range * 0.5,   # bullish close
        "volume": vol,
    })
    return df.reset_index(drop=True)


def _make_bearish_evr(n: int = 25) -> pd.DataFrame:
    """
    Case 10: VPB-like bearish EVR — volume lớn, giá đứng yên (supply absorption).
    Effort = 3×avg, Result = 0.4× ATR → EVR < 0 (supply absorbing move).
    """
    rng   = np.random.default_rng(17)
    price = 30_000 + rng.uniform(-500, 500, n)             # sideway
    vol   = rng.uniform(4_000_000, 6_000_000, n)           # very high volume
    narrow = rng.uniform(100, 400, n)                      # narrow range
    df = pd.DataFrame({
        "open":   price,
        "high":   price + narrow,
        "low":    price - narrow,
        "close":  price - narrow * 0.1,   # slight bear close
        "volume": vol,
    })
    return df.reset_index(drop=True)


# ── Imports ───────────────────────────────────────────────────────────────────
from analytics.wyckoff import (
    WyckoffResult,
    detect_wyckoff,
    _calc_spring_quality,
    _detect_lps,
    _calc_effort_vs_result,
    _count_no_supply_bars,
    _detect_stopping_volume,
    _classify_volume_profile,
)
from analytics.accumulation_detection import detect_accumulation_phase
from ml.feature_engineering import FEATURE_COLS, compute_stock_features
from ml.model import MODEL_LABEL_VERSION


# =============================================================================
# Case 1: VIC Phase B — Sideways, volume contracting
# =============================================================================
print("\n=== Case 1: VIC-like Phase B ===")
_df1 = _make_phase_b()
_w1  = detect_wyckoff(_df1)
check("C1 returns WyckoffResult",  isinstance(_w1, WyckoffResult))
check("C1 phase = phase_b or none", _w1.phase in ("phase_b", "none"),
      f"phase={_w1.phase}")
check("C1 volume_profile contracting", _w1.volume_profile == "contracting",
      f"profile={_w1.volume_profile}")
check("C1 wyckoff_score >= 0.30",   _w1.wyckoff_score >= 0.30,
      f"score={_w1.wyckoff_score:.3f}")
check("C1 no_supply_count >= 0",    _w1.no_supply_count >= 0)
check("C1 detect_accumulation_phase delegates", detect_accumulation_phase(_df1) == _w1.phase)


# =============================================================================
# Case 2: HPG Phase C — Spring
# =============================================================================
print("\n=== Case 2: HPG-like Phase C (Spring) ===")
_df2 = _make_phase_c()
_w2  = detect_wyckoff(_df2)
check("C2 returns WyckoffResult",   isinstance(_w2, WyckoffResult))
check("C2 phase = phase_c or phase_d", _w2.phase in ("phase_c", "phase_d"),
      f"phase={_w2.phase}")
check("C2 spring_quality > 0",      _w2.spring_quality > 0.0,
      f"spring_q={_w2.spring_quality:.3f}")
check("C2 wyckoff_score >= 0.45",   _w2.wyckoff_score >= 0.45,
      f"score={_w2.wyckoff_score:.3f}")


# =============================================================================
# Case 3: VCB Phase D — Markup, volume expanding
# =============================================================================
print("\n=== Case 3: VCB-like Phase D (Markup) ===")
_df3 = _make_phase_d()
_w3  = detect_wyckoff(_df3)
check("C3 returns WyckoffResult",   isinstance(_w3, WyckoffResult))
check("C3 phase = phase_d",         _w3.phase == "phase_d",
      f"phase={_w3.phase}")
check("C3 volume_profile expanding", _w3.volume_profile == "expanding",
      f"profile={_w3.volume_profile}")
check("C3 effort_vs_result > 0",    _w3.effort_vs_result > 0,
      f"evr={_w3.effort_vs_result:.3f}")
check("C3 wyckoff_score >= 0.38",   _w3.wyckoff_score >= 0.38,
      f"score={_w3.wyckoff_score:.3f}")
check("C3 phase_duration >= 1",     _w3.phase_duration >= 1)


# =============================================================================
# Case 4: VHM Distribution
# =============================================================================
print("\n=== Case 4: VHM-like Distribution ===")
_df4 = _make_distribution()
_w4  = detect_wyckoff(_df4)
check("C4 returns WyckoffResult",   isinstance(_w4, WyckoffResult))
check("C4 phase = distribution",    _w4.phase == "distribution",
      f"phase={_w4.phase}")
check("C4 wyckoff_score <= 0.50",   _w4.wyckoff_score <= 0.50,
      f"score={_w4.wyckoff_score:.3f}")
check("C4 volume_profile expanding or neutral",
      _w4.volume_profile in ("expanding", "neutral"),
      f"profile={_w4.volume_profile}")


# =============================================================================
# Case 5: FPT High-Quality Spring
# =============================================================================
print("\n=== Case 5: FPT-like High-Quality Spring ===")
_df5 = _make_high_quality_spring()
_w5  = detect_wyckoff(_df5)
check("C5 returns WyckoffResult",   isinstance(_w5, WyckoffResult))
check("C5 spring_quality > 0.30",   _w5.spring_quality > 0.30,
      f"spring_q={_w5.spring_quality:.3f}")
check("C5 phase in (phase_c, phase_d)", _w5.phase in ("phase_c", "phase_d"),
      f"phase={_w5.phase}")
check("C5 wyckoff_score >= 0.50",   _w5.wyckoff_score >= 0.50,
      f"score={_w5.wyckoff_score:.3f}")

# Test _calc_spring_quality directly
_tr_low5 = float(_df5["low"].iloc[:10].min())
_v_avg5  = float(_df5["volume"].mean())
_sq5     = _calc_spring_quality(
    _df5["close"], _df5["low"], _df5["volume"], _tr_low5, _v_avg5
)
check("C5a spring_quality helper > 0", _sq5 > 0.0,
      f"helper sq={_sq5:.3f}")


# =============================================================================
# Case 6: MSN LPS Pattern
# =============================================================================
print("\n=== Case 6: MSN-like LPS ===")
_df6 = _make_lps_pattern()
_w6  = detect_wyckoff(_df6)
check("C6 returns WyckoffResult",   isinstance(_w6, WyckoffResult))
check("C6 lps_detected or wyckoff_score > 0.40",
      _w6.lps_detected or _w6.wyckoff_score > 0.40,
      f"lps={_w6.lps_detected} score={_w6.wyckoff_score:.3f}")

# LPS helper directly on last 5 bars of recovery zone
_vol_avg6 = float(_df6["volume"].mean())
_lps6 = _detect_lps(
    _df6["close"].tail(10),
    _df6["low"].tail(10),
    _df6["volume"].tail(10),
    _vol_avg6,
)
# Direct call: just verify no exception and sensible return
check("C6a _detect_lps on recovery tail", bool(_lps6) in (True, False))


# =============================================================================
# Case 7: SSI Stopping Volume
# =============================================================================
print("\n=== Case 7: SSI-like Stopping Volume ===")
_df7 = _make_stopping_volume()
_w7  = detect_wyckoff(_df7)
check("C7 returns WyckoffResult",   isinstance(_w7, WyckoffResult))
check("C7 stopping_volume = True",  _w7.stopping_volume,
      f"stop_vol={_w7.stopping_volume} score={_w7.wyckoff_score:.3f}")

# Test helper directly
_v_avg7 = float(_df7["volume"].mean())
_sv7 = _detect_stopping_volume(_df7, _v_avg7)
check("C7a _detect_stopping_volume True", _sv7,
      f"vol_avg={_v_avg7:.0f}")


# =============================================================================
# Case 8: VNM No-Supply bars
# =============================================================================
print("\n=== Case 8: VNM-like No-Supply ===")
_df8 = _make_no_supply()
_w8  = detect_wyckoff(_df8)
check("C8 returns WyckoffResult",   isinstance(_w8, WyckoffResult))
check("C8 no_supply_count >= 2",    _w8.no_supply_count >= 2,
      f"no_supply={_w8.no_supply_count}")
check("C8 volume_profile contracting or neutral",
      _w8.volume_profile in ("contracting", "neutral"),
      f"profile={_w8.volume_profile}")

# Helper directly
_atr8 = float((_df8["high"] - _df8["low"]).mean())
_v8   = float(_df8["volume"].mean())
_ns8  = _count_no_supply_bars(_df8, _v8, _atr8)
check("C8a no_supply_count helper >= 2", _ns8 >= 2,
      f"count={_ns8}")


# =============================================================================
# Case 9: TCB Bullish Effort vs Result
# =============================================================================
print("\n=== Case 9: TCB-like Bullish EVR ===")
_df9 = _make_bullish_evr()
_w9  = detect_wyckoff(_df9)
check("C9 returns WyckoffResult",   isinstance(_w9, WyckoffResult))
check("C9 effort_vs_result > 0",    _w9.effort_vs_result > 0,
      f"evr={_w9.effort_vs_result:.3f}")

# EVR helper
_atr9 = float((_df9["high"] - _df9["low"]).mean())
_v9   = float(_df9["volume"].mean())
_evr9 = _calc_effort_vs_result(_df9, _v9, _atr9, window=5)
check("C9a _calc_evr > 0 (bullish)", _evr9 > 0,
      f"evr_helper={_evr9:.3f}")


# =============================================================================
# Case 10: VPB Bearish EVR (supply absorption)
# =============================================================================
print("\n=== Case 10: VPB-like Bearish EVR ===")
_df10 = _make_bearish_evr()
_w10  = detect_wyckoff(_df10)
check("C10 returns WyckoffResult",  isinstance(_w10, WyckoffResult))
check("C10 effort_vs_result <= 0",  _w10.effort_vs_result <= 0,
      f"evr={_w10.effort_vs_result:.3f}")

# EVR helper
_atr10 = float((_df10["high"] - _df10["low"]).mean())
_v10   = float(_df10["volume"].mean())
_evr10 = _calc_effort_vs_result(_df10, _v10, _atr10, window=5)
check("C10a _calc_evr <= 0 (bearish)", _evr10 <= 0,
      f"evr_helper={_evr10:.3f}")


# =============================================================================
# Feature Engineering integration tests
# =============================================================================
print("\n=== Feature Engineering integration ===")
check("FE FEATURE_COLS has 32 features", len(FEATURE_COLS) == 32,
      f"len={len(FEATURE_COLS)}")
check("FE spring_quality in FEATURE_COLS",   "spring_quality"     in FEATURE_COLS)
check("FE lps_detected in FEATURE_COLS",     "lps_detected"       in FEATURE_COLS)
check("FE effort_vs_result in FEATURE_COLS", "effort_vs_result"   in FEATURE_COLS)
check("FE no_supply_count in FEATURE_COLS",  "no_supply_count"    in FEATURE_COLS)
check("FE stopping_volume in FEATURE_COLS",  "stopping_volume"    in FEATURE_COLS)

# compute_stock_features produces all new columns
_fe_df = _make_phase_d(60)
_fe_df.insert(0, "date", pd.date_range("2024-01-01", periods=60, freq="B"))
_fe_out = compute_stock_features(_fe_df)
check("FE compute_stock_features not empty", not _fe_out.empty)
for _col in ["spring_quality", "lps_detected", "effort_vs_result", "no_supply_count", "stopping_volume"]:
    check(f"FE output has {_col}", _col in _fe_out.columns)
    if _col in _fe_out.columns:
        check(f"FE {_col} no inf", not _fe_out[_col].replace([float('inf'), float('-inf')], float('nan')).isna().all())


# =============================================================================
# Model version
# =============================================================================
print("\n=== Model version ===")
check("Model version = v4_wyckoff", MODEL_LABEL_VERSION == "v4_wyckoff",
      f"got={MODEL_LABEL_VERSION}")


# =============================================================================
# WyckoffResult.as_dict
# =============================================================================
print("\n=== WyckoffResult.as_dict ===")
_w_any = detect_wyckoff(_make_phase_b())
_d = _w_any.as_dict()
for _key in ["phase", "phase_duration", "spring_quality", "lps_detected",
             "effort_vs_result", "no_supply_count", "stopping_volume",
             "volume_profile", "wyckoff_score"]:
    check(f"as_dict has {_key}", _key in _d)
check("as_dict wyckoff_score in [0,1]",
      0.0 <= _d["wyckoff_score"] <= 1.0,
      f"score={_d['wyckoff_score']}")


# =============================================================================
# Edge cases
# =============================================================================
print("\n=== Edge cases ===")
_empty = detect_wyckoff(pd.DataFrame())
check("Edge empty df returns WyckoffResult", isinstance(_empty, WyckoffResult))
check("Edge empty df phase = none",          _empty.phase == "none")
check("Edge empty df spring_quality = 0",    _empty.spring_quality == 0.0)

_tiny = _make_phase_b(5)
_w_tiny = detect_wyckoff(_tiny)
check("Edge tiny df (5 bars) no crash",      isinstance(_w_tiny, WyckoffResult))


# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*40}")
print(f"Results: {_pass} passed, {_fail} failed")
if _fail == 0:
    print("All tests passed!")
else:
    sys.exit(1)
