"""
_test_sprint3.py
Sprint 3 unit tests:
  - S4 Volume Confirmation Engine
  - D3.4 Portfolio Sizing (Kelly)
  - Backtest module
"""

from __future__ import annotations

import sys
import math
import types
import numpy as np
import pandas as pd

sys.path.insert(0, "d:\\TOOL-PYTHON\\smartmoney-analysic")

# ─── helpers ───────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 30, trend: float = 0.003) -> pd.DataFrame:
    """Deterministic OHLCV — monotone uptrend, no noise."""
    closes = [10_000 * (1 + trend) ** i for i in range(n)]
    vols   = [1_000_000 + i * 10_000 for i in range(n)]
    df = pd.DataFrame({
        "open":   [c * 0.995 for c in closes],
        "high":   [c * 1.010 for c in closes],
        "low":    [c * 0.990 for c in closes],
        "close":  closes,
        "volume": vols,
    }, index=pd.date_range("2024-01-01", periods=n, freq="B"))
    return df


def _make_noisy_ohlcv(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    closes = 10_000 + np.cumsum(rng.normal(0, 200, n))
    vols   = np.maximum(100_000, rng.normal(1_000_000, 200_000, n))
    df = pd.DataFrame({
        "open":   closes * 0.995,
        "high":   closes * 1.01,
        "low":    closes * 0.99,
        "close":  closes,
        "volume": vols,
    }, index=pd.date_range("2024-01-01", periods=n, freq="B"))
    return df


PASS = 0
FAIL = 0

def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [OK] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" -- {detail}" if detail else ""))


# ─── S4 Volume Confirmation ─────────────────────────────────────────────────

print("\n=== S4 Volume Confirmation ===")

from ml.volume_confirmation import (
    VolumeConfirmation,
    compute_volume_confirmation,
    VOL_CONFIRM_THRESHOLD,
)

# T1: insufficient data returns None
df_short = _make_ohlcv(n=15)
result = compute_volume_confirmation(df_short, ticker="AAA", window=20)
check("T1 insufficient data -> None", result is None,
      f"got {type(result)}")

# T2: sufficient data returns VolumeConfirmation
df_good = _make_ohlcv(n=30)
vc = compute_volume_confirmation(df_good, ticker="BBB", window=20)
check("T2 sufficient data -> VolumeConfirmation", isinstance(vc, VolumeConfirmation),
      f"got {type(vc)}")

# T3: fields in valid range
if vc:
    check("T3a vol_surge > 0",        vc.vol_surge > 0,   f"vol_surge={vc.vol_surge}")
    check("T3b vol_quality in [0,1]",  0 <= vc.vol_quality <= 1, f"vol_quality={vc.vol_quality}")
    check("T3c obv_score in [-1,1]",  -1 <= vc.obv_score  <= 1, f"obv_score={vc.obv_score}")
    check("T3d volume_score in [-1,1]",-1 <= vc.volume_score <= 1, f"volume_score={vc.volume_score}")
    check("T3e is_confirmed is bool",  isinstance(vc.is_confirmed, bool))
    check("T3f label is str",          isinstance(vc.label, str))
    check("T3g ticker matches",        vc.ticker == "BBB")

# T4: uptrend volume -> confirmed
df_up = _make_ohlcv(n=30, trend=0.01)
# Manually inflate last 5 volumes
df_up = df_up.copy()
df_up.loc[df_up.index[-5:], "volume"] *= 3.0
vc_up = compute_volume_confirmation(df_up, ticker="UP", window=20)
if vc_up:
    check("T4 strong uptrend -> is_confirmed or vol_surge > 1",
          vc_up.is_confirmed or vc_up.vol_surge > 1.0,
          f"is_confirmed={vc_up.is_confirmed} vol_surge={vc_up.vol_surge:.2f}")

# T5: as_dict serializable
if vc:
    d = vc.as_dict()
    check("T5a as_dict returns dict", isinstance(d, dict))
    check("T5b as_dict has label",    "label" in d)
    check("T5c as_dict has is_confirmed", "is_confirmed" in d)


# ─── D3.4 Portfolio Sizing ───────────────────────────────────────────────────

print("\n=== D3.4 Portfolio Sizing (Kelly) ===")

from ml.portfolio_sizing import (
    PositionSize,
    compute_position_size,
    MIN_POSITION_PCT,
    MAX_POSITION_PCT,
    HALF_KELLY_FACTOR,
)

# T6: positive EV -> recommended_pct > 0
sz = compute_position_size(probability=0.70, rr_ratio=2.0, regime_max_positions=5)
check("T6 positive EV -> PositionSize", isinstance(sz, PositionSize))
if isinstance(sz, PositionSize):
    check("T6a kelly_pct > 0",    sz.kelly_pct > 0,  f"kelly_pct={sz.kelly_pct}")
    check("T6b half_kelly = kelly/2", abs(sz.half_kelly_pct - sz.kelly_pct * HALF_KELLY_FACTOR) < 1e-9)
    check("T6c rec in [MIN, MAX]",
          MIN_POSITION_PCT <= sz.recommended_pct <= MAX_POSITION_PCT,
          f"rec={sz.recommended_pct}")
    check("T6d rec <= 1/max_pos",  sz.recommended_pct <= 1/5 + 1e-9,
          f"rec={sz.recommended_pct:.4f}")

# T7: negative EV -> recommended_pct = 0
sz_neg = compute_position_size(probability=0.30, rr_ratio=0.5, regime_max_positions=5)
check("T7 negative EV -> rec=0", sz_neg.recommended_pct == 0.0,
      f"rec={sz_neg.recommended_pct}")

# T8: regime cap respected
sz_cap = compute_position_size(probability=0.90, rr_ratio=5.0, regime_max_positions=10)
check("T8 regime cap 1/10=10%", sz_cap.recommended_pct <= 1/10 + 1e-9,
      f"rec={sz_cap.recommended_pct:.4f}")

# T9: as_dict serializable
d_sz = sz.as_dict()
check("T9a as_dict returns dict",      isinstance(d_sz, dict))
check("T9b as_dict has kelly_pct",     "kelly_pct" in d_sz)
check("T9c as_dict has recommended_pct", "recommended_pct" in d_sz)
check("T9d as_dict has reasoning",     "reasoning" in d_sz)

# T10: capital_amount computed when price given
sz_cap2 = compute_position_size(0.70, 2.0, 5, portfolio_capital=1_000_000_000, stock_price=50_000)
check("T10 capital_amount set", sz_cap2.capital_amount > 0,
      f"capital_amount={sz_cap2.capital_amount}")
check("T10b shares set", sz_cap2.shares > 0, f"shares={sz_cap2.shares}")


# ─── Backtest module ─────────────────────────────────────────────────────────

print("\n=== Backtest module ===")

from ml.backtest import BacktestResult, run_backtest, _empty_result, TARGET_PCT, SL_PCT

# T11: _empty_result returns BacktestResult
empty = _empty_result("test note")
check("T11 _empty_result -> BacktestResult", isinstance(empty, BacktestResult))
check("T11a empty total_signals=0", empty.total_signals == 0)
check("T11b empty note set",        "test note" in (empty.note or ""))

# Build synthetic dataset + mock model for backtest
def _make_dataset(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    feature_cols = [f"f{i}" for i in range(5)]
    data = {col: rng.standard_normal(n) for col in feature_cols}
    data["ticker"]          = [f"T{i%5:02d}" for i in range(n)]
    data["date"]            = pd.date_range("2023-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
    data["forward_return_2d"] = rng.normal(0.02, 0.05, n)
    data["path_max_5d"]     = rng.uniform(0.0, 0.10, n)
    data["path_min_5d"]     = rng.uniform(-0.10, 0.0, n)
    data["sl_hit"]          = (data["path_min_5d"] < -0.05).astype(int)
    # label: 1 if path_max>=5% and path_min>-5%
    data["label"]           = (
        (np.array(data["path_max_5d"]) >= 0.05) &
        (np.array(data["path_min_5d"]) > -0.05)
    ).astype(int)
    return pd.DataFrame(data)


class _MockModel:
    """Returns fixed high probability for positive label."""
    def predict_proba(self, X):
        n = len(X)
        probs = np.full((n, 2), [0.25, 0.75])
        return probs


class _MockScaler:
    def transform(self, X):
        return X


ds = _make_dataset(120)
feature_cols = [f"f{i}" for i in range(5)]
mock_model  = _MockModel()
mock_scaler = _MockScaler()

# T12: run_backtest returns BacktestResult
bt = run_backtest(ds, mock_model, mock_scaler, feature_cols, min_prob=0.60)
check("T12 run_backtest -> BacktestResult", isinstance(bt, BacktestResult))

# T13: fields populated
check("T13a total_signals >= 0",  bt.total_signals >= 0)
check("T13b total_trades >= 0",   bt.total_trades >= 0)
check("T13c win_rate in [0,1]",   0 <= bt.win_rate <= 1,   f"win_rate={bt.win_rate}")
check("T13d precision in [0,1]",  0 <= bt.precision <= 1,  f"precision={bt.precision}")

# T14: equity_curve is Series or None
if bt.equity_curve is not None:
    check("T14 equity_curve is Series", isinstance(bt.equity_curve, pd.Series))
    check("T14b equity_curve length > 0", len(bt.equity_curve) > 0)

# T15: empty dataset -> _empty_result fallback
ds_empty = ds.iloc[:0].copy()
bt_empty = run_backtest(ds_empty, mock_model, mock_scaler, feature_cols)
check("T15 empty dataset -> fallback", bt_empty.total_signals == 0)

# T16: by_ticker dataframe
if bt.by_ticker is not None:
    check("T16 by_ticker is DataFrame", isinstance(bt.by_ticker, pd.DataFrame))


# ─── Summary ────────────────────────────────────────────────────────────────

print(f"\n{'='*40}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL:
    sys.exit(1)
else:
    print("All tests passed!")
