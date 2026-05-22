"""
_test_sprint6.py
Sprint 6 — M4 Probability Calibration unit tests
35 features · Isotonic calibration · CI · Recommendation enum · Backtest v2
"""

import sys
sys.path.insert(0, "d:/TOOL-PYTHON/smartmoney-analysic")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

_PASS = 0
_FAIL = 0

def check(name: str, cond: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  [OK] {name}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {name}" + (f" -- {detail}" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Feature Engineering — 35 features
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Feature Engineering (35 features) ===")
from ml.feature_engineering import FEATURE_COLS, compute_stock_features

check("FE FEATURE_COLS count = 35", len(FEATURE_COLS) == 35,
      f"got {len(FEATURE_COLS)}")
check("FE wyckoff_phase_score in FEATURE_COLS", "wyckoff_phase_score" in FEATURE_COLS)
check("FE phase_duration_norm in FEATURE_COLS",  "phase_duration_norm" in FEATURE_COLS)
check("FE vol_profile_score in FEATURE_COLS",    "vol_profile_score"   in FEATURE_COLS)

# Integration: compute_stock_features produces all 3 new columns
def _make_ohlcv(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    price = 100_000 + np.cumsum(rng.normal(0, 500, n))
    price = np.clip(price, 90_000, 115_000)
    df = pd.DataFrame({
        "date":   pd.date_range("2024-01-02", periods=n, freq="B"),
        "open":   price * rng.uniform(0.995, 1.005, n),
        "high":   price * rng.uniform(1.001, 1.010, n),
        "low":    price * rng.uniform(0.990, 0.999, n),
        "close":  price,
        "volume": rng.integers(500_000, 3_000_000, n).astype(float),
    })
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df

_ohlcv40 = _make_ohlcv(40)
_feat40  = compute_stock_features(_ohlcv40)

check("FE compute_stock_features not empty",            not _feat40.empty)
check("FE wyckoff_phase_score col exists",             "wyckoff_phase_score" in _feat40.columns)
check("FE phase_duration_norm col exists",             "phase_duration_norm" in _feat40.columns)
check("FE vol_profile_score col exists",               "vol_profile_score"   in _feat40.columns)
check("FE wyckoff_phase_score in [0,1]",
      _feat40["wyckoff_phase_score"].dropna().between(0, 1).all())
check("FE phase_duration_norm in [0,1]",
      _feat40["phase_duration_norm"].dropna().between(0, 1).all())
check("FE vol_profile_score in [0,1]",
      _feat40["vol_profile_score"].dropna().between(0, 1).all())
check("FE no new inf in new cols",
      not _feat40[["wyckoff_phase_score","phase_duration_norm","vol_profile_score"]]
               .isin([np.inf, -np.inf]).any().any())


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: wilson_ci
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== wilson_ci (90% CI) ===")
from ml.model import wilson_ci

_lo05, _hi05 = wilson_ci(0.5, n=50)
check("CI midpoint p=0.5 lo < 0.5",  _lo05 < 0.5,  f"lo={_lo05}")
check("CI midpoint p=0.5 hi > 0.5",  _hi05 > 0.5,  f"hi={_hi05}")
check("CI midpoint lo < hi",          _lo05 < _hi05)
check("CI lo >= 0",                   _lo05 >= 0.0)
check("CI hi <= 1",                   _hi05 <= 1.0)

_lo0, _hi0 = wilson_ci(0.0, n=50)
check("CI p=0.0 lo = 0",  _lo0 == 0.0, f"lo={_lo0}")

_lo1, _hi1 = wilson_ci(1.0, n=50)
check("CI p=1.0 hi = 1",  _hi1 == 1.0, f"hi={_hi1}")

# Wider CI for smaller n
_lo_small, _hi_small = wilson_ci(0.65, n=10)
_lo_large, _hi_large = wilson_ci(0.65, n=200)
_width_small = _hi_small - _lo_small
_width_large = _hi_large - _lo_large
check("CI width decreases with larger n", _width_small > _width_large,
      f"small={_width_small:.3f} large={_width_large:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Recommendation enum
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Recommendation enum ===")
from ml.predictor import Recommendation, _get_recommendation

check("STRONG_BUY value",  Recommendation.STRONG_BUY.value == "STRONG_BUY")
check("BUY value",         Recommendation.BUY.value        == "BUY")
check("WATCH value",       Recommendation.WATCH.value      == "WATCH")
check("HOLD value",        Recommendation.HOLD.value       == "HOLD")
check("AVOID value",       Recommendation.AVOID.value      == "AVOID")

check("rec(0.85) = STRONG_BUY", _get_recommendation(0.85) == Recommendation.STRONG_BUY)
check("rec(0.75) = BUY",        _get_recommendation(0.75) == Recommendation.BUY)
check("rec(0.65) = WATCH",      _get_recommendation(0.65) == Recommendation.WATCH)
check("rec(0.55) = HOLD",       _get_recommendation(0.55) == Recommendation.HOLD)
check("rec(0.40) = AVOID",      _get_recommendation(0.40) == Recommendation.AVOID)
check("rec boundary 0.80 = STRONG_BUY", _get_recommendation(0.80) == Recommendation.STRONG_BUY)
check("rec boundary 0.70 = BUY",        _get_recommendation(0.70) == Recommendation.BUY)
check("rec boundary 0.60 = WATCH",      _get_recommendation(0.60) == Recommendation.WATCH)
check("rec boundary 0.50 = HOLD",       _get_recommendation(0.50) == Recommendation.HOLD)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Model version + calibrator paths
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Model versioning ===")
from ml.model import MODEL_LABEL_VERSION, CALIBRATOR_PATH

check("Model version = v5_calibrated", MODEL_LABEL_VERSION == "v5_calibrated",
      f"got {MODEL_LABEL_VERSION}")
check("CALIBRATOR_PATH is Path obj", hasattr(CALIBRATOR_PATH, "suffix"))
check("CALIBRATOR_PATH ends .pkl",   CALIBRATOR_PATH.suffix == ".pkl")

# load_calibrator returns None when file absent
from ml.model import load_calibrator
_cal_none = load_calibrator()   # likely None (no artifact yet)
check("load_calibrator returns None or calibrator",
      _cal_none is None or hasattr(_cal_none, "predict"))


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: train_model returns calibrator key
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== train_model calibration integration ===")
from ml.model import train_model

def _make_synthetic_dataset(n_tickers: int = 3, n_rows: int = 60) -> pd.DataFrame:
    """Minimal synthetic dataset for train_model smoke test."""
    rng = np.random.default_rng(99)
    rows = []
    for t in range(n_tickers):
        for i in range(n_rows):
            row: dict = {"ticker": f"T{t}", "date": f"2023-{(i//22)+1:02d}-{(i%22)+1:02d}"}
            for col in FEATURE_COLS:
                row[col] = float(rng.uniform(-1, 1))
            row["label"]       = int(rng.uniform() > 0.80)
            row["path_max_5d"] = float(rng.uniform(0, 0.12))
            row["path_min_5d"] = float(rng.uniform(-0.10, 0))
            rows.append(row)
    return pd.DataFrame(rows)

_ds = _make_synthetic_dataset()
_result = train_model(_ds, save=False)

check("train_model returns calibrator key",   "calibrator" in _result)
check("train_model metrics has calibrated",   "calibrated" in _result["metrics"])
check("train_model metrics has cal_n_samples","cal_n_samples" in _result["metrics"])
check("train_model cal_n_samples > 0",        _result["metrics"].get("cal_n_samples", 0) > 0)

_cal_obj = _result.get("calibrator")
# calibrator can be IsotonicRegression (has predict) or None
check("calibrator is None or has predict",
      _cal_obj is None or hasattr(_cal_obj, "predict"))


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: BacktestResult v2 (meets_target, precision_target)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Backtest v2 ===")
from ml.backtest import run_backtest, BacktestResult

def _make_bt_dataset(n: int = 100, win_rate: float = 0.40) -> pd.DataFrame:
    """Synthetic backtest dataset — all 35 features + labels."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        row: dict = {
            "ticker": f"T{i % 5}",
            "date":   f"2024-{(i // 22) + 1:02d}-{(i % 22) + 1:02d}",
        }
        for col in FEATURE_COLS:
            row[col] = float(rng.uniform(-1, 1))
        row["label"]       = int(rng.uniform() < win_rate)
        row["path_max_5d"] = 0.06 if row["label"] else 0.01
        row["path_min_5d"] = -0.01 if row["label"] else -0.06
        rows.append(row)
    return pd.DataFrame(rows)

# Build tiny RF model for backtest
_bt_X  = _make_bt_dataset(200)
_scaler = StandardScaler()
_X_raw  = _bt_X[FEATURE_COLS].values.astype(np.float32)
_X_sc   = _scaler.fit_transform(_X_raw)
_y      = _bt_X["label"].values.astype(int)
_bt_mdl = RandomForestClassifier(n_estimators=10, random_state=1)
_bt_mdl.fit(_X_sc, _y)

_bt_ds  = _make_bt_dataset(100, win_rate=0.45)
_bt_res = run_backtest(
    dataset=_bt_ds, model=_bt_mdl, scaler=_scaler,
    feature_cols=FEATURE_COLS, min_prob=0.30,
    precision_target=0.35,
)

check("Backtest returns BacktestResult",         isinstance(_bt_res, BacktestResult))
check("Backtest has precision_target attr",      hasattr(_bt_res, "precision_target"))
check("Backtest has meets_target attr",          hasattr(_bt_res, "meets_target"))
check("Backtest precision_target == 0.35",       _bt_res.precision_target == 0.35)
check("Backtest meets_target is bool",           isinstance(_bt_res.meets_target, bool))
check("Backtest meets_target consistent",
      _bt_res.meets_target == (_bt_res.precision >= 0.35))
check("Backtest summary has precision_target",   "precision_target" in _bt_res.summary())
check("Backtest summary has meets_target",       "meets_target"     in _bt_res.summary())

# Empty-data backtest (graceful handling)
_bt_empty = run_backtest(
    dataset=pd.DataFrame(), model=_bt_mdl, scaler=_scaler,
    feature_cols=FEATURE_COLS, min_prob=0.65, precision_target=0.35,
)
check("Backtest empty dataset: total_trades=0", _bt_empty.total_trades == 0)

# With calibrator in backtest (IsotonicRegression)
from sklearn.isotonic import IsotonicRegression
_raw_probs_cal = _bt_mdl.predict_proba(_X_sc)[:, 1]
_ir_cal = IsotonicRegression(out_of_bounds="clip")
_ir_cal.fit(_raw_probs_cal[:30], _y[:30])
_bt_cal = run_backtest(
    dataset=_bt_ds, model=_bt_mdl, scaler=_scaler,
    feature_cols=FEATURE_COLS, min_prob=0.30,
    precision_target=0.35, calibrator=_ir_cal,
)
check("Backtest with calibrator: returns result",   isinstance(_bt_cal, BacktestResult))
check("Backtest with calibrator: p_calibrated col",
      "p_calibrated" in _bt_cal.trades_df.columns if not _bt_cal.trades_df.empty else True)


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: __init__ exports
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== ml.__init__ exports ===")
import ml as _ml

check("ml exports Recommendation",    hasattr(_ml, "Recommendation"))
check("ml exports load_calibrator",   hasattr(_ml, "load_calibrator"))
check("ml exports wilson_ci",         hasattr(_ml, "wilson_ci"))
check("ml exports CALIBRATOR_PATH",   hasattr(_ml, "CALIBRATOR_PATH"))
check("ml.Recommendation is enum",    issubclass(_ml.Recommendation, str))


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 40)
print(f"Results: {_PASS} passed, {_FAIL} failed")
if _FAIL == 0:
    print("All tests passed!")
else:
    sys.exit(1)
