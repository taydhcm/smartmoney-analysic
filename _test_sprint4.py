"""
_test_sprint4.py
Sprint 4 unit tests:
  - D0.2 data/db.py SQLite schema
  - D0.2 data/snapshot_logger.py
  - S4 ml/smart_money.py engine
  - Feature engineering 3 new smart money columns
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, "d:\\TOOL-PYTHON\\smartmoney-analysic")

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


# ─── data/db.py ──────────────────────────────────────────────────────────────

print("\n=== D0.2 data/db.py SQLite ===")

import importlib
import os

# Dung fixed path test DB de tranh TemporaryDirectory lock issue tren Windows
import data.db as db_mod

_TEST_DB = Path("d:\\TOOL-PYTHON\\smartmoney-analysic\\_test_snap.db")
_orig_path = db_mod.DB_PATH
_orig_dir  = db_mod._DB_DIR
db_mod.DB_PATH = _TEST_DB
db_mod._DB_DIR = _TEST_DB.parent

try:
    # Xoa DB cu neu ton tai
    for _p in [_TEST_DB, _TEST_DB.with_suffix(".db-wal"), _TEST_DB.with_suffix(".db-shm")]:
        try:
            _p.unlink(missing_ok=True)
        except Exception:
            pass

    # T1: ensure_db tao file + schema
    db_mod.ensure_db()
    check("T1 ensure_db creates file", _TEST_DB.exists())

    # T2: upsert_snapshot
    db_mod.upsert_snapshot(
        session_date="2026-05-22",
        ticker="VIC",
        foreign_buy=1_000_000,
        foreign_sell=500_000,
        foreign_net=500_000,
        total_volume=10_000_000,
        close=50_000,
    )
    sessions = db_mod.get_snapshot_sessions("VIC")
    check("T2 upsert_snapshot stores row", "2026-05-22" in sessions)

    # T3: load_snapshots TRUOC khi ghi de
    df_snap = db_mod.load_snapshots("VIC", last_n=10)
    check("T4 load_snapshots returns df",    isinstance(df_snap, pd.DataFrame))
    check("T4b load_snapshots has 1 row",    len(df_snap) == 1)
    check("T4c foreign_net correct",
          abs(float(df_snap.iloc[0]["foreign_net"]) - 500_000) < 1,
          f"got {float(df_snap.iloc[0]['foreign_net'])}")

    # T3-idempotent: ghi de voi 0 (idempotent)
    db_mod.upsert_snapshot("2026-05-22", "VIC", 0, 0, 0, 0, 0)
    check("T3 upsert idempotent no error", True)

    # T5: get_session_count
    cnt = db_mod.get_session_count("VIC")
    check("T5 get_session_count = 1", cnt == 1)

    # T6: upsert_breadth + get_last_session_date
    db_mod.upsert_breadth("2026-05-22", advance=20, decline=8, unchanged=2, total=30)
    last = db_mod.get_last_session_date()
    check("T6 get_last_session_date", last == "2026-05-22")

    # T7: set/get meta
    db_mod.set_meta("test_key", "hello_sprint4")
    val = db_mod.get_meta("test_key")
    check("T7 set/get meta", val == "hello_sprint4")
    check("T7b get_meta missing default",
          db_mod.get_meta("no_such_key", "default") == "default")

finally:
    db_mod.DB_PATH = _orig_path
    db_mod._DB_DIR = _orig_dir
    # Cleanup test DB
    for _p in [_TEST_DB, _TEST_DB.with_suffix(".db-wal"), _TEST_DB.with_suffix(".db-shm")]:
        try:
            _p.unlink(missing_ok=True)
        except Exception:
            pass


# ─── S4 ml/smart_money.py ────────────────────────────────────────────────────

print("\n=== S4 ml/smart_money.py ===")

from ml.smart_money import (
    SmartMoneySignal,
    compute_smart_money,
    compute_smart_money_features,
    _safe_net_pct,
    _linear_trend,
    MIN_SESSIONS,
    SM_CONFIRM_THRESHOLD,
)

# T8: _safe_net_pct
check("T8a net_pct positive", abs(_safe_net_pct(100, 1000) - 0.1) < 1e-9)
check("T8b net_pct clip +1",  _safe_net_pct(2000, 1000) == 1.0)
check("T8c net_pct clip -1",  _safe_net_pct(-2000, 1000) == -1.0)
check("T8d net_pct zero vol", _safe_net_pct(100, 0) == 0.0)

# T9: _linear_trend
arr_up   = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
arr_flat = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
arr_dn   = np.array([0.4, 0.3, 0.2, 0.1, 0.0])
check("T9a trend up > 0",    _linear_trend(arr_up) > 0,   f"{_linear_trend(arr_up):.3f}")
check("T9b trend flat ~ 0",  abs(_linear_trend(arr_flat)) < 1e-6)
check("T9c trend down < 0",  _linear_trend(arr_dn) < 0,   f"{_linear_trend(arr_dn):.3f}")
check("T9d trend short < 3", _linear_trend(np.array([0.1, 0.2])) == 0.0)

# T10: compute_smart_money — no SQLite data -> Insufficient data
# Mock load_snapshots de tra ve empty
with patch("data.db.load_snapshots", return_value=pd.DataFrame()):
    sm_empty = compute_smart_money("VIC")
check("T10 empty -> SmartMoneySignal", isinstance(sm_empty, SmartMoneySignal))
check("T10a empty sessions=0",        sm_empty.sessions == 0)
check("T10b empty label=Insufficient", sm_empty.label == "Insufficient data")
check("T10c empty is_confirmed=False", not sm_empty.is_confirmed)

# T11: compute_smart_money — with synthetic SQLite data (>= MIN_SESSIONS)
# Net pct = 3M / 10M = 0.30 -> score = 0.6*0.30 + 0.4*trend > 0.15
_fake_snaps = pd.DataFrame({
    "session_date": pd.date_range("2026-05-15", periods=6, freq="B"),
    "ticker":       ["VIC"] * 6,
    "foreign_buy":  [4_000_000] * 6,
    "foreign_sell": [1_000_000] * 6,
    "foreign_net":  [3_000_000] * 6,   # strong consistent buying: net_pct=0.30
    "total_volume": [10_000_000] * 6,
    "close":        [50_000] * 6,
})
with patch("data.db.load_snapshots", return_value=_fake_snaps):
    sm_acc = compute_smart_money("VIC")
check("T11 6 sessions -> SmartMoneySignal", isinstance(sm_acc, SmartMoneySignal))
check("T11a sessions=6",                    sm_acc.sessions == 6)
check("T11b foreign_net_pct > 0",           sm_acc.foreign_net_pct > 0,
      f"net_pct={sm_acc.foreign_net_pct:.4f}")
check("T11c smart_money_score > 0",         sm_acc.smart_money_score > 0,
      f"score={sm_acc.smart_money_score:.4f}")

# T12: as_dict serializable
d = sm_acc.as_dict()
check("T12a as_dict returns dict",           isinstance(d, dict))
check("T12b has smart_money_score",          "smart_money_score" in d)
check("T12c has is_confirmed",               "is_confirmed" in d)
check("T12d has sessions",                   "sessions" in d)

# T13: Accumulating label when score >= threshold
with patch("data.db.load_snapshots", return_value=_fake_snaps):
    sm2 = compute_smart_money("VIC")
check("T13 Accumulating label", sm2.label == "Accumulating",
      f"label={sm2.label} score={sm2.smart_money_score:.4f}")

# T14: Distributing scenario — net_pct = -3M/10M = -0.30 -> score < -0.15
_fake_sell = _fake_snaps.copy()
_fake_sell["foreign_net"]  = -3_000_000
_fake_sell["foreign_sell"] = 4_000_000
_fake_sell["foreign_buy"]  = 1_000_000
with patch("data.db.load_snapshots", return_value=_fake_sell):
    sm_dist = compute_smart_money("VIC")
check("T14 Distributing label", sm_dist.label == "Distributing",
      f"label={sm_dist.label} score={sm_dist.smart_money_score:.4f}")

# T15: compute_smart_money_features returns dict with 3 keys
with patch("data.db.load_snapshots", return_value=_fake_snaps):
    feat = compute_smart_money_features("VIC")
check("T15a returns dict",             isinstance(feat, dict))
check("T15b has foreign_net_pct",      "foreign_net_pct" in feat)
check("T15c has foreign_trend",        "foreign_trend" in feat)
check("T15d has smart_money_score",    "smart_money_score" in feat)


# ─── Feature Engineering: 3 new SM columns ───────────────────────────────────

print("\n=== Feature Engineering (v3.0 smart money cols) ===")

from ml.feature_engineering import FEATURE_COLS, compute_stock_features

# T16: FEATURE_COLS has 27 features
check("T16 FEATURE_COLS has 27 features", len(FEATURE_COLS) == 27,
      f"got {len(FEATURE_COLS)}")
check("T16a foreign_net_pct in list",   "foreign_net_pct"   in FEATURE_COLS)
check("T16b foreign_trend in list",     "foreign_trend"     in FEATURE_COLS)
check("T16c smart_money_score in list", "smart_money_score" in FEATURE_COLS)

# T17: compute_stock_features respects sm_features=None -> 0.0
def _make_ohlcv_clean(n: int = 40) -> pd.DataFrame:
    closes = [10_000 * (1.003 ** i) for i in range(n)]
    return pd.DataFrame({
        "date":   pd.date_range("2024-01-01", periods=n, freq="B"),
        "open":   [c * 0.995 for c in closes],
        "high":   [c * 1.010 for c in closes],
        "low":    [c * 0.990 for c in closes],
        "close":  closes,
        "volume": [1_000_000 + i * 10_000 for i in range(n)],
    })

ohlcv = _make_ohlcv_clean()
feat_df_no_sm = compute_stock_features(ohlcv, sm_features=None)
check("T17 sm_features=None -> compute ok",      not feat_df_no_sm.empty)
check("T17a foreign_net_pct=0.0",
      (feat_df_no_sm["foreign_net_pct"] == 0.0).all(),
      f"non-zero rows: {(feat_df_no_sm['foreign_net_pct'] != 0).sum()}")
check("T17b smart_money_score=0.0",
      (feat_df_no_sm["smart_money_score"] == 0.0).all())

# T18: compute_stock_features with real sm_features -> values propagated
sm_vals = {"foreign_net_pct": 0.08, "foreign_trend": 0.12, "smart_money_score": 0.20}
feat_df_sm = compute_stock_features(ohlcv, sm_features=sm_vals)
check("T18 sm_features propagated",
      abs(float(feat_df_sm["foreign_net_pct"].iloc[-1]) - 0.08) < 1e-9,
      f"got {feat_df_sm['foreign_net_pct'].iloc[-1]}")
check("T18b smart_money_score propagated",
      abs(float(feat_df_sm["smart_money_score"].iloc[-1]) - 0.20) < 1e-9)


# ─── Model version ───────────────────────────────────────────────────────────

print("\n=== Model version v3 ===")

from ml.model import MODEL_LABEL_VERSION
check("T19 model version v3", "v3" in MODEL_LABEL_VERSION,
      f"got {MODEL_LABEL_VERSION}")


# ─── Summary ────────────────────────────────────────────────────────────────

print(f"\n{'='*40}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL:
    sys.exit(1)
else:
    print("All tests passed!")
