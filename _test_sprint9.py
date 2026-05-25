"""
_test_sprint9.py
Sprint 9 — E6 Rolling IsotonicRegression Calibrator unit tests

Sections:
    1. get_calibration_data()           — 10 tests
    2. fit_rolling_calibrator()         —  9 tests
    3. load_rolling_calibrator()        —  4 tests
    4. apply_rolling_calibration()      —  6 tests
    5. get_calibrator_status()          —  7 tests
    6. needs_weekly_refit()             —  5 tests
    7. ml.__init__ exports (regression) — 10 tests
Total target: 51 tests
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "d:/TOOL-PYTHON/smartmoney-analysic")

import json
import pickle
import sqlite3
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

# ── Temp paths ─────────────────────────────────────────────────────────────────
_TMP_DIR  = Path(tempfile.mkdtemp())
_TMP_SIG  = _TMP_DIR / "signal_log.db"
_TMP_PKL  = _TMP_DIR / "rolling_cal.pkl"
_TMP_META = _TMP_DIR / "rolling_meta.json"
_TMP_SIG2 = _TMP_DIR / "signal_log2.db"   # separate DB for limit tests
_TMP_SIG3 = _TMP_DIR / "signal_log3.db"   # separate DB for sort tests

# ── Test helpers ───────────────────────────────────────────────────────────────
_PASS = 0
_FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  [OK]   {name}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {name}" + (f"  -- {detail}" if detail else ""))


def _make_signal_db(db_path: Path, records: list) -> None:
    """
    Create a minimal signal_log.db with resolved outcomes for testing.
    records: list of dict with keys: signal_date (opt), ticker (opt),
             p_raw (opt), p_calibrated (opt), outcome, pnl_pct (opt)
    """
    con = sqlite3.connect(str(db_path))
    con.executescript("""
        CREATE TABLE IF NOT EXISTS signal_log (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_date       TEXT    NOT NULL,
            ticker            TEXT    NOT NULL,
            action            TEXT    NOT NULL DEFAULT 'BUY',
            entry_price       REAL    NOT NULL DEFAULT 100.0,
            sl_price          REAL    NOT NULL DEFAULT 95.0,
            target_price      REAL    NOT NULL DEFAULT 110.0,
            rr_ratio          REAL    NOT NULL DEFAULT 2.0,
            p_calibrated      REAL    NOT NULL DEFAULT 0.0,
            p_raw             REAL    NOT NULL DEFAULT 0.0,
            recommendation    TEXT    NOT NULL DEFAULT '',
            position_size_pct REAL    NOT NULL DEFAULT 0.0,
            regime_state      TEXT    NOT NULL DEFAULT '',
            reason            TEXT    NOT NULL DEFAULT '',
            created_at        TEXT    NOT NULL DEFAULT '',
            UNIQUE (signal_date, ticker)
        );
        CREATE TABLE IF NOT EXISTS signal_outcomes (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id    INTEGER NOT NULL,
            outcome_date TEXT    NOT NULL DEFAULT '',
            outcome      TEXT    NOT NULL DEFAULT 'PENDING',
            exit_price   REAL,
            pnl_pct      REAL,
            checked_at   TEXT    NOT NULL DEFAULT '',
            UNIQUE (signal_id)
        );
    """)
    base = date(2026, 1, 1)
    for i, rec in enumerate(records):
        sig_date = rec.get("signal_date", (base + timedelta(days=i)).isoformat())
        ticker   = rec.get("ticker",       f"T{i:02d}")
        p_raw    = rec.get("p_raw",        0.60)
        p_cal    = rec.get("p_calibrated", round(p_raw * 0.9, 4))
        outcome  = rec.get("outcome",      "WIN")
        pnl      = rec.get("pnl_pct",      3.0 if outcome == "WIN" else -2.0)

        cur = con.execute(
            "INSERT OR IGNORE INTO signal_log "
            "(signal_date, ticker, p_raw, p_calibrated, created_at) VALUES (?,?,?,?,?)",
            (sig_date, ticker, p_raw, p_cal, "2026-01-25T10:00:00"),
        )
        sig_id = cur.lastrowid
        if sig_id:
            con.execute(
                "INSERT OR IGNORE INTO signal_outcomes "
                "(signal_id, outcome_date, outcome, pnl_pct, checked_at) VALUES (?,?,?,?,?)",
                (sig_id, sig_date, outcome, pnl, "2026-01-25T10:00:00"),
            )
    con.commit()
    con.close()


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: get_calibration_data()
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Section 1: get_calibration_data() ===")

import ml.rolling_calibrator as _rc

# Patch module-level paths to temps
_rc.ROLLING_CALIBRATOR_PATH = _TMP_PKL
_rc.ROLLING_META_PATH       = _TMP_META
_rc.SIGNAL_LOG_DB           = _TMP_SIG

# 1a — non-existent DB returns empty DF
from pathlib import Path as _P
_noexist = _TMP_DIR / "nope.db"
_df0 = _rc.get_calibration_data(db_path=_noexist)
check("no DB → empty DataFrame",          _df0.empty)
check("no DB → has 'label' column",       "label" in _df0.columns)
check("no DB → has 'p_raw' column",       "p_raw" in _df0.columns)
check("no DB → has 'outcome' column",     "outcome" in _df0.columns)

# 1b — seed test DB with 3 resolved + 1 PENDING + 1 EXPIRED
_seed_records = [
    {"outcome": "WIN",     "p_raw": 0.75, "pnl_pct":  5.0, "signal_date": "2026-01-10", "ticker": "VIC"},
    {"outcome": "LOSS",    "p_raw": 0.40, "pnl_pct": -3.0, "signal_date": "2026-01-11", "ticker": "HPG"},
    {"outcome": "FLAT",    "p_raw": 0.55, "pnl_pct":  0.0, "signal_date": "2026-01-12", "ticker": "VHM"},
    {"outcome": "PENDING", "p_raw": 0.65, "pnl_pct":  0.0, "signal_date": "2026-01-13", "ticker": "MSN"},
    {"outcome": "EXPIRED", "p_raw": 0.30, "pnl_pct":  0.0, "signal_date": "2026-01-14", "ticker": "VNM"},
]
_make_signal_db(_TMP_SIG, _seed_records)
_df1 = _rc.get_calibration_data(n=10, db_path=_TMP_SIG)

check("3 resolved rows returned",         len(_df1) == 3, f"got {len(_df1)}")
check("PENDING excluded",                 "PENDING" not in _df1["outcome"].values)
check("EXPIRED excluded",                 "EXPIRED" not in _df1["outcome"].values)
check("WIN → label=1",                    _df1.loc[_df1["outcome"] == "WIN", "label"].iloc[0] == 1)
check("LOSS → label=0",                   _df1.loc[_df1["outcome"] == "LOSS", "label"].iloc[0] == 0)
check("FLAT → label=0",                   _df1.loc[_df1["outcome"] == "FLAT", "label"].iloc[0] == 0)

# 1c — sorted chronologically (ASC)
_dates = _df1["signal_date"].tolist()
check("sorted ASC by signal_date",        _dates == sorted(_dates), f"got {_dates}")

# 1d — n limit
_recs_20 = [{"outcome": "WIN" if i % 2 == 0 else "LOSS", "p_raw": 0.5 + i * 0.01,
              "signal_date": f"2026-02-{i+1:02d}"} for i in range(20)]
_make_signal_db(_TMP_SIG2, _recs_20)
_df_n5 = _rc.get_calibration_data(n=5, db_path=_TMP_SIG2)
check("n=5 limit respected",              len(_df_n5) <= 5, f"got {len(_df_n5)}")

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: fit_rolling_calibrator()
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Section 2: fit_rolling_calibrator() ===")

# 2a — no DB
_r_nodb = _rc.fit_rolling_calibrator(
    db_path=_TMP_DIR / "nope.db",
    save_path=_TMP_PKL,
    meta_path=_TMP_META,
)
check("no DB → status='no_db'",           _r_nodb["status"] == "no_db",
      f"got {_r_nodb['status']}")
check("no DB → n_samples=0",              _r_nodb["n_samples"] == 0)
check("no DB → fit_at=None",              _r_nodb["fit_at"] is None)

# 2b — insufficient data (only 3 rows, min_samples=10)
_r_ins = _rc.fit_rolling_calibrator(
    n=10, min_samples=10,
    db_path=_TMP_SIG,    # has only 3 resolved
    save_path=_TMP_PKL,
    meta_path=_TMP_META,
)
check("insufficient → status='insufficient_data'",
      _r_ins["status"] == "insufficient_data", f"got {_r_ins['status']}")
check("insufficient → n_samples=3",       _r_ins["n_samples"] == 3, f"got {_r_ins['n_samples']}")

# 2c — successful fit (use DB with 20 records, min_samples=5)
_r_ok = _rc.fit_rolling_calibrator(
    n=20, min_samples=5,
    db_path=_TMP_SIG2,
    save_path=_TMP_PKL,
    meta_path=_TMP_META,
)
check("OK fit → status='ok'",             _r_ok["status"] == "ok", f"got {_r_ok['status']}")
check("OK fit → n_samples >= 5",          _r_ok["n_samples"] >= 5, f"got {_r_ok['n_samples']}")
check("OK fit → precision in [0,1]",      0.0 <= _r_ok["precision"] <= 1.0)
check("OK fit → fit_at is str",           isinstance(_r_ok["fit_at"], str))
check("OK fit → pkl created",             _TMP_PKL.exists())
check("OK fit → meta JSON created",       _TMP_META.exists())

# 2d — meta JSON structure
_meta_loaded = json.loads(_TMP_META.read_text(encoding="utf-8"))
check("meta has 'fit_at'",                "fit_at" in _meta_loaded)
check("meta has 'n_samples'",             "n_samples" in _meta_loaded)
check("meta has 'precision'",             "precision" in _meta_loaded)
check("meta n_samples matches result",    _meta_loaded["n_samples"] == _r_ok["n_samples"])

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: load_rolling_calibrator()
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Section 3: load_rolling_calibrator() ===")

# 3a — no file → None
_nofile_pkl = _TMP_DIR / "nonexistent_cal.pkl"
_loaded_none = _rc.load_rolling_calibrator(save_path=_nofile_pkl)
check("no file → None",                   _loaded_none is None)

# 3b — after fit → IsotonicRegression
_loaded_cal = _rc.load_rolling_calibrator(save_path=_TMP_PKL)
check("after fit → not None",             _loaded_cal is not None)
check("after fit → IsotonicRegression",   isinstance(_loaded_cal, IsotonicRegression))

# 3c — invalid pkl → None
_bad_pkl = _TMP_DIR / "bad.pkl"
_bad_pkl.write_bytes(b"not a valid pickle")
_loaded_bad = _rc.load_rolling_calibrator(save_path=_bad_pkl)
check("invalid pkl → None",               _loaded_bad is None)

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: apply_rolling_calibration()
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Section 4: apply_rolling_calibration() ===")

_arr_in = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

# 4a — no calibrator file (fallback_to_batch=False) → returns raw array
_nofile2 = _TMP_DIR / "nc.pkl"
_out_raw = _rc.apply_rolling_calibration(_arr_in, save_path=_nofile2, fallback_to_batch=False)
check("no calibrator → ndarray returned",  isinstance(_out_raw, np.ndarray))
check("no calibrator → same length",       len(_out_raw) == len(_arr_in))

# 4b — with valid calibrator
_out_cal = _rc.apply_rolling_calibration(_arr_in, save_path=_TMP_PKL, fallback_to_batch=False)
check("calibrated → ndarray",              isinstance(_out_cal, np.ndarray))
check("calibrated → all in [0,1]",         bool(np.all((_out_cal >= 0.0) & (_out_cal <= 1.0))),
      f"min={_out_cal.min():.4f} max={_out_cal.max():.4f}")
check("calibrated → same length as input", len(_out_cal) == len(_arr_in))

# 4c — list input works
_out_list = _rc.apply_rolling_calibration([0.2, 0.5, 0.8], save_path=_TMP_PKL, fallback_to_batch=False)
check("list input works",                  isinstance(_out_list, np.ndarray) and len(_out_list) == 3)

# 4d — scalar-like input
_out_scalar = _rc.apply_rolling_calibration([0.6], save_path=_TMP_PKL, fallback_to_batch=False)
check("scalar-like → in [0,1]",            float(_out_scalar[0]) >= 0.0 and float(_out_scalar[0]) <= 1.0)

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: get_calibrator_status()
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Section 5: get_calibrator_status() ===")

# Meta has been written by Section 2 fit; PKL exists too

# 5a — check required keys exist
_status = _rc.get_calibrator_status()
_required_keys = [
    "is_fitted", "last_fit_at", "n_samples", "precision",
    "is_stale", "needs_refit", "rolling_path_exists", "batch_path_exists",
    "drift_alert",
]
check("status has all required keys",      all(k in _status for k in _required_keys),
      f"missing: {[k for k in _required_keys if k not in _status]}")

# 5b — after fit → is_fitted=True
check("after fit → is_fitted=True",        _status["is_fitted"] is True)
check("after fit → n_samples > 0",         _status["n_samples"] > 0)
check("after fit → precision in [0,1]",    0.0 <= _status["precision"] <= 1.0)
check("after fit → rolling_path_exists=True", _status["rolling_path_exists"] is True)

# 5c — no meta file → is_fitted=False
_rc.ROLLING_META_PATH = _TMP_DIR / "no_meta.json"
_status_nometa = _rc.get_calibrator_status()
check("no meta → is_fitted=False",         _status_nometa["is_fitted"] is False)
check("no meta → needs_refit=True",        _status_nometa["needs_refit"] is True)

# Restore meta path
_rc.ROLLING_META_PATH = _TMP_META

# 5d — stale check: write old meta
_OLD_DATE = (datetime.now() - timedelta(days=10)).isoformat(timespec="seconds")
_stale_meta_path = _TMP_DIR / "stale_meta.json"
_stale_meta_path.write_text(
    json.dumps({"fit_at": _OLD_DATE, "n_samples": 15, "precision": 0.55}),
    encoding="utf-8",
)
_rc.ROLLING_META_PATH = _stale_meta_path
_status_stale = _rc.get_calibrator_status()
check("old meta → is_stale=True",          _status_stale["is_stale"] is True)

# Restore
_rc.ROLLING_META_PATH = _TMP_META

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: needs_weekly_refit()
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Section 6: needs_weekly_refit() ===")

# 6a — no pkl file
_rc.ROLLING_CALIBRATOR_PATH = _TMP_DIR / "no_cal.pkl"
_rc.ROLLING_META_PATH       = _TMP_META
check("no pkl → needs_refit=True",        _rc.needs_weekly_refit())

# 6b — no meta file
_rc.ROLLING_CALIBRATOR_PATH = _TMP_PKL
_rc.ROLLING_META_PATH       = _TMP_DIR / "no_meta2.json"
check("no meta → needs_refit=True",       _rc.needs_weekly_refit())

# 6c — fresh meta (< 7 days) → False
_fresh_meta = _TMP_DIR / "fresh_meta.json"
_NOW_STR = datetime.now().isoformat(timespec="seconds")
_fresh_meta.write_text(
    json.dumps({"fit_at": _NOW_STR, "n_samples": 20, "precision": 0.6}),
    encoding="utf-8",
)
_rc.ROLLING_META_PATH = _fresh_meta
check("fresh meta → needs_refit=False",   not _rc.needs_weekly_refit())

# 6d — stale meta (> 7 days) → True
_stale_meta2 = _TMP_DIR / "stale2.json"
_OLD = (datetime.now() - timedelta(days=8)).isoformat(timespec="seconds")
_stale_meta2.write_text(
    json.dumps({"fit_at": _OLD, "n_samples": 20, "precision": 0.6}),
    encoding="utf-8",
)
_rc.ROLLING_META_PATH = _stale_meta2
check("stale meta → needs_refit=True",    _rc.needs_weekly_refit())

# 6e — stale_days parameter: if threshold=14, 8-day-old meta is fresh
check("8d meta with stale_days=14 → False", not _rc.needs_weekly_refit(stale_days=14))

# Restore paths
_rc.ROLLING_CALIBRATOR_PATH = _TMP_PKL
_rc.ROLLING_META_PATH       = _TMP_META

# ─────────────────────────────────────────────────────────────────────────────
# Section 7: ml.__init__ exports (regression + E6 new)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Section 7: ml.__init__ exports ===")

import ml as _ml

# E6 new exports
_e6_exports = [
    "get_calibration_data",
    "fit_rolling_calibrator",
    "load_rolling_calibrator",
    "apply_rolling_calibration",
    "get_calibrator_status",
    "needs_weekly_refit",
    "ROLLING_CALIBRATOR_PATH",
    "ROLLING_META_PATH",
]
for _sym in _e6_exports:
    check(f"ml.{_sym} exported",          hasattr(_ml, _sym))

# Sprint 8 regression
_s8_exports = [
    "log_signal", "log_signals_batch", "check_pending_outcomes",
    "get_signal_stats", "SIGNAL_LOG_DB",
    "compute_trade_pnl", "compute_equity_curve",
]
for _sym in _s8_exports:
    check(f"ml.{_sym} still exported (S8 regression)", hasattr(_ml, _sym))

# Sprint 7 regression
_s7_exports = ["generate_morning_report", "compute_conviction_size", "AlertCard"]
for _sym in _s7_exports:
    check(f"ml.{_sym} still exported (S7 regression)", hasattr(_ml, _sym))

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  {_PASS} PASS | {_FAIL} FAIL | {_PASS + _FAIL} total")
if _FAIL == 0:
    print(f"  ✅ ALL {_PASS} tests PASSED")
else:
    print(f"  ❌ {_FAIL} tests FAILED")
print('='*60)

sys.exit(0 if _FAIL == 0 else 1)
