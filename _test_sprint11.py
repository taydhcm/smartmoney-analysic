"""
_test_sprint11.py
Sprint 11 — Scale Advisor + System Monitor unit tests.

Sections:
    1. ConditionResult + check_scale_conditions    — 12 tests
    2. ScaleDecision + compute_scale_recommendation— 12 tests
    3. check_d02_history_depth                     —  8 tests
    4. check_retrain_schedule                      —  8 tests
    5. check_milestone_progress                    —  6 tests
    6. run_daily_health_check + HealthStatus        — 10 tests
    7. ml.__init__ exports regression              — 12 tests
Total target: 68 tests
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "d:/TOOL-PYTHON/smartmoney-analysic")

import json
import pickle
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# ── Temp paths ─────────────────────────────────────────────────────────────────
_TMP_DIR       = Path(tempfile.mkdtemp())
# Each variant gets its own path to avoid Windows SQLite file-lock issues
_TMP_SNAP_5    = _TMP_DIR / "snapshots_5.db"
_TMP_SNAP_30   = _TMP_DIR / "snapshots_30.db"
_TMP_TRADES_10 = _TMP_DIR / "trades_10.db"
_TMP_TRADES_30 = _TMP_DIR / "trades_30.db"
_TMP_SIG_DB    = _TMP_DIR / "signal_log.db"
_TMP_META_FRESH= _TMP_DIR / "meta_fresh.pkl"
_TMP_META_STALE= _TMP_DIR / "meta_stale.pkl"
_TMP_ROLL_META = _TMP_DIR / "rolling_meta.json"
# aliases for health-check section
_TMP_SNAP_DB   = _TMP_SNAP_5
_TMP_TRADES_DB = _TMP_TRADES_10
_TMP_META_PKL  = _TMP_META_FRESH

# ── Test counter ───────────────────────────────────────────────────────────────
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


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _make_snapshots_db(path: Path, n_days: int = 5, n_tickers: int = 3) -> None:
    """Tạo snapshots.db giả với n_days phiên và n_tickers tickers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                session_date TEXT NOT NULL,
                ticker       TEXT NOT NULL,
                PRIMARY KEY (session_date, ticker)
            )
        """)
        rows = []
        for d in range(n_days):
            date_str = (datetime(2024, 1, d + 1)).strftime("%Y-%m-%d")
            for t in range(n_tickers):
                rows.append((date_str, f"T{t:02d}"))
        con.executemany("INSERT OR IGNORE INTO snapshots VALUES (?,?)", rows)
        con.commit()


def _make_trades_db(path: Path, real_closed: int = 0, paper_closed: int = 0) -> None:
    """Tạo trades.db giả."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_type TEXT,
                status     TEXT,
                ticker     TEXT,
                entry_date TEXT
            )
        """)
        rows = []
        for i in range(real_closed):
            rows.append(("real", "closed", f"T{i:02d}", "2024-01-01"))
        for i in range(paper_closed):
            rows.append(("paper", "closed", f"P{i:02d}", "2024-01-01"))
        con.executemany(
            "INSERT INTO trades(trade_type,status,ticker,entry_date) VALUES(?,?,?,?)",
            rows,
        )
        con.commit()


def _make_signal_log_db(path: Path, total: int = 10, resolved: int = 6) -> None:
    """Tạo signal_log.db giả."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS signal_log (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker  TEXT,
                outcome TEXT
            )
        """)
        rows = []
        for i in range(total):
            outcome = "win" if i < resolved else None
            rows.append((f"T{i:02d}", outcome))
        con.executemany("INSERT INTO signal_log(ticker,outcome) VALUES(?,?)", rows)
        con.commit()


def _make_meta_pkl(path: Path, age_days: float = 3.0) -> None:
    """Tạo alpha_meta.pkl giả."""
    path.parent.mkdir(parents=True, exist_ok=True)
    trained_at = (datetime.now() - timedelta(days=age_days)).isoformat()
    meta = {
        "trained_at":   trained_at,
        "label_version": "v5_calibrated",
        "cv_prec_mean": 0.62,
        "n_samples":    5000,
    }
    with open(path, "wb") as f:
        pickle.dump(meta, f)


def _make_rolling_meta(path: Path, age_days: float = 2.0) -> None:
    """Tạo rolling_meta.json giả."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fit_at = (datetime.now() - timedelta(days=age_days)).isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"fit_at": fit_at, "n_samples": 200, "precision": 0.62}, f)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: check_scale_conditions + ConditionResult
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Section 1: check_scale_conditions ===")

from ml.scale_advisor import (
    ConditionResult,
    ScaleDecision,
    SCALE_STAGES,
    compute_scale_recommendation,
    check_scale_conditions,
)

# All conditions passing
_trade_ok = {"closed_count": 30, "win_rate": 0.50}
_calibr_ok = {"precision": 0.40, "drift_alert": False}
_pnl_ok    = {"max_drawdown_pct": -10.0}

conds_ok = check_scale_conditions(_trade_ok, _calibr_ok, _pnl_ok)
check("1.1  returns list of ConditionResult",       isinstance(conds_ok, list))
check("1.2  5 conditions returned",                  len(conds_ok) == 5)
check("1.3  all conditions are ConditionResult",     all(isinstance(c, ConditionResult) for c in conds_ok))
check("1.4  all passing when stats satisfy thresholds", all(c.passed for c in conds_ok))

# Some conditions failing
_trade_fail  = {"closed_count": 10,   "win_rate": 0.30}  # closed<30, wr<0.45
_calibr_fail = {"precision": 0.20,    "drift_alert": True}
_pnl_fail    = {"max_drawdown_pct": -20.0}

conds_fail = check_scale_conditions(_trade_fail, _calibr_fail, _pnl_fail)
check("1.5  fails when closed_count < 30",          not conds_fail[0].passed)
check("1.6  fails when precision < 0.35",           not conds_fail[1].passed)
check("1.7  fails when win_rate < 0.45",            not conds_fail[2].passed)
check("1.8  fails when drawdown < -15%",            not conds_fail[3].passed)
check("1.9  fails when drift_alert=True",           not conds_fail[4].passed)

# Edge: exactly at threshold
_trade_edge  = {"closed_count": 30,  "win_rate": 0.45}
_calibr_edge = {"precision": 0.35,   "drift_alert": False}
_pnl_edge    = {"max_drawdown_pct": -15.0}

conds_edge = check_scale_conditions(_trade_edge, _calibr_edge, _pnl_edge)
check("1.10 closed_count exactly 30 → passes",      conds_edge[0].passed)
check("1.11 precision exactly 0.35 → passes",       conds_edge[1].passed)
check("1.12 drawdown exactly -15.0 → passes",       conds_edge[3].passed)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: compute_scale_recommendation + ScaleDecision
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Section 2: compute_scale_recommendation ===")

# All conditions met → can_scale=True
dec_ok = compute_scale_recommendation(_trade_ok, _calibr_ok, _pnl_ok, current_max_pct=0.20)
check("2.1  returns ScaleDecision",                  isinstance(dec_ok, ScaleDecision))
check("2.2  can_scale=True when all conditions met", dec_ok.can_scale is True)
check("2.3  recommended_pct > current when can_scale", dec_ok.recommended_pct > 0.20)
check("2.4  score == 5 when all pass",               dec_ok.score == 5)
check("2.5  total_conditions == 5",                  dec_ok.total_conditions == 5)
check("2.6  blockers is empty when can_scale",        dec_ok.blockers == [])
check("2.7  score_pct == 1.0",                       dec_ok.score_pct == 1.0)

# Not all conditions met → can_scale=False
dec_fail = compute_scale_recommendation(_trade_fail, _calibr_fail, _pnl_fail, current_max_pct=0.20)
check("2.8  can_scale=False when conditions fail",   dec_fail.can_scale is False)
check("2.9  recommended_pct == current when blocked", dec_fail.recommended_pct == 0.20)
check("2.10 blockers non-empty",                     len(dec_fail.blockers) > 0)
check("2.11 reasoning contains 'Chưa đủ'",          "Chưa đủ" in dec_fail.reasoning)

# Already at max stage → can_scale=True but no next stage
dec_max = compute_scale_recommendation(_trade_ok, _calibr_ok, _pnl_ok, current_max_pct=0.40)
check("2.12 no scale possible at max stage",         dec_max.recommended_pct == 0.40)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: check_d02_history_depth
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Section 3: check_d02_history_depth ===")

from ml.monitor import (
    HealthStatus,
    check_d02_history_depth,
    check_retrain_schedule,
    check_milestone_progress,
    run_daily_health_check,
    D02_MIN_SESSIONS,
    RETRAIN_STALE_DAYS,
)

# Non-existent DB → safe defaults
res_nodb = check_d02_history_depth(_TMP_DIR / "no_such.db")
check("3.1  returns dict when DB missing",           isinstance(res_nodb, dict))
check("3.2  session_days=0 when no DB",              res_nodb["session_days"] == 0)
check("3.3  is_ready=False when no DB",              res_nodb["is_ready"] is False)

# DB with 5 sessions — not ready
_make_snapshots_db(_TMP_SNAP_5, n_days=5, n_tickers=10)
res5 = check_d02_history_depth(_TMP_SNAP_5)
check("3.4  reads session_days correctly",           res5["session_days"] == 5)
check("3.5  reads ticker_count correctly",           res5["ticker_count"] == 10)
check("3.6  is_ready=False when < 30 sessions",     res5["is_ready"] is False)

# Separate DB with 30 sessions
_make_snapshots_db(_TMP_SNAP_30, n_days=30, n_tickers=5)
res30 = check_d02_history_depth(_TMP_SNAP_30)
check("3.7  is_ready=True when >= 30 sessions",     res30["is_ready"] is True)
check("3.8  oldest_date populated",                  res30["oldest_date"] is not None)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: check_retrain_schedule
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Section 4: check_retrain_schedule ===")

# No meta file
res_no_meta = check_retrain_schedule(_TMP_DIR / "no_such_meta.pkl")
check("4.1  returns dict when file missing",         isinstance(res_no_meta, dict))
check("4.2  retrain_due=True when no meta",          res_no_meta["retrain_due"] is True)
check("4.3  trained_at=None when no meta",           res_no_meta["trained_at"] is None)

# Fresh meta (2 days old) — use dedicated path
_make_meta_pkl(_TMP_META_FRESH, age_days=2.0)
res_fresh = check_retrain_schedule(_TMP_META_FRESH)
check("4.4  trained_at populated",                   res_fresh["trained_at"] is not None)
check("4.5  age_days close to 2",                    abs(res_fresh["age_days"] - 2.0) < 0.1)
check("4.6  retrain_due=False for fresh model",      res_fresh["retrain_due"] is False)

# Stale meta (15 days old) — separate file
_make_meta_pkl(_TMP_META_STALE, age_days=15.0)
res_stale = check_retrain_schedule(_TMP_META_STALE)
check("4.7  retrain_due=True after 15 days",         res_stale["retrain_due"] is True)
check("4.8  label_version populated",                res_stale["label_version"] == "v5_calibrated")


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: check_milestone_progress
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Section 5: check_milestone_progress ===")

# No trades DB
res_no_tr = check_milestone_progress(_TMP_DIR / "no_trades.db")
check("5.1  returns dict when no DB",                isinstance(res_no_tr, dict))
check("5.2  real_closed=0 when no DB",               res_no_tr["real_closed"] == 0)
check("5.3  milestone_30=False when no DB",          res_no_tr["milestone_30"] is False)

# 10 real closed trades
_make_trades_db(_TMP_TRADES_10, real_closed=10, paper_closed=50)
res_10 = check_milestone_progress(_TMP_TRADES_10)
check("5.4  counts only real closed trades",         res_10["real_closed"] == 10)
check("5.5  milestone_30=False with 10 trades",      res_10["milestone_30"] is False)

# 30 real closed trades — separate file
_make_trades_db(_TMP_TRADES_30, real_closed=30)
res_30 = check_milestone_progress(_TMP_TRADES_30)
check("5.6  milestone_30=True with 30 trades",       res_30["milestone_30"] is True)


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: run_daily_health_check + HealthStatus
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Section 6: run_daily_health_check ===")

# Health check uses dedicated fresh paths
_TMP_HC_SNAP  = _TMP_DIR / "hc_snapshots.db"
_TMP_HC_TRADE = _TMP_DIR / "hc_trades.db"
_TMP_HC_META  = _TMP_DIR / "hc_meta.pkl"

_make_snapshots_db(_TMP_HC_SNAP,   n_days=10, n_tickers=3)
_make_trades_db(_TMP_HC_TRADE,     real_closed=5)
_make_signal_log_db(_TMP_SIG_DB,   total=10, resolved=6)
_make_meta_pkl(_TMP_HC_META,       age_days=3.0)
_make_rolling_meta(_TMP_ROLL_META, age_days=2.0)

hs = run_daily_health_check(
    snapshots_db      = _TMP_HC_SNAP,
    trades_db         = _TMP_HC_TRADE,
    signal_log_db     = _TMP_SIG_DB,
    meta_path         = _TMP_HC_META,
    rolling_meta_path = _TMP_ROLL_META,
)
check("6.1  returns HealthStatus",                   isinstance(hs, HealthStatus))
check("6.2  d02_session_days == 10",                 hs.d02_session_days == 10)
check("6.3  d02_ready=False (10 < 30)",              hs.d02_ready is False)
check("6.4  model_trained_at not None",              hs.model_trained_at is not None)
check("6.5  retrain_due=False (3 days)",             hs.retrain_due is False)
check("6.6  e6_fitted=True",                         hs.e6_fitted is True)
check("6.7  real_trades_closed == 5",                hs.real_trades_closed == 5)
check("6.8  milestone_30=False (5 < 30)",            hs.milestone_30 is False)
check("6.9  alerts contains D0.2 warning",           any("D0.2" in a for a in hs.alerts))
check("6.10 summary() returns string",               isinstance(hs.summary(), str))


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: ml.__init__ exports regression
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Section 7: ml.__init__ exports ===")

import ml

# Sprint 11 exports
check("7.1  ScaleDecision exported",                  hasattr(ml, "ScaleDecision"))
check("7.2  ConditionResult exported",                hasattr(ml, "ConditionResult"))
check("7.3  SCALE_STAGES exported",                   hasattr(ml, "SCALE_STAGES"))
check("7.4  compute_scale_recommendation exported",   hasattr(ml, "compute_scale_recommendation"))
check("7.5  check_scale_conditions exported",         hasattr(ml, "check_scale_conditions"))
check("7.6  HealthStatus exported",                   hasattr(ml, "HealthStatus"))
check("7.7  run_daily_health_check exported",         hasattr(ml, "run_daily_health_check"))
check("7.8  check_d02_history_depth exported",        hasattr(ml, "check_d02_history_depth"))
check("7.9  check_retrain_schedule exported",         hasattr(ml, "check_retrain_schedule"))
check("7.10 check_milestone_progress exported",       hasattr(ml, "check_milestone_progress"))

# Regression: Sprint 10 still exported
check("7.11 GoLiveConfig still exported",             hasattr(ml, "GoLiveConfig"))
check("7.12 run_go_live_checks still exported",       hasattr(ml, "run_go_live_checks"))


# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Sprint 11 Tests: {_PASS} passed, {_FAIL} failed")
print(f"{'='*55}")
if _FAIL == 0:
    print("  ALL TESTS PASSED ✅")
else:
    print(f"  {_FAIL} TEST(S) FAILED ❌")
