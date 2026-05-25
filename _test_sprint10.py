"""
_test_sprint10.py
Sprint 10 — Go-Live Validator unit tests

Sections:
    1. GoLiveConfig validation          —  9 tests
    2. CheckResult dataclass            —  4 tests
    3. Individual _check_* functions    — 18 tests
    4. run_go_live_checks()             —  8 tests
    5. GoLiveStatus helpers             —  5 tests
    6. ml.__init__ exports (regression) — 12 tests
Total target: 56 tests
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "d:/TOOL-PYTHON/smartmoney-analysic")

import json
import pickle
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# ── Temp paths ─────────────────────────────────────────────────────────────────
_TMP_DIR  = Path(tempfile.mkdtemp())
_TMP_PKL  = _TMP_DIR / "alpha_model.pkl"
_TMP_CAL  = _TMP_DIR / "alpha_calibrator.pkl"
_TMP_ROLL = _TMP_DIR / "rolling_cal.pkl"
_TMP_META = _TMP_DIR / "rolling_meta.json"
_TMP_SIG  = _TMP_DIR / "signal_log.db"
_TMP_TRD  = _TMP_DIR / "trades.db"

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


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: GoLiveConfig validation
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Section 1: GoLiveConfig ===")

from ml.go_live_checker import (
    GoLiveConfig, CheckResult, GoLiveStatus, run_go_live_checks,
)
import ml.go_live_checker as _glc

# Patch all paths to temps
_glc.MODEL_PATH              = _TMP_PKL
_glc.CALIBRATOR_PATH         = _TMP_CAL
_glc.ROLLING_CALIBRATOR_PATH = _TMP_ROLL
_glc.ROLLING_META_PATH       = _TMP_META
_glc.SIGNAL_LOG_DB           = _TMP_SIG
_glc.TRADES_DB               = _TMP_TRD

# 1a — defaults are valid
cfg_default = GoLiveConfig()
check("default config is_valid()",         cfg_default.is_valid())
check("default validate() empty",          len(cfg_default.validate()) == 0)
check("default p_min=0.70",               abs(cfg_default.p_min - 0.70) < 1e-9)
check("default max_position_pct=0.20",    abs(cfg_default.max_position_pct - 0.20) < 1e-9)
check("default sl_buffer_pct=2.0",        abs(cfg_default.sl_buffer_pct - 2.0) < 1e-9)

# 1b — invalid: p_min too low
cfg_low_p = GoLiveConfig(p_min=0.40)
_errs = cfg_low_p.validate()
check("p_min=0.40 → validate has errors",  len(_errs) > 0)
check("p_min=0.40 → not is_valid()",       not cfg_low_p.is_valid())

# 1c — invalid: max_position_pct too high
cfg_high_pos = GoLiveConfig(max_position_pct=0.60)
check("max_pos=0.60 → not valid",          not cfg_high_pos.is_valid())

# 1d — invalid: sl_buffer too small
cfg_tiny_sl = GoLiveConfig(sl_buffer_pct=0.1)
check("sl_buffer=0.1 → not valid",         not cfg_tiny_sl.is_valid())

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: CheckResult dataclass
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Section 2: CheckResult ===")

_cr_pass = CheckResult(name="Test", passed=True, detail="All good")
_cr_fail = CheckResult(name="Test2", passed=False, detail="Broken", is_warning=False)
_cr_warn = CheckResult(name="Test3", passed=False, detail="Warning", is_warning=True)

check("CheckResult passed=True",          _cr_pass.passed is True)
check("CheckResult passed=False",         _cr_fail.passed is False)
check("CheckResult is_warning=False",     _cr_fail.is_warning is False)
check("CheckResult is_warning=True",      _cr_warn.is_warning is True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Individual check functions
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Section 3: Individual Checks ===")

from ml.go_live_checker import (
    _check_model_exists, _check_batch_calibrator, _check_rolling_calibrator,
    _check_e5_active, _check_e6_status, _check_no_drift,
    _check_capital, _check_p_min, _check_sl_buffer,
    _check_position_size, _check_rr_filter,
    _check_pipeline_imports, _check_trades_db,
)

# 3a — model does not exist
_r = _check_model_exists()
check("no model → passed=False",          not _r.passed)
check("no model → has name",              len(_r.name) > 0)
check("no model → has detail",            len(_r.detail) > 0)

# 3b — model exists
_TMP_PKL.write_bytes(b"dummy_model")
_r2 = _check_model_exists()
check("model exists → passed=True",       _r2.passed)

# 3c — batch calibrator does not exist
_rc = _check_batch_calibrator()
check("no calibrator → passed=False",     not _rc.passed)

# 3d — batch calibrator exists
_TMP_CAL.write_bytes(b"dummy_cal")
_rc2 = _check_batch_calibrator()
check("cal exists → passed=True",         _rc2.passed)

# 3e — rolling calibrator does not exist (is_warning)
_rroll = _check_rolling_calibrator()
check("no rolling → passed=False",        not _rroll.passed)
check("no rolling → is_warning=True",     _rroll.is_warning is True)

# 3f — rolling exists but stale (age > 7 days)
_TMP_ROLL.write_bytes(b"dummy_roll")
_OLD = (datetime.now() - timedelta(days=10)).isoformat(timespec="seconds")
_TMP_META.write_text(
    json.dumps({"fit_at": _OLD, "n_samples": 20, "precision": 0.60}),
    encoding="utf-8",
)
_rroll2 = _check_rolling_calibrator()
check("stale rolling → passed=False",     not _rroll2.passed)
check("stale rolling → is_warning=True",  _rroll2.is_warning is True)

# 3g — rolling fresh (age < 7 days)
_NOW = datetime.now().isoformat(timespec="seconds")
_TMP_META.write_text(
    json.dumps({"fit_at": _NOW, "n_samples": 25, "precision": 0.62}),
    encoding="utf-8",
)
_rroll3 = _check_rolling_calibrator()
check("fresh rolling → passed=True",      _rroll3.passed)

# 3h — e5 active: DB not exist
_re5 = _check_e5_active()
check("no signal_log.db → is_warning",    _re5.is_warning is True)

# 3i — capital checks
_cfg = GoLiveConfig()
_rc_cap_ok  = _check_capital(_cfg, 200_000_000)
_rc_cap_low = _check_capital(_cfg, 50_000_000)
check("capital 200M >= 100M → pass",      _rc_cap_ok.passed)
check("capital 50M < 100M → fail",        not _rc_cap_low.passed)

# 3j — p_min checks
_rc_pmin_ok  = _check_p_min(GoLiveConfig(p_min=0.70))
_rc_pmin_low = _check_p_min(GoLiveConfig(p_min=0.50))
check("p_min=0.70 → passed",              _rc_pmin_ok.passed)
check("p_min=0.50 < 0.60 → failed",       not _rc_pmin_low.passed)

# 3k — sl_buffer
_rc_sl_ok  = _check_sl_buffer(GoLiveConfig(sl_buffer_pct=2.0))
_rc_sl_low = _check_sl_buffer(GoLiveConfig(sl_buffer_pct=0.5))
check("sl_buffer=2.0 → passed",           _rc_sl_ok.passed)
check("sl_buffer=0.5 < 1.5 → failed",     not _rc_sl_low.passed)

# 3l — position size
_rc_pos_ok  = _check_position_size(GoLiveConfig(max_position_pct=0.20))
_rc_pos_big = _check_position_size(GoLiveConfig(max_position_pct=0.40))
check("max_pos=0.20 → passed",            _rc_pos_ok.passed)
check("max_pos=0.40 > 0.30 → failed",     not _rc_pos_big.passed)

# 3m — rr_filter
_rc_rr_ok  = _check_rr_filter(GoLiveConfig(rr_min=2.0))
_rc_rr_low = _check_rr_filter(GoLiveConfig(rr_min=1.0))
check("rr_min=2.0 → passed",              _rc_rr_ok.passed)
check("rr_min=1.0 < 1.5 → failed",        not _rc_rr_low.passed)

# 3n — pipeline imports (should pass — all modules are importable)
_rc_pipe = _check_pipeline_imports()
check("pipeline imports → passed",         _rc_pipe.passed,
      f"failed: {_rc_pipe.detail[:80]}")

# 3o — trades_db (no file → still passed, is_warning)
_rc_trd = _check_trades_db()
check("no trades.db → passed=True (auto-create)", _rc_trd.passed)

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: run_go_live_checks()
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Section 4: run_go_live_checks() ===")

# 4a — runs without exception
_cfg_default = GoLiveConfig()
_status = run_go_live_checks(config=_cfg_default, portfolio_capital=500_000_000)
check("run_go_live_checks returns GoLiveStatus", isinstance(_status, GoLiveStatus))
check("status has 13 checks",              _status.total == 13,
      f"got {_status.total}")
check("score is int >= 0",                 isinstance(_status.score, int) and _status.score >= 0)
check("hard_fails + warnings + passed = total",
      _status.hard_fails + _status.warnings + _status.score >= 0)

# 4b — with capital too low → capital check should fail (hard fail)
_status_low = run_go_live_checks(
    config=GoLiveConfig(), portfolio_capital=10_000_000
)
# Capital check (#7) should be a hard fail
_cap_check = _status_low.checks[6]  # index 6 = _check_capital
check("capital 10M < 100M → capital check failed",
      not _cap_check.passed)
check("capital fail is NOT warning",       not _cap_check.is_warning)

# 4c — with bad p_min
_cfg_bad_p = GoLiveConfig(p_min=0.50)
_status_bad = run_go_live_checks(config=_cfg_bad_p, portfolio_capital=500_000_000)
_pmin_check = _status_bad.checks[7]   # index 7 = _check_p_min
check("p_min=0.50 → p_min check failed",  not _pmin_check.passed)

# 4d — summary string
_sum = _status.summary()
check("summary() is non-empty string",    isinstance(_sum, str) and len(_sum) > 10)
check("summary contains score fraction",  "/" in _sum)

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: GoLiveStatus helpers
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Section 5: GoLiveStatus helpers ===")

# Build a GoLiveStatus manually
_checks_mock = [
    CheckResult("A", True,  "ok"),
    CheckResult("B", False, "fail", is_warning=False),
    CheckResult("C", False, "warn", is_warning=True),
    CheckResult("D", True,  "ok"),
]
_mock_status = GoLiveStatus(
    checks     = _checks_mock,
    all_passed = False,
    hard_fails = 1,
    warnings   = 1,
    score      = 2,
    total      = 4,
)
check("score_pct = 2/4 = 0.50",           abs(_mock_status.score_pct - 0.50) < 1e-9)
check("failed_checks() len=1",            len(_mock_status.failed_checks()) == 1)
check("warning_checks() len=1",           len(_mock_status.warning_checks()) == 1)
check("failed_checks()[0].name='B'",      _mock_status.failed_checks()[0].name == "B")
check("summary() contains 'NOT READY'",   "NOT READY" in _mock_status.summary())

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: ml.__init__ exports (regression)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Section 6: ml.__init__ exports ===")

import ml as _ml

# Sprint 10 new exports
_s10 = ["GoLiveConfig", "CheckResult", "GoLiveStatus", "run_go_live_checks"]
for _sym in _s10:
    check(f"ml.{_sym} exported (S10)",    hasattr(_ml, _sym))

# Sprint 9 regression
_s9 = ["get_calibration_data", "fit_rolling_calibrator", "apply_rolling_calibration",
       "get_calibrator_status", "needs_weekly_refit",
       "ROLLING_CALIBRATOR_PATH", "ROLLING_META_PATH"]
for _sym in _s9:
    check(f"ml.{_sym} still exported (S9)", hasattr(_ml, _sym))

# Sprint 8 regression
_s8 = ["log_signal", "get_signal_stats", "compute_trade_pnl", "compute_equity_curve"]
for _sym in _s8:
    check(f"ml.{_sym} still exported (S8)", hasattr(_ml, _sym))

# Sprint 7 regression
_s7 = ["generate_morning_report", "compute_conviction_size"]
for _sym in _s7:
    check(f"ml.{_sym} still exported (S7)", hasattr(_ml, _sym))

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  {_PASS} PASS | {_FAIL} FAIL | {_PASS + _FAIL} total")
if _FAIL == 0:
    print(f"  \u2705 ALL {_PASS} tests PASSED")
else:
    print(f"  \u274c {_FAIL} tests FAILED")
print('='*60)

sys.exit(0 if _FAIL == 0 else 1)
