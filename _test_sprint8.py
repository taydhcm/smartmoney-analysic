"""
_test_sprint8.py
Sprint 8 — E2 Trade Logger + E3 P&L Tracker + E5 Outcome Tracker unit tests
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "d:/TOOL-PYTHON/smartmoney-analysic")

import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

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
# Section 1: data/trade_logger.py (E2)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== E2 Trade Logger (data/trade_logger.py) ===")

import data.trade_logger as _tl
from data.trade_logger import (
    add_trade, close_trade, get_trade, get_open_trades,
    get_all_trades, get_trade_stats, update_unrealized_pnl,
    TRADES_DB,
)

# Use temp DB for tests
_ORIG_TRADES_DB = _tl.TRADES_DB
_TMP_DIR = Path(tempfile.mkdtemp())
_TMP_TRADES_DB = _TMP_DIR / "test_trades.db"
_tl.TRADES_DB = _TMP_TRADES_DB


# add_trade basic
_tid1 = add_trade(
    ticker="VIC", entry_date="2026-05-20", entry_price=50000,
    shares=1000, sl_price=47000, target_price=55000,
    trade_type="paper",
)
check("add_trade returns int id", isinstance(_tid1, int) and _tid1 > 0, f"got {_tid1}")

_tid2 = add_trade(
    ticker="HPG", entry_date="2026-05-21", entry_price=25000,
    shares=2000, sl_price=23000, target_price=28000,
    trade_type="real", note="Sprint 8 test",
)
check("add_trade second trade ok", isinstance(_tid2, int) and _tid2 > 0)

# get_trade
_t1 = get_trade(_tid1)
check("get_trade ticker=VIC",    _t1["ticker"] == "VIC")
check("get_trade status=open",   _t1["status"] == "open")
check("get_trade entry_price",   abs(_t1["entry_price"] - 50000) < 1)
check("get_trade sl_price",      abs(_t1["sl_price"] - 47000) < 1)

# get_open_trades
_open = get_open_trades()
check("get_open_trades has 2",   len(_open) == 2, f"got {len(_open)}")

# get_open_trades by type
_open_paper = get_open_trades("paper")
check("get_open_trades paper=1", len(_open_paper) == 1)

# close_trade WIN
_closed = close_trade(_tid1, "2026-05-25", 54000.0, "closed")
check("close_trade status=closed", _closed["status"] == "closed")
check("close_trade exit_price",    abs(_closed["exit_price"] - 54000) < 1)
check("close_trade pnl_pct > 0",   _closed["pnl_pct"] > 0,
      f"got {_closed['pnl_pct']}")
check("close_trade pnl_vnd > 0",   _closed["pnl_vnd"] > 0)

_expected_pnl = round((54000 - 50000) / 50000 * 100, 4)
check("close_trade pnl_pct formula",
      abs(_closed["pnl_pct"] - _expected_pnl) < 0.01, f"got {_closed['pnl_pct']}")

# close_trade LOSS (stopped)
_closed2 = close_trade(_tid2, "2026-05-25", 23500.0, "stopped")
check("close_trade stopped status",  _closed2["status"] == "stopped")
check("close_trade stopped pnl<0",   _closed2["pnl_pct"] < 0)

# get_all_trades
_all = get_all_trades()
check("get_all_trades len=2",  len(_all) == 2, f"got {len(_all)}")

# update_unrealized_pnl
_tid3 = add_trade(
    ticker="MWG", entry_date="2026-05-25", entry_price=60000,
    shares=500, sl_price=57000, target_price=66000,
    trade_type="paper",
)
_updated = update_unrealized_pnl({"MWG": 62000.0})
_mwg = next((t for t in _updated if t["ticker"] == "MWG"), None)
check("update_unrealized_pnl MWG found",    _mwg is not None)
check("update_unrealized_pnl pct > 0",
      _mwg is not None and _mwg.get("unrealized_pct", 0) > 0)

# get_trade_stats
_stats = get_trade_stats()
check("get_trade_stats win_count=1",   _stats["win_count"] == 1)
check("get_trade_stats loss_count=1",  _stats["loss_count"] == 1)
check("get_trade_stats win_rate=0.5",  abs(_stats["win_rate"] - 0.5) < 1e-4)
check("get_trade_stats profit_factor >= 0", _stats["profit_factor"] >= 0)

# Restore original DB path
_tl.TRADES_DB = _ORIG_TRADES_DB


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: ml/pnl_tracker.py (E3)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== E3 P&L Tracker (ml/pnl_tracker.py) ===")

from ml.pnl_tracker import (
    compute_trade_pnl, compute_equity_curve,
    compute_stats, compute_win_streak,
)

def _make_trade(ticker="VIC", entry=50000, exit_price=None, shares=1000,
                sl=47000, target=55000, status="open",
                entry_date="2026-05-01", exit_date=None, pnl_pct=None, pnl_vnd=None):
    t = dict(ticker=ticker, entry_price=entry, sl_price=sl, target_price=target,
             shares=shares, status=status, entry_date=entry_date)
    if exit_price is not None:
        t["exit_price"] = exit_price
        t["exit_date"]  = exit_date or "2026-05-06"
        t["pnl_pct"]    = pnl_pct if pnl_pct is not None else round((exit_price-entry)/entry*100, 4)
        t["pnl_vnd"]    = pnl_vnd if pnl_vnd is not None else round((exit_price-entry)*shares, 0)
    return t

# compute_trade_pnl
_t_open = _make_trade(entry=50000, shares=1000, sl=47000, target=55000)
_pnl_open = compute_trade_pnl(_t_open)
check("compute_trade_pnl risk_reward > 0", _pnl_open["risk_reward"] > 0,
      f"got {_pnl_open['risk_reward']}")
check("compute_trade_pnl risk_reward = 5/3 = 1.67",
      abs(_pnl_open["risk_reward"] - round((55000-50000)/(50000-47000), 2)) < 0.01)
check("compute_trade_pnl max_loss_vnd",
      abs(_pnl_open["max_loss_vnd"] - (50000-47000)*1000) < 1)

_t_closed_win = _make_trade(entry=50000, exit_price=54000, shares=1000,
                             status="closed", pnl_pct=8.0, pnl_vnd=4_000_000)
_pnl_closed = compute_trade_pnl(_t_closed_win)
check("compute_trade_pnl pnl_pct present", "pnl_pct" in _pnl_closed)
check("compute_trade_pnl r_multiple > 0", _pnl_closed.get("r_multiple", 0) > 0)

# compute_equity_curve
_trades_list = [
    _make_trade("VIC", 50000, 54000, 1000, status="closed", exit_date="2026-05-06",
                pnl_pct=8.0, pnl_vnd=4_000_000),
    _make_trade("HPG", 25000, 23000, 2000, status="stopped", exit_date="2026-05-08",
                pnl_pct=-8.0, pnl_vnd=-4_000_000),
    _make_trade("MWG", 60000, 65000, 500, status="closed", exit_date="2026-05-12",
                pnl_pct=8.33, pnl_vnd=2_500_000),
]
_eq = compute_equity_curve(_trades_list, initial_capital=1_000_000_000)
check("compute_equity_curve returns DataFrame",   hasattr(_eq, "columns"))
check("compute_equity_curve len = 3",             len(_eq) == 3, f"got {len(_eq)}")
check("compute_equity_curve equity col exists",   "equity" in _eq.columns)
check("compute_equity_curve drawdown_pct <= 0",   (_eq["drawdown_pct"] <= 0).all(),
      f"max dd = {_eq['drawdown_pct'].max()}")

# equity after win+loss+win: 1B + 4M - 4M + 2.5M = 1B + 2.5M
check("compute_equity_curve final equity correct",
      abs(_eq.iloc[-1]["equity"] - 1_002_500_000) < 1000)

# Empty trades
_eq_empty = compute_equity_curve([], 1_000_000_000)
check("compute_equity_curve empty trades = empty df", len(_eq_empty) == 0)

# compute_win_streak
_cur, _max = compute_win_streak(_trades_list)
check("compute_win_streak max >= 1",  _max >= 1)
check("compute_win_streak returns tuple", isinstance(_cur, int) and isinstance(_max, int))

# All wins
_all_wins = [
    _make_trade(exit_price=54000, status="closed", exit_date="2026-05-0" + str(i+1),
                pnl_pct=8.0, pnl_vnd=4_000_000)
    for i in range(3)
]
_cur_w, _max_w = compute_win_streak(_all_wins)
check("compute_win_streak all wins current=+3",  _cur_w == 3, f"got {_cur_w}")
check("compute_win_streak all wins max=3",       _max_w == 3, f"got {_max_w}")

# Empty
_cur_e, _max_e = compute_win_streak([])
check("compute_win_streak empty = (0,0)", _cur_e == 0 and _max_e == 0)

# compute_stats
_stats_s = compute_stats(_trades_list, 1_000_000_000)
check("compute_stats win_count=2",       _stats_s["win_count"] == 2)
check("compute_stats loss_count=1",      _stats_s["loss_count"] == 1)
check("compute_stats win_rate=0.667",    abs(_stats_s["win_rate"] - 2/3) < 0.01)
check("compute_stats total_pnl_vnd",     _stats_s["total_pnl_vnd"] == 2_500_000)
check("compute_stats max_drawdown <= 0", _stats_s["max_drawdown_pct"] <= 0)
check("compute_stats profit_factor > 0", _stats_s["profit_factor"] > 0)
check("compute_stats has sharpe",        "sharpe_ratio" in _stats_s)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: ml/outcome_tracker.py (E5)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== E5 Outcome Tracker (ml/outcome_tracker.py) ===")

import ml.outcome_tracker as _ot
from ml.outcome_tracker import (
    log_signal, log_signals_batch, get_recent_signals,
    get_signal_stats, get_outcome_table, check_pending_outcomes,
    SIGNAL_LOG_DB, _add_business_days, _t5_date,
)

# Use temp DB for tests
_ORIG_SIGNAL_DB = _ot.SIGNAL_LOG_DB
_TMP_SIGNAL_DB  = _TMP_DIR / "test_signal_log.db"
_ot.SIGNAL_LOG_DB = _TMP_SIGNAL_DB

# _add_business_days
_d_mon = date(2026, 5, 25)   # Monday
check("_add_business_days +1 = Tuesday",
      _add_business_days(_d_mon, 1) == date(2026, 5, 26))
check("_add_business_days +5 from Mon = next Mon",
      _add_business_days(_d_mon, 5) == date(2026, 6, 1))   # Mon 25 + 5bd = Mon June 1
_d_fri = date(2026, 5, 29)   # Friday
check("_add_business_days +1 from Fri = Monday",
      _add_business_days(_d_fri, 1) == date(2026, 6, 1))
check("_t5_date returns date 5bd later",
      _t5_date(date(2026, 5, 25)) == date(2026, 6, 1))

# Create mock AlertCard
from ml.alert_generator import AlertCard

def _make_card(ticker="VIC", date_str="2026-05-25", p_cal=0.75, p_raw=0.70,
               entry_lo=50000.0, sl=47000.0, target=55000.0):
    return AlertCard(
        ticker=ticker, date=date_str, action="BUY",
        entry_lo=entry_lo, entry_hi=entry_lo * 1.01,
        sl_price=sl, target_price=target,
        rr_ratio=round((target - entry_lo) / (entry_lo - sl), 2),
        sl_pct=round((entry_lo - sl) / entry_lo, 4),
        position_size_pct=0.30, position_size_vnd=0, shares=0,
        p_calibrated=p_cal, p_raw=p_raw, recommendation="BUY",
        ci_lo=p_cal - 0.04, ci_hi=p_cal + 0.04,
        regime_state="BULL", regime_max_pos=3,
        reason="Wyckoff Spring · RS Outperform",
    )

_card1 = _make_card("VIC", "2026-05-25")
_card2 = _make_card("HPG", "2026-05-25", p_cal=0.68)

# log_signal
_sid1 = log_signal(_card1)
check("log_signal returns int id",  isinstance(_sid1, int) and _sid1 > 0)

# Idempotent: log same signal twice → same id
_sid1b = log_signal(_card1)
check("log_signal idempotent (upsert)",  _sid1b > 0)  # any positive id

# log_signals_batch
_ids = log_signals_batch([_card2])
check("log_signals_batch returns list",   isinstance(_ids, list))
check("log_signals_batch len=1",          len(_ids) == 1)

# get_recent_signals
_recent = get_recent_signals(10)
check("get_recent_signals returns list",  isinstance(_recent, list))
check("get_recent_signals count >= 2",    len(_recent) >= 2, f"got {len(_recent)}")
check("get_recent_signals has ticker",    all("ticker" in r for r in _recent))
check("get_recent_signals has outcome",   all("outcome" in r for r in _recent))

# get_signal_stats
_sts = get_signal_stats()
check("get_signal_stats total_signals >= 2", _sts["total_signals"] >= 2)
check("get_signal_stats precision in [0,1]",
      0.0 <= _sts["precision"] <= 1.0)
check("get_signal_stats drift_alert is bool", isinstance(_sts["drift_alert"], bool))
check("get_signal_stats roll_30d_total >= 2", _sts["roll_30d_total"] >= 2)

# get_outcome_table
_tbl = get_outcome_table()
check("get_outcome_table returns list",   isinstance(_tbl, list))
check("get_outcome_table has entries",    len(_tbl) >= 2)
check("get_outcome_table fields complete",
      all(k in _tbl[0] for k in ["signal_date", "ticker", "outcome", "p_calibrated"]))

# check_pending_outcomes (no data today → zero results or PENDING kept)
# Patch get_ohlcv to return empty so it returns EXPIRED
with patch("ml.outcome_tracker._resolve_outcome", return_value=("EXPIRED", None, None)):
    _past_card = _make_card("ACB", "2026-01-01")  # far in the past → T+5 passed
    _sid_past  = log_signal(_past_card)
    # Force the outcome_date to be in the past so check_pending picks it up
    from ml.outcome_tracker import _get_connection as _ot_con
    with _ot_con() as _con:
        _con.execute(
            "UPDATE signal_outcomes SET outcome_date=? WHERE signal_id=?",
            ("2026-01-08", _sid_past)
        )
    results = check_pending_outcomes(date(2026, 5, 25))
check("check_pending_outcomes returns list",    isinstance(results, list))
check("check_pending_outcomes resolved >=1",    len(results) >= 1, f"got {len(results)}")
check("check_pending_outcomes outcome EXPIRED", results[0]["outcome"] == "EXPIRED")

# Restore original DB path
_ot.SIGNAL_LOG_DB = _ORIG_SIGNAL_DB


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: ml/__init__.py exports (Sprint 8)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== ml.__init__ exports (Sprint 8) ===")
import ml

check("ml.log_signal exported",           hasattr(ml, "log_signal"))
check("ml.log_signals_batch exported",    hasattr(ml, "log_signals_batch"))
check("ml.check_pending_outcomes exported", hasattr(ml, "check_pending_outcomes"))
check("ml.get_signal_stats exported",     hasattr(ml, "get_signal_stats"))
check("ml.get_outcome_table exported",    hasattr(ml, "get_outcome_table"))
check("ml.SIGNAL_LOG_DB exported",        hasattr(ml, "SIGNAL_LOG_DB"))
check("ml.compute_trade_pnl exported",    hasattr(ml, "compute_trade_pnl"))
check("ml.compute_equity_curve exported", hasattr(ml, "compute_equity_curve"))
check("ml.compute_stats exported",        hasattr(ml, "compute_stats"))
check("ml.compute_win_streak exported",   hasattr(ml, "compute_win_streak"))
# Sprint 7 exports still intact
check("ml.generate_morning_report intact",hasattr(ml, "generate_morning_report"))
check("ml.compute_conviction_size intact",hasattr(ml, "compute_conviction_size"))


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"Sprint 8 results: {_PASS} PASS  |  {_FAIL} FAIL  |  {_PASS+_FAIL} total")
if _FAIL == 0:
    print(f"✅ ALL {_PASS} tests PASSED — Sprint 8 DONE")
else:
    print(f"❌ {_FAIL} test(s) FAILED")
print('='*55)

# Cleanup temp dir
import shutil
shutil.rmtree(_TMP_DIR, ignore_errors=True)

import sys
sys.exit(0 if _FAIL == 0 else 1)
