"""
_test_sprint7.py
Sprint 7 — D3.4/D3.5 Position Sizer + Alert Generator unit tests
compute_conviction_size · AlertCard · _build_reason · PortfolioUsage · generate_morning_report · ml.__init__ exports
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "d:/TOOL-PYTHON/smartmoney-analysic")

import numpy as np

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
# Section 1: compute_conviction_size (D3.4 Sprint 7)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== compute_conviction_size (D3.4) ===")
from ml.portfolio_sizing import compute_conviction_size

# p >= 0.70 → base=0.30, capped by regime
_cv_p85_2 = compute_conviction_size(0.85, regime_max_positions=2)
check("CS p=0.85, max_pos=2 → 0.30 (regime_cap=0.50 > 0.30, no cap)",
      abs(_cv_p85_2 - 0.30) < 1e-6,
      f"got {_cv_p85_2}")
_cv_p80_5 = compute_conviction_size(0.80, regime_max_positions=5)
check("CS p=0.80, max_pos=5 → capped to 0.20 (1/5)",
      abs(_cv_p80_5 - 0.20) < 1e-6,
      f"got {_cv_p80_5}")

_cv_p80_3 = compute_conviction_size(0.80, regime_max_positions=3)
check("CS p=0.80, max_pos=3 -> 0.30 (regime_cap=0.333 > 0.30, no cap)",
      abs(_cv_p80_3 - 0.30) < 1e-6,
      f"got {_cv_p80_3}")

_cv_p75_2 = compute_conviction_size(0.75, regime_max_positions=2)
check("CS p=0.75, max_pos=2 → 0.30 (1/2=0.50 > 0.30)",
      abs(_cv_p75_2 - 0.30) < 1e-6,
      f"got {_cv_p75_2}")

_cv_p65_2 = compute_conviction_size(0.65, regime_max_positions=2)
check("CS p=0.65, max_pos=2 → 0.20 (standard, below 0.70 threshold)",
      abs(_cv_p65_2 - 0.20) < 1e-6,
      f"got {_cv_p65_2}")

_cv_p70_5 = compute_conviction_size(0.70, regime_max_positions=5)
check("CS p=0.70 (exactly at threshold) → 0.20 (regime cap 1/5)",
      abs(_cv_p70_5 - 0.20) < 1e-6,
      f"got {_cv_p70_5}")

_cv_p50_5 = compute_conviction_size(0.50, regime_max_positions=5)
check("CS p=0.50 → 0.20 (standard, capped by 1/5=0.20)",
      abs(_cv_p50_5 - 0.20) < 1e-6,
      f"got {_cv_p50_5}")

_cv_p90_1 = compute_conviction_size(0.90, regime_max_positions=1)
check("CS p=0.90, max_pos=1 → 1.0 (1/1=1.0 > 0.30, returns 0.30)",
      abs(_cv_p90_1 - 0.30) < 1e-6,
      f"got {_cv_p90_1}")

_cv_clamped = compute_conviction_size(1.5, regime_max_positions=5)
check("CS p=1.5 (clipped to 1.0) → valid result",
      0.0 <= _cv_clamped <= 1.0,
      f"got {_cv_clamped}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: AlertCard generation
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== AlertCard generation (D3.5) ===")
from ml.alert_generator import generate_alert, AlertCard

def _make_pick(
    ticker="VIC",
    p_raw=0.72,
    p_cal=0.75,
    rec="BUY",
    entry_lo=48000.0,
    entry_hi=49000.0,
    sl_price=46000.0,
    target_price=52000.0,
    rr_ratio=2.0,
    sl_pct=0.05,
    max_pos=3,
    regime="BULL",
    wyckoff=None,
    sm=None,
    rs=None,
    vc=None,
) -> dict:
    return {
        "ticker":         ticker,
        "probability":    p_raw,
        "p_calibrated":   p_cal,
        "ci_lo":          p_cal - 0.04,
        "ci_hi":          p_cal + 0.04,
        "recommendation": rec,
        "entry": {
            "entry_low":    entry_lo,
            "entry_high":   entry_hi,
            "sl_price":     sl_price,
            "target_price": target_price,
            "rr_ratio":     rr_ratio,
            "sl_pct":       sl_pct,
        },
        "regime": {
            "regime":        regime,
            "max_positions": max_pos,
        },
        "wyckoff": wyckoff or {},
        "sm":      sm      or {"label": "Neutral", "smart_money_score": 0.0},
        "rs":      rs      or {"rs_label": "Neutral", "rs_5d": 0.0},
        "vc":      vc      or {"vol_surge": 1.0, "is_confirmed": False},
    }

_pick1 = _make_pick(p_cal=0.75, max_pos=3)
_alert1 = generate_alert(_pick1, portfolio_capital=0.0)
check("AlertCard returned (not None)",     _alert1 is not None)
check("AlertCard is AlertCard instance",   isinstance(_alert1, AlertCard))
check("AlertCard ticker = VIC",            _alert1.ticker == "VIC")
check("AlertCard action = BUY",            _alert1.action == "BUY")
check("AlertCard entry_lo correct",        abs(_alert1.entry_lo - 48000.0) < 1.0)
check("AlertCard sl_price correct",        abs(_alert1.sl_price - 46000.0) < 1.0)
check("AlertCard target_price correct",    abs(_alert1.target_price - 52000.0) < 1.0)

# p=0.75 ≥ 0.70 → base=0.30, regime_cap=1/3≈0.333 → size=0.30
_expected_size = compute_conviction_size(0.75, 3)
check("AlertCard position_size_pct = conviction size",
      abs(_alert1.position_size_pct - _expected_size) < 1e-6,
      f"got {_alert1.position_size_pct}, expected {_expected_size}")

check("AlertCard position_size_vnd=0 when capital=0",
      _alert1.position_size_vnd == 0.0)

# With capital
_alert_cap = generate_alert(_make_pick(p_cal=0.80, max_pos=2), portfolio_capital=500_000_000.0)
check("AlertCard position_size_vnd > 0 when capital provided",
      _alert_cap is not None and _alert_cap.position_size_vnd > 0)

# Missing entry → returns None
_pick_no_entry = _make_pick()
_pick_no_entry["entry"] = {}
check("AlertCard None when entry missing",  generate_alert(_pick_no_entry) is None)

# as_dict keys
_d = _alert1.as_dict()
check("AlertCard as_dict has all required keys",
      all(k in _d for k in ["ticker", "action", "entry_lo", "sl_price",
                             "position_size_pct", "p_calibrated", "reason"]))


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: _build_reason
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== _build_reason (D3.5) ===")
from ml.alert_generator import _build_reason

# Spring quality high → mentions spring
_pick_spring = _make_pick(
    wyckoff={"spring_quality": 0.80, "lps_detected": False, "wyckoff_score": 0.70,
             "stopping_volume": False, "effort_vs_result": 0.10, "no_supply_count": 0.0}
)
_r_spring = _build_reason(_pick_spring)
check("reason includes 'Spring' when spring_quality >= 0.65",
      "Spring" in _r_spring, f"got: {_r_spring!r}")

# LPS detected
_pick_lps = _make_pick(
    wyckoff={"spring_quality": 0.0, "lps_detected": True, "wyckoff_score": 0.65,
             "stopping_volume": False, "effort_vs_result": 0.0, "no_supply_count": 0.0}
)
_r_lps = _build_reason(_pick_lps)
check("reason includes 'LPS' when lps_detected=True",
      "LPS" in _r_lps, f"got: {_r_lps!r}")

# Smart Money accumulating
_pick_sm = _make_pick(
    wyckoff={"spring_quality": 0.0, "lps_detected": False, "wyckoff_score": 0.3,
             "stopping_volume": False, "effort_vs_result": 0.0, "no_supply_count": 0.0},
    sm={"label": "Accumulating", "smart_money_score": 0.30},
)
_r_sm = _build_reason(_pick_sm)
check("reason includes ngoại/tích lũy when SM Accumulating",
      "tích lũy" in _r_sm or "Acc" in _r_sm, f"got: {_r_sm!r}")

# RS Outperform
_pick_rs = _make_pick(
    wyckoff={"spring_quality": 0.0, "lps_detected": False, "wyckoff_score": 0.3,
             "stopping_volume": False, "effort_vs_result": 0.0, "no_supply_count": 0.0},
    rs={"rs_label": "Outperform", "rs_5d": 2.5},
)
_r_rs = _build_reason(_pick_rs)
check("reason includes RS when rs_label=Outperform and rs_5d>1",
      "RS" in _r_rs, f"got: {_r_rs!r}")

# Fallback when all signals empty
_pick_bare = _make_pick(
    wyckoff={"spring_quality": 0.0, "lps_detected": False, "wyckoff_score": 0.0,
             "stopping_volume": False, "effort_vs_result": 0.0, "no_supply_count": 0.0},
    sm={"label": "Neutral", "smart_money_score": 0.0},
    rs={"rs_label": "Neutral", "rs_5d": 0.0},
    vc={"vol_surge": 0.8, "is_confirmed": False},
)
_r_bare = _build_reason(_pick_bare)
check("reason non-empty even with all signals empty (fallback)",
      len(_r_bare) > 0, f"got: {_r_bare!r}")

# Max 4 parts
_pick_all = _make_pick(
    wyckoff={"spring_quality": 0.80, "lps_detected": True, "wyckoff_score": 0.90,
             "stopping_volume": True, "effort_vs_result": 0.50, "no_supply_count": 0.50},
    sm={"label": "Accumulating", "smart_money_score": 0.50},
    rs={"rs_label": "Outperform", "rs_5d": 3.0},
    vc={"vol_surge": 2.0, "is_confirmed": True},
)
_r_all = _build_reason(_pick_all)
check("reason has at most 4 parts (separated by ·)",
      _r_all.count("·") <= 3, f"got {_r_all.count('·')} dots: {_r_all!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: PortfolioUsage
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== PortfolioUsage (D3.5) ===")
from ml.alert_generator import compute_portfolio_usage, PortfolioUsage

_alerts_empty = []
_usage_empty  = compute_portfolio_usage(_alerts_empty, max_positions=3, capital=0.0)
check("PortfolioUsage empty picks → picks_count=0",  _usage_empty.picks_count == 0)
check("PortfolioUsage empty picks → total_pct=0",    _usage_empty.total_allocated_pct == 0.0)
check("PortfolioUsage empty picks → is_full=False",  not _usage_empty.is_full)
check("PortfolioUsage is PortfolioUsage instance",   isinstance(_usage_empty, PortfolioUsage))

# Build 2 alerts with capital
_p1 = _make_pick("VIC", p_cal=0.75, max_pos=3)
_p2 = _make_pick("HPG", p_cal=0.65, max_pos=3)
_a1 = generate_alert(_p1, portfolio_capital=1_000_000_000)
_a2 = generate_alert(_p2, portfolio_capital=1_000_000_000)
_alerts2 = [a for a in [_a1, _a2] if a is not None]
_usage2   = compute_portfolio_usage(_alerts2, max_positions=3, capital=1_000_000_000)
check("PortfolioUsage 2 picks → picks_count=2",
      _usage2.picks_count == 2, f"got {_usage2.picks_count}")
check("PortfolioUsage total_pct <= 1.0",
      0.0 <= _usage2.total_allocated_pct <= 1.0,
      f"got {_usage2.total_allocated_pct}")
check("PortfolioUsage remaining_pct = 1 - total_pct",
      abs(_usage2.remaining_pct - (1.0 - _usage2.total_allocated_pct)) < 1e-6)
check("PortfolioUsage capital_deployed_vnd > 0",
      _usage2.capital_deployed_vnd > 0)
check("PortfolioUsage capital_remaining_vnd + deployed ≈ capital",
      abs(_usage2.capital_deployed_vnd + _usage2.capital_remaining_vnd - 1_000_000_000) < 1000)

# is_full: 3 picks with max_positions=3
_p3 = _make_pick("MWG", p_cal=0.72, max_pos=3)
_a3 = generate_alert(_p3, portfolio_capital=1_000_000_000)
_alerts3 = [a for a in [_a1, _a2, _a3] if a is not None]
_usage3   = compute_portfolio_usage(_alerts3, max_positions=3, capital=1_000_000_000)
check("PortfolioUsage is_full=True when picks_count >= max_positions",
      _usage3.is_full, f"picks={_usage3.picks_count}, max={_usage3.max_positions}")

# as_dict
_ud = _usage2.as_dict()
check("PortfolioUsage as_dict has all keys",
      all(k in _ud for k in ["picks_count", "max_positions", "total_allocated_pct",
                              "remaining_pct", "capital_deployed_vnd",
                              "capital_remaining_vnd", "is_full"]))


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: generate_morning_report
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== generate_morning_report (D3.5) ===")
from ml.alert_generator import generate_morning_report

_picks_ok = [
    _make_pick("VIC", p_cal=0.80, max_pos=3),
    _make_pick("HPG", p_cal=0.68, max_pos=3),
    _make_pick("MWG", p_cal=0.71, max_pos=3),
]
_report = generate_morning_report(_picks_ok, capital=0.0)
check("generate_morning_report returns list",  isinstance(_report, list))
check("generate_morning_report 3 valid picks → 3 cards",
      len(_report) == 3, f"got {len(_report)}")
check("generate_morning_report cards are AlertCard instances",
      all(isinstance(c, AlertCard) for c in _report))

# With capital
_report_cap = generate_morning_report(_picks_ok, capital=500_000_000)
check("generate_morning_report with capital → cards have vnd > 0",
      all(c.position_size_vnd > 0 for c in _report_cap))

# Empty input
_report_empty = generate_morning_report([], capital=0.0)
check("generate_morning_report [] → []",  _report_empty == [])

# Pick with missing entry → excluded
_pick_no_e = _make_pick("ACB")
_pick_no_e["entry"] = {}
_report_mixed = generate_morning_report([_pick_no_e, _picks_ok[0]], capital=0.0)
check("generate_morning_report: pick with no entry excluded",
      len(_report_mixed) == 1, f"got {len(_report_mixed)}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: ml.__init__ exports
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== ml.__init__ exports (Sprint 7) ===")
import ml

check("ml.compute_conviction_size exported",   hasattr(ml, "compute_conviction_size"))
check("ml.AlertCard exported",                 hasattr(ml, "AlertCard"))
check("ml.PortfolioUsage exported",            hasattr(ml, "PortfolioUsage"))
check("ml.generate_alert exported",            hasattr(ml, "generate_alert"))
check("ml.generate_morning_report exported",   hasattr(ml, "generate_morning_report"))
check("ml.compute_portfolio_usage exported",   hasattr(ml, "compute_portfolio_usage"))

# Smoke test via ml namespace
_alert_via_ml = ml.generate_alert(_picks_ok[0], portfolio_capital=0.0)
check("ml.generate_alert() callable and returns AlertCard",
      isinstance(_alert_via_ml, ml.AlertCard))

_size_via_ml = ml.compute_conviction_size(0.75, regime_max_positions=2)
check("ml.compute_conviction_size(0.75, 2) = 0.30",
      abs(_size_via_ml - 0.30) < 1e-6, f"got {_size_via_ml}")

# Sprint 6 exports still intact
check("ml.Recommendation still exported",      hasattr(ml, "Recommendation"))
check("ml.wilson_ci still exported",           hasattr(ml, "wilson_ci"))
check("ml.load_calibrator still exported",     hasattr(ml, "load_calibrator"))


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"Sprint 7 results: {_PASS} PASS  |  {_FAIL} FAIL  |  {_PASS+_FAIL} total")
if _FAIL == 0:
    print(f"✅ ALL {_PASS} tests PASSED — Sprint 7 DONE")
else:
    print(f"❌ {_FAIL} test(s) FAILED")
print('='*55)
sys.exit(0 if _FAIL == 0 else 1)
