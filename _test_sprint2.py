"""
_test_sprint2.py — Unit tests for Sprint 2: S2 + S5 + D3.2 + D3.3
"""
import sys, pathlib, datetime
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import pandas as pd

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _make_ohlcv(n=60, start_price=20_000.0, trend=0.001, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    close = start_price * np.cumprod(1 + rng.normal(trend, 0.01, n))
    high  = close * (1 + rng.uniform(0.005, 0.015, n))
    low   = close * (1 - rng.uniform(0.005, 0.015, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    vol   = rng.integers(100_000, 5_000_000, n).astype(float)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": open_, "high": high, "low": low, "close": close, "volume": vol,
    })


def _make_clean_ohlcv(n=80, start_price=20_000.0, daily_return=0.002) -> pd.DataFrame:
    """Perfectly trending OHLCV — no noise, deterministic for threshold tests."""
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    close = start_price * np.cumprod(np.full(n, 1 + daily_return))
    high  = close * 1.01
    low   = close * 0.99
    open_ = np.roll(close, 1); open_[0] = start_price
    vol   = np.full(n, 1_000_000.0)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": open_, "high": high, "low": low, "close": close, "volume": vol,
    })


# Clean (noise-free) data for label threshold tests
vn30_clean   = _make_clean_ohlcv(n=80, start_price=1200.0, daily_return=0.0005)
stock_strong = _make_clean_ohlcv(n=80, start_price=25_000.0, daily_return=0.005)   # strong outperform
stock_weak   = _make_clean_ohlcv(n=80, start_price=15_000.0, daily_return=-0.005)  # underperform

# Noisy data for rank / SL tests
vn30_df    = _make_ohlcv(n=80, start_price=1200.0, trend=0.0005, seed=0)
stock_bull = _make_ohlcv(n=80, start_price=25_000.0, trend=0.003, seed=1)
stock_bear = _make_ohlcv(n=80, start_price=15_000.0, trend=-0.003, seed=2)
stock_neut = _make_ohlcv(n=80, start_price=18_000.0, trend=0.0005, seed=3)

print("=" * 60)
print("Sprint 2 Tests")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# T1: compute_stock_rs — basic
# ─────────────────────────────────────────────────────────────────────────────
from ml.relative_strength import compute_stock_rs, rank_by_rs, RSInfo

rs_strong = compute_stock_rs(stock_strong, vn30_clean, ticker="STRONG")
rs_weak   = compute_stock_rs(stock_weak,   vn30_clean, ticker="WEAK")
rs_bull   = compute_stock_rs(stock_bull, vn30_df, ticker="BULL")
rs_bear   = compute_stock_rs(stock_bear, vn30_df, ticker="BEAR")
rs_neut   = compute_stock_rs(stock_neut, vn30_df, ticker="NEUT")

assert rs_strong is not None, "T1a: strong RS should not be None"
assert rs_weak   is not None, "T1b: weak RS should not be None"

# Use clean (no-noise) data for label threshold assertions
assert rs_strong.rs_score > rs_weak.rs_score, (
    f"T1c: strong RS score {rs_strong.rs_score:.4f} should be > weak {rs_weak.rs_score:.4f}"
)
assert rs_strong.rs_label == "Outperform",   f"T1d: strong should Outperform, got {rs_strong.rs_label} (score={rs_strong.rs_score:.4f})"
assert rs_weak.rs_label   == "Underperform", f"T1e: weak should Underperform, got {rs_weak.rs_label} (score={rs_weak.rs_score:.4f})"

# Use noisy data for ordering assertions (less strict)
assert rs_bull is not None and rs_bear is not None and rs_neut is not None
assert rs_bull.rs_score > rs_bear.rs_score, (
    f"T1f: bull RS score {rs_bull.rs_score:.4f} should > bear {rs_bear.rs_score:.4f}"
)
print(f"T1 compute_stock_rs: STRONG={rs_strong.rs_score:+.4f} ({rs_strong.rs_label}), "
      f"WEAK={rs_weak.rs_score:+.4f} ({rs_weak.rs_label}), "
      f"BULL={rs_bull.rs_score:+.4f} ({rs_bull.rs_label})")

# ─────────────────────────────────────────────────────────────────────────────
# T2: compute_stock_rs — insufficient data → None
# ─────────────────────────────────────────────────────────────────────────────
short_df = _make_ohlcv(n=10, seed=99)
rs_short = compute_stock_rs(short_df, vn30_df, ticker="SHORT")
assert rs_short is None, "T2: < 21 phien phai tra ve None"
print("T2 insufficient data -> None: PASS")

# ─────────────────────────────────────────────────────────────────────────────
# T3: rank_by_rs — cross-sectional percentile
# ─────────────────────────────────────────────────────────────────────────────
rs_list = [rs_bull, rs_bear, rs_neut]
ranked  = rank_by_rs(rs_list)
ranks = {r.ticker: r.rs_rank for r in ranked}
assert ranks["BULL"] > ranks["NEUT"], f"T3a: BULL rank {ranks['BULL']:.2f} should > NEUT {ranks['NEUT']:.2f}"
assert ranks["BULL"] > ranks["BEAR"], f"T3b: BULL rank {ranks['BULL']:.2f} should > BEAR {ranks['BEAR']:.2f}"
assert abs(max(ranks.values()) - 1.0) < 1e-6, f"T3c: max rank should be 1.0, got {max(ranks.values())}"
assert abs(min(ranks.values()) - 0.0) < 1e-6, f"T3d: min rank should be 0.0, got {min(ranks.values())}"

print(f"T3 rank_by_rs: BULL={ranks['BULL']:.2f}, NEUT={ranks['NEUT']:.2f}, BEAR={ranks['BEAR']:.2f} (PASS)")

# ─────────────────────────────────────────────────────────────────────────────
# T4: rank_by_rs — single element edge case
# ─────────────────────────────────────────────────────────────────────────────
single = [compute_stock_rs(stock_bull, vn30_df, ticker="X")]
ranked1 = rank_by_rs(single)
assert ranked1[0].rs_rank == 0.5, f"T4: single element rank should be 0.5, got {ranked1[0].rs_rank}"
print("T4 rank_by_rs single element: PASS")

# ─────────────────────────────────────────────────────────────────────────────
# T5: compute_entry_zone — basic
# ─────────────────────────────────────────────────────────────────────────────
from ml.entry_timing import compute_entry_zone, EntryZone, MAX_SL_PCT

entry = compute_entry_zone(stock_bull)
assert entry is not None, "T5a: entry zone should not be None"
assert entry.entry_low < entry.entry_high, "T5b: entry_low < entry_high"
assert entry.sl_price < entry.entry_low, "T5c: SL must be below entry_low"
assert entry.target_price > entry.entry_high, "T5d: target must be above entry_high"
assert 0 < entry.sl_pct <= MAX_SL_PCT + 0.001, f"T5e: sl_pct={entry.sl_pct:.4f} must be <= {MAX_SL_PCT}"
assert entry.rr_ratio > 0, "T5f: rr_ratio must be positive"

print(f"T5 compute_entry_zone: entry={entry.entry_low:,.0f}-{entry.entry_high:,.0f}, "
      f"SL={entry.sl_price:,.0f} (-{entry.sl_pct:.1%}), "
      f"target={entry.target_price:,.0f}, R:R=1:{entry.rr_ratio:.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# T6: EntryZone.is_valid() — D3.3 filter
# ─────────────────────────────────────────────────────────────────────────────
assert entry.is_valid(), f"T6a: bull entry should be valid (sl_pct={entry.sl_pct:.2%} rr={entry.rr_ratio:.2f})"

# Tạo entry zone với SL quá rộng (vượt MAX_SL_PCT)
wide_entry = EntryZone(
    entry_low=20_000, entry_high=20_060,
    sl_price=18_000,              # sl_pct = (20030 - 18000)/20030 ≈ 10.1%
    target_price=21_030,
    sl_pct=0.101, rr_ratio=0.50,
    support=18_100, resistance=22_000, atr=200, in_entry_zone=True,
)
assert not wide_entry.is_valid(), f"T6b: wide SL entry should be invalid (sl={wide_entry.sl_pct:.1%})"

# Tạo entry với R:R thấp
low_rr_entry = EntryZone(
    entry_low=20_000, entry_high=20_060,
    sl_price=19_500, target_price=20_600,
    sl_pct=0.026, rr_ratio=0.50,   # R:R clearly below MIN_RR=0.80
    support=19_500, resistance=21_000, atr=150, in_entry_zone=True,
)
assert not low_rr_entry.is_valid(), "T6c: R:R < 0.80 should be invalid"

print("T6 EntryZone.is_valid() D3.3 filter: PASS")

# ─────────────────────────────────────────────────────────────────────────────
# T7: compute_entry_zone — insufficient data → None
# ─────────────────────────────────────────────────────────────────────────────
entry_short = compute_entry_zone(_make_ohlcv(n=5, seed=10))
assert entry_short is None, "T7: < 15 phien phai tra ve None"
print("T7 insufficient data -> None: PASS")

# ─────────────────────────────────────────────────────────────────────────────
# T8: rs_info.as_dict() — serializable
# ─────────────────────────────────────────────────────────────────────────────
import json
d = rs_bull.as_dict()
assert "ticker" in d and "rs_score" in d and "rs_rank" in d
json_str = json.dumps(d)   # phải không throw
assert "BULL" in json_str
print("T8 RSInfo.as_dict() serializable: PASS")

# ─────────────────────────────────────────────────────────────────────────────
# T9: entry.as_dict() — includes is_valid
# ─────────────────────────────────────────────────────────────────────────────
ed = entry.as_dict()
assert "is_valid" in ed, "T9a: as_dict phải có is_valid"
assert "sl_price" in ed and "target_price" in ed
json.dumps(ed)    # serializable
print("T9 EntryZone.as_dict() serializable: PASS")

# ─────────────────────────────────────────────────────────────────────────────
print()
print("ALL SPRINT 2 TESTS PASSED")
