"""Quick sanity check for ml/trade_log.py"""
import sys, tempfile, sqlite3
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent))

import ml.trade_log as tl

# Redirect to tmp DB
tmp = Path(tempfile.mkdtemp())
tl.TRADE_LOG_DB   = tmp / "test.db"
tl._ARTIFACTS_DIR = tmp

picks = [
    {"ticker": "VNM", "probability": 0.72, "p_calibrated": 0.68,
     "pattern": "Spring", "confidence": "high",
     "entry": {"entry_low": 79000, "entry_high": 80000}},
    {"ticker": "VIC", "probability": 0.65, "p_calibrated": 0.60,
     "pattern": "LPS", "confidence": "medium",
     "entry": {"entry_low": 50000, "entry_high": 51000}},
]
today = date.today().isoformat()

n1 = tl.log_alpha_signals(picks, signal_date=today)
assert n1 == 2, f"Expected 2 inserts, got {n1}"
print(f"[OK] insert: {n1}")

n2 = tl.log_alpha_signals(picks, signal_date=today)
assert n2 == 0, f"Expected 0 (duplicate), got {n2}"
print(f"[OK] duplicate ignored: {n2}")

df = tl.get_track_record(days=5)
assert len(df) == 2, f"Expected 2 rows, got {len(df)}"
assert set(df["outcome"]) == {"PENDING"}
print(f"[OK] get_track_record: {len(df)} rows, outcome=PENDING")

stats = tl.get_summary_stats()
assert stats["total"]    == 2
assert stats["pending"]  == 2
assert stats["resolved"] == 0
print(f"[OK] get_summary_stats: {stats}")

# entry_price midpoint
with sqlite3.connect(str(tl.TRADE_LOG_DB)) as con:
    row = con.execute("SELECT entry_price FROM alpha_signals WHERE ticker='VNM'").fetchone()
assert row and abs(row[0] - 79500.0) < 1, f"entry_price VNM expected ~79500, got {row}"
print(f"[OK] entry_price midpoint VNM = {row[0]}")

wwr = tl.get_weekly_win_rate()
assert hasattr(wwr, "columns")
print(f"[OK] get_weekly_win_rate: {len(wwr)} rows")

tck = tl.get_ticker_stats()
assert hasattr(tck, "columns")
print(f"[OK] get_ticker_stats: {len(tck)} rows (expected 0 since no WIN/LOSS)")

# pending count
pc = tl.get_pending_count()
assert pc == 2
print(f"[OK] get_pending_count: {pc}")

print("\n=== trade_log sanity check: ALL PASSED ===")
