"""
retrain_v5.py — Sprint 6: Retrain model v5_calibrated với 35 features.

Chạy: python retrain_v5.py
"""

from __future__ import annotations

import sys
import time
import logging
import pickle
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("retrain_v5")

# ── Verify working dir ────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Imports ──────────────────────────────────────────────────────────────────
from config.constants import VN30_TICKERS
from ml.dataset_builder import build_dataset
from ml.model import (
    MODEL_LABEL_VERSION,
    CALIBRATOR_PATH,
    FEATURE_COLS,
    train_model,
    load_calibrator,
)
from ml.backtest import run_backtest
from ml.feature_engineering import FEATURE_COLS as FE_FEATURE_COLS

# ── Sanity checks ─────────────────────────────────────────────────────────────
assert len(FE_FEATURE_COLS) == 35, f"Expected 35 features, got {len(FE_FEATURE_COLS)}"
log.info("MODEL_LABEL_VERSION = %s", MODEL_LABEL_VERSION)
log.info("Feature count       = %d", len(FE_FEATURE_COLS))
log.info("Tickers             = %d (VN30)", len(VN30_TICKERS))

# ── Build dataset ─────────────────────────────────────────────────────────────
log.info("=" * 55)
log.info("STEP 1 — Build dataset (period=6m, %d tickers)...", len(VN30_TICKERS))
t0 = time.time()

last_pct = [-1.0]


def _progress(pct: float, msg: str) -> None:
    if pct - last_pct[0] >= 0.05 or pct >= 0.99:
        log.info("  [%.0f%%] %s", pct * 100, msg)
        last_pct[0] = pct


dataset = build_dataset(
    tickers=VN30_TICKERS,
    period="6m",
    progress_callback=_progress,
)

elapsed = time.time() - t0
log.info("Dataset built: %d rows | %d tickers | %.1fs", len(dataset), dataset["ticker"].nunique(), elapsed)

if dataset.empty:
    log.error("Dataset rỗng — kiểm tra kết nối API")
    sys.exit(1)

n_pos = int(dataset["label"].sum())
n_tot = len(dataset)
log.info("Label rate: %.1f%% (y=1 = forward return ≥ +5%%)", 100 * n_pos / n_tot)

# ── Train model ───────────────────────────────────────────────────────────────
log.info("=" * 55)
log.info("STEP 2 — Train model (v5_calibrated, 35 features)...")
t1 = time.time()

result = train_model(dataset, save=True)

elapsed2 = time.time() - t1
m = result["metrics"]

log.info("Training done in %.1fs", elapsed2)
log.info("  model_type   : %s", m["model_type"])
log.info("  cv_auc_mean  : %.4f ± %.4f", m["cv_auc_mean"], m["cv_auc_std"])
log.info("  cv_prec_mean : %.4f", m["cv_prec_mean"])
log.info("  n_samples    : %d", m["n_samples"])
log.info("  n_features   : %s", m.get("n_features", "?"))
log.info("  label_version: %s", m.get("label_version", "?"))

# Calibration info
if "calibrated" in m:
    log.info("  calibrated   : %s", m["calibrated"])
    log.info("  cal_n_samples: %d", m.get("cal_n_samples", 0))
else:
    log.warning("  calibrated key NOT in metrics!")

# Fold details
for fd in m.get("fold_details", []):
    log.info(
        "  Fold %d — AUC %.3f | Precision %.3f | Recall %.3f",
        fd["fold"], fd["auc"], fd["precision"], fd["recall"],
    )

# ── Verify calibrator saved ───────────────────────────────────────────────────
log.info("=" * 55)
log.info("STEP 3 — Verify artifacts...")

cal = load_calibrator()
if cal is not None:
    log.info("  alpha_calibrator.pkl : LOADED OK — %s", type(cal).__name__)
else:
    log.warning("  alpha_calibrator.pkl : NOT FOUND — calibration sẽ không hoạt động")

artifacts_dir = ROOT / "ml" / "artifacts"
for f in sorted(artifacts_dir.glob("*.pkl")):
    log.info("  %s  (%d KB)", f.name, f.stat().st_size // 1024)

# ── In-sample backtest v2 ─────────────────────────────────────────────────────
log.info("=" * 55)
log.info("STEP 4 — Backtest v2 (in-sample, precision_target=0.35)...")

model_obj   = result["model"]
scaler_obj  = result["scaler"]
cal_obj     = result.get("calibrator")

from sklearn.preprocessing import StandardScaler  # already a dependency

bt = run_backtest(
    dataset=dataset,
    model=model_obj,
    scaler=scaler_obj,
    feature_cols=FE_FEATURE_COLS,
    min_prob=0.65,
    precision_target=0.35,
    calibrator=cal_obj,
)

s = bt.summary()
_prec = bt.precision
_meets = bt.meets_target
log.info("  total_trades    : %d", s["total_trades"])
log.info("  precision       : %.3f  (target >= 0.35)", _prec)
log.info("  avg_return      : %s", s.get("avg_return", "n/a"))
log.info("  meets_target    : %s", _meets)

if _meets:
    log.info("  PASS - Precision >= 0.35")
else:
    log.warning("  FAIL - Precision %.3f < 0.35", _prec)

# ── Summary ───────────────────────────────────────────────────────────────────
log.info("=" * 55)
log.info("SPRINT 6 RETRAIN COMPLETE")
log.info("  Model version: %s", MODEL_LABEL_VERSION)
log.info("  Features     : %d", len(FE_FEATURE_COLS))
log.info("  Calibrator   : %s", "OK" if cal is not None else "MISSING")
log.info("  Backtest     : precision=%.3f  meets_target=%s", _prec, _meets)
log.info("")
log.info("Bước tiếp theo:")
log.info("  1. Khởi động Streamlit: streamlit run Home.py")
log.info("  2. Vào trang Alpha Signals → chạy prediction")
log.info("  3. Kiểm tra p_calibrated, CI, Recommendation badge")
