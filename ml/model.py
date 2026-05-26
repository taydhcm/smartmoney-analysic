"""
ml/model.py
Train, evaluate và persist GBM model dự đoán alpha signal.

Strategy:
  - LightGBM (preferred) → XGBoost → RandomForest (fallback)
  - TimeSeriesSplit cross-validation (không shuffle để tránh look-ahead)
  - scale_pos_weight để xử lý class imbalance (y=1 thường ~10-20%)
"""

from __future__ import annotations

import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV  # noqa: F401 (kept for reference)
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .feature_engineering import FEATURE_COLS

log = logging.getLogger(__name__)

# ── Artifact paths ─────────────────────────────────────────────────────────────
_ARTIFACTS_DIR  = Path(__file__).parent / "artifacts"
MODEL_PATH      = _ARTIFACTS_DIR / "alpha_model.pkl"
SCALER_PATH     = _ARTIFACTS_DIR / "alpha_scaler.pkl"
META_PATH       = _ARTIFACTS_DIR / "alpha_meta.pkl"
CALIBRATOR_PATH = _ARTIFACTS_DIR / "alpha_calibrator.pkl"  # Isotonic calibrator (v5)

# ── Model versioning ────────────────────────────────────────────────────────────
# Tăng khi thay đổi: label definition, feature list, hoặc training schema.
# Artifact có version khác → bị reject tự động trong load_model().
MODEL_LABEL_VERSION = "v6_calibrated"  # v6.0: +3 Proprietary flow features (38 total) + SSI iBoard


# ── Model factory ──────────────────────────────────────────────────────────────

def _make_model(pos_weight: float = 5.0):
    """
    Tạo GBM model với hyperparameter đã tuned cho VN30 alpha prediction.
    Thử LightGBM → XGBoost → RandomForest theo thứ tự.
    """
    try:
        import lightgbm as lgb  # noqa: F401
        return lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=24,
            min_child_samples=15,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=pos_weight,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
    except ImportError:
        log.info("LightGBM không khả dụng, thử XGBoost...")

    try:
        from xgboost import XGBClassifier  # noqa: F401
        return XGBClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.75,
            colsample_bytree=0.7,
            scale_pos_weight=pos_weight,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False,
            n_jobs=-1,
        )
    except ImportError:
        log.info("XGBoost không khả dụng, dùng RandomForest fallback...")

    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


# ── Training ───────────────────────────────────────────────────────────────────

def train_model(dataset: pd.DataFrame, save: bool = True) -> dict:
    """
    Train model trên dataset, cross-validate với TimeSeriesSplit.

    Parameters
    ----------
    dataset : Output của build_dataset() với cột FEATURE_COLS + "label".
    save    : Nếu True, lưu model + scaler vào ml/artifacts/.

    Returns
    -------
    dict với keys: model, scaler, metrics, feature_importance.
    """
    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Sort by date để TimeSeriesSplit có ý nghĩa (no look-ahead)
    dataset = dataset.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Chỉ giữ rows có label + features đầy đủ
    # Đồng thời loại rows có inf (pct_change từ 0, division edge cases)
    feat_data = dataset[FEATURE_COLS + ["label"]].copy()
    feat_data = feat_data.replace([np.inf, -np.inf], np.nan).dropna()
    X_raw = feat_data[FEATURE_COLS].values.astype(np.float64)
    # Clip outliers trước khi ép sang float32 để tránh overflow
    X_raw = np.clip(X_raw, -1e6, 1e6)
    X_raw = X_raw.astype(np.float32)
    y     = feat_data["label"].values.astype(int)

    if len(X_raw) < 50:
        raise ValueError(f"Không đủ dữ liệu để train: chỉ có {len(X_raw)} rows")

    n_pos   = y.sum()
    n_neg   = len(y) - n_pos
    pos_rate = float(n_pos / max(len(y), 1))
    pos_weight = max(n_neg / max(n_pos, 1), 1.0)

    log.info(
        "Training: %d samples | pos=%d (%.1f%%) | pos_weight=%.1f",
        len(X_raw), n_pos, 100 * pos_rate, pos_weight,
    )

    # Scale
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # TimeSeriesSplit cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    fold_metrics: list[dict] = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        fold_model = _make_model(pos_weight=pos_weight)
        fold_model.fit(X_tr, y_tr)

        probs = fold_model.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.65).astype(int)

        auc       = roc_auc_score(y_val, probs) if y_val.sum() > 0 else 0.5
        precision = precision_score(y_val, preds, zero_division=0)
        recall    = recall_score(y_val, preds, zero_division=0)

        fold_metrics.append({"fold": fold + 1, "auc": auc, "precision": precision, "recall": recall})
        log.info("Fold %d: AUC=%.3f  Precision=%.2f  Recall=%.2f", fold + 1, auc, precision, recall)

    # Final model trên toàn bộ data
    final_model = _make_model(pos_weight=pos_weight)
    final_model.fit(X_scaled, y)

    # ── Isotonic Regression Calibration (v5.0, Sprint 6) ─────────────────────
    # Calibrate trên 20% cuối (chronological holdout) — cv='prefit' vì
    # final_model đã fit toàn bộ data trước rồi.
    calibrator = None
    cal_n = max(30, len(X_scaled) // 5)
    X_cal_hold = X_scaled[-cal_n:]
    y_cal_hold = y[-cal_n:]
    if y_cal_hold.sum() >= 3 and (len(y_cal_hold) - y_cal_hold.sum()) >= 3:
        try:
            # Dùng IsotonicRegression trực tiếp (tương đương CalibratedClassifierCV isotonic).
            # Map raw_prob → calibrated_prob trên holdout set.
            raw_probs_hold = final_model.predict_proba(X_cal_hold)[:, 1]
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(raw_probs_hold, y_cal_hold)
            log.info("Isotonic calibrator fitted on %d holdout samples", cal_n)
        except Exception as exc:
            log.warning("Calibration failed (non-critical): %s", exc)
            calibrator = None

    # Feature importance
    feat_imp: dict[str, float] = {}
    if hasattr(final_model, "feature_importances_"):
        raw_imp = final_model.feature_importances_
        total   = raw_imp.sum() + 1e-9
        feat_imp = {col: float(imp / total) for col, imp in zip(FEATURE_COLS, raw_imp)}

    auc_vals  = [m["auc"] for m in fold_metrics]
    prec_vals = [m["precision"] for m in fold_metrics]

    metrics = {
        "trained_at":        datetime.now().isoformat(),
        "n_samples":         int(len(X_raw)),
        "n_tickers":         int(dataset["ticker"].nunique()) if "ticker" in dataset.columns else 0,
        "pos_rate":          float(pos_rate),
        "pos_weight":        float(pos_weight),
        "cv_auc_mean":       float(np.mean(auc_vals)),
        "cv_auc_std":        float(np.std(auc_vals)),
        "cv_prec_mean":      float(np.mean(prec_vals)),
        "fold_details":      fold_metrics,
        "model_type":        type(final_model).__name__,
        "calibrated":        calibrator is not None,
        "cal_n_samples":     int(cal_n),
        "feature_importance": feat_imp,
        # ── Compatibility signature ────────────────────────────────────────────
        # Dùng để phát hiện model stale khi FEATURE_COLS hoặc label thay đổi.
        "feature_cols":      list(FEATURE_COLS),
        "n_features":        len(FEATURE_COLS),
        "label_version":     MODEL_LABEL_VERSION,
    }

    if save:
        with open(MODEL_PATH,  "wb") as f:
            pickle.dump(final_model, f, protocol=5)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f, protocol=5)
        with open(META_PATH, "wb") as f:
            pickle.dump(metrics, f, protocol=5)
        if calibrator is not None:
            with open(CALIBRATOR_PATH, "wb") as f:
                pickle.dump(calibrator, f, protocol=5)
            log.info("Calibrator saved → %s", CALIBRATOR_PATH)
        log.info("Model saved → %s", MODEL_PATH)

    return {
        "model":              final_model,
        "scaler":             scaler,
        "metrics":            metrics,
        "feature_importance": feat_imp,
        "calibrator":         calibrator,
    }


# ── Load ───────────────────────────────────────────────────────────────────────

def is_model_compatible() -> tuple[bool, str]:
    """
    Kiểm tra artifact hiện tại có tương thích với FEATURE_COLS và label schema không.

    Returns
    -------
    (True, "OK") nếu tương thích.
    (False, reason_str) nếu không tương thích — cần retrain.
    """
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return False, "Chưa có artifact (chưa train lần nào)"

    if not META_PATH.exists():
        return False, "Không có metadata — artifact cũ (trước v2.0), cần retrain"

    try:
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
    except Exception as exc:
        return False, f"Không đọc được metadata: {exc}"

    # Kiểm tra label version
    saved_label = meta.get("label_version", "v1_t2_return")
    if saved_label != MODEL_LABEL_VERSION:
        return False, (
            f"Label version thay đổi: artifact='{saved_label}' ≠ current='{MODEL_LABEL_VERSION}'. "
            "Cần retrain với M2 path-dependent label mới."
        )

    # Kiểm tra feature list
    saved_cols = meta.get("feature_cols")
    if saved_cols is None:
        # Artifact cũ không lưu feature_cols → không thể verify
        return False, "Artifact cũ không lưu feature_cols — cần retrain"

    if saved_cols != list(FEATURE_COLS):
        added   = [c for c in FEATURE_COLS if c not in saved_cols]
        removed = [c for c in saved_cols    if c not in FEATURE_COLS]
        parts = []
        if added:   parts.append(f"thêm {len(added)} feature: {added}")
        if removed: parts.append(f"bỏ {len(removed)} feature: {removed}")
        return False, f"FEATURE_COLS thay đổi ({'; '.join(parts)}) — cần retrain"

    return True, "OK"


def load_model() -> tuple | None:
    """
    Load trained model + scaler từ artifacts.

    Returns
    -------
    (model, scaler, meta) nếu tương thích và load được.
    None nếu:
      - Chưa có artifact
      - Artifact không tương thích (feature/label mismatch) → user cần retrain
    """
    compatible, reason = is_model_compatible()
    if not compatible:
        log.warning("[MODEL] Không load: %s", reason)
        return None

    try:
        with open(MODEL_PATH,  "rb") as f:
            model  = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        return model, scaler, meta
    except Exception as exc:
        log.warning("Không load được model: %s", exc)
        return None


def model_exists() -> bool:
    """True nếu artifact tồn tại VÀ tương thích với FEATURE_COLS + label hiện tại."""
    compatible, _ = is_model_compatible()
    return compatible


def load_calibrator():
    """
    Load isotonic calibrator nếu đã train (Sprint 6).

    Returns
    -------
    CalibratedClassifierCV instance hoặc None nếu chưa có.
    """
    if not CALIBRATOR_PATH.exists():
        return None
    try:
        with open(CALIBRATOR_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        log.warning("Không load được calibrator: %s", exc)
        return None


def wilson_ci(p: float, n: int = 50, z: float = 1.645) -> tuple[float, float]:
    """
    90% Wilson score confidence interval cho xác suất p.

    Parameters
    ----------
    p : Calibrated probability [0, 1].
    n : Effective sample size (default 50 = tiêu biểu cho time-series CV fold).
    z : Z-score (1.645 = 90%, 1.96 = 95%).

    Returns
    -------
    (ci_lo, ci_hi) — both clipped to [0, 1].
    """
    if n <= 0:
        half = min(0.15, p, 1 - p)
        return (round(max(0.0, p - half), 4), round(min(1.0, p + half), 4))
    denom  = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * float(np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)))
    return (round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4))
