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
_ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH     = _ARTIFACTS_DIR / "alpha_model.pkl"
SCALER_PATH    = _ARTIFACTS_DIR / "alpha_scaler.pkl"
META_PATH      = _ARTIFACTS_DIR / "alpha_meta.pkl"


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
    clean = dataset[FEATURE_COLS + ["label"]].dropna()
    X_raw = clean[FEATURE_COLS].values.astype(np.float32)
    y     = clean["label"].values.astype(int)

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

    # Feature importance
    feat_imp: dict[str, float] = {}
    if hasattr(final_model, "feature_importances_"):
        raw_imp = final_model.feature_importances_
        total   = raw_imp.sum() + 1e-9
        feat_imp = {col: float(imp / total) for col, imp in zip(FEATURE_COLS, raw_imp)}

    auc_vals  = [m["auc"] for m in fold_metrics]
    prec_vals = [m["precision"] for m in fold_metrics]

    metrics = {
        "trained_at":    datetime.now().isoformat(),
        "n_samples":     int(len(X_raw)),
        "n_tickers":     int(dataset["ticker"].nunique()) if "ticker" in dataset.columns else 0,
        "pos_rate":      float(pos_rate),
        "pos_weight":    float(pos_weight),
        "cv_auc_mean":   float(np.mean(auc_vals)),
        "cv_auc_std":    float(np.std(auc_vals)),
        "cv_prec_mean":  float(np.mean(prec_vals)),
        "fold_details":  fold_metrics,
        "model_type":    type(final_model).__name__,
        "feature_importance": feat_imp,
    }

    if save:
        with open(MODEL_PATH,  "wb") as f:
            pickle.dump(final_model, f, protocol=5)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f, protocol=5)
        with open(META_PATH, "wb") as f:
            pickle.dump(metrics, f, protocol=5)
        log.info("Model saved → %s", MODEL_PATH)

    return {
        "model":              final_model,
        "scaler":             scaler,
        "metrics":            metrics,
        "feature_importance": feat_imp,
    }


# ── Load ───────────────────────────────────────────────────────────────────────

def load_model() -> tuple | None:
    """
    Load trained model + scaler từ artifacts.
    Returns (model, scaler, meta) hoặc None nếu chưa train.
    """
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None
    try:
        with open(MODEL_PATH,  "rb") as f:
            model  = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        meta: dict = {}
        if META_PATH.exists():
            with open(META_PATH, "rb") as f:
                meta = pickle.load(f)
        return model, scaler, meta
    except Exception as exc:
        log.warning("Không load được model: %s", exc)
        return None


def model_exists() -> bool:
    """Kiểm tra xem model đã được train chưa."""
    return MODEL_PATH.exists() and SCALER_PATH.exists()
