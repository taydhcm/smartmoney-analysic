"""
ml/rolling_calibrator.py
E6 Rolling IsotonicRegression Calibrator — Sprint 9.

Re-fit IsotonicRegression trên 90 resolved outcomes gần nhất từ signal_log.db
để thích nghi với điều kiện thị trường hiện tại (weekly schedule).

Artifacts
---------
    ml/artifacts/alpha_calibrator_rolling.pkl        — fitted IsotonicRegression
    ml/artifacts/alpha_calibrator_rolling_meta.json  — metadata: fit_at, n_samples, precision

Workflow Hybrid 4-bước
-----------------------
    Bước 1 — LightGBM batch retrain (retrain_v5.py, mỗi 2 tuần)
    Bước 2 — E5 Outcome Tracker cung cấp signal → T+5 data (signal_log.db)
    Bước 3 — E6 Rolling Calibrator (module này): re-fit isotonic weekly
    Bước 4 — Drift Alert: rolling precision < 0.30 × n>=10 → banner đỏ UI → manual retrain

Public API
----------
    get_calibration_data(n, db_path)                          → pd.DataFrame
    fit_rolling_calibrator(n, min_samples, db_path, ...)      → dict (status, n_samples, precision, fit_at)
    load_rolling_calibrator(save_path)                        → IsotonicRegression | None
    apply_rolling_calibration(p_raw_array, save_path, ...)    → np.ndarray  (clamped [0,1])
    get_calibrator_status(db_path)                            → dict
    needs_weekly_refit(stale_days)                            → bool

Constants (monkey-patchable for tests)
---------------------------------------
    ROLLING_CALIBRATOR_PATH
    ROLLING_META_PATH
    SIGNAL_LOG_DB
"""

from __future__ import annotations

import json
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from utils.logger import get_logger

log = get_logger(__name__)

# ── Artifact paths (monkey-patchable for tests) ────────────────────────────────
_ARTIFACTS_DIR          = Path(__file__).parent / "artifacts"
ROLLING_CALIBRATOR_PATH = _ARTIFACTS_DIR / "alpha_calibrator_rolling.pkl"
ROLLING_META_PATH       = _ARTIFACTS_DIR / "alpha_calibrator_rolling_meta.json"

# Batch calibrator path (fallback when rolling not available)
_BATCH_CALIBRATOR_PATH  = _ARTIFACTS_DIR / "alpha_calibrator.pkl"

# Signal DB (monkey-patchable for tests)
_DB_DIR       = Path(__file__).resolve().parent.parent / "data" / "db"
SIGNAL_LOG_DB = _DB_DIR / "signal_log.db"


# ── Data retrieval ─────────────────────────────────────────────────────────────

def get_calibration_data(
    n: int = 90,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Query signal_log.db cho resolved outcomes (WIN / LOSS / FLAT).

    Parameters
    ----------
    n        : số outcomes tối đa (lấy n gần nhất theo signal_date DESC).
    db_path  : override DB path (mặc định dùng module-level SIGNAL_LOG_DB).

    Returns
    -------
    DataFrame với columns:
        signal_date, ticker, p_raw, p_calibrated, label, outcome, pnl_pct

    label = 1 nếu WIN, 0 nếu LOSS hoặc FLAT.
    PENDING và EXPIRED bị loại trừ.
    Rows được sort theo signal_date ASC (chronological) cho isotonic regression.
    """
    _empty = pd.DataFrame(
        columns=["signal_date", "ticker", "p_raw", "p_calibrated",
                 "label", "outcome", "pnl_pct"]
    )

    _path = db_path if db_path is not None else SIGNAL_LOG_DB
    _path = Path(str(_path))
    if not _path.exists():
        return _empty

    try:
        con = sqlite3.connect(str(_path), check_same_thread=False)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT sl.signal_date, sl.ticker,
                   sl.p_raw, sl.p_calibrated,
                   so.outcome, so.pnl_pct
            FROM signal_log sl
            JOIN signal_outcomes so ON so.signal_id = sl.id
            WHERE so.outcome NOT IN ('PENDING', 'EXPIRED')
            ORDER BY sl.signal_date DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()
        con.close()
    except Exception as exc:
        log.warning("get_calibration_data DB error: %s", exc)
        return _empty

    if not rows:
        return _empty

    df = pd.DataFrame([dict(r) for r in rows])
    df["label"] = (df["outcome"] == "WIN").astype(int)
    # Đảo về chronological order (DESC → ASC) cho proper isotonic fit
    df = df.sort_values("signal_date").reset_index(drop=True)
    return df


# ── Fit & persist ──────────────────────────────────────────────────────────────

def fit_rolling_calibrator(
    n: int = 90,
    min_samples: int = 10,
    db_path: Optional[Path] = None,
    save_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
) -> dict:
    """
    Fit IsotonicRegression(increasing=True) trên last n resolved outcomes.
    Lưu calibrator pkl và metadata JSON.

    Parameters
    ----------
    n           : số outcomes sử dụng (lấy n gần nhất).
    min_samples : số samples tối thiểu để fit (mặc định 10).
    db_path     : override signal_log.db path.
    save_path   : override rolling calibrator pkl path.
    meta_path   : override rolling meta JSON path.

    Returns
    -------
    dict với keys:
        status         : "ok" | "insufficient_data" | "no_db"
        n_samples      : int   (số outcomes được dùng)
        precision      : float (tỷ lệ WIN trong training set)
        fit_at         : str | None (ISO datetime)
        calibrator_path: str   (đường dẫn pkl đã lưu)
    """
    _db   = Path(str(db_path   if db_path   is not None else SIGNAL_LOG_DB))
    _save = Path(str(save_path if save_path is not None else ROLLING_CALIBRATOR_PATH))
    _meta = Path(str(meta_path if meta_path is not None else ROLLING_META_PATH))

    _result_base = {"status": "ok", "n_samples": 0, "precision": 0.0,
                    "fit_at": None, "calibrator_path": str(_save)}

    # Check DB tồn tại
    if not _db.exists():
        log.warning("fit_rolling_calibrator: DB không tìm thấy tại %s", _db)
        return {**_result_base, "status": "no_db"}

    df = get_calibration_data(n=n, db_path=_db)
    n_samples = len(df)
    if n_samples < min_samples:
        log.info("fit_rolling_calibrator: không đủ data (%d < %d)", n_samples, min_samples)
        return {**_result_base, "status": "insufficient_data", "n_samples": n_samples}

    X = df["p_raw"].values.astype(float)
    y = df["label"].values.astype(float)

    # Fit IsotonicRegression: higher p_raw → higher calibrated probability
    cal = IsotonicRegression(increasing=True, out_of_bounds="clip")
    cal.fit(X, y)

    precision = float(y.mean())   # fraction WIN trong training set
    fit_at    = datetime.now().isoformat(timespec="seconds")

    # Lưu calibrator pkl
    _save.parent.mkdir(parents=True, exist_ok=True)
    with open(_save, "wb") as fh:
        pickle.dump(cal, fh)

    # Lưu metadata JSON
    meta = {
        "fit_at":           fit_at,
        "n_samples":        n_samples,
        "n_requested":      n,
        "precision":        round(precision, 4),
        "calibrator_path":  str(_save),
    }
    _meta.parent.mkdir(parents=True, exist_ok=True)
    _meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info(
        "fit_rolling_calibrator: fitted %d samples, precision=%.3f, saved→%s",
        n_samples, precision, _save.name,
    )
    return {
        "status":           "ok",
        "n_samples":        n_samples,
        "precision":        precision,
        "fit_at":           fit_at,
        "calibrator_path":  str(_save),
    }


# ── Load & apply ───────────────────────────────────────────────────────────────

def load_rolling_calibrator(
    save_path: Optional[Path] = None,
) -> Optional[IsotonicRegression]:
    """
    Load rolling calibrator từ pkl.

    Returns
    -------
    IsotonicRegression nếu file tồn tại và hợp lệ, else None.
    """
    _save = Path(str(save_path if save_path is not None else ROLLING_CALIBRATOR_PATH))
    if not _save.exists():
        return None
    try:
        with open(_save, "rb") as fh:
            cal = pickle.load(fh)
        if not isinstance(cal, IsotonicRegression):
            log.warning("load_rolling_calibrator: unexpected type %s", type(cal))
            return None
        return cal
    except Exception as exc:
        log.warning("load_rolling_calibrator error: %s", exc)
        return None


def apply_rolling_calibration(
    p_raw_array,
    save_path: Optional[Path] = None,
    fallback_to_batch: bool = True,
) -> np.ndarray:
    """
    Áp dụng rolling calibration cho raw probabilities từ LightGBM.
    Fallback sang batch calibrator nếu rolling chưa có.
    Output được clamp trong [0.0, 1.0].

    Parameters
    ----------
    p_raw_array      : array-like của raw probabilities (float, [0,1]).
    save_path        : override rolling calibrator pkl path.
    fallback_to_batch: nếu True, dùng batch calibrator khi rolling không có.

    Returns
    -------
    np.ndarray shape (n,), clipped [0.0, 1.0].
    """
    arr = np.asarray(p_raw_array, dtype=float).ravel().clip(0.0, 1.0)

    # Thử rolling calibrator trước
    cal = load_rolling_calibrator(save_path=save_path)

    # Fallback sang batch calibrator nếu cần
    if cal is None and fallback_to_batch:
        _batch = Path(str(_BATCH_CALIBRATOR_PATH))
        if _batch.exists():
            try:
                with open(_batch, "rb") as fh:
                    cal = pickle.load(fh)
                log.debug("apply_rolling_calibration: using batch calibrator fallback")
            except Exception as exc:
                log.warning("apply_rolling_calibration batch fallback error: %s", exc)
                cal = None

    if cal is None:
        log.warning("apply_rolling_calibration: không có calibrator, trả về raw values")
        return arr.copy()

    try:
        result = np.asarray(cal.transform(arr.reshape(-1)), dtype=float).clip(0.0, 1.0)
        return result
    except Exception as exc:
        log.warning("apply_rolling_calibration transform error: %s", exc)
        return arr.copy()


# ── Status & scheduling ────────────────────────────────────────────────────────

def get_calibrator_status(db_path: Optional[Path] = None) -> dict:
    """
    Trả về trạng thái calibrator cho UI dashboard.

    Returns
    -------
    dict với keys:
        is_fitted           : bool   (có rolling calibrator không)
        last_fit_at         : str | None  (ISO datetime của lần fit gần nhất)
        n_samples           : int    (số outcomes dùng để fit)
        precision           : float  (tỷ lệ WIN trong training set)
        is_stale            : bool   (True nếu > 7 ngày kể từ last fit)
        needs_refit         : bool   (True nếu chưa fit hoặc stale)
        rolling_path_exists : bool
        batch_path_exists   : bool
        drift_alert         : bool   (từ E5 get_signal_stats — best-effort)
    """
    _rp = Path(str(ROLLING_CALIBRATOR_PATH))
    _mp = Path(str(ROLLING_META_PATH))
    _bp = Path(str(_BATCH_CALIBRATOR_PATH))

    rolling_exists = _rp.exists()
    batch_exists   = _bp.exists()

    is_fitted   = False
    last_fit_at = None
    n_samples   = 0
    precision   = 0.0
    is_stale    = True   # mặc định stale nếu chưa có meta

    if _mp.exists():
        try:
            meta        = json.loads(_mp.read_text(encoding="utf-8"))
            last_fit_at = meta.get("fit_at")
            n_samples   = int(meta.get("n_samples", 0))
            precision   = float(meta.get("precision", 0.0))
            is_fitted   = True
            if last_fit_at:
                age_days = (
                    datetime.now() - datetime.fromisoformat(last_fit_at)
                ).total_seconds() / 86400
                is_stale = age_days >= 7
        except Exception as exc:
            log.warning("get_calibrator_status: meta parse error: %s", exc)

    needs_refit = not is_fitted or is_stale

    # Drift Alert từ E5 (best-effort, may fail if no DB)
    drift_alert = False
    try:
        from .outcome_tracker import get_signal_stats
        stats       = get_signal_stats()
        drift_alert = bool(stats.get("drift_alert", False))
    except Exception:
        pass

    return {
        "is_fitted":            is_fitted,
        "last_fit_at":          last_fit_at,
        "n_samples":            n_samples,
        "precision":            precision,
        "is_stale":             is_stale,
        "needs_refit":          needs_refit,
        "rolling_path_exists":  rolling_exists,
        "batch_path_exists":    batch_exists,
        "drift_alert":          drift_alert,
    }


def needs_weekly_refit(stale_days: int = 7) -> bool:
    """
    Kiểm tra xem rolling calibrator có cần re-fit không.

    Returns True nếu:
        - Chưa có rolling calibrator pkl hoặc meta file
        - Lần fit gần nhất > stale_days ngày (mặc định 7)

    Note: được gọi tự động trong UI để hiển thị cảnh báo.
    """
    _rp = Path(str(ROLLING_CALIBRATOR_PATH))
    _mp = Path(str(ROLLING_META_PATH))

    if not _rp.exists() or not _mp.exists():
        return True

    try:
        meta         = json.loads(_mp.read_text(encoding="utf-8"))
        last_fit_str = meta.get("fit_at")
        if not last_fit_str:
            return True
        last_fit = datetime.fromisoformat(last_fit_str)
        age_days = (datetime.now() - last_fit).total_seconds() / 86400
        return age_days >= stale_days
    except Exception:
        return True
