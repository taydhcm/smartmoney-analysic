"""
ml/monitor.py
Sprint 11 — Daily Health Monitor & Milestone Tracker.

Chạy health check hằng ngày để đảm bảo hệ thống hoạt động bình thường:
  - D0.2 snapshot depth (cần >= 30 phiên)
  - Model retrain schedule (cảnh báo khi > 14 ngày)
  - E6 Rolling Calibrator freshness
  - E5 Signal outcome pending
  - Real trade milestone (mốc 30 lệnh real)
  - Scale readiness summary

Public API
----------
    HealthStatus      — dataclass: toàn bộ trạng thái hệ thống
    run_daily_health_check(capital) → HealthStatus
    check_d02_history_depth(db_path)  → dict
    check_retrain_schedule(meta_path) → dict
    check_milestone_progress(db_path) → dict
"""

from __future__ import annotations

import pickle
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)

# ── Monkey-patchable paths (for testing) ──────────────────────────────────────
_ROOT_DIR         = Path(__file__).resolve().parents[1]
SNAPSHOTS_DB      = _ROOT_DIR / "data" / "db" / "snapshots.db"
TRADES_DB         = _ROOT_DIR / "data" / "db" / "trades.db"
SIGNAL_LOG_DB     = _ROOT_DIR / "data" / "db" / "signal_log.db"
META_PATH         = _ROOT_DIR / "ml"  / "artifacts" / "alpha_meta.pkl"
ROLLING_META_PATH = _ROOT_DIR / "ml"  / "artifacts" / "alpha_calibrator_rolling_meta.json"

# ── Thresholds ────────────────────────────────────────────────────────────────
D02_MIN_SESSIONS   = 30    # phiên tối thiểu để chạy được S4 features
RETRAIN_STALE_DAYS = 14    # ngày tối đa trước khi cần retrain
E6_STALE_DAYS      = 7     # ngày tối đa trước khi cần refit E6
MILESTONE_REAL     = 30    # số lệnh real đóng để sang Stage 2


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class HealthStatus:
    """Trạng thái sức khoẻ toàn bộ hệ thống — kết quả run_daily_health_check."""

    timestamp:          str             # ISO datetime khi chạy check

    # D0.2 History Depth
    d02_session_days:   int             # số phiên distinct đã log
    d02_tickers:        int             # số ticker distinct
    d02_oldest_date:    Optional[str]   # phiên đầu tiên
    d02_newest_date:    Optional[str]   # phiên gần nhất
    d02_ready:          bool            # >= D02_MIN_SESSIONS

    # Model Retrain Schedule
    model_trained_at:   Optional[str]   # ISO timestamp train lần cuối
    model_age_days:     Optional[float] # ngày kể từ khi train
    retrain_due:        bool            # >= 14 ngày hoặc chưa có model

    # E6 Rolling Calibrator
    e6_fitted:          bool            # đã fit lần nào chưa
    e6_last_fit_days:   Optional[float] # ngày kể từ lần fit gần nhất
    e6_stale:           bool            # >= E6_STALE_DAYS

    # E5 Signal Outcomes
    total_signals:      int
    pending_outcomes:   int             # chưa resolve
    resolved_outcomes:  int

    # Trade Milestone (Real Trades)
    real_trades_closed: int             # số lệnh real đã đóng
    milestone_30:       bool            # >= 30
    milestone_pct:      float           # 0–100, progress to milestone

    # Scale Readiness (quick flag)
    scale_ready:        bool

    # Summary alerts (human-readable)
    alerts:             list[str] = field(default_factory=list)

    @property
    def overall_healthy(self) -> bool:
        """Hệ thống hoạt động bình thường (không có critical alert)."""
        return not self.alerts

    def summary(self) -> str:
        lines = [
            f"[{self.timestamp[:10]}] Health Check",
            f"  D0.2 : {self.d02_session_days} phiên ({'OK' if self.d02_ready else 'THIẾU'})",
            f"  Model: {'Chưa có' if not self.model_trained_at else f'{self.model_age_days:.0f}d ago'}"
                + (' ⚠ RETRAIN DUE' if self.retrain_due else ''),
            f"  E6   : {'Fitted' if self.e6_fitted else 'Chưa fit'}"
                + (' ⚠ STALE' if self.e6_stale else ''),
            f"  Trades: {self.real_trades_closed}/30 real lệnh "
                + ('✅' if self.milestone_30 else f'({self.milestone_pct:.0f}%)'),
            f"  Scale: {'✅ Sẵn sàng scale' if self.scale_ready else '⏳ Chưa đủ điều kiện'}",
        ]
        if self.alerts:
            lines.append("  Alerts:")
            for a in self.alerts:
                lines.append(f"    ⚠ {a}")
        return "\n".join(lines)


# ── Individual checks ──────────────────────────────────────────────────────────

def check_d02_history_depth(db_path: Optional[Path] = None) -> dict:
    """
    Kiểm tra độ sâu lịch sử D0.2 snapshot.

    Returns
    -------
    dict:
        session_days   (int)
        ticker_count   (int)
        oldest_date    (str | None)
        newest_date    (str | None)
        is_ready       (bool)  — session_days >= D02_MIN_SESSIONS
    """
    path = Path(db_path) if db_path else SNAPSHOTS_DB
    result = dict(
        session_days=0, ticker_count=0,
        oldest_date=None, newest_date=None,
        is_ready=False,
    )
    if not path.exists():
        return result
    try:
        with sqlite3.connect(str(path)) as con:
            cur = con.execute(
                "SELECT COUNT(DISTINCT session_date), COUNT(DISTINCT ticker), "
                "MIN(session_date), MAX(session_date) FROM snapshots"
            )
            row = cur.fetchone()
        if row:
            result["session_days"]  = int(row[0] or 0)
            result["ticker_count"]  = int(row[1] or 0)
            result["oldest_date"]   = row[2]
            result["newest_date"]   = row[3]
            result["is_ready"]      = result["session_days"] >= D02_MIN_SESSIONS
    except Exception as exc:
        log.warning("check_d02_history_depth failed: %s", exc)
    return result


def check_retrain_schedule(meta_path: Optional[Path] = None) -> dict:
    """
    Kiểm tra lịch retrain model.

    Returns
    -------
    dict:
        trained_at     (str | None)     ISO timestamp
        age_days       (float | None)   ngày kể từ khi train
        retrain_due    (bool)           age >= RETRAIN_STALE_DAYS hoặc không có model
        label_version  (str | None)
        cv_prec_mean   (float | None)
    """
    path = Path(meta_path) if meta_path else META_PATH
    result = dict(
        trained_at=None, age_days=None,
        retrain_due=True, label_version=None,
        cv_prec_mean=None,
    )
    if not path.exists():
        return result
    try:
        with open(path, "rb") as f:
            meta = pickle.load(f)
        trained_at_str   = meta.get("trained_at")
        result["label_version"] = meta.get("label_version")
        result["cv_prec_mean"]  = meta.get("cv_prec_mean")
        if trained_at_str:
            result["trained_at"] = trained_at_str
            trained_dt = datetime.fromisoformat(trained_at_str)
            age = (datetime.now() - trained_dt).total_seconds() / 86400
            result["age_days"]    = round(age, 2)
            result["retrain_due"] = age >= RETRAIN_STALE_DAYS
        else:
            result["retrain_due"] = True
    except Exception as exc:
        log.warning("check_retrain_schedule failed: %s", exc)
        result["retrain_due"] = True
    return result


def check_milestone_progress(db_path: Optional[Path] = None) -> dict:
    """
    Kiểm tra mốc 30 lệnh real đã đóng.

    Returns
    -------
    dict:
        real_closed    (int)
        milestone_30   (bool)
        milestone_pct  (float)   0–100
    """
    path = Path(db_path) if db_path else TRADES_DB
    result = dict(real_closed=0, milestone_30=False, milestone_pct=0.0)
    if not path.exists():
        return result
    try:
        with sqlite3.connect(str(path)) as con:
            cur = con.execute(
                "SELECT COUNT(*) FROM trades "
                "WHERE trade_type='real' AND status IN ('closed','stopped')"
            )
            row = cur.fetchone()
        count = int(row[0]) if row else 0
        result["real_closed"]   = count
        result["milestone_30"]  = count >= MILESTONE_REAL
        result["milestone_pct"] = min(100.0, count / MILESTONE_REAL * 100)
    except Exception as exc:
        log.warning("check_milestone_progress failed: %s", exc)
    return result


def _check_e5_signals(db_path: Optional[Path] = None) -> dict:
    """Đọc signal log để lấy thống kê E5 đơn giản."""
    path = Path(db_path) if db_path else SIGNAL_LOG_DB
    result = dict(total=0, pending=0, resolved=0)
    if not path.exists():
        return result
    try:
        with sqlite3.connect(str(path)) as con:
            cur = con.execute(
                "SELECT COUNT(*), "
                "SUM(CASE WHEN outcome IS NULL THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END) "
                "FROM signal_log"
            )
            row = cur.fetchone()
        if row:
            result["total"]    = int(row[0] or 0)
            result["pending"]  = int(row[1] or 0)
            result["resolved"] = int(row[2] or 0)
    except Exception as exc:
        log.warning("_check_e5_signals failed: %s", exc)
    return result


def _check_e6_calibrator(rolling_meta_path: Optional[Path] = None) -> dict:
    """Đọc E6 rolling calibrator metadata."""
    import json as _json
    path = Path(rolling_meta_path) if rolling_meta_path else ROLLING_META_PATH
    result = dict(fitted=False, last_fit_days=None, stale=True)
    if not path.exists():
        return result
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = _json.load(f)
        fit_at = meta.get("fit_at")
        if fit_at:
            fit_dt  = datetime.fromisoformat(fit_at)
            age     = (datetime.now() - fit_dt).total_seconds() / 86400
            result["fitted"]        = True
            result["last_fit_days"] = round(age, 2)
            result["stale"]         = age >= E6_STALE_DAYS
    except Exception as exc:
        log.warning("_check_e6_calibrator failed: %s", exc)
    return result


# ── Main health check ──────────────────────────────────────────────────────────

def run_daily_health_check(
    capital:           float      = 0.0,
    snapshots_db:      Optional[Path] = None,
    trades_db:         Optional[Path] = None,
    signal_log_db:     Optional[Path] = None,
    meta_path:         Optional[Path] = None,
    rolling_meta_path: Optional[Path] = None,
) -> HealthStatus:
    """
    Chạy toàn bộ health checks và trả về HealthStatus.

    Parameters (tất cả optional, dùng để override paths trong test)
    ----------
    capital           : tổng vốn — dùng để kiểm tra min_capital nếu cần
    snapshots_db      : override SNAPSHOTS_DB
    trades_db         : override TRADES_DB
    signal_log_db     : override SIGNAL_LOG_DB
    meta_path         : override META_PATH
    rolling_meta_path : override ROLLING_META_PATH
    """
    ts = datetime.now().isoformat()

    # D0.2
    d02 = check_d02_history_depth(snapshots_db)

    # Model retrain
    retrain = check_retrain_schedule(meta_path)

    # E6
    e6 = _check_e6_calibrator(rolling_meta_path)

    # E5
    sigs = _check_e5_signals(signal_log_db)

    # Trade milestone
    ms = check_milestone_progress(trades_db)

    # Scale readiness: quick flag (without full ScaleDecision compute)
    scale_ready = ms["milestone_30"]  # simplified — requires 30+ closed real trades

    # Build alerts
    alerts: list[str] = []
    if not d02["is_ready"]:
        alerts.append(
            f"D0.2: chỉ có {d02['session_days']}/{D02_MIN_SESSIONS} phiên snapshot. "
            "Cần chạy daily_snapshot thêm."
        )
    if retrain["retrain_due"]:
        if retrain["trained_at"]:
            alerts.append(
                f"Model: đã {retrain['age_days']:.0f} ngày chưa retrain "
                f"(ngưỡng {RETRAIN_STALE_DAYS}d). Chạy retrain khi có đủ dữ liệu mới."
            )
        else:
            alerts.append("Model: chưa có artifact. Cần train model trước.")
    if e6["stale"] and not e6["fitted"]:
        alerts.append("E6 Rolling Calibrator: chưa được fit. Mở trang Self-Learning để fit.")
    elif e6["stale"] and e6["fitted"]:
        alerts.append(
            f"E6 Rolling Calibrator: {e6['last_fit_days']:.0f} ngày chưa refit "
            f"(ngưỡng {E6_STALE_DAYS}d)."
        )

    status = HealthStatus(
        timestamp         = ts,
        d02_session_days  = d02["session_days"],
        d02_tickers       = d02["ticker_count"],
        d02_oldest_date   = d02["oldest_date"],
        d02_newest_date   = d02["newest_date"],
        d02_ready         = d02["is_ready"],
        model_trained_at  = retrain["trained_at"],
        model_age_days    = retrain["age_days"],
        retrain_due       = retrain["retrain_due"],
        e6_fitted         = e6["fitted"],
        e6_last_fit_days  = e6["last_fit_days"],
        e6_stale          = e6["stale"],
        total_signals     = sigs["total"],
        pending_outcomes  = sigs["pending"],
        resolved_outcomes = sigs["resolved"],
        real_trades_closed= ms["real_closed"],
        milestone_30      = ms["milestone_30"],
        milestone_pct     = ms["milestone_pct"],
        scale_ready       = scale_ready,
        alerts            = alerts,
    )
    log.info("HealthStatus: d02=%d retrain_due=%s scale_ready=%s alerts=%d",
             d02["session_days"], retrain["retrain_due"], scale_ready, len(alerts))
    return status
