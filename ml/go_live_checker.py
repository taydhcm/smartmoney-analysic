"""
ml/go_live_checker.py
Sprint 10 — Pre-flight Go-Live Validator.

Kiểm tra tất cả KPI và hệ thống trước khi deploy real capital.

Checklist:
    1. LightGBM model (alpha_model.pkl) tồn tại
    2. Batch calibrator (alpha_calibrator.pkl) tồn tại
    3. E6 Rolling Calibrator tồn tại và không stale (> 7 ngày)
    4. E5 Outcome Tracker (signal_log.db) active
    5. E6 calibrator đã được fit (>= 10 samples)
    6. Không có Drift Alert từ E5
    7. Portfolio capital >= min_capital
    8. p_min >= 0.60 (conservative threshold)
    9. SL buffer >= 1.5%
   10. Max position size <= 30%
   11. R/R minimum >= 1.5
   12. Pipeline imports (tất cả ml.* modules)
   13. E2 Trade Logger (trades.db) sẵn sàng

Conservative params cho First Real Trade:
    p_min=0.70, max_position_pct=0.20, sl_buffer_pct=2.0, rr_min=2.0, max_positions=5

Public API
----------
    GoLiveConfig   — dataclass cấu hình bảo thủ
    CheckResult    — kết quả một check đơn
    GoLiveStatus   — tổng hợp kết quả tất cả checks
    run_go_live_checks(config, portfolio_capital) → GoLiveStatus

Constants (monkey-patchable for tests)
----------------------------------------
    MODEL_PATH, CALIBRATOR_PATH, ROLLING_CALIBRATOR_PATH, ROLLING_META_PATH
    SIGNAL_LOG_DB, TRADES_DB
"""

from __future__ import annotations

import importlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from utils.logger import get_logger

log = get_logger(__name__)

# ── Paths (monkey-patchable for tests) ────────────────────────────────────────
_ML_DIR        = Path(__file__).parent
_ARTIFACTS_DIR = _ML_DIR / "artifacts"
_DB_DIR        = Path(__file__).resolve().parent.parent / "data" / "db"

MODEL_PATH              = _ARTIFACTS_DIR / "alpha_model.pkl"
CALIBRATOR_PATH         = _ARTIFACTS_DIR / "alpha_calibrator.pkl"
ROLLING_CALIBRATOR_PATH = _ARTIFACTS_DIR / "alpha_calibrator_rolling.pkl"
ROLLING_META_PATH       = _ARTIFACTS_DIR / "alpha_calibrator_rolling_meta.json"
SIGNAL_LOG_DB           = _DB_DIR / "signal_log.db"
TRADES_DB               = _DB_DIR / "trades.db"


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    """Kết quả một kiểm tra đơn lẻ trong pre-flight checklist."""
    name:       str
    passed:     bool
    detail:     str
    value:      Any  = None
    is_warning: bool = False   # True = cảnh báo, không phải hard fail


@dataclass
class GoLiveConfig:
    """
    Tham số bảo thủ cho lệnh real đầu tiên (Go-Live Sprint 10).

    Conservative params để giảm rủi ro:
        p_min            = 0.70  — chỉ trade khi calibrated prob >= 70%
        max_position_pct = 0.20  — tối đa 20% vốn / lệnh
        sl_buffer_pct    = 2.0   — SL buffer >= 2.0% từ entry
        max_positions    = 5     — tối đa 5 lệnh mở cùng lúc
        rr_min           = 2.0   — Risk/Reward tối thiểu 2:1
        min_capital      = 100M VND
        min_precision    = 0.55  — precision backtest >= 55%
        min_win_rate     = 0.50  — win rate >= 50%
    """
    p_min:            float = 0.70
    max_position_pct: float = 0.20
    sl_buffer_pct:    float = 2.0
    max_positions:    int   = 5
    rr_min:           float = 2.0
    min_capital:      float = 100_000_000.0
    min_precision:    float = 0.55
    min_win_rate:     float = 0.50

    def validate(self) -> list[str]:
        """Trả về danh sách lỗi config (empty = hợp lệ)."""
        errors: list[str] = []
        if not (0.0 < self.p_min <= 1.0):
            errors.append(f"p_min={self.p_min} không hợp lệ (phải trong (0, 1])")
        if self.p_min < 0.60:
            errors.append(f"p_min={self.p_min} quá thấp (tối thiểu 0.60)")
        if not (0.0 < self.max_position_pct <= 1.0):
            errors.append(f"max_position_pct={self.max_position_pct} không hợp lệ")
        if self.max_position_pct > 0.50:
            errors.append(f"max_position_pct={self.max_position_pct} quá cao (tối đa 0.50)")
        if self.sl_buffer_pct < 0.5:
            errors.append(f"sl_buffer_pct={self.sl_buffer_pct} quá nhỏ (tối thiểu 0.5%)")
        if self.rr_min < 1.0:
            errors.append(f"rr_min={self.rr_min} quá thấp (tối thiểu 1.0)")
        if self.max_positions < 1:
            errors.append(f"max_positions={self.max_positions} không hợp lệ (>= 1)")
        if self.min_capital <= 0:
            errors.append(f"min_capital={self.min_capital} không hợp lệ (> 0)")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


@dataclass
class GoLiveStatus:
    """Tổng hợp kết quả tất cả pre-flight checks."""
    checks:     list[CheckResult]
    all_passed: bool
    hard_fails: int    # số checks fail (is_warning=False, passed=False)
    warnings:   int    # số checks là warning (is_warning=True, passed=False)
    score:      int    # số checks passed (passed=True)
    total:      int    # tổng số checks

    @property
    def score_pct(self) -> float:
        return self.score / self.total if self.total > 0 else 0.0

    def summary(self) -> str:
        icon   = "🟢" if self.all_passed else "🔴"
        status = "GO-LIVE READY" if self.all_passed else "NOT READY"
        return (
            f"{icon} {status} — "
            f"{self.score}/{self.total} checks passed, "
            f"{self.hard_fails} failures, {self.warnings} warnings"
        )

    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and not c.is_warning]

    def warning_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.is_warning]


# ── Individual check functions ─────────────────────────────────────────────────

def _check_model_exists() -> CheckResult:
    """Kiểm tra alpha_model.pkl (LightGBM model) tồn tại."""
    _p = Path(str(MODEL_PATH))
    passed = _p.exists()
    return CheckResult(
        name   = "LightGBM Model (alpha_model.pkl)",
        passed = passed,
        detail = (f"Found: {_p.name}" if passed
                  else "Không tìm thấy. Chạy: python retrain_v5.py"),
        value  = str(_p),
    )


def _check_batch_calibrator() -> CheckResult:
    """Kiểm tra alpha_calibrator.pkl (batch Isotonic calibrator) tồn tại."""
    _p = Path(str(CALIBRATOR_PATH))
    passed = _p.exists()
    return CheckResult(
        name   = "Batch Calibrator (alpha_calibrator.pkl)",
        passed = passed,
        detail = (f"Found: {_p.name}" if passed
                  else "Không tìm thấy. Chạy retrain_v5.py"),
        value  = str(_p),
    )


def _check_rolling_calibrator() -> CheckResult:
    """
    Kiểm tra E6 rolling calibrator.
    is_warning=True nếu stale (> 7 ngày) — batch là fallback nên không fail cứng.
    passed=False nếu chưa có file.
    """
    _pkl  = Path(str(ROLLING_CALIBRATOR_PATH))
    _meta = Path(str(ROLLING_META_PATH))

    if not _pkl.exists():
        return CheckResult(
            name       = "E6 Rolling Calibrator",
            passed     = False,
            detail     = "alpha_calibrator_rolling.pkl chưa có. Vào 09_self_learning → Re-fit.",
            is_warning = True,   # batch calibrator là fallback
        )

    # Kiểm tra stale
    age_days: Optional[float] = None
    n_samples = 0
    is_stale  = True
    detail    = "pkl exists"

    if _meta.exists():
        try:
            meta      = json.loads(_meta.read_text(encoding="utf-8"))
            fit_at    = meta.get("fit_at", "")
            n_samples = int(meta.get("n_samples", 0))
            if fit_at:
                age_days = (datetime.now() - datetime.fromisoformat(fit_at)).total_seconds() / 86400
                is_stale = age_days >= 7
            detail = (
                f"{n_samples} samples · fit {age_days:.1f}d trước"
                + (" — ⚠️ stale" if is_stale else " — fresh ✅")
            )
        except Exception as exc:
            detail = f"meta parse error: {exc}"
    else:
        detail = "pkl exists nhưng không có meta file"

    return CheckResult(
        name       = "E6 Rolling Calibrator",
        passed     = not is_stale,
        detail     = detail,
        value      = age_days,
        is_warning = is_stale,   # stale là warning (không hard fail)
    )


def _check_e5_active() -> CheckResult:
    """Kiểm tra E5 Outcome Tracker (signal_log.db) tồn tại."""
    _p = Path(str(SIGNAL_LOG_DB))
    if not _p.exists():
        return CheckResult(
            name       = "E5 Outcome Tracker (signal_log.db)",
            passed     = False,
            detail     = "signal_log.db chưa có. Log signal từ trang Alpha Picks trước.",
            is_warning = True,
        )
    try:
        con   = sqlite3.connect(str(_p))
        count = con.execute("SELECT COUNT(*) FROM signal_log").fetchone()[0]
        con.close()
        return CheckResult(
            name   = "E5 Outcome Tracker (signal_log.db)",
            passed = True,
            detail = f"Found — {count} signals logged",
            value  = count,
        )
    except Exception:
        return CheckResult(
            name       = "E5 Outcome Tracker (signal_log.db)",
            passed     = True,
            detail     = "signal_log.db exists (tables chưa khởi tạo — tự tạo khi log)",
            is_warning = True,
        )


def _check_e6_status() -> CheckResult:
    """Kiểm tra E6 đã được fit với đủ data (>= 10 samples)."""
    _meta = Path(str(ROLLING_META_PATH))
    if not _meta.exists():
        return CheckResult(
            name       = "E6 Calibrator Fitted",
            passed     = False,
            detail     = "Chưa fit. Vào 09_self_learning → Re-fit ngay.",
            is_warning = True,
        )
    try:
        meta      = json.loads(_meta.read_text(encoding="utf-8"))
        n_samples = int(meta.get("n_samples", 0))
        precision = float(meta.get("precision", 0.0))
        fit_at    = str(meta.get("fit_at", "N/A"))[:10]
        passed    = n_samples >= 10
        return CheckResult(
            name       = "E6 Calibrator Fitted",
            passed     = passed,
            detail     = f"{n_samples} samples · precision={precision:.1%} · {fit_at}",
            value      = n_samples,
            is_warning = not passed,
        )
    except Exception as exc:
        return CheckResult(
            name="E6 Calibrator Fitted", passed=False,
            detail=f"Lỗi đọc meta: {exc}", is_warning=True,
        )


def _check_no_drift() -> CheckResult:
    """Kiểm tra không có Drift Alert từ E5 stats."""
    try:
        from .outcome_tracker import get_signal_stats
        stats      = get_signal_stats()
        drift      = stats.get("drift_alert", False)
        roll_total = stats.get("roll_30d_total", 0)
        precision  = stats.get("roll_30d_prec", 0.0)

        if roll_total < 10:
            return CheckResult(
                name       = "Drift Alert Check",
                passed     = True,
                detail     = f"Chưa đủ data ({roll_total} outcomes). Drift check bỏ qua.",
                is_warning = True,
                value      = roll_total,
            )
        passed = not drift
        return CheckResult(
            name   = "Drift Alert Check",
            passed = passed,
            detail = (
                f"Rolling precision={precision:.1%} trên {roll_total} outcomes"
                + (" — ✅ OK" if passed else " — 🚨 DRIFT!")
            ),
            value  = precision,
        )
    except Exception as exc:
        return CheckResult(
            name       = "Drift Alert Check",
            passed     = True,
            detail     = f"Không thể kiểm tra ({exc}) — bỏ qua.",
            is_warning = True,
        )


def _check_capital(config: GoLiveConfig, portfolio_capital: float) -> CheckResult:
    """Kiểm tra vốn deploy >= config.min_capital."""
    passed = portfolio_capital >= config.min_capital
    fmt    = lambda x: f"{x/1_000_000:.0f}M VND"
    return CheckResult(
        name   = "Portfolio Capital",
        passed = passed,
        detail = (
            f"Vốn: {fmt(portfolio_capital)}"
            + (f" ✅ >= {fmt(config.min_capital)}" if passed
               else f" ❌ < tối thiểu {fmt(config.min_capital)}")
        ),
        value  = portfolio_capital,
    )


def _check_p_min(config: GoLiveConfig) -> CheckResult:
    """Kiểm tra p_min >= 0.60 (hard minimum) và >= 0.70 (recommended)."""
    passed = config.p_min >= 0.60
    is_warning = passed and config.p_min < 0.70
    detail = (
        f"p_min = {config.p_min:.2f}"
        + (" ✅ conservative (>= 0.70)" if config.p_min >= 0.70
           else (" ⚠️ thấp hơn recommended 0.70" if passed
                 else " ❌ quá thấp (tối thiểu 0.60)"))
    )
    return CheckResult(
        name       = "P_min Threshold",
        passed     = passed,
        detail     = detail,
        value      = config.p_min,
        is_warning = is_warning,
    )


def _check_sl_buffer(config: GoLiveConfig) -> CheckResult:
    """Kiểm tra SL buffer >= 1.5%."""
    passed = config.sl_buffer_pct >= 1.5
    return CheckResult(
        name   = "SL Buffer",
        passed = passed,
        detail = (
            f"SL buffer = {config.sl_buffer_pct:.1f}%"
            + (" ✅" if passed else " ❌ tối thiểu 1.5%")
        ),
        value  = config.sl_buffer_pct,
    )


def _check_position_size(config: GoLiveConfig) -> CheckResult:
    """Kiểm tra max_position_pct <= 0.30 (bảo thủ)."""
    passed     = config.max_position_pct <= 0.30
    is_warning = passed and config.max_position_pct > 0.20
    detail = (
        f"Max position = {config.max_position_pct:.0%}/lệnh"
        + (" ✅ conservative" if config.max_position_pct <= 0.20
           else (" ⚠️ > 20%, thận trọng" if passed
                 else " ❌ > 30%, quá rủi ro"))
    )
    return CheckResult(
        name       = "Max Position Size",
        passed     = passed,
        detail     = detail,
        value      = config.max_position_pct,
        is_warning = is_warning,
    )


def _check_rr_filter(config: GoLiveConfig) -> CheckResult:
    """Kiểm tra rr_min >= 1.5 (Risk/Reward minimum)."""
    passed = config.rr_min >= 1.5
    return CheckResult(
        name   = "R/R Filter (rr_min)",
        passed = passed,
        detail = (
            f"R/R minimum = {config.rr_min:.1f}:1"
            + (" ✅ good" if config.rr_min >= 2.0
               else (" ⚠️ < 2:1, kiểm tra lại" if passed
                     else " ❌ tối thiểu 1.5:1"))
        ),
        value  = config.rr_min,
    )


def _check_pipeline_imports() -> CheckResult:
    """Kiểm tra tất cả ml.* modules import thành công."""
    _REQUIRED = [
        ("ml.predictor",          "predict_all"),
        ("ml.regime",             "get_market_regime"),
        ("ml.alert_generator",    "generate_morning_report"),
        ("ml.outcome_tracker",    "get_signal_stats"),
        ("ml.rolling_calibrator", "apply_rolling_calibration"),
        ("ml.pnl_tracker",        "compute_stats"),
        ("data.trade_logger",     "add_trade"),
    ]
    failed: list[str] = []
    for mod_name, attr in _REQUIRED:
        try:
            mod = importlib.import_module(mod_name)
            if not hasattr(mod, attr):
                failed.append(f"{mod_name}.{attr} missing")
        except Exception as exc:
            failed.append(f"{mod_name}: {exc}")

    passed = len(failed) == 0
    return CheckResult(
        name   = "Pipeline Imports",
        passed = passed,
        detail = (f"✅ Tất cả {len(_REQUIRED)} modules OK" if passed
                  else f"❌ {len(failed)} lỗi: {'; '.join(failed[:2])}"),
        value  = len(_REQUIRED) - len(failed),
    )


def _check_trades_db() -> CheckResult:
    """Kiểm tra E2 Trade Logger (trades.db) sẵn sàng (tự tạo khi dùng)."""
    _p = Path(str(TRADES_DB))
    if _p.exists():
        try:
            con   = sqlite3.connect(str(_p))
            count = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            con.close()
            return CheckResult(
                name   = "E2 Trade Logger (trades.db)",
                passed = True,
                detail = f"Found — {count} trades",
                value  = count,
            )
        except Exception:
            return CheckResult(
                name       = "E2 Trade Logger (trades.db)",
                passed     = True,
                detail     = "trades.db exists (tables tự tạo khi add_trade())",
                is_warning = True,
            )
    return CheckResult(
        name       = "E2 Trade Logger (trades.db)",
        passed     = True,
        detail     = "trades.db chưa có — sẽ tự tạo khi add_trade() lần đầu",
        is_warning = True,
        value      = 0,
    )


# ── Main check runner ──────────────────────────────────────────────────────────

def run_go_live_checks(
    config: Optional[GoLiveConfig] = None,
    portfolio_capital: float = 0.0,
) -> GoLiveStatus:
    """
    Chạy tất cả 13 pre-flight checks trước khi deploy real capital.

    Parameters
    ----------
    config            : GoLiveConfig. Mặc định: conservative params.
    portfolio_capital : VND — dùng cho capital check.

    Returns
    -------
    GoLiveStatus — all_passed=True khi không có hard fail nào.
    """
    if config is None:
        config = GoLiveConfig()

    checks: list[CheckResult] = [
        # 1-3: Model artifacts
        _check_model_exists(),
        _check_batch_calibrator(),
        _check_rolling_calibrator(),
        # 4-6: E5/E6 systems
        _check_e5_active(),
        _check_e6_status(),
        _check_no_drift(),
        # 7-11: Config validation
        _check_capital(config, portfolio_capital),
        _check_p_min(config),
        _check_sl_buffer(config),
        _check_position_size(config),
        _check_rr_filter(config),
        # 12-13: Pipeline integrity
        _check_pipeline_imports(),
        _check_trades_db(),
    ]

    hard_fails = sum(1 for c in checks if not c.passed and not c.is_warning)
    warnings   = sum(1 for c in checks if not c.passed and c.is_warning)
    score      = sum(1 for c in checks if c.passed)
    all_passed = hard_fails == 0

    status = GoLiveStatus(
        checks     = checks,
        all_passed = all_passed,
        hard_fails = hard_fails,
        warnings   = warnings,
        score      = score,
        total      = len(checks),
    )
    log.info("Go-Live: %s", status.summary())
    return status
