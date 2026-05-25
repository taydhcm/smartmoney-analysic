"""
ml/scale_advisor.py
Sprint 11 — Capital Scaling Decision Engine.

Tính toán xem có nên tăng vốn deploy không, dựa trên:
  - Số lệnh real đã đóng (cần >= 30)
  - Rolling precision E6 (cần >= 0.35)
  - Win rate thực tế (cần >= 0.45)
  - Max drawdown (cần <= 15%)
  - Không có Drift Alert

Khi TẤT CẢ điều kiện đạt → `can_scale=True`
→ Gợi ý tăng max_position_pct từ 20% → 30%, rồi → 40% (theo stage)

Public API
----------
    ScaleDecision  — dataclass: can_scale, recommended_pct, score, blockers, reasoning
    SCALE_STAGES   — list[dict]: định nghĩa từng mức scale
    compute_scale_recommendation(trade_stats, calibr_status, pnl_stats, current_max_pct)
        → ScaleDecision
    check_scale_conditions(trade_stats, calibr_status, pnl_stats)
        → dict  (mỗi condition: name, passed, value, threshold)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)

# ── Scale stages ───────────────────────────────────────────────────────────────
# Từng bước tăng vốn theo lộ trình bảo thủ
SCALE_STAGES: list[dict] = [
    {
        "stage":            1,
        "label":            "Go-Live (Conservative)",
        "max_position_pct": 0.20,
        "max_positions":    5,
        "min_trades":       0,
        "description":      "Khởi đầu — 20% vốn/lệnh, P_min=0.70",
    },
    {
        "stage":            2,
        "label":            "Confirmed Edge",
        "max_position_pct": 0.30,
        "max_positions":    5,
        "min_trades":       30,
        "description":      "30+ lệnh, precision≥0.35, drawdown≤15% — tăng lên 30%",
    },
    {
        "stage":            3,
        "label":            "Full Deployment",
        "max_position_pct": 0.40,
        "max_positions":    6,
        "min_trades":       60,
        "description":      "60+ lệnh, precision≥0.40, drawdown≤10% — full scale",
    },
]

# ── Scale conditions ──────────────────────────────────────────────────────────
_CONDITIONS = [
    # (key, label, threshold, higher_is_better)
    ("closed_count",       "Real trades closed ≥ 30",     30,     True),
    ("roll_30d_prec",      "Rolling precision ≥ 0.35",    0.35,   True),
    ("win_rate",           "Win rate ≥ 0.45",             0.45,   True),
    ("max_drawdown_pct",   "Max drawdown ≤ -15%",         -15.0,  False),   # value must be >= -15
    ("drift_alert_clear",  "Drift Alert = False",         1.0,    True),    # 1=clear, 0=alert
]


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class ConditionResult:
    """Kết quả một điều kiện scale."""
    key:       str
    label:     str
    passed:    bool
    value:     float
    threshold: float


@dataclass
class ScaleDecision:
    """Kết quả tính toán nên scale vốn hay không."""
    can_scale:        bool
    recommended_pct:  float          # max_position_pct gợi ý
    current_pct:      float          # max_position_pct hiện tại
    current_stage:    int            # 1 | 2 | 3
    next_stage:       Optional[dict] # stage tiếp theo (None nếu đang ở max)
    score:            int            # số điều kiện đạt
    total_conditions: int            # tổng số điều kiện
    conditions:       list[ConditionResult]
    blockers:         list[str]      # điều kiện chưa đạt (human-readable)
    reasoning:        str            # tóm tắt để UI hiển thị

    @property
    def score_pct(self) -> float:
        return self.score / self.total_conditions if self.total_conditions > 0 else 0.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _current_stage(current_pct: float) -> int:
    """Trả về stage hiện tại dựa vào max_position_pct."""
    for s in reversed(SCALE_STAGES):
        if current_pct >= s["max_position_pct"] - 0.001:
            return s["stage"]
    return 1


def _next_stage(current_stage: int) -> Optional[dict]:
    """Trả về stage tiếp theo, None nếu đang ở max."""
    for s in SCALE_STAGES:
        if s["stage"] == current_stage + 1:
            return s
    return None


# ── Core functions ─────────────────────────────────────────────────────────────

def check_scale_conditions(
    trade_stats:   dict,
    calibr_status: dict,
    pnl_stats:     dict,
) -> list[ConditionResult]:
    """
    Kiểm tra từng điều kiện scale.

    Parameters
    ----------
    trade_stats   : từ data.trade_logger.get_trade_stats(trade_type="real")
    calibr_status : từ ml.rolling_calibrator.get_calibrator_status()
    pnl_stats     : từ ml.pnl_tracker.compute_stats()

    Returns
    -------
    list[ConditionResult] — một kết quả cho mỗi điều kiện
    """
    closed_count  = int(trade_stats.get("closed_count", 0))
    roll_prec     = float(calibr_status.get("precision", 0.0))
    win_rate      = float(trade_stats.get("win_rate", 0.0))
    max_dd        = float(pnl_stats.get("max_drawdown_pct", -100.0))
    drift_alert   = bool(calibr_status.get("drift_alert", True))
    drift_clear   = 0.0 if drift_alert else 1.0   # 1=clear, 0=alert

    raw = {
        "closed_count":      closed_count,
        "roll_30d_prec":     roll_prec,
        "win_rate":          win_rate,
        "max_drawdown_pct":  max_dd,
        "drift_alert_clear": drift_clear,
    }

    results: list[ConditionResult] = []
    for key, label, threshold, higher_is_better in _CONDITIONS:
        val    = raw[key]
        passed = (val >= threshold) if higher_is_better else (val >= threshold)
        results.append(ConditionResult(
            key=key, label=label,
            passed=passed, value=val, threshold=threshold,
        ))
    return results


def compute_scale_recommendation(
    trade_stats:     dict,
    calibr_status:   dict,
    pnl_stats:       dict,
    current_max_pct: float = 0.20,
) -> ScaleDecision:
    """
    Tính toán xem có nên tăng vốn deploy không.

    Parameters
    ----------
    trade_stats      : từ data.trade_logger.get_trade_stats(trade_type="real")
    calibr_status    : từ ml.rolling_calibrator.get_calibrator_status()
    pnl_stats        : từ ml.pnl_tracker.compute_stats()
    current_max_pct  : max_position_pct hiện tại (mặc định 0.20 = stage 1)

    Returns
    -------
    ScaleDecision với can_scale, recommended_pct, blockers, reasoning
    """
    conditions   = check_scale_conditions(trade_stats, calibr_status, pnl_stats)
    score        = sum(1 for c in conditions if c.passed)
    can_scale    = score == len(conditions)  # tất cả phải đạt
    blockers     = [c.label for c in conditions if not c.passed]

    cur_stage  = _current_stage(current_max_pct)
    nxt_stage  = _next_stage(cur_stage)
    rec_pct    = nxt_stage["max_position_pct"] if (can_scale and nxt_stage) else current_max_pct

    # Reasoning
    closed = int(trade_stats.get("closed_count", 0))
    prec   = float(calibr_status.get("precision", 0.0))
    dd     = float(pnl_stats.get("max_drawdown_pct", 0.0))
    if can_scale and nxt_stage:
        reasoning = (
            f"✅ Đủ điều kiện scale: {closed} lệnh, precision={prec:.1%}, "
            f"drawdown={dd:.1f}% — tăng từ {current_max_pct:.0%} → {rec_pct:.0%}"
        )
    elif can_scale and not nxt_stage:
        reasoning = (
            f"✅ Tất cả điều kiện đạt nhưng đã ở stage tối đa ({current_max_pct:.0%}). "
            "Giữ nguyên params."
        )
    else:
        needed   = len(conditions) - score
        reasoning = (
            f"❌ Chưa đủ điều kiện scale ({score}/{len(conditions)} passed, "
            f"{needed} blockers). "
            f"Tiếp tục trade với max_pos={current_max_pct:.0%}."
        )

    decision = ScaleDecision(
        can_scale        = can_scale,
        recommended_pct  = rec_pct,
        current_pct      = current_max_pct,
        current_stage    = cur_stage,
        next_stage       = nxt_stage,
        score            = score,
        total_conditions = len(conditions),
        conditions       = conditions,
        blockers         = blockers,
        reasoning        = reasoning,
    )
    log.info("ScaleDecision: can_scale=%s score=%d/%d", can_scale, score, len(conditions))
    return decision
