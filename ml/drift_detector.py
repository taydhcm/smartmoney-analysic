"""
ml/drift_detector.py
Module E — Model Drift Detector (out-of-sample, dựa trên trade_log.db).

Nguồn dữ liệu: ml/artifacts/trade_log.db (Module B)
Không dùng signal_log.db (E5) — chỉ theo dõi alpha signal outcomes.

Logic
-----
  1. Lấy N giao dịch gần nhất đã resolved (WIN/LOSS) từ alpha_signals
  2. Tính rolling win rate cho cửa sổ N (mặc định = 20)
  3. Nếu win_rate < threshold (mặc định 25%) và n_resolved >= MIN_SAMPLES (10) → DRIFT
  4. Tính trend bằng cách so sánh window hiện tại với window trước đó

Public API
----------
    DriftStatus          — dataclass kết quả
    get_drift_status(window, threshold_pct) → DriftStatus
    get_rolling_series(n_total)             → pd.DataFrame  (daily rolling win rate)
    DRIFT_WINDOW                            — 20 (số trade cửa sổ)
    DRIFT_THRESHOLD_PCT                     — 25.0 (% alert)

Monkey-patchable
----------------
    TRADE_LOG_DB   — path tới DB (dùng cho test)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)

# ── Constants (monkey-patchable) ──────────────────────────────────────────────
DRIFT_WINDOW        = 20     # số trade gần nhất để tính win rate
DRIFT_THRESHOLD_PCT = 25.0   # threshold cảnh báo
MIN_SAMPLES         = 10     # cần ít nhất N resolved trước khi alert

_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
TRADE_LOG_DB   = _ARTIFACTS_DIR / "trade_log.db"


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class DriftStatus:
    """Trạng thái drift của model dựa trên track record thực tế."""

    is_drifting:      bool     # True = win rate < threshold trong window gần nhất
    win_rate_pct:     float    # Rolling win rate hiện tại (%)
    n_resolved:       int      # Tổng số trades đã resolved (WIN/LOSS)
    n_window:         int      # Cửa sổ đánh giá (default 20)
    threshold_pct:    float    # Ngưỡng alert (default 25%)
    has_enough_data:  bool     # n_resolved >= MIN_SAMPLES
    trend:            str      # 'improving' / 'stable' / 'degrading' / 'insufficient'
    trend_delta_pct:  float    # Δ win rate giữa window hiện tại và window trước (percentage points)
    message:          str      # Mô tả ngắn gọn
    last_checked:     str      # ISO timestamp

    @property
    def badge(self) -> str:
        """Badge ngắn cho sidebar."""
        if not self.has_enough_data:
            return f"⚪ Model (chưa đủ data: {self.n_resolved}/{MIN_SAMPLES})"
        if self.is_drifting:
            return f"🔴 DRIFT! Win: {self.win_rate_pct:.0f}% < {self.threshold_pct:.0f}%"
        return f"🟢 Model OK — Win: {self.win_rate_pct:.0f}%"

    @property
    def trend_icon(self) -> str:
        return {"improving": "📈", "stable": "➡️", "degrading": "📉",
                "insufficient": "❓"}.get(self.trend, "❓")

    def as_dict(self) -> dict:
        return {
            "is_drifting":     self.is_drifting,
            "win_rate_pct":    self.win_rate_pct,
            "n_resolved":      self.n_resolved,
            "n_window":        self.n_window,
            "threshold_pct":   self.threshold_pct,
            "has_enough_data": self.has_enough_data,
            "trend":           self.trend,
            "trend_delta_pct": self.trend_delta_pct,
            "message":         self.message,
            "last_checked":    self.last_checked,
        }


def _no_data_status() -> DriftStatus:
    """Trả về DriftStatus khi chưa có data."""
    return DriftStatus(
        is_drifting=False,
        win_rate_pct=0.0,
        n_resolved=0,
        n_window=DRIFT_WINDOW,
        threshold_pct=DRIFT_THRESHOLD_PCT,
        has_enough_data=False,
        trend="insufficient",
        trend_delta_pct=0.0,
        message="Chưa có dữ liệu resolved. Bấm 'Resolve outcomes' sau T+5.",
        last_checked=datetime.now().isoformat(timespec="seconds"),
    )


# ── Core function ─────────────────────────────────────────────────────────────

def get_drift_status(
    window: int = DRIFT_WINDOW,
    threshold_pct: float = DRIFT_THRESHOLD_PCT,
) -> DriftStatus:
    """
    Tính trạng thái drift từ trade_log.db.

    Parameters
    ----------
    window        : Số trade gần nhất để tính rolling win rate.
    threshold_pct : Win rate (%) dưới ngưỡng này → DRIFT alert.

    Returns
    -------
    DriftStatus với đầy đủ thông tin.
    """
    db = TRADE_LOG_DB
    if not db.exists():
        return _no_data_status()

    try:
        with sqlite3.connect(str(db)) as con:
            # Lấy 2x window trade gần nhất đã resolved để tính trend
            rows = con.execute(
                """
                SELECT outcome
                FROM alpha_signals
                WHERE outcome IN ('WIN', 'LOSS')
                ORDER BY signal_date DESC, id DESC
                LIMIT ?
                """,
                (window * 2,),
            ).fetchall()

            n_total_resolved = con.execute(
                "SELECT COUNT(*) FROM alpha_signals WHERE outcome IN ('WIN','LOSS')"
            ).fetchone()[0]

    except Exception as e:
        log.warning("get_drift_status DB error: %s", e)
        return _no_data_status()

    n_resolved = int(n_total_resolved)
    outcomes   = [r[0] for r in rows]   # newest first

    if n_resolved < MIN_SAMPLES:
        return DriftStatus(
            is_drifting=False,
            win_rate_pct=0.0,
            n_resolved=n_resolved,
            n_window=window,
            threshold_pct=threshold_pct,
            has_enough_data=False,
            trend="insufficient",
            trend_delta_pct=0.0,
            message=f"Chưa đủ data — {n_resolved}/{MIN_SAMPLES} trades resolved.",
            last_checked=datetime.now().isoformat(timespec="seconds"),
        )

    # ── Cửa sổ hiện tại (window trades gần nhất) ────────────────────────────
    current_window = outcomes[:window]
    wins_current   = current_window.count("WIN")
    wr_current     = wins_current / len(current_window) * 100

    # ── Cửa sổ trước đó (window trades kế tiếp) ─────────────────────────────
    prev_window = outcomes[window:window * 2]
    if len(prev_window) >= 5:
        wins_prev = prev_window.count("WIN")
        wr_prev   = wins_prev / len(prev_window) * 100
        delta     = round(wr_current - wr_prev, 1)
        if delta >= 5:
            trend = "improving"
        elif delta <= -5:
            trend = "degrading"
        else:
            trend = "stable"
    else:
        wr_prev = 0.0
        delta   = 0.0
        trend   = "insufficient"

    is_drifting = wr_current < threshold_pct
    wr_pct      = round(wr_current, 1)

    # ── Message ──────────────────────────────────────────────────────────────
    if is_drifting:
        msg = (
            f"🚨 DRIFT ALERT — Win rate {wr_pct:.0f}% < {threshold_pct:.0f}% "
            f"trong {len(current_window)} trades gần nhất. "
            f"Cân nhắc retrain model với dữ liệu mới hơn."
        )
    else:
        trend_txt = {
            "improving":   f"↑ tăng {abs(delta):.0f}pp so với window trước",
            "degrading":   f"↓ giảm {abs(delta):.0f}pp so với window trước",
            "stable":      "ổn định",
            "insufficient":"(chưa đủ data để so sánh trend)",
        }.get(trend, "")
        msg = f"Model ổn định — Win rate {wr_pct:.0f}% ({trend_txt}). {n_resolved} trades đã resolved."

    log.info(
        "DriftStatus: win_rate=%.1f%% n=%d window=%d drift=%s trend=%s",
        wr_pct, n_resolved, window, is_drifting, trend
    )

    return DriftStatus(
        is_drifting=is_drifting,
        win_rate_pct=wr_pct,
        n_resolved=n_resolved,
        n_window=window,
        threshold_pct=threshold_pct,
        has_enough_data=True,
        trend=trend,
        trend_delta_pct=delta,
        message=msg,
        last_checked=datetime.now().isoformat(timespec="seconds"),
    )


# ── Rolling win rate series ───────────────────────────────────────────────────

def get_rolling_series(n_total: int = 100) -> "pd.DataFrame":
    """
    Trả về DataFrame rolling win rate theo thời gian để vẽ biểu đồ.

    Mỗi ngày tính win rate trên window=20 giao dịch tính đến ngày đó.

    Columns: signal_date, cum_resolved, cum_wins, cumulative_win_rate_pct,
             rolling_20_win_rate_pct (NaN nếu chưa đủ 20 trades).
    """
    import pandas as pd

    db = TRADE_LOG_DB
    if not db.exists():
        return pd.DataFrame(columns=[
            "signal_date", "cum_resolved", "cum_wins",
            "cumulative_win_rate_pct", "rolling_20_win_rate_pct",
        ])

    try:
        with sqlite3.connect(str(db)) as con:
            df = pd.read_sql_query(
                """
                SELECT signal_date, outcome
                FROM alpha_signals
                WHERE outcome IN ('WIN', 'LOSS')
                ORDER BY signal_date ASC, id ASC
                LIMIT ?
                """,
                con,
                params=(n_total,),
            )
    except Exception as e:
        log.warning("get_rolling_series error: %s", e)
        return pd.DataFrame()

    if df.empty:
        return df

    # Tính theo từng ngày (groupby date, sort ascending)
    daily = (
        df.groupby("signal_date")
        .agg(wins=("outcome", lambda x: (x == "WIN").sum()), trades=("outcome", "count"))
        .reset_index()
        .sort_values("signal_date")
    )

    daily["cum_wins"]     = daily["wins"].cumsum()
    daily["cum_resolved"] = daily["trades"].cumsum()
    daily["cumulative_win_rate_pct"] = (daily["cum_wins"] / daily["cum_resolved"] * 100).round(1)

    # Rolling 20-trade win rate (expand window thực sự rolling theo row tích lũy)
    # Dùng cumsum approach: rolling trên từng giao dịch, rồi group theo ngày
    # Đơn giản hơn: dùng rolling trên cumulative data
    daily["rolling_20_win_rate_pct"] = (
        daily["cum_wins"].diff(20).fillna(daily["cum_wins"])
        / daily["cum_resolved"].diff(20).fillna(daily["cum_resolved"])
        * 100
    ).where(daily["cum_resolved"] >= 20).round(1)

    return daily[[
        "signal_date", "cum_resolved", "cum_wins",
        "cumulative_win_rate_pct", "rolling_20_win_rate_pct",
    ]]
