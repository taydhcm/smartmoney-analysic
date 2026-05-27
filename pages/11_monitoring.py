"""
pages/11_monitoring.py
Sprint 11 — Long-term Monitoring Dashboard.

Theo dõi sức khoẻ hệ thống hằng ngày:
  - KPI row: D0.2 depth, model age, E6 freshness, real trade count
  - Daily Health Status: alerts + summary
  - 30-Trade Milestone: progress bar + remaining
  - Scale Readiness: 5 conditions table + ScaleDecision banner
  - Retrain Reminder: badge + instructions
  - D0.2 History Depth: snapshot stats
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Monitoring (S11)", layout="wide")
st.title("📈 Long-term Monitoring  (Sprint 11)")
st.caption("Daily Health · Milestone Tracker · Scale Advisor · Retrain Reminder")

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from ml.monitor import run_daily_health_check, D02_MIN_SESSIONS, MILESTONE_REAL, RETRAIN_STALE_DAYS, E6_STALE_DAYS
    from ml.scale_advisor import compute_scale_recommendation, check_scale_conditions, SCALE_STAGES
    from data.trade_logger import get_trade_stats, get_all_trades
    from ml.rolling_calibrator import get_calibrator_status
    from ml.pnl_tracker import compute_stats, compute_equity_curve
    _BACKEND_OK = True
except Exception as _e:
    _BACKEND_OK = False
    st.error(f"Lỗi import backend: {_e}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    capital_m = st.number_input(
        "Portfolio Capital (Triệu VND)",
        min_value=0, max_value=10_000, value=500, step=50,
    )
    capital = capital_m * 1_000_000

    current_max_pct = st.slider(
        "Max position / lệnh hiện tại (%)", 5, 40, 20, 5,
    ) / 100

    st.divider()
    if st.button("🔄 Refresh", type="primary", width='stretch'):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.subheader("🔗 Quick Links")
    st.page_link("pages/09_self_learning.py", label="→ E6 Rolling Calibrator")
    st.page_link("pages/10_go_live.py",       label="→ Go-Live Validator")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Đang tải dữ liệu…")
def _load(capital: float, current_max_pct: float):
    health = run_daily_health_check(capital=capital)

    trade_stats_all  = get_trade_stats()
    trade_stats_real = get_trade_stats(trade_type="real")

    calibr_status = get_calibrator_status()
    # drift_alert may come from outcome_tracker — attach to calibr_status dict
    try:
        from ml.outcome_tracker import get_signal_stats
        sig_stats = get_signal_stats()
        calibr_status["drift_alert"] = sig_stats.get("drift_alert", False)
    except Exception:
        calibr_status.setdefault("drift_alert", False)

    real_trades = get_all_trades(trade_type="real")
    closed_real  = [t for t in real_trades if t.get("status") in ("closed", "stopped")]

    pnl_stats = {}
    if closed_real:
        pnl_stats = compute_stats(closed_real, initial_capital=capital if capital > 0 else 1_000_000_000)

    scale = compute_scale_recommendation(
        trade_stats    = trade_stats_real,
        calibr_status  = calibr_status,
        pnl_stats      = pnl_stats,
        current_max_pct= current_max_pct,
    )
    return health, trade_stats_real, calibr_status, pnl_stats, scale


with st.spinner("Đang kiểm tra hệ thống…"):
    try:
        health, trade_stats_real, calibr_status, pnl_stats, scale = _load(capital, current_max_pct)
    except Exception as exc:
        st.error(f"Lỗi khi chạy health check: {exc}")
        st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — KPI Row
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📊 Daily Health Status")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(
    "📅 D0.2 Snapshot Days",
    f"{health.d02_session_days}",
    delta=f"cần {D02_MIN_SESSIONS}" if not health.d02_ready else "✅ OK",
    delta_color="inverse" if not health.d02_ready else "normal",
)
k2.metric(
    "🧠 Model Age",
    f"{health.model_age_days:.0f}d" if health.model_age_days is not None else "N/A",
    delta="RETRAIN" if health.retrain_due else "Fresh",
    delta_color="inverse" if health.retrain_due else "normal",
)
k3.metric(
    "🎯 E6 Calibrator",
    f"{health.e6_last_fit_days:.0f}d" if health.e6_last_fit_days is not None else "Chưa fit",
    delta="STALE" if health.e6_stale else "Fresh",
    delta_color="inverse" if health.e6_stale else "normal",
)
k4.metric(
    "📊 Real Trades Closed",
    f"{health.real_trades_closed}",
    delta="✅ Milestone đạt" if health.milestone_30 else f"{30 - health.real_trades_closed} còn lại",
    delta_color="normal" if health.milestone_30 else "off",
)
k5.metric(
    "🚀 Scale Ready",
    "Có" if health.scale_ready else "Chưa",
    delta_color="normal" if health.scale_ready else "inverse",
)

# Alerts
if health.alerts:
    st.warning("**⚠️ Cảnh báo hệ thống:**")
    for alert in health.alerts:
        st.warning(f"• {alert}")
else:
    st.success("✅ Tất cả hệ thống hoạt động bình thường.")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — 30-Trade Milestone
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader(f"🎯 Milestone: {MILESTONE_REAL} Lệnh Real")

ms_col1, ms_col2 = st.columns([2, 1])
with ms_col1:
    st.progress(
        health.milestone_pct / 100,
        text=f"{health.real_trades_closed}/{MILESTONE_REAL} lệnh real đã đóng "
             f"({health.milestone_pct:.0f}%)",
    )
    if health.milestone_30:
        st.success(
            "🏆 **Milestone đạt!** Đủ dữ liệu để đánh giá scale vốn."
        )
    else:
        remaining = MILESTONE_REAL - health.real_trades_closed
        st.info(
            f"⏳ Cần thêm **{remaining} lệnh real** (closed/stopped) để đủ điều kiện "
            "đánh giá scale vốn."
        )

with ms_col2:
    st.subheader("E5 Signal Stats")
    st.metric("Tổng tín hiệu",  health.total_signals)
    st.metric("Chờ outcome",     health.pending_outcomes)
    st.metric("Đã resolved",    health.resolved_outcomes)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Scale Readiness
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📈 Scale Readiness")

# Banner
if scale.can_scale and scale.next_stage:
    st.success(
        f"🚀 **CÓ THỂ SCALE!** {scale.reasoning}\n\n"
        f"Stage {scale.current_stage} → Stage {scale.next_stage['stage']}: "
        f"**{scale.next_stage['label']}** ({scale.next_stage['description']})"
    )
elif scale.can_scale and not scale.next_stage:
    st.info(f"✅ {scale.reasoning}")
else:
    st.warning(
        f"⏳ **Chưa đủ điều kiện scale** ({scale.score}/{scale.total_conditions} điều kiện)\n\n"
        f"{scale.reasoning}"
    )

# Score bar
st.progress(
    scale.score_pct,
    text=f"Scale Score: {scale.score}/{scale.total_conditions}",
)

# Conditions table
import pandas as pd

cond_rows = []
for c in scale.conditions:
    status = "✅" if c.passed else "❌"
    if c.key == "max_drawdown_pct":
        val_str = f"{c.value:.1f}%"
        thr_str = f"≥ {c.threshold:.0f}%"
    elif c.key == "drift_alert_clear":
        val_str = "Clear" if c.value >= 1 else "ALERT"
        thr_str = "Clear (0 alerts)"
    elif c.key in ("roll_30d_prec", "win_rate"):
        val_str = f"{c.value:.1%}"
        thr_str = f"≥ {c.threshold:.0%}"
    else:
        val_str = str(int(c.value))
        thr_str = f"≥ {int(c.threshold)}"
    cond_rows.append({
        "Status":     status,
        "Điều kiện":  c.label,
        "Giá trị":    val_str,
        "Ngưỡng":     thr_str,
    })

st.dataframe(
    pd.DataFrame(cond_rows),
    hide_index=True,
    column_config={
        "Status":    st.column_config.TextColumn(width="small"),
        "Điều kiện": st.column_config.TextColumn(width="medium"),
        "Giá trị":   st.column_config.TextColumn(width="small"),
        "Ngưỡng":    st.column_config.TextColumn(width="small"),
    },
)

# Stage Roadmap
st.subheader("🗺️ Scale Stage Roadmap")
stage_cols = st.columns(len(SCALE_STAGES))
for col, stage in zip(stage_cols, SCALE_STAGES):
    is_current = stage["stage"] == scale.current_stage
    badge = "🟢" if is_current else ("✅" if stage["stage"] < scale.current_stage else "⬜")
    col.markdown(
        f"**{badge} Stage {stage['stage']}: {stage['label']}**\n\n"
        f"- Max pos: **{stage['max_position_pct']:.0%}**\n"
        f"- Min trades: {stage['min_trades']}\n"
        f"- {stage['description']}"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Retrain Reminder
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("🔁 Bi-weekly Retrain Reminder")

r1, r2 = st.columns([1, 2])
with r1:
    if health.retrain_due:
        st.error(
            f"⚠️ **RETRAIN DUE**\n\n"
            f"{'Model chưa được train.' if not health.model_trained_at else f'Đã {health.model_age_days:.0f} ngày kể từ lần train cuối (ngưỡng: {RETRAIN_STALE_DAYS}d).'}"
        )
    else:
        st.success(
            f"✅ Model còn tươi\n\n"
            f"Train lúc: `{(health.model_trained_at or '')[:10]}`\n\n"
            f"Độ tuổi: **{health.model_age_days:.0f}** ngày"
        )

with r2:
    st.info(
        "**Hướng dẫn retrain:**\n"
        "```\n"
        "python scripts/train_pipeline.py --retrain\n"
        "```\n"
        f"Khuyến nghị: retrain mỗi **{RETRAIN_STALE_DAYS} ngày** (2 tuần) "
        "hoặc khi có Drift Alert từ E5/E6."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — D0.2 History Depth
# ═══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📅 D0.2 Snapshot History")

d_cols = st.columns(4)
d_cols[0].metric("Số phiên đã log",   health.d02_session_days, delta=f"cần {D02_MIN_SESSIONS}")
d_cols[1].metric("Số ticker",         health.d02_tickers)
d_cols[2].metric("Phiên đầu tiên",   health.d02_oldest_date or "—")
d_cols[3].metric("Phiên gần nhất",   health.d02_newest_date or "—")

if health.d02_ready:
    st.success(f"✅ D0.2 đủ lịch sử ({health.d02_session_days} ≥ {D02_MIN_SESSIONS} phiên).")
else:
    pct = min(100, health.d02_session_days / D02_MIN_SESSIONS * 100)
    st.progress(pct / 100, text=f"Progress: {health.d02_session_days}/{D02_MIN_SESSIONS} phiên ({pct:.0f}%)")
    st.warning(
        f"D0.2 cần thêm **{D02_MIN_SESSIONS - health.d02_session_days} phiên** "
        "để đủ dữ liệu cho S4 features. Chạy `scripts/daily_snapshot.py` mỗi ngày giao dịch."
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Sprint 11 · Scale Advisor + System Monitor · "
    f"Timestamp: `{health.timestamp[:19]}`"
)
