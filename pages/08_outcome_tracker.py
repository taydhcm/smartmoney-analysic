"""
pages/08_outcome_tracker.py
E5 Outcome Tracker UI — Sprint 8.

Hiển thị:
  - Tất cả signal đã phát ra (signal_log)
  - Kết quả T+5 (WIN/LOSS/PENDING/FLAT/EXPIRED)
  - Drift Alert khi rolling precision < 0.30
  - Nút "🔍 Kiểm tra T+5 ngay" để trigger check_pending_outcomes()
  - Stats: precision, avg_pnl, win_rate_30d

Hybrid Self-Learning (E5 → E6 Sprint 9):
  Drift Alert banner → nhắc retrain E6 Rolling Calibrator.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Outcome Tracker (E5)", layout="wide")
st.title("🎯 Outcome Tracker  (E5)")
st.caption(
    "Ghi nhận signal phát ra · Auto-check T+5 outcome · Hybrid Self-Learning foundation · Sprint 8"
)

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from ml.outcome_tracker import (
        check_pending_outcomes,
        get_outcome_table,
        get_recent_signals,
        get_signal_stats,
        log_signals_batch,
    )
    _BACKEND_OK = True
except Exception as _e:
    _BACKEND_OK = False
    st.error(f"Lỗi import backend: {_e}")
    st.stop()

# ── Signal Stats & Drift Alert ────────────────────────────────────────────────
stats = get_signal_stats()

# Drift Alert Banner (Hybrid Self-Learning)
if stats["drift_alert"]:
    st.error(
        "🚨 **DRIFT ALERT — Hybrid Self-Learning**\n\n"
        f"Rolling precision 30 ngày: **{stats['roll_30d_prec']:.1%}** "
        f"(trên {stats['roll_30d_total']} signal) — dưới ngưỡng 30%.\n\n"
        "Hệ thống cần **retrain model** hoặc re-fit **E6 Rolling Calibrator** (Sprint 9)."
    )
elif stats["total_signals"] > 0:
    prec_color = "success" if stats["precision"] >= 0.40 else "warning" if stats["precision"] >= 0.30 else "error"
    getattr(st, prec_color)(
        f"📊 Precision tổng thể: **{stats['precision']:.1%}**  |  "
        f"Rolling 30d: **{stats['roll_30d_prec']:.1%}**  |  "
        f"Avg PnL: **{stats['avg_pnl_pct']:+.2f}%**"
    )

# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tổng signal",        stats["total_signals"])
c2.metric("Đã resolve",         stats["resolved"])
c3.metric("Wins",               stats["wins"])
c4.metric("Precision",          f"{stats['precision']:.1%}")
c5.metric("Avg PnL/signal",     f"{stats['avg_pnl_pct']:+.2f}%")

# ── Actions ───────────────────────────────────────────────────────────────────
st.markdown("---")
col_act1, col_act2, col_act3 = st.columns([2, 2, 3])

with col_act1:
    if st.button("🔍 Kiểm tra T+5 ngay", type="primary", width='stretch',
                 help="Tự động resolve signal PENDING có T+5 <= hôm nay"):
        with st.spinner("Đang check outcome T+5..."):
            try:
                results = check_pending_outcomes(date.today())
                if results:
                    st.success(f"✅ Đã resolve **{len(results)}** signal:")
                    for r in results:
                        icon = "🟢" if r["outcome"] == "WIN" else "🔴" if r["outcome"] == "LOSS" else "⚪"
                        pnl = f"{r['pnl_pct']:+.2f}%" if r.get("pnl_pct") is not None else ""
                        st.caption(f"{icon} {r['ticker']} ({r['signal_date']}) → **{r['outcome']}** {pnl}")
                else:
                    st.info("Không có signal nào cần resolve hôm nay.")
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi check outcome: {e}")

with col_act2:
    if st.button("🌅 Log signal từ Alpha Picks", width='stretch',
                 help="Lấy alert cards từ session state (nếu đã chạy prediction)"):
        _picks_key = "alpha_picks"
        _alerts_key = "alpha_alerts"
        try:
            from ml.alert_generator import generate_morning_report
            picks = st.session_state.get(_picks_key, [])
            if not picks:
                st.warning("Chưa có alpha picks. Chạy prediction ở trang Alpha Signals trước.")
            else:
                alerts = generate_morning_report(picks, capital=0.0)
                if alerts:
                    ids = log_signals_batch(alerts)
                    st.success(f"✅ Đã log **{len(ids)}** signal vào signal_log.db")
                    st.rerun()
                else:
                    st.warning("Không tạo được alert cards từ picks hiện tại.")
        except Exception as e:
            st.error(f"Lỗi: {e}")

with col_act3:
    st.caption(
        "💡 **Workflow**: Chạy Alpha Signals → Log signal → T+5 tự động resolve lúc 15:15\n\n"
        "Precision < 30% × 10 ngày liên tục → DRIFT ALERT → Re-fit Calibrator (Sprint 9)"
    )

# ── Signal Outcome Table ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Signal Log & Outcomes")

_outcome_filter = st.multiselect(
    "Lọc outcome",
    options=["WIN", "LOSS", "PENDING", "FLAT", "EXPIRED"],
    default=["WIN", "LOSS", "PENDING"],
)

table = get_outcome_table()
if not table:
    st.info("Chưa có signal nào. Chạy Alpha Signals và bấm 'Log signal từ Alpha Picks'.")
else:
    df = pd.DataFrame(table)
    if _outcome_filter:
        df = df[df["outcome"].isin(_outcome_filter)]

    if df.empty:
        st.info(f"Không có signal nào với outcome: {', '.join(_outcome_filter)}")
    else:
        # Styling
        _OUTCOME_ICON = {
            "WIN":     "🟢 WIN",
            "LOSS":    "🔴 LOSS",
            "FLAT":    "⚪ FLAT",
            "PENDING": "⏳ PENDING",
            "EXPIRED": "❓ EXPIRED",
        }
        df["outcome_icon"] = df["outcome"].map(_OUTCOME_ICON).fillna(df["outcome"])
        df["pnl_display"]  = df["pnl_pct"].apply(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"
        )
        df["p_cal_display"] = df["p_calibrated"].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "—"
        )

        display_cols = {
            "signal_date":    "Ngày signal",
            "ticker":         "Ticker",
            "recommendation": "Recommendation",
            "entry_price":    "Entry",
            "sl_price":       "SL",
            "target_price":   "Target",
            "p_cal_display":  "P_cal",
            "regime_state":   "Regime",
            "outcome_icon":   "Outcome",
            "exit_price":     "Exit",
            "pnl_display":    "PnL%",
            "outcome_date":   "T+5 Date",
            "reason":         "Lý do",
        }

        st.dataframe(
            df[[c for c in display_cols if c in df.columns]].rename(columns=display_cols),
            hide_index=True,
        )

        # Outcome distribution chart
        if len(df) >= 3:
            outcome_counts = df["outcome"].value_counts()
            _COLORS = {
                "WIN": "#00cc66", "LOSS": "#ff4444",
                "FLAT": "#888888", "PENDING": "#ffaa00", "EXPIRED": "#aaaaaa"
            }
            fig_pie = go.Figure(go.Pie(
                labels=outcome_counts.index.tolist(),
                values=outcome_counts.values.tolist(),
                marker_colors=[_COLORS.get(o, "#888") for o in outcome_counts.index],
                hole=0.4,
                textinfo="label+percent+value",
            ))
            fig_pie.update_layout(
                title="Phân phối Outcome",
                height=300,
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            # Precision over time (resolved signals by month)
            resolved_df = df[df["outcome"].isin(["WIN", "LOSS", "FLAT"])].copy()
            if len(resolved_df) >= 5 and "signal_date" in resolved_df.columns:
                resolved_df["month"] = pd.to_datetime(resolved_df["signal_date"]).dt.to_period("M").astype(str)
                monthly = (
                    resolved_df.groupby("month")
                    .agg(wins=("outcome", lambda x: (x == "WIN").sum()),
                         total=("outcome", "count"))
                    .reset_index()
                )
                monthly["precision"] = monthly["wins"] / monthly["total"]

                fig_prec = go.Figure()
                fig_prec.add_trace(go.Bar(
                    x=monthly["month"], y=monthly["total"],
                    name="Signals", marker_color="#4488ff", opacity=0.4,
                ))
                fig_prec.add_trace(go.Scatter(
                    x=monthly["month"], y=monthly["precision"],
                    name="Precision", mode="lines+markers",
                    line=dict(color="#ffaa00", width=2),
                    yaxis="y2",
                ))
                fig_prec.add_hline(
                    y=0.30, yref="y2", line_dash="dash", line_color="red",
                    opacity=0.6, annotation_text="Drift Alert 30%",
                )
                fig_prec.update_layout(
                    title="Precision theo tháng",
                    height=260,
                    margin=dict(l=0, r=0, t=40, b=0),
                    yaxis=dict(title="# Signals"),
                    yaxis2=dict(title="Precision", overlaying="y", side="right",
                                tickformat=".0%", range=[0, 1]),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                ch1, ch2 = st.columns([1, 2])
                ch1.plotly_chart(fig_pie)
                ch2.plotly_chart(fig_prec)
            else:
                st.plotly_chart(fig_pie)
