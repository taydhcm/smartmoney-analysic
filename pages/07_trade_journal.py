"""
pages/07_trade_journal.py
E2 Trade Logger + E3 P&L Tracker UI — Sprint 8.

Tabs:
  1. 📝 Nhập Lệnh   — form thêm trade paper/real + đóng lệnh đang mở
  2. 📊 Nhật Ký     — bảng tất cả trades + close button
  3. 📈 P&L & Stats — equity curve, win rate, streak, max drawdown
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Trade Journal", layout="wide")
st.title("📒 Trade Journal  (E2 + E3)")
st.caption("Nhật ký lệnh paper/real · P&L Tracker · Equity Curve · Sprint 8")

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from data.trade_logger import (
        add_trade, close_trade, get_all_trades, get_open_trades, get_trade_stats,
        update_unrealized_pnl,
    )
    from ml.pnl_tracker import compute_equity_curve, compute_stats, compute_win_streak
    _BACKEND_OK = True
except Exception as _e:
    _BACKEND_OK = False
    st.error(f"Lỗi import backend: {_e}")
    st.stop()

# ── Sidebar: vốn ban đầu ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cấu hình")
    init_capital = st.number_input(
        "Vốn ban đầu (triệu VND)",
        min_value=100,
        max_value=100_000,
        value=1_000,
        step=100,
    ) * 1_000_000
    trade_type_filter = st.radio(
        "Loại lệnh hiển thị",
        options=["Tất cả", "paper", "real"],
        horizontal=True,
    )
    st.divider()
    if st.button("🔄 Làm mới dữ liệu"):
        st.rerun()

_ttype = None if trade_type_filter == "Tất cả" else trade_type_filter

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_entry, tab_journal, tab_pnl = st.tabs(["📝 Nhập Lệnh", "📊 Nhật Ký", "📈 P&L & Stats"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Nhập Lệnh
# ─────────────────────────────────────────────────────────────────────────────
with tab_entry:
    st.subheader("➕ Thêm lệnh mới")

    with st.form("form_add_trade", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            f_ticker       = st.text_input("Ticker", value="", placeholder="VIC").upper()
            f_entry_price  = st.number_input("Giá vào lệnh (VND)", min_value=0, value=50000, step=100)
            f_sl_price     = st.number_input("Stop-Loss (VND)",     min_value=0, value=47000, step=100)
            f_target_price = st.number_input("Target (VND)",        min_value=0, value=55000, step=100)
        with col2:
            f_shares      = st.number_input("Số cổ phiếu", min_value=100, value=1000, step=100)
            f_entry_date  = st.date_input("Ngày vào lệnh", value=date.today())
            f_trade_type  = st.radio("Loại lệnh", ["paper", "real"], horizontal=True)
            f_note        = st.text_area("Ghi chú", height=68)
        submitted = st.form_submit_button("✅ Thêm lệnh", type="primary", width='stretch')

    if submitted:
        if not f_ticker:
            st.error("Vui lòng nhập ticker.")
        elif f_entry_price <= 0:
            st.error("Giá vào lệnh phải > 0.")
        elif f_sl_price >= f_entry_price:
            st.error("Stop-Loss phải thấp hơn giá vào.")
        elif f_target_price <= f_entry_price:
            st.error("Target phải cao hơn giá vào.")
        else:
            try:
                tid = add_trade(
                    ticker       = f_ticker,
                    entry_date   = f_entry_date.isoformat(),
                    entry_price  = f_entry_price,
                    shares       = f_shares,
                    sl_price     = f_sl_price,
                    target_price = f_target_price,
                    trade_type   = f_trade_type,
                    note         = f_note,
                )
                rr = round((f_target_price - f_entry_price) / (f_entry_price - f_sl_price), 2)
                st.success(
                    f"✅ Đã thêm lệnh **{f_ticker}** (id={tid})  |  "
                    f"Entry: {f_entry_price:,}  |  SL: {f_sl_price:,}  |  "
                    f"Target: {f_target_price:,}  |  R:R 1:{rr}"
                )
            except Exception as e:
                st.error(f"Lỗi: {e}")

    st.divider()
    st.subheader("🔒 Đóng lệnh đang mở")
    open_trades = get_open_trades(_ttype)
    if not open_trades:
        st.info("Không có lệnh đang mở.")
    else:
        open_df = pd.DataFrame(open_trades)
        for _, row in open_df.iterrows():
            with st.expander(
                f"📌 #{int(row['id'])} **{row['ticker']}**  ×{int(row['shares']):,} cp  "
                f"@ {row['entry_price']:,.0f}  [{row['trade_type'].upper()}]  {row['entry_date']}"
            ):
                cl1, cl2, cl3 = st.columns(3)
                cl1.metric("Entry", f"{row['entry_price']:,.0f}")
                cl2.metric("SL",    f"{row['sl_price']:,.0f}")
                cl3.metric("Target",f"{row['target_price']:,.0f}")
                with st.form(f"close_{int(row['id'])}"):
                    cx1, cx2 = st.columns(2)
                    c_exit_price = cx1.number_input(
                        "Giá đóng (VND)", min_value=0,
                        value=int(row["entry_price"]), step=100,
                        key=f"ep_{int(row['id'])}",
                    )
                    c_exit_date  = cx2.date_input(
                        "Ngày đóng", value=date.today(),
                        key=f"ed_{int(row['id'])}",
                    )
                    c_status = st.radio(
                        "Lý do đóng", ["closed", "stopped"],
                        horizontal=True, key=f"st_{int(row['id'])}",
                    )
                    if st.form_submit_button("🔒 Đóng lệnh", type="primary"):
                        try:
                            closed_t = close_trade(
                                int(row["id"]), c_exit_date.isoformat(), c_exit_price, c_status
                            )
                            pnl_pct = closed_t.get("pnl_pct", 0) or 0
                            pnl_vnd = closed_t.get("pnl_vnd", 0) or 0
                            _fn = st.success if pnl_pct > 0 else st.error if pnl_pct < 0 else st.info
                            _fn(
                                f"Đóng **{row['ticker']}** @ {c_exit_price:,}  |  "
                                f"PnL: **{pnl_pct:+.2f}%**  ({pnl_vnd:+,.0f} VND)"
                            )
                            st.rerun()
                        except Exception as e:
                            st.error(f"Lỗi: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Nhật Ký
# ─────────────────────────────────────────────────────────────────────────────
with tab_journal:
    st.subheader("📋 Nhật Ký Lệnh")
    all_trades = get_all_trades(_ttype)
    if not all_trades:
        st.info("Chưa có lệnh nào. Hãy nhập lệnh đầu tiên ở tab 'Nhập Lệnh'.")
    else:
        df_all = pd.DataFrame(all_trades)
        # Màu theo status
        _STATUS_ICON = {"open": "🟡 Open", "closed": "✅ Closed", "stopped": "🛑 Stopped"}
        df_all["status_icon"] = df_all["status"].map(_STATUS_ICON).fillna(df_all["status"])
        df_all["pnl_display"] = df_all["pnl_pct"].apply(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "—"
        )
        df_all["pnl_vnd_display"] = df_all["pnl_vnd"].apply(
            lambda x: f"{x:+,.0f}" if pd.notna(x) else "—"
        )

        st.dataframe(
            df_all[[
                "id", "trade_type", "ticker", "entry_date", "entry_price",
                "shares", "sl_price", "target_price", "status_icon",
                "exit_date", "exit_price", "pnl_display", "pnl_vnd_display", "note"
            ]].rename(columns={
                "id": "ID", "trade_type": "Loại", "ticker": "Ticker",
                "entry_date": "Ngày vào", "entry_price": "Giá vào",
                "shares": "Cổ phiếu", "sl_price": "SL", "target_price": "Target",
                "status_icon": "Trạng thái", "exit_date": "Ngày đóng",
                "exit_price": "Giá đóng", "pnl_display": "PnL%",
                "pnl_vnd_display": "PnL (VND)", "note": "Ghi chú"
            }),
            hide_index=True,
        )

        # Quick stats bar
        _stats = get_trade_stats(_ttype)
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Tổng lệnh",   _stats["total"])
        mc2.metric("Đang mở",     _stats["open_count"])
        mc3.metric("Win Rate",    f"{_stats['win_rate']:.0%}")
        mc4.metric("PnL Total",   f"{_stats['total_pnl_vnd']:+,.0f} VND")
        mc5.metric("Profit Factor", f"{_stats['profit_factor']:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — P&L & Stats
# ─────────────────────────────────────────────────────────────────────────────
with tab_pnl:
    st.subheader("📈 P&L Dashboard")

    all_trades = get_all_trades(_ttype)
    if not all_trades:
        st.info("Chưa có dữ liệu. Nhập và đóng ít nhất 1 lệnh để xem stats.")
    else:
        stats = compute_stats(all_trades, init_capital)

        # KPI row 1
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Tổng lệnh",   f"{stats['total']}")
        k2.metric("Win Rate",     f"{stats['win_rate']:.0%}",
                  delta=f"{stats['win_count']}W / {stats['loss_count']}L")
        k3.metric("Avg Win",      f"{stats['avg_win_pct']:+.2f}%")
        k4.metric("Avg Loss",     f"{stats['avg_loss_pct']:+.2f}%")
        k5.metric("Profit Factor",f"{stats['profit_factor']:.2f}")
        k6.metric("Sharpe",       f"{stats['sharpe_ratio']:.2f}")

        # KPI row 2
        k7, k8, k9, k10 = st.columns(4)
        k7.metric("Max Drawdown",   f"{stats['max_drawdown_pct']:+.2f}%")
        k8.metric("Total PnL (VND)", f"{stats['total_pnl_vnd']:+,.0f}")
        k9.metric("Streak hiện tại",
                  f"{'📈' if stats['current_streak'] > 0 else '📉'} "
                  f"{abs(stats['current_streak'])} lệnh",
                  delta="thắng liên tiếp" if stats['current_streak'] > 0 else "thua liên tiếp",
                  delta_color="normal" if stats['current_streak'] > 0 else "inverse")
        k10.metric("Max Win Streak", f"{stats['max_win_streak']} lệnh")

        # Equity curve chart
        eq_df = compute_equity_curve(all_trades, init_capital)
        if not eq_df.empty:
            st.markdown("---")
            st.markdown("**📉 Equity Curve**")
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=eq_df["date"],
                y=eq_df["equity"],
                mode="lines+markers",
                name="Equity",
                line=dict(color="#00cc66", width=2),
                fill="tozeroy",
                fillcolor="rgba(0,204,102,0.1)",
                hovertemplate="<b>%{x}</b><br>Equity: %{y:,.0f} VND<extra></extra>",
            ))
            fig_eq.add_hline(
                y=init_capital, line_dash="dash", line_color="white",
                opacity=0.4, annotation_text="Vốn ban đầu",
            )
            fig_eq.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(tickformat=",.0f"),
            )
            st.plotly_chart(fig_eq)

            # Drawdown chart
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Bar(
                x=eq_df["date"],
                y=eq_df["drawdown_pct"] * 100,
                name="Drawdown %",
                marker_color="#ff4444",
                hovertemplate="<b>%{x}</b><br>DD: %{y:.2f}%<extra></extra>",
            ))
            fig_dd.update_layout(
                title="Drawdown %",
                height=180,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_dd)

        # PnL distribution
        closed_trades = [t for t in all_trades if t.get("pnl_pct") is not None]
        if len(closed_trades) >= 3:
            pnl_vals = [t["pnl_pct"] for t in closed_trades]
            fig_dist = go.Figure(go.Histogram(
                x=pnl_vals,
                nbinsx=15,
                marker_color=["#00cc66" if p > 0 else "#ff4444" for p in pnl_vals],
                name="PnL %",
            ))
            fig_dist.update_layout(
                title="Phân phối PnL %",
                height=200,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_dist)
