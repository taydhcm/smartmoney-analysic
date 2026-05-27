"""
pages/12_institutional_flow.py
Sprint 12 — Trang Dashboard Dòng Tiền Tổ Chức (Institutional Flow Dashboard).

Hiển thị:
  - Dòng tiền Nước Ngoài (khối ngoại) — từ D0.2 SQLite
  - Dòng tiền Tự Doanh — từ SSI iBoard (real-time fetch)
  - Combined Institutional Score per ticker
  - Bảng so sánh tất cả VN30
  - Trạng thái kết nối SSI
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import streamlit as st
import pandas as pd
import numpy as np

# Load .env trước mọi import
try:
    from analytics.secret_manager import load_env
    load_env()
except Exception:
    pass

st.set_page_config(
    page_title="S12 Institutional Flow",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 S12 — Dòng Tiền Tổ Chức")
st.caption(
    "Kết hợp dòng tiền **Nước Ngoài** (D0.2 SQLite) + **Tự Doanh** (SSI iBoard) "
    "→ Combined Institutional Score"
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cài đặt")

    from config.constants import VN30_TICKERS
    selected_tickers = st.multiselect(
        "Chọn mã theo dõi",
        options=VN30_TICKERS,
        default=VN30_TICKERS[:10],
    )
    if not selected_tickers:
        selected_tickers = VN30_TICKERS[:10]

    show_all = st.checkbox("Hiện toàn bộ VN30", value=False)
    if show_all:
        selected_tickers = VN30_TICKERS

    st.divider()
    st.subheader("🔌 Trạng thái SSI iBoard")
    test_ssi = st.button("Kiểm tra kết nối SSI")


# ── SSI connection test ────────────────────────────────────────────────────────
if test_ssi:
    with st.spinner("Đang kết nối SSI iBoard..."):
        try:
            from analytics.ssi_iboard import get_token, fetch_investor_flow
            token = get_token()
            if token:
                st.sidebar.success("✅ SSI iBoard: Kết nối OK")
                # Test fetch 1 ticker
                test_df = fetch_investor_flow("ACB", limit=3)
                if not test_df.empty:
                    st.sidebar.info(f"📊 Test ACB: {len(test_df)} phiên data")
                else:
                    st.sidebar.warning("⚠️ Auth OK nhưng chưa lấy được data ACB")
            else:
                st.sidebar.error("❌ SSI iBoard: Đăng nhập thất bại")
                st.sidebar.caption("Kiểm tra credentials trong .env")
        except Exception as exc:
            st.sidebar.error(f"❌ Lỗi: {exc}")


# ── Main data loading ──────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)   # cache 30 phút
def load_institutional_flow(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Load institutional flow cho tất cả tickers đã chọn."""
    from ml.smart_money import compute_institutional_flow
    rows = []
    for ticker in tickers:
        try:
            flow = compute_institutional_flow(ticker)
            rows.append({
                "Mã":                    ticker,
                "F.Net 5D (%)":          round(flow.foreign.foreign_net_5d * 100, 2),
                "F.Trend":               round(flow.foreign.foreign_trend, 3),
                "Ngoại Label":           flow.foreign.label,
                "TD.Net 5D (%)":         round(flow.proprietary.proprietary_net_5d * 100, 2),
                "TD.Trend":              round(flow.proprietary.prop_trend, 3),
                "TD Label":              flow.proprietary.label,
                "Combined Score":        round(flow.combined_institutional_score, 3),
                "Combined Label":        flow.combined_label,
                "Data Sessions":         flow.foreign.sessions,
                "TD Sessions":           flow.proprietary.sessions,
            })
        except Exception as exc:
            rows.append({
                "Mã": ticker,
                "F.Net 5D (%)": 0.0,
                "F.Trend": 0.0,
                "Ngoại Label": "Lỗi",
                "TD.Net 5D (%)": 0.0,
                "TD.Trend": 0.0,
                "TD Label": str(exc)[:30],
                "Combined Score": 0.0,
                "Combined Label": "Lỗi",
                "Data Sessions": 0,
                "TD Sessions": 0,
            })
    df = pd.DataFrame(rows)
    return df.sort_values("Combined Score", ascending=False).reset_index(drop=True)


# ── KPI summary row ────────────────────────────────────────────────────────────
st.subheader("📊 Tổng quan Institutional Flow")

with st.spinner("Đang tải dữ liệu dòng tiền tổ chức..."):
    flow_df = load_institutional_flow(tuple(selected_tickers))

if flow_df.empty:
    st.warning("Chưa có dữ liệu. Hãy chạy Daily Snapshot trước.")
    st.stop()

# KPI cards
col1, col2, col3, col4, col5 = st.columns(5)

buying_strong = (flow_df["Combined Score"] >= 0.15).sum()
buying_light  = ((flow_df["Combined Score"] >= 0.05) & (flow_df["Combined Score"] < 0.15)).sum()
neutral       = ((flow_df["Combined Score"] > -0.05) & (flow_df["Combined Score"] < 0.05)).sum()
selling_light = ((flow_df["Combined Score"] <= -0.05) & (flow_df["Combined Score"] > -0.15)).sum()
selling_strong = (flow_df["Combined Score"] <= -0.15).sum()

with col1:
    st.metric("🟢 Mua mạnh", buying_strong, help="Combined Score ≥ 0.15")
with col2:
    st.metric("🔵 Mua nhẹ", buying_light, help="0.05 ≤ Score < 0.15")
with col3:
    st.metric("⚪ Trung lập", neutral, help="-0.05 < Score < 0.05")
with col4:
    st.metric("🟡 Bán nhẹ", selling_light, help="-0.15 < Score ≤ -0.05")
with col5:
    st.metric("🔴 Bán mạnh", selling_strong, help="Score ≤ -0.15")


# ── Main table ─────────────────────────────────────────────────────────────────
st.subheader("📋 Bảng Dòng Tiền Chi Tiết")

# Color coding
def color_combined(val):
    if isinstance(val, (int, float)):
        if val >= 0.15:
            return "background-color: #1a7a4a; color: white"
        elif val >= 0.05:
            return "background-color: #2d9e6b; color: white"
        elif val <= -0.15:
            return "background-color: #c0392b; color: white"
        elif val <= -0.05:
            return "background-color: #e74c3c; color: white"
    return ""

def color_net(val):
    if isinstance(val, (int, float)):
        if val >= 2:
            return "color: #27ae60; font-weight: bold"
        elif val <= -2:
            return "color: #e74c3c; font-weight: bold"
    return ""

styled_df = (
    flow_df.style
    .applymap(color_combined, subset=["Combined Score"])
    .applymap(color_net, subset=["F.Net 5D (%)", "TD.Net 5D (%)"])
    .format({
        "F.Net 5D (%)": "{:+.2f}%",
        "TD.Net 5D (%)": "{:+.2f}%",
        "F.Trend": "{:+.3f}",
        "TD.Trend": "{:+.3f}",
        "Combined Score": "{:+.3f}",
    })
)

st.dataframe(styled_df, height=600)


# ── Top movers ────────────────────────────────────────────────────────────────
col_buy, col_sell = st.columns(2)

with col_buy:
    st.subheader("🟢 Top Tổ Chức Mua Ròng")
    top_buy = flow_df[flow_df["Combined Score"] > 0].head(5)
    if top_buy.empty:
        st.info("Không có mã nào tổ chức mua ròng hôm nay")
    else:
        for _, row in top_buy.iterrows():
            with st.container():
                c1, c2, c3 = st.columns([2, 2, 3])
                c1.metric(row["Mã"], f"{row['Combined Score']:+.3f}")
                c2.caption(f"🌍 NN: {row['F.Net 5D (%)']:+.1f}%")
                c3.caption(f"🏦 TD: {row['TD.Net 5D (%)']:+.1f}%  |  {row['Combined Label']}")

with col_sell:
    st.subheader("🔴 Top Tổ Chức Bán Ròng")
    top_sell = flow_df[flow_df["Combined Score"] < 0].tail(5).iloc[::-1]
    if top_sell.empty:
        st.info("Không có mã nào tổ chức bán ròng hôm nay")
    else:
        for _, row in top_sell.iterrows():
            with st.container():
                c1, c2, c3 = st.columns([2, 2, 3])
                c1.metric(row["Mã"], f"{row['Combined Score']:+.3f}", delta_color="inverse")
                c2.caption(f"🌍 NN: {row['F.Net 5D (%)']:+.1f}%")
                c3.caption(f"🏦 TD: {row['TD.Net 5D (%)']:+.1f}%  |  {row['Combined Label']}")


# ── Score distribution chart ───────────────────────────────────────────────────
st.subheader("📈 Phân bố Combined Institutional Score")
try:
    import plotly.express as px
    fig = px.bar(
        flow_df.sort_values("Combined Score", ascending=True),
        x="Mã",
        y="Combined Score",
        color="Combined Score",
        color_continuous_scale=["#c0392b", "#e74c3c", "#95a5a6", "#2ecc71", "#27ae60"],
        color_continuous_midpoint=0,
        title="Combined Score = 0.50×Ngoại + 0.35×TựDoanh5D + 0.15×TDTrend",
        height=400,
    )
    fig.add_hline(y=0.15,  line_dash="dash", line_color="green",  annotation_text="Mua mạnh")
    fig.add_hline(y=-0.15, line_dash="dash", line_color="red",    annotation_text="Bán mạnh")
    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig)
except ImportError:
    st.bar_chart(flow_df.set_index("Mã")["Combined Score"])


# ── Data quality info ──────────────────────────────────────────────────────────
with st.expander("ℹ️ Data Quality & Methodology"):
    td_sessions = flow_df["TD Sessions"]
    has_td = (td_sessions > 0).sum()
    no_td  = (td_sessions == 0).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Mã có data Tự Doanh", has_td)
    col2.metric("Mã chưa có data TD", no_td,
                help="Chưa có SSI data hoặc chưa log đủ phiên")
    col3.metric("TB sessions ngoại", f"{flow_df['Data Sessions'].mean():.0f}")

    st.markdown("""
    **Nguồn dữ liệu:**
    - 🌍 **Khối Ngoại**: D0.2 SQLite snapshots (log mỗi ngày lúc 15:05)
    - 🏦 **Tự Doanh**: SSI iBoard API (fetch live khi xem trang này)

    **Combined Institutional Score:**
    ```
    combined = 0.50 × smart_money_score
             + 0.35 × proprietary_net_5d
             + 0.15 × prop_trend
    ```

    **Khi SSI không khả dụng:** `proprietary_net_pct = prop_trend = 0.0`
    → Combined Score = 0.50 × foreign_score (degraded gracefully)

    **Cách đọc Combined Score:**
    | Score | Ý nghĩa |
    |---|---|
    | ≥ +0.15 | Tổ chức mua ròng mạnh |
    | +0.05 đến +0.15 | Tổ chức mua ròng nhẹ |
    | -0.05 đến +0.05 | Trung lập |
    | -0.15 đến -0.05 | Tổ chức bán ròng nhẹ |
    | ≤ -0.15 | Tổ chức bán ròng mạnh |
    """)
