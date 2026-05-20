"""
pages/01_market_overview.py
Trang Tổng Quan Thị Trường: VNIndex, breadth, top khối ngoại/tự doanh hôm nay.
"""

import streamlit as st
from data.market_data import get_market_breadth, get_index_data
from data.foreign_flow import get_top_foreign_net
from data.proprietary_trading import get_top_tu_doan_net
from data.sector_data import get_sector_flow_summary
from ui.components.charts import (
    candlestick_volume_chart, sector_heatmap, market_breadth_donut
)
from ui.components.tables import top_foreign_net_table, sector_flow_table

st.set_page_config(page_title="Tổng Quan Thị Trường", layout="wide")
st.title("🏛️ Tổng Quan Thị Trường")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    exchange = st.selectbox("Sàn giao dịch", ["HOSE", "HNX", "UPCOM"], index=0)
    period   = st.selectbox("Kỳ phân tích", ["1w", "2w", "1m", "3m"],
                            format_func=lambda x: {"1w":"1 tuần","2w":"2 tuần",
                                                   "1m":"1 tháng","3m":"3 tháng"}[x])
    if st.button("🔄 Làm mới dữ liệu"):
        from utils.cache import clear_cache
        clear_cache()
        st.rerun()

# ── Row 1: Chỉ số + Breadth ───────────────────────────────────────────────────
col_idx, col_breadth = st.columns([2, 1])

with col_idx:
    st.subheader("📈 VNINDEX")
    with st.spinner("Đang tải dữ liệu chỉ số..."):
        idx_df = get_index_data("VNINDEX", period)
    if not idx_df.empty:
        from data.volume_analysis import enrich_with_volume_indicators
        idx_df = enrich_with_volume_indicators(idx_df)
        st.plotly_chart(
            candlestick_volume_chart(idx_df, "VNINDEX"),
            use_container_width=True,
        )
    else:
        st.warning("Không tải được dữ liệu VNINDEX")

with col_breadth:
    st.subheader("📊 Độ Rộng Thị Trường")
    with st.spinner("Đang tải breadth..."):
        breadth = get_market_breadth()
    st.plotly_chart(market_breadth_donut(breadth), use_container_width=True)
    col_a, col_b = st.columns(2)
    col_a.metric("Trần", breadth.get("ceiling", 0), help="Số mã kịch trần")
    col_b.metric("Sàn",  breadth.get("floor",   0), help="Số mã kịch sàn")

# ── Row 2: Top Khối Ngoại ────────────────────────────────────────────────────
st.divider()
st.subheader("💰 Khối Ngoại Hôm Nay")
with st.spinner("Đang tải dữ liệu khối ngoại..."):
    top_foreign = get_top_foreign_net(exchange)
top_foreign_net_table(top_foreign)

# ── Row 3: Top Tự Doanh ──────────────────────────────────────────────────────
st.divider()
st.subheader("🏦 Tự Doanh Hôm Nay")
with st.spinner("Đang tải dữ liệu tự doanh..."):
    top_tu_doan = get_top_tu_doan_net(exchange)
top_foreign_net_table(top_tu_doan)  # cùng format

# ── Row 4: Sector Heatmap ────────────────────────────────────────────────────
st.divider()
st.subheader("🔄 Dòng Tiền Theo Ngành")
with st.spinner("Đang phân tích sector rotation (có thể mất 30-60s)..."):
    sector_df = get_sector_flow_summary(period)

if not sector_df.empty:
    st.plotly_chart(sector_heatmap(sector_df), use_container_width=True)
    sector_flow_table(sector_df)
else:
    st.warning("Không đủ dữ liệu sector")
