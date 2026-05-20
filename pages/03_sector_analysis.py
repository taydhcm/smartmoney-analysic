"""
pages/03_sector_analysis.py
Phân tích dòng tiền theo ngành: heatmap, rotation, top picks.
"""

import streamlit as st
from data.sector_data import get_sector_flow_summary, get_sector_top_picks
from ui.components.charts import sector_heatmap
from ui.components.tables import sector_flow_table
from config.constants import SECTOR_MAP

st.set_page_config(page_title="Phân Tích Ngành", layout="wide")
st.title("🔄 Phân Tích Dòng Tiền Theo Ngành")

with st.sidebar:
    period = st.selectbox("Kỳ phân tích", ["1w", "1m", "3m"], index=1,
                          format_func=lambda x: {"1w":"1 tuần","1m":"1 tháng","3m":"3 tháng"}[x])

# ── Sector Heatmap ────────────────────────────────────────────────────────────
st.subheader("🗺️ Heatmap Dòng Tiền Ngành")
with st.spinner("Đang tính toán sector rotation (30-60s)..."):
    sector_df = get_sector_flow_summary(period)

if not sector_df.empty:
    st.plotly_chart(sector_heatmap(sector_df), use_container_width=True)

    st.subheader("📊 Bảng Xếp Hạng Ngành")
    sector_flow_table(sector_df)

    # Highlight top & bottom
    col_top, col_bot = st.columns(2)
    with col_top:
        st.success("🟢 **Ngành Đang Nhận Dòng Tiền Mạnh Nhất:**")
        for _, row in sector_df.head(3).iterrows():
            st.write(f"• **{row['sector']}** – Score: {row['score']*100:.0f}/100")
    with col_bot:
        st.error("🔴 **Ngành Đang Bị Dòng Tiền Rút Ra:**")
        for _, row in sector_df.tail(3).iterrows():
            st.write(f"• **{row['sector']}** – Score: {row['score']*100:.0f}/100")
else:
    st.warning("Không đủ dữ liệu sector")

# ── Top picks theo ngành ──────────────────────────────────────────────────────
st.divider()
st.subheader("🏆 Top Picks Theo Ngành")

selected_sector = st.selectbox("Chọn ngành", list(SECTOR_MAP.keys()))

with st.spinner(f"Tìm top picks ngành {selected_sector}..."):
    picks_df = get_sector_top_picks(selected_sector, period)

if not picks_df.empty:
    st.markdown(f"**Top cổ phiếu – {selected_sector} ({period})**")
    display = picks_df.copy()
    display["Khối ngoại (tỷ)"] = (display["foreign_net_val"] / 1e9).round(1)
    display["Tự doanh (tỷ)"]   = (display["tu_doan_net_val"] / 1e9).round(1)
    display["RelVol TB"]       = display["avg_rel_vol"].round(2)
    st.dataframe(
        display[["ticker", "Khối ngoại (tỷ)", "Tự doanh (tỷ)", "RelVol TB"]],
        use_container_width=True, hide_index=True,
    )
    st.caption(f"Thành phần ngành: {', '.join(SECTOR_MAP[selected_sector])}")
else:
    st.info(f"Không đủ dữ liệu cho ngành {selected_sector}")
