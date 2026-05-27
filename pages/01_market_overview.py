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
from ui.components.tables import top_flow_table, sector_flow_table

st.set_page_config(page_title="Tổng Quan Thị Trường", layout="wide")
st.title("🏛️ Tổng Quan Thị Trường")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    exchange = st.selectbox("Sàn giao dịch", ["HOSE", "HNX", "UPCOM"], index=0)
    period   = st.selectbox("Kỳ phân tích", ["1w", "2w", "1m", "3m"],
                            format_func=lambda x: {"1w":"1 tuần","2w":"2 tuần",
                                                   "1m":"1 tháng","3m":"3 tháng"}[x])
    st.divider()
    data_mode = st.radio(
        "Nguồn dữ liệu khối ngoại / tự doanh",
        options=["live", "db"],
        format_func=lambda x: {
            "live": "🔴 Phiên gần nhất (FiinMarket SSI)",
            "db":   "📦 Phiên trước (D0.2 lưu trữ)",
        }[x],
        index=0,
        help=(
            "**FiinMarket SSI**: Real-time, cập nhật mỗi 5 phút. "
            "Trả về phiên gần nhất đã hoàn tất (tức phiên hôm qua nếu thị trường chưa mở).\n\n"
            "**D0.2 lưu trữ**: Lấy từ SQLite snapshots.db — luôn có data phiên trước dù "
            "FiinMarket chưa cập nhật."
        ),
    )
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

# ── Helper: load data based on mode ──────────────────────────────────────────
def _get_data_for_mode(mode: str, exchange_: str):
    """
    Trả về (foreign_data, prop_data, data_date, source_label, is_empty).

    foreign_data / prop_data : {"buy": DataFrame, "sell": DataFrame}
    data_date                : "YYYY-MM-DD" hoặc "N/A"
    source_label             : mô tả nguồn dữ liệu
    is_empty                 : True nếu cả foreign lẫn prop đều rỗng
    """
    import pandas as pd

    if mode == "live":
        # ── FiinMarket SSI ────────────────────────────────────────────────
        with st.spinner("Đang tải dữ liệu FiinMarket SSI..."):
            foreign_data = get_top_foreign_net(exchange_)
            prop_data    = get_top_tu_doan_net(exchange_)

        # Lấy ngày khối ngoại từ batch (tất cả tickers đã có date đúng sau fix)
        foreign_date = "N/A"
        try:
            from analytics.ssi_foreign import get_foreign_batch
            batch = get_foreign_batch("VNINDEX")
            if batch:
                for v in batch.values():
                    d = v.get("date", "")
                    if d and d >= "2020-01-01":
                        foreign_date = d
                        break
        except Exception:
            pass

        # Lấy ngày tự doanh (thường là phiên hoàn tất gần nhất = hôm qua)
        prop_date = prop_data.get("date", "N/A") if isinstance(prop_data, dict) else "N/A"

        f_empty = foreign_data.get("buy", pd.DataFrame()).empty and \
                  foreign_data.get("sell", pd.DataFrame()).empty
        p_empty = prop_data.get("buy", pd.DataFrame()).empty and \
                  prop_data.get("sell", pd.DataFrame()).empty
        is_empty = f_empty and p_empty

        source_label = "FiinMarket SSI"
        return foreign_data, prop_data, foreign_date, prop_date, source_label, is_empty

    else:
        # ── D0.2 SQLite — phiên trước ─────────────────────────────────────
        with st.spinner("Đang tải dữ liệu D0.2 SQLite..."):
            from data.db import get_top_movers_from_db
            movers = get_top_movers_from_db(top_n=15)

        session_date = movers.get("session_date") or "N/A"
        foreign_data = movers["foreign"]
        prop_data    = movers["proprietary"]
        f_empty = foreign_data.get("buy", pd.DataFrame()).empty and \
                  foreign_data.get("sell", pd.DataFrame()).empty
        p_empty = prop_data.get("buy", pd.DataFrame()).empty and \
                  prop_data.get("sell", pd.DataFrame()).empty
        is_empty = f_empty and p_empty
        source_label = "D0.2 SQLite"
        return foreign_data, prop_data, session_date, session_date, source_label, is_empty


foreign_data, prop_data, foreign_date, prop_date, source_label, is_empty = _get_data_for_mode(data_mode, exchange)

# ── Auto-fallback warning khi FiinMarket rỗng đầu giờ ───────────────────────
if data_mode == "live" and is_empty:
    st.warning(
        "⏰ **FiinMarket SSI chưa có dữ liệu phiên hôm nay** (thường xảy ra trước 9:15 AM). "
        "Hãy chuyển sang **📦 Phiên trước (D0.2 lưu trữ)** ở sidebar để xem dữ liệu phiên giao dịch gần nhất.",
        icon="⚠️",
    )
    # Auto-load fallback từ SQLite
    from data.db import get_top_movers_from_db
    movers = get_top_movers_from_db(top_n=15)
    if movers["session_date"]:
        foreign_data  = movers["foreign"]
        prop_data     = movers["proprietary"]
        foreign_date  = movers["session_date"]
        prop_date     = movers["session_date"]
        source_label  = f"D0.2 SQLite (auto-fallback)"

# ── Date badges ───────────────────────────────────────────────────────────────
st.divider()
badge_color = "#d63031" if data_mode == "live" else "#6c5ce7"
# Hiển thị riêng date cho foreign vs proprietary (có thể khác nhau)
if foreign_date == prop_date:
    date_info = f"📅 Phiên: <b>{foreign_date}</b>"
else:
    date_info = f"📅 Khối ngoại: <b>{foreign_date}</b> &nbsp;|&nbsp; Tự doanh: <b>{prop_date}</b>"
st.markdown(
    f"<span style='background:{badge_color};color:white;padding:3px 10px;"
    f"border-radius:12px;font-size:0.85em'>{date_info} &nbsp;|&nbsp; {source_label}</span>",
    unsafe_allow_html=True,
)

# ── Row 2: Top Khối Ngoại ────────────────────────────────────────────────────
st.subheader("💰 Khối Ngoại")
# live = VND value (tỷ); db = giá trị gốc DB (VND mới hoặc shares cũ — hiện tỷ cho đồng nhất)
top_flow_table(
    foreign_data,
    buy_label="🟢 Khối Ngoại Mua Ròng",
    sell_label="🔴 Khối Ngoại Bán Ròng",
    unit_label="tỷ VND",
    divisor=1e9,
)

# ── Row 3: Top Tự Doanh ──────────────────────────────────────────────────────
st.divider()
st.subheader("🏦 Tự Doanh")
# FiinMarket prop = khối lượng CP; D0.2 prop = khối lượng CP (đều là shares)
top_flow_table(
    prop_data,
    buy_label="🟢 Tự Doanh Mua Ròng",
    sell_label="🔴 Tự Doanh Bán Ròng",
    unit_label="nghìn CP",
    divisor=1e3,
)

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
