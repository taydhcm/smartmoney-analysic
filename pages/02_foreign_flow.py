"""
pages/02_foreign_flow.py
Phân tích chi tiết dòng tiền khối ngoại: trend 30 phiên, room ngoại alert.
"""

import streamlit as st
import pandas as pd
from data.foreign_flow import get_foreign_flow, get_foreign_room, get_top_foreign_net
from ui.components.charts import foreign_flow_bar_chart
from ui.components.tables import foreign_flow_table
from config.constants import VN30_TICKERS

st.set_page_config(page_title="Khối Ngoại", layout="wide")
st.title("💰 Phân Tích Dòng Tiền Khối Ngoại")

with st.sidebar:
    period = st.selectbox("Kỳ phân tích", ["1w", "2w", "1m", "3m"],
                          index=0,
                          format_func=lambda x: {"1w":"1 tuần","2w":"2 tuần",
                                                 "1m":"1 tháng","3m":"3 tháng"}[x])
    exchange = st.selectbox("Sàn", ["HOSE", "HNX", "UPCOM"])

# ── Top mua/bán ròng ──────────────────────────────────────────────────────────
st.subheader("📊 Top Khối Ngoại Mua/Bán Ròng Mạnh Nhất Hôm Nay")
with st.spinner("Đang tải..."):
    top = get_top_foreign_net(exchange, top_n=15)

from ui.components.tables import top_foreign_net_table
top_foreign_net_table(top)

# ── Chi tiết 1 mã ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Phân Tích Chi Tiết Mã Cụ Thể")

ticker = st.selectbox("Chọn mã cổ phiếu", sorted(VN30_TICKERS), index=0)

col_room, col_flow = st.columns([1, 3])

with col_room:
    st.markdown("#### 🚪 Foreign Room")
    with st.spinner(f"Kiểm tra room {ticker}..."):
        room = get_foreign_room(ticker)

    if room.get("remaining_pct") is not None:
        remaining = room["remaining_pct"]
        if room.get("alert"):
            st.error(f"⚠️ Room còn lại: **{remaining}%** – GẦN ĐẦY!")
        elif remaining < 15:
            st.warning(f"Room còn lại: **{remaining}%**")
        else:
            st.success(f"Room còn lại: **{remaining}%**")

        st.metric("Max room", f"{room['max_room_pct']}%")
        st.metric("Đang dùng", f"{room['used_pct']}%")
    else:
        st.info("Không có dữ liệu room")

with col_flow:
    st.markdown(f"#### 📈 Dòng Tiền Ngoại {ticker} – {period}")
    with st.spinner(f"Tải dữ liệu khối ngoại {ticker}..."):
        ff_df = get_foreign_flow(ticker, period)

    if not ff_df.empty:
        days_map = {"1w": 7, "2w": 14, "1m": 30, "3m": 90}
        st.plotly_chart(foreign_flow_bar_chart(ff_df, ticker, days=days_map.get(period, 7)),
                        use_container_width=True)
        foreign_flow_table(ff_df, f"Chi tiết {ticker}")
    else:
        st.info(f"Không có dữ liệu khối ngoại cho {ticker} — dữ liệu sẽ tích lũy dần sau mỗi phiên.")

# ── Foreign Room Alert Dashboard ──────────────────────────────────────────────
st.divider()
st.subheader("⚠️ Cảnh Báo Room Ngoại Gần Đầy (VN30)")

with st.spinner("Kiểm tra room toàn VN30..."):
    # 1 batch KBS call cho toàn bộ VN30, thay vì 30 call riêng lẻ
    try:
        from vnstock.api.trading import Trading  # type: ignore
        board = Trading(source="KBS").price_board(symbols_list=VN30_TICKERS)
    except Exception:
        board = pd.DataFrame()

    room_alerts = []
    if not board.empty:
        ticker_col = next(
            (c for c in board.columns if c.lower() in ("symbol", "ticker", "code")),
            board.columns[0]
        )
        for _, row in board.iterrows():
            try:
                tkr = str(row[ticker_col])
                fp = float(pd.to_numeric(row.get("foreign_ownership_ratio", 0), errors="coerce") or 0)
                used_pct = fp if fp > 1 else fp * 100
                remaining = round(49.0 - used_pct, 2)
                room_alerts.append({
                    "ticker":        tkr,
                    "remaining_pct": remaining,
                    "used_pct":      round(used_pct, 2),
                    "max_room_pct":  49.0,
                    "alert":         remaining < 5.0,
                })
            except Exception:
                pass
    else:
        # fallback: gọi từng ticker qua get_foreign_room (chậm hơn)
        for t in VN30_TICKERS:
            r = get_foreign_room(t)
            if r.get("remaining_pct") is not None:
                room_alerts.append(r)

if room_alerts:
    alert_df = pd.DataFrame(room_alerts)
    alert_df = alert_df.sort_values("remaining_pct")
    st.dataframe(
        alert_df[["ticker", "remaining_pct", "used_pct", "max_room_pct", "alert"]],
        use_container_width=True, hide_index=True,
    )
else:
    st.info("Không có dữ liệu room alerts")
