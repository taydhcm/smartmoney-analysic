"""
pages/04_stock_detail.py
Chi tiết 1 mã: biểu đồ, Smart Money Score, OBV, accumulation signal.
"""

import streamlit as st
from data.market_data import get_ohlcv
from data.foreign_flow import get_foreign_flow, get_foreign_room
from data.proprietary_trading import get_tu_doan_flow
from data.volume_analysis import enrich_with_volume_indicators
from analytics.smart_money_signals import calculate_smart_money_score
from analytics.accumulation_detection import (
    detect_accumulation_phase, detect_stealth_accumulation, get_accumulation_summary
)
from news.cafef_scraper import search_cafef_news
from news.sentiment import aggregate_sentiment
from ui.components.charts import (
    candlestick_volume_chart, foreign_flow_bar_chart, smart_money_gauge
)
from ui.components.tables import foreign_flow_table
from config.constants import VN30_TICKERS

st.set_page_config(page_title="Chi Tiết Cổ Phiếu", layout="wide")
st.title("🔍 Chi Tiết Cổ Phiếu")

# ── Controls ──────────────────────────────────────────────────────────────────
with st.sidebar:
    ticker = st.text_input("Mã cổ phiếu", value="VIC").upper().strip()
    period = st.selectbox("Kỳ phân tích", ["1w", "2w", "1m", "3m"], index=0,
                          format_func=lambda x: {"1w":"1 tuần","2w":"2 tuần",
                                                 "1m":"1 tháng","3m":"3 tháng"}[x])

if not ticker:
    st.info("Nhập mã cổ phiếu ở sidebar để bắt đầu")
    st.stop()

st.markdown(f"## 📊 {ticker}")

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner(f"Đang tải dữ liệu {ticker}..."):
    ohlcv   = enrich_with_volume_indicators(get_ohlcv(ticker, period))
    ff_df   = get_foreign_flow(ticker, "1w")   # luôn lấy 1 tuần cho foreign flow
    td_df   = get_tu_doan_flow(ticker, "1w")
    room    = get_foreign_room(ticker)
    news    = search_cafef_news(ticker, limit=100, days=7)  # tất cả tin 7 ngày
    sent    = aggregate_sentiment(news)

# ── Smart Money Score ─────────────────────────────────────────────────────────
f_net    = ff_df["net_val"].sum()   if (not ff_df.empty and "net_val" in ff_df.columns) else 0
# p_net: dùng net_val nếu có (VNDirect); khi SSI iBoard fallback (net_val=0, net_vol≠0)
# thì quy đổi net_vol × close_price từ OHLCV để có giá trị tỷ VND xấp xỉ
if not td_df.empty and "net_val" in td_df.columns and td_df["net_val"].sum() != 0:
    p_net = td_df["net_val"].sum()
elif not td_df.empty and "net_vol" in td_df.columns:
    close_px = float(ohlcv["close"].iloc[-1]) if (not ohlcv.empty and "close" in ohlcv.columns) else 0
    p_net = td_df["net_vol"].sum() * close_px
else:
    p_net = 0
avg_val  = ohlcv["volume"].mean() * ohlcv["close"].mean() if not ohlcv.empty else 1e9
rv_mean  = ohlcv["rel_vol"].mean()  if (not ohlcv.empty and "rel_vol" in ohlcv.columns) else 1.0
mfi_last = float(ohlcv["mfi"].iloc[-1]) if (not ohlcv.empty and "mfi" in ohlcv.columns) else 50.0

import numpy as np
obv_trend = 0.0
if not ohlcv.empty and "obv" in ohlcv.columns and ohlcv["obv"].iloc[0] != 0:
    obv_trend = (ohlcv["obv"].iloc[-1] - ohlcv["obv"].iloc[0]) / abs(ohlcv["obv"].iloc[0])

acc_phase = detect_accumulation_phase(ohlcv)
score_result = calculate_smart_money_score(
    ticker=ticker,
    foreign_net_val=f_net, tu_doan_net_val=p_net,
    avg_daily_val=avg_val, rel_vol_mean=rv_mean,
    obv_trend=obv_trend, mfi_last=mfi_last,
    sentiment_avg=sent["avg_score"],
    accumulation_phase=acc_phase,
)

# ── Row 1: Gauge + Key Metrics ────────────────────────────────────────────────
col_gauge, col_metrics = st.columns([1, 2])
with col_gauge:
    st.plotly_chart(smart_money_gauge(score_result["score"], ticker))
    st.markdown(f"**{score_result['grade']}**")

with col_metrics:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Khối ngoại net", f"{f_net/1e9:.1f} tỷ")
    # Ghi chú nguồn tự doanh: SSI iBoard (xấp xỉ) hoặc VNDirect (chính xác)
    _td_src = ""
    if not td_df.empty and "net_val" in td_df.columns and td_df["net_val"].sum() == 0 and td_df["net_vol"].sum() != 0:
        _td_src = " ~"  # ký hiệu xấp xỉ khi dùng SSI fallback
    m2.metric("Tự doanh net" + _td_src, f"{p_net/1e9:.1f} tỷ")
    m3.metric("MFI",            f"{mfi_last:.0f}")
    m4.metric("RelVol TB",      f"{rv_mean:.2f}x")

    room_pct = room.get("remaining_pct")
    if room_pct is not None:
        st.info(f"🚪 Foreign room còn lại: **{room_pct}%**" +
                (" ⚠️" if room.get("alert") else ""))

    acc_summary = get_accumulation_summary(ticker, ohlcv)
    st.info(f"🔍 {acc_summary}")

# ── Row 2: Candlestick ────────────────────────────────────────────────────────
st.divider()
if not ohlcv.empty:
    st.plotly_chart(candlestick_volume_chart(ohlcv, ticker))

# ── Row 3: Foreign Flow ───────────────────────────────────────────────────────
st.divider()
st.subheader("💰 Dòng Tiền Khối Ngoại (1 tuần gần nhất)")
st.caption("Nguồn: KBS price-board – tích lũy từ đầu phiên hiện tại.")
st.plotly_chart(foreign_flow_bar_chart(ff_df, ticker, days=7))
if not ff_df.empty:
    foreign_flow_table(ff_df, title="Khối Ngoại")

# ── Row 4: Tin tức ────────────────────────────────────────────────────────────
st.divider()
st.subheader("📰 Tin Tức & Sentiment (7 ngày gần nhất)")

sent_color = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(sent["label"], "⚪")
st.markdown(
    f"**Sentiment:** {sent_color} {sent['label'].upper()} "
    f"(score: {sent['avg_score']:+.2f}) | "
    f"{sent['bullish_count']} tích cực / {sent['bearish_count']} tiêu cực "
    f"/ {sent['article_count']} bài"
)

if news:
    st.caption(f"Tìm thấy {len(news)} bài viết liên quan trong 7 ngày gần nhất.")
    for n in news:
        with st.expander(n.get("title", "")[:100]):
            st.write(n.get("summary", "")[:500])
            st.caption(f"Nguồn: {n.get('source','cafef')} | {n.get('published','')}")
            if n.get("url"):
                st.markdown(f"[Đọc thêm]({n['url']})")
else:
    st.info(f"Không tìm thấy tin nào liên quan đến {ticker} trong 7 ngày gần nhất.")
