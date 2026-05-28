"""
pages/03_sector_analysis.py
Phân tích dòng tiền theo ngành: heatmap, rotation, top picks.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data.sector_data import get_sector_flow_summary, get_sector_top_picks
from ui.components.charts import sector_heatmap
from ui.components.tables import sector_flow_table
from config.constants import SECTOR_MAP


# ── Helper: tổng hợp dòng tiền tổ chức 5 phiên cho 1 ngành ──────────────────

def _get_sector_foreign_5d(tickers: list[str]) -> pd.DataFrame:
    """
    Tổng hợp dòng tiền khối ngoại (net_val VND) theo ngày, 5 phiên gần nhất,
    cho toàn bộ tickers trong ngành.
    Trả về DataFrame: date, foreign_net_val (VND).
    """
    from data.foreign_flow import get_foreign_flow
    rows: list[dict] = []
    for ticker in tickers:
        try:
            df = get_foreign_flow(ticker, "1w")
            if df.empty or "net_val" not in df.columns or "date" not in df.columns:
                continue
            for _, r in df.iterrows():
                rows.append({"date": str(r["date"])[:10], "net_val": float(r.get("net_val", 0) or 0)})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["date", "foreign_net_val"])
    agg = (
        pd.DataFrame(rows)
        .groupby("date", as_index=False)["net_val"]
        .sum()
        .rename(columns={"net_val": "foreign_net_val"})
        .sort_values("date")
        .tail(5)
        .reset_index(drop=True)
    )
    return agg


def _get_sector_tu_doan_5d(sector: str, tickers: list[str]) -> pd.DataFrame:
    """
    Lấy dữ liệu tự doanh ngành 5 phiên gần nhất:
      1. Fetch phiên hôm nay từ SSI iBoard và lưu vào DB.
      2. Đọc 5 phiên gần nhất từ DB để hiển thị.
    Trả về DataFrame: session_date, buy_k_shares, sell_k_shares, net_k_shares, ticker_count.
    """
    from datetime import date as _date
    from analytics.ssi_iboard import get_proprietary_batch
    from data.db import upsert_sector_tu_doan, load_sector_tu_doan_5d

    # Bước 1: fetch phiên hôm nay và lưu DB
    total_buy = total_sell = total_net = 0.0
    date_str = _date.today().isoformat()
    found = 0
    for _group in ("VNINDEX", "VN30"):
        batch = get_proprietary_batch(_group)
        for ticker in tickers:
            if ticker in batch:
                e = batch[ticker]
                total_buy  += e.get("proprietary_buy",  0)
                total_sell += e.get("proprietary_sell", 0)
                total_net  += e.get("proprietary_net",  0)
                _d = e.get("date", "")
                if _d and _d >= "2020-01-01":
                    date_str = _d
                found += 1
        if found > 0:
            break

    if found > 0:
        try:
            upsert_sector_tu_doan(
                session_date  = date_str,
                sector        = sector,
                buy_k_shares  = total_buy  / 1_000,
                sell_k_shares = total_sell / 1_000,
                net_k_shares  = total_net  / 1_000,
                ticker_count  = found,
            )
        except Exception:
            pass  # DB write failure không làm crash UI

    # Bước 2: đọc 5 phiên từ DB
    return load_sector_tu_doan_5d(sector, last_n=5)

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
    st.plotly_chart(sector_heatmap(sector_df))

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
        display[["ticker", "Khối ngoại (tỷ)", "Tự doanh (tỷ)", "RelVol TB"]], hide_index=True,
    )
    st.caption(f"Thành phần ngành: {', '.join(SECTOR_MAP[selected_sector])}")

    # ── Dòng tiền tổ chức 5 phiên ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"#### 🏦 Dòng Tiền Tổ Chức Ngành **{selected_sector}** — 5 Phiên Gần Nhất")
    st.caption("Nguồn: FiinMarket (khối ngoại) · SSI iBoard (tự doanh phiên hiện tại)")

    _sector_tickers = SECTOR_MAP.get(selected_sector, [])
    col_foreign, col_tudoan = st.columns([3, 2])

    with col_foreign:
        st.markdown("**📊 Khối ngoại — Net giá trị 5 phiên (tỷ VND)**")
        with st.spinner("Đang tải dữ liệu khối ngoại..."):
            _fgn_df = _get_sector_foreign_5d(_sector_tickers)

        if not _fgn_df.empty:
            _fgn_df["net_ty"] = (_fgn_df["foreign_net_val"] / 1e9).round(2)
            _fgn_df["color"]  = _fgn_df["net_ty"].apply(lambda v: "#26a69a" if v >= 0 else "#ef5350")

            _fig = go.Figure(go.Bar(
                x=_fgn_df["date"],
                y=_fgn_df["net_ty"],
                marker_color=_fgn_df["color"],
                text=_fgn_df["net_ty"].apply(lambda v: f"{v:+.1f}"),
                textposition="outside",
            ))
            _fig.update_layout(
                height=260,
                margin=dict(l=0, r=0, t=20, b=0),
                yaxis_title="Tỷ VND",
                xaxis_title="",
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            _fig.update_yaxes(zeroline=True, zerolinecolor="#888", zerolinewidth=1)
            st.plotly_chart(_fig, use_container_width=True)

            # Bảng tóm tắt
            _total_net = _fgn_df["net_ty"].sum()
            _trend_icon = "🟢" if _total_net >= 0 else "🔴"
            _trend_text = "MUA RÒNG" if _total_net >= 0 else "BÁN RÒNG"
            st.markdown(
                f"{_trend_icon} **Tổng net 5 phiên: {_total_net:+.1f} tỷ — Khối ngoại đang {_trend_text}**"
            )
            st.dataframe(
                _fgn_df[["date", "net_ty"]].rename(columns={"date": "Ngày", "net_ty": "Net (tỷ VND)"}),
                hide_index=True, use_container_width=True,
            )
        else:
            st.info("Không đủ dữ liệu lịch sử khối ngoại cho ngành này.")

    with col_tudoan:
        st.markdown("**⚙️ Tự doanh — Net khối lượng 5 phiên (nghìn CP)**")
        with st.spinner("Đang tải dữ liệu tự doanh..."):
            _td_df = _get_sector_tu_doan_5d(selected_sector, _sector_tickers)

        if not _td_df.empty:
            _td_df["color"] = _td_df["net_k_shares"].apply(
                lambda v: "#26a69a" if v >= 0 else "#ef5350"
            )
            _td_fig = go.Figure(go.Bar(
                x=_td_df["session_date"],
                y=_td_df["net_k_shares"],
                marker_color=_td_df["color"],
                text=_td_df["net_k_shares"].apply(lambda v: f"{v:+,.0f}"),
                textposition="outside",
            ))
            _td_fig.update_layout(
                height=260,
                margin=dict(l=0, r=0, t=20, b=0),
                yaxis_title="Nghìn CP",
                xaxis_title="",
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            _td_fig.update_yaxes(zeroline=True, zerolinecolor="#888", zerolinewidth=1)
            st.plotly_chart(_td_fig, use_container_width=True)

            _td_total = _td_df["net_k_shares"].sum()
            _td_icon  = "🟢" if _td_total >= 0 else "🔴"
            _td_text  = "MUA RÒNG" if _td_total >= 0 else "BÁN RÒNG"
            st.markdown(
                f"{_td_icon} **Tổng net 5 phiên: {_td_total:+,.0f} nghìn CP — Tự doanh đang {_td_text}**"
            )
            st.dataframe(
                _td_df[["session_date", "net_k_shares", "buy_k_shares", "sell_k_shares", "ticker_count"]].rename(
                    columns={
                        "session_date":  "Ngày",
                        "net_k_shares":  "Net (nghìn CP)",
                        "buy_k_shares":  "Mua (nghìn CP)",
                        "sell_k_shares": "Bán (nghìn CP)",
                        "ticker_count":  "Số mã",
                    }
                ),
                hide_index=True, use_container_width=True,
            )
            st.caption("⚠️ Tự doanh được lưu hàng phiên vào DB; lịch sử tăng dần theo số ngày sử dụng.")
        else:
            st.info("Chưa có lịch sử tự doanh trong DB. Dữ liệu sẽ được tích lũy từ hôm nay.")
else:
    st.info(f"Không đủ dữ liệu cho ngành {selected_sector}")
