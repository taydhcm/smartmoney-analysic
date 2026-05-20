"""
ui/components/tables.py
Streamlit table components tái sử dụng.
"""

from __future__ import annotations
import pandas as pd
import streamlit as st

from utils.formatters import fmt_value_vnd, fmt_pct, color_sign


def foreign_flow_table(df: pd.DataFrame, title: str = "Khối Ngoại") -> None:
    """Hiển thị bảng dòng tiền ngoại với màu sắc."""
    if df.empty:
        st.info(f"{title}: Không có dữ liệu")
        return

    st.subheader(title)

    display = df.copy()
    for col in ["buy_val", "sell_val", "net_val"]:
        if col in display.columns:
            display[col] = display[col].apply(fmt_value_vnd)

    st.dataframe(display, use_container_width=True, hide_index=True)


def top_foreign_net_table(result: dict) -> None:
    """Bảng top khối ngoại mua ròng / bán ròng."""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🟢 Khối Ngoại Mua Ròng")
        df_buy = result.get("buy", pd.DataFrame())
        if not df_buy.empty:
            df_buy = df_buy.copy()
            if "net_val" in df_buy.columns:
                df_buy["net_val (tỷ)"] = (df_buy["net_val"] / 1e9).round(1)
            st.dataframe(df_buy[["ticker", "net_val (tỷ)"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("Không có dữ liệu")

    with col2:
        st.markdown("#### 🔴 Khối Ngoại Bán Ròng")
        df_sell = result.get("sell", pd.DataFrame())
        if not df_sell.empty:
            df_sell = df_sell.copy()
            if "net_val" in df_sell.columns:
                df_sell["net_val (tỷ)"] = (df_sell["net_val"] / 1e9).round(1)
            st.dataframe(df_sell[["ticker", "net_val (tỷ)"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("Không có dữ liệu")


def score_table(df: pd.DataFrame) -> None:
    """Bảng Smart Money Score với màu grade."""
    if df.empty:
        st.info("Không có dữ liệu score")
        return

    display = df[["ticker", "score", "grade"]].copy()
    display["score"] = display["score"].apply(lambda x: f"{x:.1f}/100")
    st.dataframe(display, use_container_width=True, hide_index=True)


def sector_flow_table(df: pd.DataFrame) -> None:
    """Bảng dòng tiền ngành."""
    if df.empty:
        st.info("Không có dữ liệu ngành")
        return

    display = df[["sector", "foreign_net_val", "tu_doan_net_val", "avg_rel_vol", "score"]].copy()
    display["Khối ngoại (tỷ)"] = (display["foreign_net_val"] / 1e9).round(1)
    display["Tự doanh (tỷ)"]   = (display["tu_doan_net_val"] / 1e9).round(1)
    display["RelVol TB"]       = display["avg_rel_vol"].round(2)
    display["Score"]           = (display["score"] * 100).round(1)

    st.dataframe(
        display[["sector", "Khối ngoại (tỷ)", "Tự doanh (tỷ)", "RelVol TB", "Score"]],
        use_container_width=True, hide_index=True,
    )
