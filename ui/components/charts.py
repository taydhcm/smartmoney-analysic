"""
ui/components/charts.py
Plotly chart components tái sử dụng trên toàn bộ Streamlit pages.
"""

from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def candlestick_volume_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Biểu đồ nến + volume với OBV overlay."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=[f"{ticker} – Giá", "Volume", "OBV"],
        vertical_spacing=0.04,
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.get("date", df.index),
            open=df["open"], high=df["high"],
            low=df["low"],   close=df["close"],
            name="Giá", increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # VWAP
    if "vwap" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.get("date", df.index), y=df["vwap"],
                       name="VWAP", line=dict(color="orange", dash="dash", width=1)),
            row=1, col=1,
        )

    # Volume bars
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(
        go.Bar(x=df.get("date", df.index), y=df["volume"],
               name="Volume", marker_color=colors, opacity=0.8),
        row=2, col=1,
    )

    # OBV
    if "obv" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.get("date", df.index), y=df["obv"],
                       name="OBV", line=dict(color="#ab47bc", width=1.5)),
            row=3, col=1,
        )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_dark",
        margin=dict(l=40, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.02),
    )
    return fig


def foreign_flow_bar_chart(
    df: pd.DataFrame,
    ticker: str,
    days: int = 7,
) -> go.Figure:
    """Biểu đồ cột dòng tiền ngoại – x-axis luôn trải đủ `days` ngày."""
    from datetime import date, timedelta

    if df.empty or "net_val" not in df.columns:
        # Vẫn trả về figure trống có x-axis 7 ngày
        fig = go.Figure()
        end_dt   = date.today()
        start_dt = end_dt - timedelta(days=days)
        fig.update_layout(
            title=f"{ticker} – Khối ngoại Net (tỷ đồng) – {days} ngày gần nhất",
            xaxis=dict(range=[str(start_dt), str(end_dt)], title="Ngày"),
            yaxis_title="Tỷ đồng",
            template="plotly_dark",
            height=300,
            margin=dict(l=40, r=20, t=40, b=20),
        )
        return fig

    colors = ["#26a69a" if v > 0 else "#ef5350" for v in df["net_val"]]
    fig = go.Figure()

    # Cột net buy/sell
    fig.add_trace(go.Bar(
        x=df.get("date", df.index),
        y=df["net_val"] / 1e9,
        marker_color=colors,
        name="Net (tỷ đ)",
    ))

    # Overlay buy / sell nếu có cột
    if "buy_vol" in df.columns and "sell_vol" in df.columns:
        fig.add_trace(go.Bar(
            x=df.get("date", df.index),
            y=df["buy_vol"] / 1e6,
            marker_color="#26a69a",
            opacity=0.4,
            name="Mua (tr.cp)",
            yaxis="y2",
        ))
        fig.add_trace(go.Bar(
            x=df.get("date", df.index),
            y=-df["sell_vol"] / 1e6,
            marker_color="#ef5350",
            opacity=0.4,
            name="Bán (tr.cp)",
            yaxis="y2",
        ))

    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=days)

    fig.update_layout(
        title=f"{ticker} – Khối ngoại Net (tỷ đồng) – {days} ngày gần nhất",
        xaxis=dict(range=[str(start_dt), str(end_dt)], title="Ngày"),
        yaxis=dict(title="Tỷ đồng"),
        yaxis2=dict(title="Triệu cp", overlaying="y", side="right", showgrid=False),
        template="plotly_dark",
        height=320,
        margin=dict(l=40, r=60, t=40, b=20),
        barmode="overlay",
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def sector_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap sector rotation theo score."""
    if df.empty or "score" not in df.columns:
        return go.Figure()

    fig = px.treemap(
        df,
        path=["sector"],
        values="ticker_count",
        color="score",
        color_continuous_scale=["#ef5350", "#ffcc80", "#26a69a"],
        color_continuous_midpoint=0.5,
        hover_data={"foreign_net_val": True, "tu_doan_net_val": True},
        title="Sector Rotation – Smart Money Score",
    )
    fig.update_layout(template="plotly_dark", height=400,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def smart_money_gauge(score: float, ticker: str) -> go.Figure:
    """Gauge chart cho Smart Money Score."""
    color = (
        "#26a69a" if score >= 70 else
        "#ffcc80" if score >= 50 else
        "#ef5350"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": f"{ticker} Smart Money Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0,  40], "color": "#2d1b1b"},
                {"range": [40, 55], "color": "#2d2d1b"},
                {"range": [55, 75], "color": "#1b2d2d"},
                {"range": [75, 100], "color": "#1b2d1b"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        template="plotly_dark", height=250,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def market_breadth_donut(breadth: dict) -> go.Figure:
    """Donut chart độ rộng thị trường."""
    labels = ["Tăng", "Giảm", "Đứng"]
    values = [
        breadth.get("advance", 0),
        breadth.get("decline", 0),
        breadth.get("unchanged", 0),
    ]
    colors = ["#26a69a", "#ef5350", "#78909c"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.5,
        marker_colors=colors,
        textinfo="label+percent",
    ))
    fig.update_layout(
        title="Độ Rộng Thị Trường",
        template="plotly_dark", height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    return fig
