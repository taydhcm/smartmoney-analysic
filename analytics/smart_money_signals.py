"""
analytics/smart_money_signals.py
Engine tổng hợp tín hiệu Smart Money → Smart Money Score (0–100) cho mỗi ticker.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from config.constants import SIGNAL_WEIGHTS, PERIOD_DAYS
from utils.logger import get_logger

log = get_logger(__name__)


def _normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize → [0, 1]"""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - mn) / (mx - mn)


def compute_foreign_score(foreign_net_val: float, avg_daily_val: float) -> float:
    """
    Điểm dòng tiền ngoại: 0–1
    Tính relative: net mua so với trung bình thanh khoản hàng ngày.
    """
    if avg_daily_val <= 0:
        return 0.5
    ratio = foreign_net_val / avg_daily_val
    # Sigmoid để chuẩn hóa
    return float(1 / (1 + np.exp(-ratio * 2)))


def compute_tu_doan_score(tu_doan_net_val: float, avg_daily_val: float) -> float:
    """Điểm tự doanh: 0–1"""
    if avg_daily_val <= 0:
        return 0.5
    ratio = tu_doan_net_val / avg_daily_val
    return float(1 / (1 + np.exp(-ratio * 3)))


def compute_volume_score(rel_vol_mean: float, obv_trend: float, mfi_last: float) -> float:
    """
    Điểm volume: kết hợp relative volume, OBV trend, MFI.
    - rel_vol_mean: trung bình relative volume (1.0 = bình thường)
    - obv_trend: hệ số góc OBV / giá trị OBV đầu kỳ (>0 = tích lũy)
    - mfi_last: MFI phiên cuối (0–100)
    """
    rv_score  = min(rel_vol_mean / 3.0, 1.0)           # cap at 3x
    obv_score = float(1 / (1 + np.exp(-obv_trend * 5)))
    mfi_score = mfi_last / 100.0

    # Nếu MFI > 80 (quá mua) – giảm score
    if mfi_last > 80:
        mfi_score = 0.4
    elif mfi_last < 20:  # quá bán – tiềm năng đảo chiều
        mfi_score = 0.7

    return (rv_score * 0.4 + obv_score * 0.4 + mfi_score * 0.2)


def compute_news_score(sentiment_avg: float) -> float:
    """sentiment_avg: [-1, 1] → [0, 1]"""
    return (sentiment_avg + 1) / 2


def compute_accumulation_score(accumulation_phase: str | None) -> float:
    """Điểm từ phát hiện mẫu tích lũy Wyckoff."""
    phase_scores = {
        "phase_b":   0.7,   # Accumulation B – volume giảm, giá ổn định
        "phase_c":   0.85,  # Spring – test đáy giả
        "phase_d":   0.95,  # Markup bắt đầu
        "none":      0.4,
        None:        0.4,
    }
    return phase_scores.get(accumulation_phase, 0.4)


def calculate_smart_money_score(
    ticker: str,
    foreign_net_val: float      = 0.0,
    tu_doan_net_val: float      = 0.0,
    avg_daily_val:   float      = 1e9,
    rel_vol_mean:    float      = 1.0,
    obv_trend:       float      = 0.0,
    mfi_last:        float      = 50.0,
    sentiment_avg:   float      = 0.0,
    accumulation_phase: str | None = None,
) -> dict:
    """
    Tính Smart Money Score tổng hợp (0–100) cho 1 ticker.
    Trả về dict: {ticker, score, grade, component_scores}
    """
    w = SIGNAL_WEIGHTS

    scores = {
        "foreign_net_buy":      compute_foreign_score(foreign_net_val, avg_daily_val),
        "tu_doan_net_buy":      compute_tu_doan_score(tu_doan_net_val, avg_daily_val),
        "volume_surge":         compute_volume_score(rel_vol_mean, obv_trend, mfi_last),
        "news_sentiment":       compute_news_score(sentiment_avg),
        "accumulation_pattern": compute_accumulation_score(accumulation_phase),
    }

    final = sum(scores[k] * w[k] for k in scores) * 100

    grade = (
        "🟢 TÍCH LŨY MẠNH"   if final >= 75 else
        "🟡 THEO DÕI"         if final >= 55 else
        "⚪ TRUNG TÍNH"       if final >= 40 else
        "🔴 PHÂN PHỐI"
    )

    return {
        "ticker":            ticker,
        "score":             round(final, 1),
        "grade":             grade,
        "component_scores":  {k: round(v * 100, 1) for k, v in scores.items()},
    }


def batch_score_tickers(tickers: list[str], period: str = "1m") -> pd.DataFrame:
    """
    Tính Smart Money Score cho nhiều tickers.
    Trả về DataFrame: ticker, score, grade, + component columns.
    """
    from data.foreign_flow import get_foreign_flow
    from data.proprietary_trading import get_tu_doan_flow
    from data.volume_analysis import enrich_with_volume_indicators
    from data.market_data import get_ohlcv
    from news.cafef_scraper import search_cafef_news
    from news.sentiment import aggregate_sentiment
    from analytics.accumulation_detection import detect_accumulation_phase

    records = []
    for ticker in tickers:
        try:
            ff    = get_foreign_flow(ticker, period)
            td    = get_tu_doan_flow(ticker, period)
            ohlcv = enrich_with_volume_indicators(get_ohlcv(ticker, period))
            news  = search_cafef_news(ticker, limit=5)
            sent  = aggregate_sentiment(news)

            f_net = ff["net_val"].sum()   if (not ff.empty and "net_val" in ff.columns) else 0
            p_net = td["net_val"].sum()   if (not td.empty and "net_val" in td.columns) else 0

            avg_val   = ohlcv["volume"].mean() * ohlcv["close"].mean() if not ohlcv.empty else 1e9
            rv_mean   = ohlcv["rel_vol"].mean() if (not ohlcv.empty and "rel_vol" in ohlcv.columns) else 1.0

            obv_trend = 0.0
            if not ohlcv.empty and "obv" in ohlcv.columns and ohlcv["obv"].iloc[0] != 0:
                obv_trend = (ohlcv["obv"].iloc[-1] - ohlcv["obv"].iloc[0]) / abs(ohlcv["obv"].iloc[0])

            mfi_last = float(ohlcv["mfi"].iloc[-1]) if (not ohlcv.empty and "mfi" in ohlcv.columns) else 50.0

            acc_phase = detect_accumulation_phase(ohlcv)

            result = calculate_smart_money_score(
                ticker=ticker,
                foreign_net_val=f_net,
                tu_doan_net_val=p_net,
                avg_daily_val=avg_val,
                rel_vol_mean=rv_mean,
                obv_trend=obv_trend,
                mfi_last=mfi_last,
                sentiment_avg=sent["avg_score"],
                accumulation_phase=acc_phase,
            )
            records.append(result)

        except Exception as exc:
            log.warning("batch_score_tickers(%s) lỗi: %s", ticker, exc)
            records.append({"ticker": ticker, "score": 0.0, "grade": "N/A", "component_scores": {}})

    df = pd.DataFrame(records).sort_values("score", ascending=False)
    return df.reset_index(drop=True)
