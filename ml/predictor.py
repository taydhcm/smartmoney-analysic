"""
ml/predictor.py
Prediction pipeline: tính P_alpha = P(return_T+2 >= 5%) cho tất cả tickers.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from config.constants import VN30_TICKERS
from data.market_data import get_ohlcv, get_index_data
from .feature_engineering import compute_stock_features, FEATURE_COLS
from .model import load_model

log = logging.getLogger(__name__)

# Threshold mặc định để chọn alpha picks
DEFAULT_MIN_PROB   = 0.65
DEFAULT_MAX_RSI    = 80.0   # loại cổ phiếu overbought rõ ràng


def predict_today(
    tickers: list[str] | None = None,
    period: str = "3m",
    min_probability: float = DEFAULT_MIN_PROB,
    max_rsi: float = DEFAULT_MAX_RSI,
    progress_callback: Callable[[float, str], None] | None = None,
) -> list[dict]:
    """
    Dự đoán xác suất đạt >5% return trong T+2 cho tất cả tickers hôm nay.

    Parameters
    ----------
    tickers           : Danh sách mã. Mặc định = VN30_TICKERS.
    period            : Kỳ lấy OHLCV ("3m" đủ cho feature computation).
    min_probability   : Ngưỡng tối thiểu để xuất hiện trong alpha picks.
    max_rsi           : Loại cổ phiếu có RSI > threshold (overbought).
    progress_callback : fn(pct, msg) để cập nhật UI.

    Returns
    -------
    list[dict] sorted by probability giảm dần, chỉ gồm picks >= min_probability.
    """
    if tickers is None:
        tickers = VN30_TICKERS

    loaded = load_model()
    if loaded is None:
        log.warning("Chưa có model – hãy train trước")
        return []

    model, scaler, _meta = loaded

    if progress_callback:
        progress_callback(0.0, "Đang tải VN30 index...")
    vn30_df = get_index_data("VN30", period=period)

    results: list[dict] = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i / total, f"Dự đoán {ticker} ({i+1}/{total})...")
        try:
            ohlcv = get_ohlcv(ticker, period=period)
            if ohlcv.empty or len(ohlcv) < 25:
                continue

            feat_df = compute_stock_features(ohlcv, vn30_df)
            if feat_df.empty:
                continue

            # Row cuối = phiên gần nhất (hôm nay hoặc hôm qua)
            last = feat_df.iloc[-1]

            # Extract features — đảm bảo đúng thứ tự FEATURE_COLS
            raw_vals = np.array(
                [float(last.get(col, 0.0) or 0.0) for col in FEATURE_COLS],
                dtype=np.float32,
            )
            raw_vals = np.nan_to_num(raw_vals, nan=0.0, posinf=3.0, neginf=-3.0)

            X      = scaler.transform(raw_vals.reshape(1, -1))
            prob   = float(model.predict_proba(X)[0, 1])

            rsi        = float(last.get("rsi_14", 50) or 50)
            vol_ratio  = float(last.get("volume_ratio_5d", 1.0) or 1.0)
            acc_score  = float(last.get("accumulation_score", 0.3) or 0.3)
            div_score  = float(last.get("divergence_score", 0.0) or 0.0)
            pp_score   = float(last.get("pull_push_score", 0.0) or 0.0)
            rel_str    = float(last.get("relative_strength", 0.0) or 0.0)
            ret_1d     = float(last.get("return_1d", 0.0) or 0.0)

            # Loại overbought (RSI > threshold)
            if rsi > max_rsi:
                log.debug("Bỏ qua %s: RSI=%.1f (overbought)", ticker, rsi)
                continue

            results.append({
                "ticker":             ticker,
                "probability":        round(prob, 4),
                "expected_return":    _estimate_return(prob),
                "pattern":            _describe_pattern(acc_score, div_score, pp_score),
                "confidence":         _confidence_label(prob),
                "rsi":                round(rsi, 1),
                "volume_ratio_5d":    round(vol_ratio, 2),
                "relative_strength":  round(rel_str * 100, 2),   # in %
                "return_1d_pct":      round(ret_1d * 100, 2),    # in %
                "accumulation_score": round(acc_score, 3),
                "divergence_score":   round(div_score, 3),
                "pull_push_score":    round(pp_score, 3),
            })

        except Exception as exc:
            log.warning("predict_today(%s) lỗi: %s", ticker, exc)

    if progress_callback:
        progress_callback(1.0, "Hoàn tất dự đoán.")

    # Lọc theo threshold và sort
    picks = [r for r in results if r["probability"] >= min_probability]
    picks.sort(key=lambda x: x["probability"], reverse=True)

    log.info(
        "Prediction: %d tickers → %d picks (min_prob=%.2f)",
        len(results), len(picks), min_probability,
    )
    return picks


def predict_all(
    tickers: list[str] | None = None,
    period: str = "3m",
    progress_callback: Callable[[float, str], None] | None = None,
) -> pd.DataFrame:
    """
    Trả về full DataFrame với P_alpha cho TẤT CẢ tickers (không lọc threshold).
    Dùng cho phân tích toàn thị trường.
    """
    if tickers is None:
        tickers = VN30_TICKERS

    # Gọi predict_today với threshold = 0 để lấy tất cả
    all_results = predict_today(
        tickers=tickers,
        period=period,
        min_probability=0.0,
        max_rsi=100.0,
        progress_callback=progress_callback,
    )
    if not all_results:
        return pd.DataFrame()
    return pd.DataFrame(all_results).sort_values("probability", ascending=False)


def get_feature_importance() -> dict[str, float]:
    """Feature importance từ trained model, sorted by importance."""
    loaded = load_model()
    if loaded is None:
        return {}
    model, _, meta = loaded
    if meta.get("feature_importance"):
        return dict(sorted(meta["feature_importance"].items(), key=lambda x: x[1], reverse=True))
    if hasattr(model, "feature_importances_"):
        raw = model.feature_importances_
        total = raw.sum() + 1e-9
        return dict(
            sorted(
                {col: float(imp / total) for col, imp in zip(FEATURE_COLS, raw)}.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )
    return {}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _describe_pattern(acc: float, div: float, pp: float) -> str:
    """Mô tả pattern bằng tiếng Việt."""
    parts: list[str] = []
    if acc >= 0.75:
        parts.append("tích lũy Wyckoff mạnh")
    elif acc >= 0.55:
        parts.append("tích lũy nhẹ")
    if div > 0.25:
        parts.append("volume divergence")
    if pp > 0.40:
        parts.append("smart money kéo")
    elif pp < -0.30:
        parts.append("cảnh báo đạp giá")
    return " + ".join(parts) if parts else "không rõ mẫu"


def _estimate_return(prob: float) -> str:
    if prob >= 0.80:
        return "+7% đến +12%"
    elif prob >= 0.70:
        return "+5% đến +9%"
    elif prob >= 0.65:
        return "+5% đến +7%"
    else:
        return "<5%"


def _confidence_label(prob: float) -> str:
    if prob >= 0.80:
        return "high"
    elif prob >= 0.70:
        return "medium"
    else:
        return "low"
