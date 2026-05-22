"""
ml/predictor.py
Prediction pipeline: tính P_alpha = P(đạt target +5% trong T+5 không chạm SL -5%).

v2.0: Tích hợp D3.1 Regime Gate:
  - BEAR  → block toàn bộ long signal, trả về []
  - Các state khác → tự động nâng min_probability theo regime
  - Mỗi pick có thêm trường 'regime' với đầy đủ context

v2.1: Sprint 2 — S2 + S5 + D3.2 + D3.3
  - S2  Relative Strength Engine: multi-timeframe RS vs VN30, cross-sectional rank
  - S5  Entry Timing Engine: entry zone, SL price, target, R:R per pick
  - D3.2 Cross-sectional Ranker: composite_score = 0.6×P + 0.4×rs_rank
  - D3.3 SL Filter: reject nếu sl_pct > 7% hoặc R:R < 0.8

v2.2: Sprint 3 — S4 + D3.4
  - S4  Volume Confirmation Engine: vol_surge, vol_quality, OBV score per pick
  - D3.4 Portfolio Sizing: Kelly Criterion position size per pick

v2.3: Sprint 4 — D0.2 + S4 Smart Money
  - D0.2 SQLite logger ghi daily snapshot 15:05
  - S4 Smart Money Flow: foreign_net_pct, foreign_trend, smart_money_score per pick
  - compute_stock_features nhan them sm_features tu D0.2 SQLite

v2.4: Sprint 6 — M4 Probability Calibration
  - Isotonic Regression calibration → p_calibrated per pick
  - 90% Wilson score confidence interval [ci_lo, ci_hi]
  - Recommendation enum: STRONG_BUY | BUY | WATCH | HOLD | AVOID
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Callable

import numpy as np
import pandas as pd

from config.constants import VN30_TICKERS
from data.market_data import get_ohlcv, get_index_data
from .feature_engineering import compute_stock_features, FEATURE_COLS
from .model import load_model, load_calibrator, wilson_ci
from .regime import RegimeInfo, RegimeState, get_market_regime
from .relative_strength import RSInfo, compute_stock_rs, rank_by_rs
from .entry_timing import EntryZone, compute_entry_zone
from .volume_confirmation import VolumeConfirmation, compute_volume_confirmation
from .portfolio_sizing import PositionSize, compute_position_size
from .smart_money import SmartMoneySignal, compute_smart_money, compute_smart_money_features

log = logging.getLogger(__name__)

# ── Recommendation enum (Sprint 6) ───────────────────────────────────────────────────────────────────────────────────

class Recommendation(str, Enum):
    """Kết luận đầu tư dựa trên xác suất đã calibrate + composite score."""
    STRONG_BUY = "STRONG_BUY"   # p_cal >= 0.80
    BUY        = "BUY"          # p_cal >= 0.70
    WATCH      = "WATCH"        # p_cal >= 0.60
    HOLD       = "HOLD"         # p_cal >= 0.50
    AVOID      = "AVOID"        # p_cal < 0.50


# Threshold mặc định (có thể bị override bởi Regime Gate)
DEFAULT_MIN_PROB = 0.65
DEFAULT_MAX_RSI  = 80.0   # loại cổ phiếu overbought rõ ràng

# Trọng số composite score D3.2
_W_PROB = 0.60
_W_RS   = 0.40


def predict_today(
    tickers: list[str] | None = None,
    period: str = "3m",
    min_probability: float = DEFAULT_MIN_PROB,
    max_rsi: float = DEFAULT_MAX_RSI,
    progress_callback: Callable[[float, str], None] | None = None,
    enable_regime_gate: bool = True,
) -> list[dict]:
    """
    Dự đoán xác suất đạt target (+5%, không chạm SL -5%) trong 5 phiên cho tất cả tickers.

    Parameters
    ----------
    tickers            : Danh sách mã. Mặc định = VN30_TICKERS.
    period             : Kỳ lấy OHLCV ("3m" đủ cho feature computation).
    min_probability    : Ngưỡng P tối thiểu (có thể bị nâng bởi regime gate).
    max_rsi            : Loại cổ phiếu có RSI > threshold (overbought).
    progress_callback  : fn(pct, msg) để cập nhật UI.
    enable_regime_gate : True → D3.1 gate tự động adjust threshold.
                         False → bỏ qua regime (dùng cho analysis/debug).

    Returns
    -------
    list[dict] sorted by probability giảm dần.
    Trả về [] khi regime = BEAR (enable_regime_gate=True).
    Mỗi dict có trường 'regime' chứa đầy đủ context từ S3.
    """
    if tickers is None:
        tickers = VN30_TICKERS

    loaded = load_model()
    if loaded is None:
        log.warning("Chưa có model – hãy train trước")
        return []

    model, scaler, _meta = loaded
    calibrator = load_calibrator()   # None nếu chưa có (Sprint 6)

    if progress_callback:
        progress_callback(0.0, "Đang tải VN30 + xác định Market Regime...")
    vn30_df = get_index_data("VN30", period=period)

    # ── D3.1 Regime Gate ─────────────────────────────────────────────────────
    regime: RegimeInfo = get_market_regime(vn30_df)

    if enable_regime_gate:
        effective_min_prob = max(min_probability, regime.min_prob)

        if not regime.is_tradeable:
            log.warning(
                "[REGIME GATE] State=%s → TẮT toàn bộ long signal. %s",
                regime.state.value, regime.reason,
            )
            if progress_callback:
                progress_callback(1.0, f"🔴 Regime {regime.state.value}: không vào lệnh long.")
            return []

        if effective_min_prob > min_probability:
            log.info(
                "[REGIME GATE] State=%s → nâng P_min %.2f → %.2f",
                regime.state.value, min_probability, effective_min_prob,
            )
    else:
        effective_min_prob = min_probability

    regime_dict = regime.as_dict()

    # ── Pass 1: tính P_alpha + RS cho mỗi ticker ─────────────────────────────
    # Lưu raw_results (chưa lọc threshold) + RSInfo list để rank cross-sectional
    raw_results:  list[dict]    = []
    rs_info_list: list[RSInfo]  = []
    ohlcv_cache:  dict[str, pd.DataFrame] = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(
                0.05 + 0.80 * (i / total),
                f"Phân tích {ticker} ({i+1}/{total})...",
            )
        try:
            ohlcv = get_ohlcv(ticker, period=period)
            if ohlcv.empty or len(ohlcv) < 25:
                continue

            # S4 Smart Money features tu D0.2 SQLite (0.0 khi chua du 5 phien)
            try:
                sm_feat = compute_smart_money_features(ticker)
            except Exception:
                sm_feat = None

            feat_df = compute_stock_features(ohlcv, vn30_df, sm_features=sm_feat)
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

            X    = scaler.transform(raw_vals.reshape(1, -1))
            prob = float(model.predict_proba(X)[0, 1])

            # Calibrated probability (Sprint 6) — fallback = raw prob
            if calibrator is not None:
                try:
                    p_cal = float(calibrator.predict([prob])[0])
                    p_cal = float(np.clip(p_cal, 0.0, 1.0))
                except Exception:
                    p_cal = prob
            else:
                p_cal = prob
            ci_lo, ci_hi = wilson_ci(p_cal, n=50)

            rsi       = float(last.get("rsi_14", 50) or 50)
            vol_ratio = float(last.get("volume_ratio_5d", 1.0) or 1.0)
            acc_score = float(last.get("accumulation_score", 0.3) or 0.3)
            div_score = float(last.get("divergence_score", 0.0) or 0.0)
            pp_score  = float(last.get("pull_push_score", 0.0) or 0.0)
            rel_str   = float(last.get("relative_strength", 0.0) or 0.0)
            ret_1d    = float(last.get("return_1d", 0.0) or 0.0)
            # S1 Wyckoff VSA v2.0 features
            spring_q  = float(last.get("spring_quality",   0.0) or 0.0)
            lps_det   = float(last.get("lps_detected",     0.0) or 0.0)
            evr       = float(last.get("effort_vs_result", 0.0) or 0.0)
            no_sup    = float(last.get("no_supply_count",  0.0) or 0.0)
            stop_vol  = float(last.get("stopping_volume",  0.0) or 0.0)

            # Loại overbought (RSI > threshold)
            if rsi > max_rsi:
                log.debug("Bỏ qua %s: RSI=%.1f (overbought)", ticker, rsi)
                continue

            # S2: Relative Strength (single stock)
            rs_info = compute_stock_rs(ohlcv, vn30_df, ticker=ticker)
            if rs_info is not None:
                rs_info_list.append(rs_info)

            ohlcv_cache[ticker] = ohlcv

            raw_results.append({
                "ticker":             ticker,
                "probability":        round(prob, 4),
                "p_calibrated":       round(p_cal, 4),
                "ci_lo":              ci_lo,
                "ci_hi":              ci_hi,
                "recommendation":     _get_recommendation(p_cal).value,
                "expected_return":    _estimate_return(prob),
                "pattern":            _describe_pattern(acc_score, div_score, pp_score, spring_q, lps_det),
                "confidence":         _confidence_label(prob),
                "rsi":                round(rsi, 1),
                "volume_ratio_5d":    round(vol_ratio, 2),
                "relative_strength":  round(rel_str * 100, 2),   # in %
                "return_1d_pct":      round(ret_1d * 100, 2),    # in %
                "accumulation_score": round(acc_score, 3),
                "divergence_score":   round(div_score, 3),
                "pull_push_score":    round(pp_score, 3),
                # S1 Wyckoff VSA v2.0
                "wyckoff": {
                    "wyckoff_score":    round(acc_score, 3),
                    "spring_quality":   round(spring_q, 3),
                    "lps_detected":     bool(lps_det > 0.5),
                    "effort_vs_result": round(evr, 3),
                    "no_supply_count":  round(no_sup, 3),
                    "stopping_volume":  bool(stop_vol > 0.5),
                },
                "regime":             regime_dict,                # D3.1 context
                "_rs_info":           rs_info,                    # internal, removed below
            })

        except Exception as exc:
            log.warning("predict_today(%s) lỗi: %s", ticker, exc)

    if progress_callback:
        progress_callback(0.86, "D3.2 Cross-sectional RS ranking...")

    # ── D3.2: Cross-sectional RS rank (universe-wide) ─────────────────────────
    if rs_info_list:
        rank_by_rs(rs_info_list)
    rs_lookup: dict[str, RSInfo] = {r.ticker: r for r in rs_info_list}

    if progress_callback:
        progress_callback(0.90, "S5 Entry Timing + D3.3 SL Filter...")

    # ── Pass 2: merge RS rank + entry zone + filter ───────────────────────────
    filtered: list[dict] = []

    for row in raw_results:
        ticker  = row["ticker"]
        prob    = row["probability"]
        rs_info = row.pop("_rs_info")   # lấy ra khỏi output

        # Gán RS info (rs_rank đã được cập nhật sau cross-sectional rank)
        final_rs = rs_lookup.get(ticker, rs_info)
        if final_rs is not None:
            row["rs"] = final_rs.as_dict()
        else:
            row["rs"] = {
                "ticker": ticker, "rs_1d": 0.0, "rs_5d": 0.0, "rs_20d": 0.0,
                "rs_60d": 0.0, "rs_trend": 0.0, "rs_score": 0.0,
                "rs_rank": 0.5, "rs_label": "Neutral",
            }

        rs_rank = row["rs"]["rs_rank"]

        # D3.2 Composite score
        row["composite_score"] = round(_W_PROB * prob + _W_RS * rs_rank, 4)

        # S5: Entry Timing
        ohlcv = ohlcv_cache.get(ticker)
        entry = compute_entry_zone(ohlcv) if ohlcv is not None else None

        if entry is not None:
            row["entry"] = entry.as_dict()
            # D3.3 SL Filter (chỉ áp dụng khi enable_regime_gate bật)
            if enable_regime_gate and not entry.is_valid():
                log.debug(
                    "[D3.3 SL FILTER] %s bi loai: sl_pct=%.1f%% rr=%.2f",
                    ticker, entry.sl_pct * 100, entry.rr_ratio,
                )
                continue
        else:
            row["entry"] = None

        # S4: Volume Confirmation
        vc = compute_volume_confirmation(ohlcv, ticker=ticker) if ohlcv is not None else None
        row["vc"] = vc.as_dict() if vc is not None else None

        # S4 Smart Money Flow (D0.2 SQLite)
        sm = compute_smart_money(ticker)
        row["sm"] = sm.as_dict()

        # D3.4: Portfolio Sizing (Kelly)
        rr = entry.rr_ratio if entry is not None else 1.0
        row["sizing"] = compute_position_size(
            probability=prob,
            rr_ratio=rr,
            regime_max_positions=regime.max_positions,
        ).as_dict()

        # Lọc probability threshold
        if prob < effective_min_prob:
            continue

        filtered.append(row)

    if progress_callback:
        progress_callback(1.0, "Hoàn tất dự đoán.")

    # Sort theo composite_score (D3.2): 60% P_alpha + 40% RS rank
    filtered.sort(key=lambda x: x["composite_score"], reverse=True)

    log.info(
        "Prediction: %d tickers → %d raw → %d picks "
        "(min_prob=%.2f effective=%.2f regime=%s)",
        len(tickers), len(raw_results), len(filtered),
        min_probability, effective_min_prob, regime.state.value,
    )
    return filtered


def predict_all(
    tickers: list[str] | None = None,
    period: str = "3m",
    progress_callback: Callable[[float, str], None] | None = None,
) -> pd.DataFrame:
    """
    Trả về full DataFrame với P_alpha cho TẤT CẢ tickers (không lọc threshold).
    Dùng cho phân tích toàn thị trường — regime gate bị tắt.
    """
    if tickers is None:
        tickers = VN30_TICKERS

    all_results = predict_today(
        tickers=tickers,
        period=period,
        min_probability=0.0,
        max_rsi=100.0,
        progress_callback=progress_callback,
        enable_regime_gate=False,   # analysis mode: không block
    )
    if not all_results:
        return pd.DataFrame()
    return pd.DataFrame(all_results).sort_values("probability", ascending=False)


def get_current_regime(period: str = "3m") -> RegimeInfo:
    """
    Trả về Market Regime hiện tại dựa trên VN30 OHLCV gần nhất.
    Dùng để hiển thị trên UI mà không cần chạy toàn bộ prediction pipeline.
    """
    vn30_df = get_index_data("VN30", period=period)
    return get_market_regime(vn30_df)


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

def _describe_pattern(acc: float, div: float, pp: float,
                      spring_q: float = 0.0, lps: float = 0.0) -> str:
    """Mô tả pattern bằng tiếng Việt (S1 Wyckoff v2.0 + OHLCV proxy)."""
    parts: list[str] = []
    if spring_q >= 0.60:
        parts.append("Spring chất lượng cao")
    elif spring_q >= 0.30:
        parts.append("Spring detected")
    if lps > 0.5:
        parts.append("LPS confirmed")
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


def _get_recommendation(p_cal: float) -> Recommendation:
    """Chuyển p_calibrated thành Recommendation enum."""
    if p_cal >= 0.80:
        return Recommendation.STRONG_BUY
    elif p_cal >= 0.70:
        return Recommendation.BUY
    elif p_cal >= 0.60:
        return Recommendation.WATCH
    elif p_cal >= 0.50:
        return Recommendation.HOLD
    else:
        return Recommendation.AVOID
