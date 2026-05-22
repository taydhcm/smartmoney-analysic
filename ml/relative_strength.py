"""
ml/relative_strength.py
S2 Market Relative Strength Engine.

Đo sức mạnh tương đối (RS) của từng cổ phiếu so với VN30:
  - Multi-timeframe: 1d, 5d, 20d, 60d
  - RS trend: RS line đang cải thiện hay xấu đi (độ dốc 12 phiên gần nhất)
  - Cross-sectional rank: percentile trong VN30 universe tại predict_today
  - Composite RS score: tổng hợp → [-1, +1]

Không look-ahead: tất cả tính từ dữ liệu hiện tại trở về trước.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RSInfo:
    """Kết quả RS cho 1 cổ phiếu tại 1 thời điểm."""
    ticker: str
    rs_1d:    float   # Excess return 1 ngày vs VN30 (%)
    rs_5d:    float   # Excess return 5 ngày vs VN30 (%)
    rs_20d:   float   # Excess return 20 ngày vs VN30 (%)
    rs_60d:   float   # Excess return 60 ngày vs VN30 (%)
    rs_trend: float   # Độ dốc RS line (+ = đang cải thiện, - = yếu dần)
    rs_score: float   # Composite score [-1, +1]
    rs_rank:  float   # Percentile [0, 1] trong universe — set bởi rank_by_rs()
    rs_label: str     # "Outperform" / "Neutral" / "Underperform"

    def as_dict(self) -> dict:
        return asdict(self)


# ── Thresholds ────────────────────────────────────────────────────────────────

_OUTPERFORM_THRESHOLD  =  0.20   # rs_score ≥ 0.20 → Outperform
_UNDERPERFORM_THRESHOLD = -0.20  # rs_score ≤ -0.20 → Underperform

# Weights cho composite score (tổng = 1.0)
_W_1D  = 0.15
_W_5D  = 0.35
_W_20D = 0.35
_W_TREND = 0.15

# Normalisation constants (divisor đưa từng thành phần về khoảng [-1, 1])
_NORM_1D   = 0.020   # 2% daily excess → max contribution
_NORM_5D   = 0.050   # 5% weekly excess → max contribution
_NORM_20D  = 0.100   # 10% monthly excess → max contribution
_NORM_TREND = 0.030  # trend slope normaliser


def compute_stock_rs(
    stock_df: pd.DataFrame,
    vn30_df:  pd.DataFrame,
    ticker:   str = "",
) -> RSInfo | None:
    """
    Tính RS của 1 cổ phiếu vs VN30.

    Parameters
    ----------
    stock_df : OHLCV 1 ticker (cột bắt buộc: date, close).
    vn30_df  : OHLCV VN30 (cột bắt buộc: date, close).
    ticker   : Mã CK (dùng cho log + RSInfo).

    Returns
    -------
    RSInfo hoặc None nếu không đủ dữ liệu (< 21 phiên).
    """
    if stock_df is None or stock_df.empty or vn30_df is None or vn30_df.empty:
        return None

    try:
        stock = (
            stock_df.copy()
            .sort_values("date")
            .set_index("date")["close"]
            .astype(float)
        )
        vn30 = (
            vn30_df.copy()
            .sort_values("date")
            .set_index("date")["close"]
            .astype(float)
        )
    except Exception as exc:
        log.warning("compute_stock_rs(%s): data prep lỗi: %s", ticker, exc)
        return None

    if len(stock) < 21:
        log.debug("compute_stock_rs(%s): không đủ phiên (%d < 21)", ticker, len(stock))
        return None

    # Align VN30 vào trading dates của stock (forward-fill gaps)
    vn30_aligned = vn30.reindex(stock.index, method="ffill").bfill()

    # ── Excess returns ────────────────────────────────────────────────────────
    def _excess_pct(n: int) -> float:
        if len(stock) <= n or len(vn30_aligned) <= n:
            return 0.0
        s_ret = float(stock.pct_change(n).iloc[-1])
        v_ret = float(vn30_aligned.pct_change(n).iloc[-1])
        # Bỏ NaN (phiên đầu tiên không có prev)
        if np.isnan(s_ret) or np.isnan(v_ret):
            return 0.0
        return s_ret - v_ret

    rs_1d  = _excess_pct(1)
    rs_5d  = _excess_pct(5)
    rs_20d = _excess_pct(20)
    rs_60d = _excess_pct(60) if len(stock) > 62 else _excess_pct(20)

    # ── RS trend (độ dốc RS line 12 phiên gần nhất) ───────────────────────────
    # RS line = tỷ số stock/VN30 (không normalized — chỉ cần slope)
    ratio_line = (stock / (vn30_aligned + 1e-9)).iloc[-14:]
    if len(ratio_line) >= 5:
        x = np.arange(len(ratio_line), dtype=float)
        y = ratio_line.values.astype(float)
        # Loại bỏ NaN trước khi fit
        mask = ~np.isnan(y)
        if mask.sum() >= 4:
            slope = float(np.polyfit(x[mask], y[mask], 1)[0])
            # Normalize về scale tương đương: slope / mean_ratio
            mean_ratio = float(ratio_line.mean())
            rs_trend = slope / (abs(mean_ratio) + 1e-9)
        else:
            rs_trend = 0.0
    else:
        rs_trend = 0.0

    # ── Composite score ───────────────────────────────────────────────────────
    rs_score = float(np.clip(
        _W_1D   * np.clip(rs_1d   / _NORM_1D,   -1.0, 1.0)
        + _W_5D   * np.clip(rs_5d   / _NORM_5D,   -1.0, 1.0)
        + _W_20D  * np.clip(rs_20d  / _NORM_20D,  -1.0, 1.0)
        + _W_TREND * np.clip(rs_trend / _NORM_TREND, -1.0, 1.0),
        -1.0, 1.0,
    ))

    rs_label = (
        "Outperform"   if rs_score >= _OUTPERFORM_THRESHOLD
        else "Underperform" if rs_score <= _UNDERPERFORM_THRESHOLD
        else "Neutral"
    )

    return RSInfo(
        ticker=ticker,
        rs_1d=round(rs_1d * 100, 3),    # đổi sang %
        rs_5d=round(rs_5d * 100, 3),
        rs_20d=round(rs_20d * 100, 3),
        rs_60d=round(rs_60d * 100, 3),
        rs_trend=round(rs_trend, 6),
        rs_score=round(rs_score, 4),
        rs_rank=0.5,                      # placeholder — set bởi rank_by_rs()
        rs_label=rs_label,
    )


def rank_by_rs(rs_list: list[RSInfo]) -> list[RSInfo]:
    """
    Tính cross-sectional percentile rank trong universe.

    rs_rank = 0.0 (yếu nhất) → 1.0 (mạnh nhất) dựa theo rs_score.
    Cập nhật trực tiếp vào từng RSInfo object.

    Parameters
    ----------
    rs_list : Danh sách RSInfo từ compute_stock_rs().

    Returns
    -------
    Cùng list, đã cập nhật rs_rank.
    """
    if not rs_list:
        return rs_list

    scores = np.array([r.rs_score for r in rs_list], dtype=float)
    n = len(scores)

    if n == 1:
        rs_list[0].rs_rank = 0.5
        return rs_list

    # Ordinal rank → normalize về [0, 1]
    ordinal_ranks = scores.argsort().argsort().astype(float)
    percentiles   = ordinal_ranks / (n - 1)

    for rs_info, pct in zip(rs_list, percentiles):
        rs_info.rs_rank = round(float(pct), 4)

    return rs_list
