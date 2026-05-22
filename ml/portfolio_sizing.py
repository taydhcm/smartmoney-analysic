"""
ml/portfolio_sizing.py
D3.4 Portfolio Sizing — Kelly Criterion.

f* = (p * b - q) / b
  where b = R:R ratio (win/loss), p = P_alpha, q = 1-p

Conservative approach:
  - Dùng Half-Kelly (f*/2) làm điểm khởi đầu
  - Cap theo regime max_positions (vốn phân bổ đều)
  - Hard cap: MIN 2%, MAX 20% mỗi vị thế

Nếu Kelly âm → EV âm → không vào lệnh (position = 0).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict

from utils.logger import get_logger

log = get_logger(__name__)

# ── Position size limits ──────────────────────────────────────────────────────
MIN_POSITION_PCT  = 0.02   # 2% vốn tối thiểu
MAX_POSITION_PCT  = 0.20   # 20% vốn tối đa mỗi vị thế
HALF_KELLY_FACTOR = 0.50   # Conservative: dùng nửa Kelly


@dataclass
class PositionSize:
    """Khuyến nghị vị thế cho 1 cổ phiếu."""
    kelly_pct:        float   # Full Kelly fraction [0, 1]
    half_kelly_pct:   float   # Half-Kelly (conservative)
    recommended_pct:  float   # Sau khi áp cap + regime
    max_positions:    int     # Từ Regime Engine
    capital_amount:   float   # Số tiền (VND) nếu portfolio_capital được cung cấp
    shares:           int     # Số cổ phiếu lô 100 nếu stock_price được cung cấp
    reasoning:        str     # Giải thích

    def as_dict(self) -> dict:
        return asdict(self)


def compute_position_size(
    probability:          float,
    rr_ratio:             float,
    regime_max_positions: int   = 5,
    portfolio_capital:    float = 0.0,
    stock_price:          float = 0.0,
) -> PositionSize:
    """
    Tính position size theo Kelly Criterion.

    Parameters
    ----------
    probability           : P_alpha từ ML model [0, 1].
    rr_ratio              : Risk:Reward ratio (target_gain / sl_loss).
    regime_max_positions  : Số vị thế tối đa đồng thời (từ RegimeInfo).
    portfolio_capital     : Tổng vốn (VND). 0 → không tính tiền cụ thể.
    stock_price           : Giá cổ phiếu. 0 → không tính khối lượng.

    Returns
    -------
    PositionSize
    """
    p = float(np.clip(probability, 0.0, 1.0))
    q = 1.0 - p
    b = max(0.10, float(rr_ratio))   # R:R phải dương

    # Full Kelly: f* = (p*b - q) / b = p - q/b
    kelly_full = (p * b - q) / b

    if kelly_full <= 0:
        reason = (
            f"Kelly < 0 ({kelly_full:.2%}) — EV âm với P={p:.0%}, R:R={b:.2f}. "
            "Không vào lệnh."
        )
        return PositionSize(
            kelly_pct=round(kelly_full, 4),
            half_kelly_pct=0.0,
            recommended_pct=0.0,
            max_positions=regime_max_positions,
            capital_amount=0.0,
            shares=0,
            reasoning=reason,
        )

    kelly_half = kelly_full * HALF_KELLY_FACTOR

    # Regime cap: phân bổ đều tối đa
    regime_cap = 1.0 / max(regime_max_positions, 1)

    # Recommended: half-Kelly, capped by regime & hard limits
    cap_used = min(MAX_POSITION_PCT, regime_cap)
    recommended = float(np.clip(kelly_half, MIN_POSITION_PCT, cap_used))

    # Build reasoning
    parts = [f"Kelly={kelly_full:.1%}", f"Half-Kelly={kelly_half:.1%}"]
    if kelly_half > MAX_POSITION_PCT:
        parts.append(f"cap max {MAX_POSITION_PCT:.0%}")
    elif kelly_half > regime_cap:
        parts.append(f"cap regime ({regime_max_positions} pos → {regime_cap:.0%}/pos)")
    elif kelly_half < MIN_POSITION_PCT:
        parts.append(f"floor min {MIN_POSITION_PCT:.0%}")
    reasoning = " → ".join(parts) + f" → De xuat: {recommended:.1%}"

    # VND amounts
    capital_amount = round(portfolio_capital * recommended, 0) if portfolio_capital > 0 else 0.0
    shares = (
        int(capital_amount / stock_price / 100) * 100
        if stock_price > 0 and capital_amount > 0
        else 0
    )

    return PositionSize(
        kelly_pct=round(kelly_full, 4),
        half_kelly_pct=round(kelly_half, 4),
        recommended_pct=round(recommended, 4),
        max_positions=regime_max_positions,
        capital_amount=capital_amount,
        shares=shares,
        reasoning=reasoning,
    )
