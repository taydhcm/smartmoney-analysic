"""
ml/alert_generator.py
D3.5 Alert Generator — Sprint 7.

Tạo morning report cards từ picks output của predict_today():
    - AlertCard  : entry zone, SL giá cụ thể, target, conviction size, auto reason
    - PortfolioUsage : tracker tổng % deployed / VND còn lại
    - generate_morning_report() : batch generation từ danh sách picks
    - compute_portfolio_usage() : tổng hợp usage toàn danh mục

Conviction sizing rule (D3.4 Sprint 7):
    p_cal >= 0.70  →  30% vốn  (HIGH CONVICTION)
    p_cal <  0.70  →  20% vốn  (STANDARD)
    Bounded by: min(conviction_size, 1/regime_max_positions)

Auto reason builder (_build_reason):
    Ưu tiên: Wyckoff Spring/LPS → Smart Money → RS Outperform → Vol Surge → fallback
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime

from utils.logger import get_logger
from .portfolio_sizing import compute_conviction_size

log = get_logger(__name__)


# ── AlertCard ─────────────────────────────────────────────────────────────────

@dataclass
class AlertCard:
    """Morning report card cho 1 pick."""
    ticker:             str
    date:               str          # YYYY-MM-DD
    action:             str          # "BUY"
    # Price levels (S5 Entry Timing)
    entry_lo:           float
    entry_hi:           float
    sl_price:           float
    target_price:       float
    rr_ratio:           float
    sl_pct:             float        # % distance to SL (positive)
    # Conviction sizing (D3.4)
    position_size_pct:  float        # 0.30 hoặc 0.20 (conviction-based)
    position_size_vnd:  float        # VND amount (0 nếu capital không cung cấp)
    shares:             int          # lot size (bội số 100)
    # Probabilities (M4)
    p_calibrated:       float
    p_raw:              float
    recommendation:     str          # Recommendation enum value (str)
    ci_lo:              float
    ci_hi:              float
    # Regime context
    regime_state:       str
    regime_max_pos:     int
    # Auto reason
    reason:             str          # 1-liner tự động

    def as_dict(self) -> dict:
        return asdict(self)


# ── PortfolioUsage ────────────────────────────────────────────────────────────

@dataclass
class PortfolioUsage:
    """Tracker tổng usage portfolio từ danh sách AlertCards."""
    picks_count:           int
    max_positions:         int
    total_allocated_pct:   float    # tổng % đã phân bổ
    remaining_pct:         float    # % còn lại (max 1.0 - allocated)
    capital_deployed_vnd:  float    # tổng VND (0 nếu capital = 0)
    capital_remaining_vnd: float
    is_full:               bool     # True nếu picks_count >= max_positions

    def as_dict(self) -> dict:
        return asdict(self)


# ── Core functions ────────────────────────────────────────────────────────────

def generate_alert(
    pick: dict,
    portfolio_capital: float = 0.0,
) -> AlertCard | None:
    """
    Tạo AlertCard từ 1 pick dict (output của predict_today()).

    Parameters
    ----------
    pick               : Dict từ predict_today(). Cần có 'entry' và 'regime'.
    portfolio_capital  : Tổng vốn (VND). 0 → không tính tiền cụ thể.

    Returns
    -------
    AlertCard hoặc None nếu pick thiếu entry zone data hợp lệ.
    """
    entry  = pick.get("entry") or {}
    regime = pick.get("regime") or {}

    entry_lo     = float(entry.get("entry_low",    0.0))
    entry_hi     = float(entry.get("entry_high",   0.0))
    sl_price     = float(entry.get("sl_price",     0.0))
    target_price = float(entry.get("target_price", 0.0))
    rr_ratio     = float(entry.get("rr_ratio",     1.0))
    sl_pct       = float(entry.get("sl_pct",       0.05))

    # Cần entry zone hợp lệ để tạo AlertCard
    if entry_lo <= 0 or sl_price <= 0:
        log.debug("generate_alert(%s): thiếu entry zone data", pick.get("ticker", "?"))
        return None

    p_cal          = float(pick.get("p_calibrated", pick.get("probability", 0.5)))
    p_raw          = float(pick.get("probability",  0.5))
    regime_max_pos = int(regime.get("max_positions", 5))

    # D3.4 Conviction-based sizing (Sprint 7)
    size_pct = compute_conviction_size(p_cal, regime_max_pos)

    # VND + shares (lot 100)
    capital_amount = round(portfolio_capital * size_pct) if portfolio_capital > 0 else 0.0
    ref_price = (entry_lo + entry_hi) / 2 if entry_hi > entry_lo else entry_lo
    shares = (
        int(capital_amount / ref_price / 100) * 100
        if ref_price > 0 and capital_amount > 0
        else 0
    )

    return AlertCard(
        ticker            = str(pick.get("ticker", "?")),
        date              = datetime.now().strftime("%Y-%m-%d"),
        action            = "BUY",
        entry_lo          = entry_lo,
        entry_hi          = entry_hi,
        sl_price          = sl_price,
        target_price      = target_price,
        rr_ratio          = rr_ratio,
        sl_pct            = sl_pct,
        position_size_pct = size_pct,
        position_size_vnd = capital_amount,
        shares            = shares,
        p_calibrated      = p_cal,
        p_raw             = p_raw,
        recommendation    = str(pick.get("recommendation", "WATCH")),
        ci_lo             = float(pick.get("ci_lo", max(p_cal - 0.05, 0.0))),
        ci_hi             = float(pick.get("ci_hi", min(p_cal + 0.05, 1.0))),
        regime_state      = str(regime.get("regime", "NEUTRAL")),
        regime_max_pos    = regime_max_pos,
        reason            = _build_reason(pick),
    )


def generate_morning_report(
    picks: list[dict],
    capital: float = 0.0,
) -> list[AlertCard]:
    """
    Tạo danh sách AlertCard từ toàn bộ picks (output của predict_today()).

    Parameters
    ----------
    picks   : list[dict] từ predict_today(), sorted by composite_score.
    capital : Tổng vốn (VND). 0 → không tính tiền cụ thể.

    Returns
    -------
    list[AlertCard] — chỉ các pick có đủ entry zone data hợp lệ.
    """
    alerts: list[AlertCard] = []
    for pick in picks:
        card = generate_alert(pick, portfolio_capital=capital)
        if card is not None:
            alerts.append(card)
    log.info(
        "generate_morning_report: %d picks → %d alert cards",
        len(picks), len(alerts),
    )
    return alerts


def compute_portfolio_usage(
    alerts: list[AlertCard],
    max_positions: int,
    capital: float = 0.0,
) -> PortfolioUsage:
    """
    Tính tổng portfolio usage từ danh sách AlertCards.

    Parameters
    ----------
    alerts        : Danh sách AlertCard từ generate_morning_report().
    max_positions : Số vị thế tối đa (từ RegimeInfo.max_positions).
    capital       : Tổng vốn (VND). 0 → chỉ tính %.

    Returns
    -------
    PortfolioUsage
    """
    _max = max(max_positions, 1)
    total_pct   = min(sum(a.position_size_pct for a in alerts), 1.0)
    remaining   = max(0.0, 1.0 - total_pct)
    total_vnd   = sum(a.position_size_vnd for a in alerts) if capital > 0 else 0.0
    remain_vnd  = max(0.0, capital - total_vnd)            if capital > 0 else 0.0

    return PortfolioUsage(
        picks_count           = len(alerts),
        max_positions         = _max,
        total_allocated_pct   = round(total_pct, 4),
        remaining_pct         = round(remaining, 4),
        capital_deployed_vnd  = total_vnd,
        capital_remaining_vnd = remain_vnd,
        is_full               = len(alerts) >= _max,
    )


# ── Auto reason builder ───────────────────────────────────────────────────────

def _build_reason(pick: dict) -> str:
    """
    Tự động tạo 1 câu lý do ngắn gọn dựa trên top signals trong pick.

    Thứ tự ưu tiên:
        1. Wyckoff pattern (Spring Quality / LPS / Phase D score)
        2. Stopping Volume
        3. Effort vs Result (demand mạnh)
        4. Smart Money (ngoại tích lũy)
        5. Relative Strength (Outperform)
        6. Volume surge
        7. Fallback: P_calibrated + recommendation
    """
    parts: list[str] = []

    # ── S1 Wyckoff VSA ─────────────────────────────────────────────────────────
    wyk       = pick.get("wyckoff") or {}
    spring_q  = float(wyk.get("spring_quality",   0.0))
    lps_det   = bool( wyk.get("lps_detected",     False))
    wyk_score = float(wyk.get("wyckoff_score",    0.0))
    stop_vol  = bool( wyk.get("stopping_volume",  False))
    evr       = float(wyk.get("effort_vs_result", 0.0))
    no_sup    = float(wyk.get("no_supply_count",  0.0))

    if spring_q >= 0.65:
        parts.append(f"Wyckoff Spring ({spring_q:.0%})")
    elif lps_det:
        parts.append("LPS confirmed")
    elif wyk_score >= 0.60:
        parts.append(f"Wyckoff tích lũy ({wyk_score:.0%})")

    if stop_vol:
        parts.append("Stopping Volume")
    if evr > 0.25 and len(parts) < 3:
        parts.append("Demand > Supply")
    if no_sup >= 0.30 and len(parts) < 3:
        parts.append(f"No-supply {no_sup:.0%}")

    # ── S4 Smart Money ─────────────────────────────────────────────────────────
    sm       = pick.get("sm") or {}
    sm_label = sm.get("label", "Neutral")
    sm_score = float(sm.get("smart_money_score", 0.0))
    if sm_label == "Accumulating" and sm_score > 0.15:
        parts.append("Ngoại tích lũy")

    # ── S2 Relative Strength ───────────────────────────────────────────────────
    rs       = pick.get("rs") or {}
    rs_label = rs.get("rs_label", "Neutral")
    rs_5d    = float(rs.get("rs_5d", 0.0))
    if rs_label == "Outperform" and rs_5d > 1.0 and len(parts) < 3:
        parts.append(f"RS +{rs_5d:.1f}% vs VN30")

    # ── S4 Volume Confirmation ─────────────────────────────────────────────────
    vc          = pick.get("vc") or {}
    vol_surge   = float(vc.get("vol_surge",    0.0))
    vc_confirmed = bool(vc.get("is_confirmed", False))
    if vc_confirmed and vol_surge >= 1.5 and len(parts) < 4:
        parts.append(f"Vol surge {vol_surge:.1f}×")

    # ── Fallback ───────────────────────────────────────────────────────────────
    if not parts:
        p_cal = float(pick.get("p_calibrated", pick.get("probability", 0.5)))
        rec   = pick.get("recommendation", "WATCH")
        parts.append(f"P_cal={p_cal:.0%} ({rec})")

    return " · ".join(parts[:4])   # giới hạn 4 phần cho gọn
