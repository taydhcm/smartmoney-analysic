"""
ml/smart_money.py
S4 Smart Money Flow Engine — phân tích dòng tiền tổ chức từ D0.2 SQLite.

Sprint 12: Bổ sung dòng tiền Tự Doanh (proprietary trading) từ SSI iBoard.

Outputs per ticker:
  ── Foreign flow (SQLite D0.2) ──
  - foreign_net_pct    : net foreign vol / total vol hôm nay [-1, +1]
  - foreign_net_5d     : TB 5 phiên gần nhất [-1, +1]
  - foreign_trend      : slope of foreign net pct 5 phiên [-1, +1]
  - smart_money_score  : 0.6*foreign_net_5d + 0.4*foreign_trend [-1, +1]

  ── Proprietary flow (SSI iBoard) ──  Sprint 12
  - proprietary_net_pct  : net prop vol / (total prop vol) hôm nay [-1, +1]
  - proprietary_net_5d   : TB 5 phiên [-1, +1]
  - prop_trend           : slope 5 phiên [-1, +1]

  ── Combined institutional score ──  Sprint 12
  - combined_institutional_score = 0.50*smart_money_score
                                 + 0.35*proprietary_net_5d
                                 + 0.15*prop_trend
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict

from utils.logger import get_logger

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MIN_SESSIONS = 5              # so phien toi thieu de tinh duoc S4
SM_CONFIRM_THRESHOLD = 0.15   # smart_money_score > this -> "Accumulating"
SM_DISTRIB_THRESHOLD = -0.15  # smart_money_score < this -> "Distributing"

# Sprint 12: proprietary weights
PROP_CONFIRM_THRESHOLD = 0.10   # prop net 5d > this → "Tích lũy tự doanh"
PROP_DISTRIB_THRESHOLD = -0.10  # prop net 5d < this → "Bán ròng tự doanh"

# Combined institutional score weights
_W_FOREIGN = 0.50
_W_PROP_5D = 0.35
_W_PROP_TR = 0.15

_W_5D    = 0.60
_W_TREND = 0.40


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class SmartMoneySignal:
    """Ket qua phan tich S4 Smart Money Flow cho 1 ticker."""

    ticker:             str
    sessions:           int       # so phien SQLite co san
    foreign_net_pct:    float     # hom nay [-1, +1]
    foreign_net_5d:     float     # TB 5 phien [-1, +1]
    foreign_trend:      float     # slope normalized [-1, +1]
    smart_money_score:  float     # composite [-1, +1]
    is_confirmed:       bool
    label:              str       # "Accumulating"/"Distributing"/"Neutral"/"Insufficient data"

    def as_dict(self) -> dict:
        d = asdict(self)
        # Round floats
        for k in ("foreign_net_pct", "foreign_net_5d", "foreign_trend", "smart_money_score"):
            d[k] = round(d[k], 4)
        return d


# ── Sprint 12: Proprietary Signal ────────────────────────────────────────────

@dataclass
class ProprietarySignal:
    """Kết quả phân tích dòng tiền Tự Doanh (Sprint 12)."""

    ticker:               str
    sessions:             int       # số phiên có data tự doanh
    proprietary_net_pct:  float     # hôm nay [-1, +1]
    proprietary_net_5d:   float     # TB 5 phiên [-1, +1]
    prop_trend:           float     # slope 5 phiên [-1, +1]
    label:                str       # "Tích lũy tự doanh"/"Bán ròng tự doanh"/"Trung lập"/"Không có data"

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("proprietary_net_pct", "proprietary_net_5d", "prop_trend"):
            d[k] = round(d[k], 4)
        return d


@dataclass
class InstitutionalFlowSignal:
    """Combined foreign + proprietary institutional signal (Sprint 12)."""

    ticker:                        str
    foreign:                       SmartMoneySignal
    proprietary:                   ProprietarySignal
    combined_institutional_score:  float   # [-1, +1]
    combined_label:                str     # aggregated interpretation

    def as_dict(self) -> dict:
        return {
            "ticker":                       self.ticker,
            "foreign":                      self.foreign.as_dict(),
            "proprietary":                  self.proprietary.as_dict(),
            "combined_institutional_score": round(self.combined_institutional_score, 4),
            "combined_label":               self.combined_label,
        }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _safe_net_pct(foreign_net: float, total_volume: float) -> float:
    """foreign_net / total_volume, clip [-1, +1]."""
    if total_volume < 1:
        return 0.0
    return float(np.clip(foreign_net / total_volume, -1.0, 1.0))


def _linear_trend(values: np.ndarray) -> float:
    """
    Slope cua duong hoi quy tuyen tinh, normalized boi max(|values|).
    Tra ve 0.0 neu it hon 3 diem.
    """
    n = len(values)
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=float)
    # Ordinary least squares
    xm = x - x.mean()
    ym = values - values.mean()
    denom = (xm ** 2).sum()
    if denom < 1e-9:
        return 0.0
    slope = (xm * ym).sum() / denom
    # Normalize: slope trong 1 buoc / spread (so sanh voi range gia tri)
    spread = float(np.abs(values).max()) or 1.0
    return float(np.clip(slope / spread, -1.0, 1.0))


def _label_from_score(score: float, sessions: int) -> str:
    if sessions < MIN_SESSIONS:
        return "Insufficient data"
    if score >= SM_CONFIRM_THRESHOLD:
        return "Accumulating"
    if score <= SM_DISTRIB_THRESHOLD:
        return "Distributing"
    return "Neutral"


# ── Public API ────────────────────────────────────────────────────────────────

def compute_smart_money(ticker: str, last_n: int = 20) -> SmartMoneySignal:
    """
    Tinh S4 Smart Money Signal cho 1 ticker tu SQLite.

    Parameters
    ----------
    ticker : Ma co phieu.
    last_n : So phien SQLite toi da can lay (20 du de tinh trend on dinh).

    Returns
    -------
    SmartMoneySignal — luon tra ve (khong tra None), dung truong 'sessions'
    va 'label' de phan biet "Insufficient data".
    """
    try:
        from data.db import load_snapshots
        df = load_snapshots(ticker, last_n=last_n)
    except Exception as exc:
        log.warning("compute_smart_money(%s): loi doc SQLite: %s", ticker, exc)
        return _empty_signal(ticker, sessions=0)

    sessions = len(df)

    if sessions == 0:
        return _empty_signal(ticker, sessions=0)

    # Tinh foreign_net_pct tung phien
    df["net_pct"] = df.apply(
        lambda r: _safe_net_pct(r["foreign_net"], r["total_volume"]), axis=1
    )

    # Hom nay (dong cuoi cung)
    today_net_pct = float(df["net_pct"].iloc[-1])

    # TB 5 phien gan nhat
    window5 = df["net_pct"].values[-min(5, sessions):]
    foreign_net_5d = float(np.clip(np.nanmean(window5), -1.0, 1.0))

    # Trend qua 5 phien (slope normalized)
    foreign_trend = _linear_trend(window5)

    # Composite score
    smart_money_score = float(np.clip(
        _W_5D * foreign_net_5d + _W_TREND * foreign_trend,
        -1.0, 1.0,
    ))

    is_confirmed = (sessions >= MIN_SESSIONS) and (smart_money_score >= SM_CONFIRM_THRESHOLD)
    label = _label_from_score(smart_money_score, sessions)

    return SmartMoneySignal(
        ticker            = ticker,
        sessions          = sessions,
        foreign_net_pct   = round(today_net_pct,      4),
        foreign_net_5d    = round(foreign_net_5d,     4),
        foreign_trend     = round(foreign_trend,      4),
        smart_money_score = round(smart_money_score,  4),
        is_confirmed      = is_confirmed,
        label             = label,
    )


def _empty_signal(ticker: str, sessions: int = 0) -> SmartMoneySignal:
    return SmartMoneySignal(
        ticker            = ticker,
        sessions          = sessions,
        foreign_net_pct   = 0.0,
        foreign_net_5d    = 0.0,
        foreign_trend     = 0.0,
        smart_money_score = 0.0,
        is_confirmed      = False,
        label             = "Insufficient data",
    )


def compute_smart_money_features(ticker: str) -> dict[str, float]:
    """
    Tra ve 3 feature values cho ML pipeline.
    Dung khi khong can full SmartMoneySignal object.

    Returns
    -------
    {"foreign_net_pct": float, "foreign_trend": float, "smart_money_score": float}
    Gia tri = 0.0 khi chua du data.
    """
    sm = compute_smart_money(ticker)
    return {
        "foreign_net_pct":   sm.foreign_net_5d,      # dung 5d TB on dinh hon
        "foreign_trend":     sm.foreign_trend,
        "smart_money_score": sm.smart_money_score,
    }


# ── Sprint 12: Proprietary flow functions ─────────────────────────────────────

def _prop_label(net_5d: float, sessions: int) -> str:
    if sessions < 1:
        return "Không có data"
    if net_5d >= PROP_CONFIRM_THRESHOLD:
        return "Tích lũy tự doanh"
    if net_5d <= PROP_DISTRIB_THRESHOLD:
        return "Bán ròng tự doanh"
    return "Trung lập"


def _empty_prop_signal(ticker: str) -> ProprietarySignal:
    return ProprietarySignal(
        ticker               = ticker,
        sessions             = 0,
        proprietary_net_pct  = 0.0,
        proprietary_net_5d   = 0.0,
        prop_trend           = 0.0,
        label                = "Không có data",
    )


def compute_proprietary_signal(ticker: str, last_n: int = 20) -> ProprietarySignal:
    """
    Tính dòng tiền Tự Doanh từ SQLite (cột proprietary_*) hoặc SSI iBoard.

    Strategy:
    1. Lấy từ SQLite (đã log bởi daily_snapshot với SSI data)
    2. Nếu SQLite chưa có → gọi SSI iBoard trực tiếp
    3. Nếu SSI không khả dụng → trả về signal rỗng (0.0)

    Returns
    -------
    ProprietarySignal — luôn trả về (không None).
    """
    ticker = ticker.upper()

    # ── Bước 1: từ SQLite ────────────────────────────────────────────────────
    try:
        from data.db import load_snapshots
        df = load_snapshots(ticker, last_n=last_n)
        if not df.empty and "proprietary_buy" in df.columns:
            prop_df = df[["proprietary_buy", "proprietary_sell",
                          "proprietary_net", "total_volume"]].copy()
            # Chỉ dùng rows có data tự doanh (> 0)
            has_data = (prop_df["proprietary_buy"] > 0) | (prop_df["proprietary_sell"] > 0)
            prop_df = prop_df[has_data]
            if len(prop_df) >= 1:
                return _calc_prop_signal(ticker, prop_df)
    except Exception as exc:
        log.debug("compute_proprietary_signal(%s) SQLite: %s", ticker, exc)

    # ── Bước 2: từ SSI iBoard trực tiếp ─────────────────────────────────────
    try:
        from analytics.ssi_iboard import fetch_investor_flow
        ssi_df = fetch_investor_flow(ticker, limit=last_n)
        if not ssi_df.empty:
            return _calc_prop_signal_from_ssi(ticker, ssi_df)
    except Exception as exc:
        log.debug("compute_proprietary_signal(%s) SSI: %s", ticker, exc)

    # ── Bước 3: fallback rỗng ────────────────────────────────────────────────
    return _empty_prop_signal(ticker)


def _calc_prop_signal(ticker: str, df) -> ProprietarySignal:
    """Tính ProprietarySignal từ DataFrame có prop+total_volume columns."""
    # Normalize net bởi total volume
    def safe_net_pct(buy, sell, total):
        gross = buy + sell
        if gross < 1:
            return 0.0
        return float(np.clip((buy - sell) / max(total, gross), -1.0, 1.0))

    net_pcts = df.apply(
        lambda r: safe_net_pct(
            r.get("proprietary_buy", 0),
            r.get("proprietary_sell", 0),
            r.get("total_volume", 0),
        ), axis=1
    ).values

    sessions = len(net_pcts)
    today_pct = float(net_pcts[-1]) if sessions > 0 else 0.0
    window5   = net_pcts[-min(5, sessions):]
    net_5d    = float(np.clip(np.nanmean(window5), -1.0, 1.0))
    trend     = _linear_trend(window5)

    return ProprietarySignal(
        ticker               = ticker,
        sessions             = sessions,
        proprietary_net_pct  = round(today_pct, 4),
        proprietary_net_5d   = round(net_5d,    4),
        prop_trend           = round(trend,      4),
        label                = _prop_label(net_5d, sessions),
    )


def _calc_prop_signal_from_ssi(ticker: str, ssi_df) -> ProprietarySignal:
    """Tính ProprietarySignal từ SSI iBoard DataFrame."""
    def safe_net_pct_ssi(row):
        buy  = float(row.get("proprietary_buy",  0) or 0)
        sell = float(row.get("proprietary_sell", 0) or 0)
        for_b = float(row.get("foreign_buy",  0) or 0)
        for_s = float(row.get("foreign_sell", 0) or 0)
        ret_b = float(row.get("retail_buy",   0) or 0)
        ret_s = float(row.get("retail_sell",  0) or 0)
        total = buy + sell + for_b + for_s + ret_b + ret_s
        if total < 1:
            return 0.0
        return float(np.clip((buy - sell) / total, -1.0, 1.0))

    net_pcts = ssi_df.apply(safe_net_pct_ssi, axis=1).values
    sessions = len(net_pcts)
    today_pct = float(net_pcts[-1]) if sessions > 0 else 0.0
    window5   = net_pcts[-min(5, sessions):]
    net_5d    = float(np.clip(np.nanmean(window5), -1.0, 1.0))
    trend     = _linear_trend(window5)

    return ProprietarySignal(
        ticker               = ticker,
        sessions             = sessions,
        proprietary_net_pct  = round(today_pct, 4),
        proprietary_net_5d   = round(net_5d,    4),
        prop_trend           = round(trend,      4),
        label                = _prop_label(net_5d, sessions),
    )


def compute_institutional_flow(ticker: str) -> InstitutionalFlowSignal:
    """
    Tính combined institutional flow (foreign + proprietary) cho 1 ticker.

    combined_institutional_score = 0.50 × smart_money_score
                                 + 0.35 × proprietary_net_5d
                                 + 0.15 × prop_trend
    """
    foreign = compute_smart_money(ticker)
    prop    = compute_proprietary_signal(ticker)

    combined = float(np.clip(
        _W_FOREIGN * foreign.smart_money_score
        + _W_PROP_5D * prop.proprietary_net_5d
        + _W_PROP_TR * prop.prop_trend,
        -1.0, 1.0,
    ))

    # Combined label
    if combined >= 0.15:
        clabel = "Tổ chức mua ròng mạnh"
    elif combined >= 0.05:
        clabel = "Tổ chức mua ròng nhẹ"
    elif combined <= -0.15:
        clabel = "Tổ chức bán ròng mạnh"
    elif combined <= -0.05:
        clabel = "Tổ chức bán ròng nhẹ"
    else:
        clabel = "Tổ chức trung lập"

    return InstitutionalFlowSignal(
        ticker                       = ticker,
        foreign                      = foreign,
        proprietary                  = prop,
        combined_institutional_score = round(combined, 4),
        combined_label               = clabel,
    )


def compute_institutional_flow_features(ticker: str) -> dict[str, float]:
    """
    Trả về 6 feature values cho ML pipeline (Sprint 12).

    Returns
    -------
    dict with keys:
      foreign_net_pct, foreign_trend, smart_money_score   (backward compat)
      proprietary_net_pct, prop_trend, combined_institutional_score  (Sprint 12)
    """
    flow = compute_institutional_flow(ticker)
    return {
        # Legacy foreign features (backward compat)
        "foreign_net_pct":             flow.foreign.foreign_net_5d,
        "foreign_trend":               flow.foreign.foreign_trend,
        "smart_money_score":           flow.foreign.smart_money_score,
        # Sprint 12: proprietary + combined
        "proprietary_net_pct":         flow.proprietary.proprietary_net_5d,
        "prop_trend":                  flow.proprietary.prop_trend,
        "combined_institutional_score": flow.combined_institutional_score,
    }

