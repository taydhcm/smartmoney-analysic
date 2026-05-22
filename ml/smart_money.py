"""
ml/smart_money.py
S4 Smart Money Flow Engine — phan tich dong tien khoi ngoai tu D0.2 SQLite.

Yeu cau >= 5 phien SQLite data (MIN_SESSIONS_FOR_S4).
Khi chua du data: tra ve SmartMoneySignal voi is_confirmed=False, label="Insufficient data".

Outputs per ticker:
  - foreign_net_pct    : net foreign vol / total vol hom nay [-1, +1]
  - foreign_net_5d     : TB 5 phien gan nhat cua foreign_net_pct [-1, +1]
  - foreign_trend      : slope of foreign_net_pct qua 5 phien (normalized) [-1, +1]
  - smart_money_score  : 0.6 * foreign_net_5d + 0.4 * foreign_trend [-1, +1]
  - is_confirmed       : score > SM_CONFIRM_THRESHOLD
  - label              : "Accumulating" / "Distributing" / "Neutral" / "Insufficient data"
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
