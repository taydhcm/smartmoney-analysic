"""
ml/trade_log.py
Module B — Alpha Signal Trade Log (SQLite persistence).

Mục tiêu:
  - Ghi toàn bộ tín hiệu từ predict_today() vào DB mỗi ngày
  - Sau T+5 → tự động resolve WIN/LOSS bằng cách fetch giá thực từ vnstock
  - Cung cấp track record out-of-sample theo thời gian thực

DB: ml/artifacts/trade_log.db

Table: alpha_signals
    id                  INTEGER PRIMARY KEY AUTOINCREMENT
    signal_date         TEXT    YYYY-MM-DD (ngày phát signal)
    ticker              TEXT
    p_alpha             REAL    xác suất raw (probability)
    p_calibrated        REAL    xác suất sau calibration
    pattern             TEXT    mô tả pattern (Spring, LPS, ...)
    confidence          TEXT    'high' | 'medium' | 'low'
    entry_price         REAL    giá tham chiếu (midpoint entry zone hoặc close)
    target_pct          REAL    0.05 = +5% target
    sl_pct              REAL    0.05 = -5% stop-loss
    outcome             TEXT    'PENDING' | 'WIN' | 'LOSS' | 'EXPIRED'
    exit_price          REAL    giá thoát thực tế (khi resolved)
    gross_return_pct    REAL    return trước phí (%)
    net_return_pct      REAL    return sau phí (%)
    round_trip_cost_pct REAL    phí round-trip đã áp dụng (%)
    resolve_date        TEXT    ngày T+5 trading (ngày resolve)
    recorded_at         TEXT    timestamp ghi vào DB
    UNIQUE (signal_date, ticker)
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_PCT = 0.05   # +5% win condition
SL_PCT     = 0.05   # -5% loss condition

_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
TRADE_LOG_DB   = _ARTIFACTS_DIR / "trade_log.db"

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS alpha_signals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_date         TEXT    NOT NULL,
    ticker              TEXT    NOT NULL,
    p_alpha             REAL    NOT NULL DEFAULT 0,
    p_calibrated        REAL    NOT NULL DEFAULT 0,
    pattern             TEXT    NOT NULL DEFAULT '',
    confidence          TEXT    NOT NULL DEFAULT '',
    entry_price         REAL    NOT NULL DEFAULT 0,
    target_pct          REAL    NOT NULL DEFAULT 0.05,
    sl_pct              REAL    NOT NULL DEFAULT 0.05,
    outcome             TEXT    NOT NULL DEFAULT 'PENDING',
    exit_price          REAL,
    gross_return_pct    REAL,
    net_return_pct      REAL,
    round_trip_cost_pct REAL    NOT NULL DEFAULT 0.4,
    resolve_date        TEXT    NOT NULL,
    recorded_at         TEXT    NOT NULL,
    UNIQUE (signal_date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_tl_date    ON alpha_signals (signal_date);
CREATE INDEX IF NOT EXISTS idx_tl_ticker  ON alpha_signals (ticker, signal_date);
CREATE INDEX IF NOT EXISTS idx_tl_outcome ON alpha_signals (outcome);
CREATE INDEX IF NOT EXISTS idx_tl_resolve ON alpha_signals (resolve_date, outcome);
"""


# ── DB helpers ────────────────────────────────────────────────────────────────

def _ensure_db() -> Path:
    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(TRADE_LOG_DB)) as con:
        con.executescript(_DDL)
        con.commit()
    return TRADE_LOG_DB


@contextmanager
def _get_connection():
    _ensure_db()
    con = sqlite3.connect(str(TRADE_LOG_DB), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ── Business day helpers ──────────────────────────────────────────────────────

def _add_business_days(d: date, n: int) -> date:
    """Cộng n ngày giao dịch (T2–T6) vào ngày d."""
    current = d
    added = 0
    while added < n:
        current += timedelta(days=1)
        if current.weekday() < 5:
            added += 1
    return current


def _t5_date(signal_date: date) -> date:
    """Trả về ngày T+5 giao dịch."""
    return _add_business_days(signal_date, 5)


# ── Log signals ───────────────────────────────────────────────────────────────

def log_alpha_signals(
    picks: list[dict],
    signal_date: str | None = None,
    round_trip_cost: float = 0.004,
) -> int:
    """
    Ghi danh sách picks từ predict_today() vào alpha_signals.

    Parameters
    ----------
    picks           : Output của predict_today() — list[dict].
    signal_date     : Ngày phát signal (YYYY-MM-DD). Mặc định = today.
    round_trip_cost : Phí round-trip (0.004 = 0.40%).

    Returns
    -------
    Số signal được insert mới (bỏ qua duplicate).
    """
    if not picks:
        return 0

    today_str    = signal_date or date.today().isoformat()
    today_d      = date.fromisoformat(today_str)
    resolve_str  = _t5_date(today_d).isoformat()
    now_str      = datetime.now().isoformat(timespec="seconds")
    cost_pct     = round(round_trip_cost * 100, 4)

    rows = []
    for pick in picks:
        ticker = pick.get("ticker", "")
        if not ticker:
            continue

        p_alpha      = float(pick.get("probability", 0))
        p_cal        = float(pick.get("p_calibrated", p_alpha))
        pattern      = str(pick.get("pattern", ""))
        confidence   = str(pick.get("confidence", ""))

        # Entry price: midpoint của entry zone, fallback = 0
        entry_dict   = pick.get("entry") or {}
        entry_lo     = float(entry_dict.get("entry_low",  0) or 0)
        entry_hi     = float(entry_dict.get("entry_high", 0) or 0)
        if entry_lo > 0 and entry_hi > 0:
            entry_price = round((entry_lo + entry_hi) / 2, 2)
        elif entry_lo > 0:
            entry_price = entry_lo
        else:
            entry_price = 0.0

        rows.append((
            today_str, ticker, p_alpha, p_cal,
            pattern, confidence, entry_price,
            TARGET_PCT, SL_PCT,
            "PENDING",          # outcome
            None,               # exit_price
            None,               # gross_return_pct
            None,               # net_return_pct
            cost_pct,
            resolve_str,
            now_str,
        ))

    if not rows:
        return 0

    inserted = 0
    with _get_connection() as con:
        for row in rows:
            cur = con.execute(
                """
                INSERT OR IGNORE INTO alpha_signals
                    (signal_date, ticker, p_alpha, p_calibrated,
                     pattern, confidence, entry_price,
                     target_pct, sl_pct,
                     outcome, exit_price,
                     gross_return_pct, net_return_pct, round_trip_cost_pct,
                     resolve_date, recorded_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                row,
            )
            inserted += cur.rowcount

    log.info("log_alpha_signals: %d/%d signals inserted (%s)", inserted, len(rows), today_str)
    return inserted


# ── Resolve outcomes ──────────────────────────────────────────────────────────

def resolve_pending_outcomes(
    round_trip_cost: float = 0.004,
    as_of_date: date | None = None,
) -> int:
    """
    Resolve tất cả signal PENDING có resolve_date <= as_of_date.
    Fetch giá thực từ vnstock, xác định WIN/LOSS.

    Logic path-dependent (giống dataset_builder):
      - Duyệt từng phiên T+1..T+5:
          if high >= entry * (1 + TARGET_PCT) → WIN (chạm target trước)
          if low  <= entry * (1 - SL_PCT)     → LOSS (chạm SL trước)
      - Sau T+5 nếu chưa resolve: dùng close[T+5]
          close >= entry * (1 + TARGET_PCT) → WIN
          close <= entry * (1 - SL_PCT)     → LOSS
          else                              → EXPIRED

    Returns
    -------
    Số signal đã resolve.
    """
    if as_of_date is None:
        as_of_date = date.today()

    resolved = 0
    with _get_connection() as con:
        rows = con.execute(
            """
            SELECT id, signal_date, ticker, entry_price, target_pct, sl_pct, round_trip_cost_pct
            FROM alpha_signals
            WHERE outcome = 'PENDING'
              AND resolve_date <= ?
            ORDER BY signal_date
            """,
            (as_of_date.isoformat(),),
        ).fetchall()

    for row in rows:
        sig_id      = row["id"]
        sig_date    = date.fromisoformat(row["signal_date"])
        ticker      = row["ticker"]
        entry_price = float(row["entry_price"] or 0)
        tgt_pct     = float(row["target_pct"] or TARGET_PCT)
        sl_pct_     = float(row["sl_pct"]     or SL_PCT)
        cost_pct    = float(row["round_trip_cost_pct"] or round_trip_cost * 100)

        if entry_price <= 0:
            _update_signal_outcome(sig_id, "EXPIRED", None, None, None)
            resolved += 1
            continue

        outcome, exit_price, gross_pct = _resolve_path_dependent(
            ticker, sig_date, entry_price, tgt_pct, sl_pct_
        )
        if gross_pct is not None:
            net_pct = round(gross_pct - cost_pct, 3)
        else:
            net_pct = None

        _update_signal_outcome(sig_id, outcome, exit_price, gross_pct, net_pct)
        resolved += 1
        log.info("Resolved %s %s → %s (gross=%.2f%%)", ticker, sig_date, outcome, gross_pct or 0)

    return resolved


def _resolve_path_dependent(
    ticker: str,
    sig_date: date,
    entry_price: float,
    target_pct: float,
    sl_pct: float,
) -> tuple[str, Optional[float], Optional[float]]:
    """Trả về (outcome, exit_price, gross_return_pct)."""
    try:
        import pandas as pd
        from data.market_data import get_ohlcv

        t1_date = _add_business_days(sig_date, 1)
        t5_date = _t5_date(sig_date)

        ohlcv = get_ohlcv(ticker, period="1m")
        if ohlcv is None or ohlcv.empty:
            return "EXPIRED", None, None

        # Normalize date column
        if "date" not in ohlcv.columns:
            ohlcv = ohlcv.reset_index()
        ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.date

        window = ohlcv[
            (ohlcv["date"] >= t1_date) & (ohlcv["date"] <= t5_date)
        ].copy()

        if window.empty:
            return "PENDING", None, None

        target_price = entry_price * (1 + target_pct)
        sl_price     = entry_price * (1 - sl_pct)

        # Path-dependent: duyệt từng phiên theo thứ tự thời gian
        window_sorted = window.sort_values("date")
        for _, day in window_sorted.iterrows():
            high = float(day.get("high", day.get("close", entry_price)))
            low  = float(day.get("low",  day.get("close", entry_price)))

            if low <= sl_price and high >= target_price:
                # Cả 2 trong cùng phiên → ưu tiên SL (conservative)
                pnl = round((sl_price - entry_price) / entry_price * 100, 3)
                return "LOSS", round(sl_price, 2), pnl
            elif high >= target_price:
                pnl = round((target_price - entry_price) / entry_price * 100, 3)
                return "WIN", round(target_price, 2), pnl
            elif low <= sl_price:
                pnl = round((sl_price - entry_price) / entry_price * 100, 3)
                return "LOSS", round(sl_price, 2), pnl

        # T+5 qua mà chưa resolve → dùng close T+5
        last_close = float(window_sorted.iloc[-1].get("close", entry_price))
        pnl = round((last_close - entry_price) / entry_price * 100, 3)
        if last_close >= target_price:
            return "WIN", round(last_close, 2), pnl
        elif last_close <= sl_price:
            return "LOSS", round(last_close, 2), pnl
        else:
            return "EXPIRED", round(last_close, 2), pnl

    except Exception as e:
        log.warning("_resolve_path_dependent %s %s: %s", ticker, sig_date, e)
        return "EXPIRED", None, None


def _update_signal_outcome(
    sig_id: int,
    outcome: str,
    exit_price: Optional[float],
    gross_return_pct: Optional[float],
    net_return_pct: Optional[float],
) -> None:
    with _get_connection() as con:
        con.execute(
            """
            UPDATE alpha_signals
            SET outcome=?, exit_price=?,
                gross_return_pct=?, net_return_pct=?
            WHERE id=?
            """,
            (outcome, exit_price, gross_return_pct, net_return_pct, sig_id),
        )


# ── Query helpers ─────────────────────────────────────────────────────────────

def get_track_record(days: int = 90) -> "pd.DataFrame":
    """
    Trả về DataFrame tất cả signals trong `days` ngày gần nhất.

    Columns: signal_date, ticker, p_alpha, p_calibrated, pattern, confidence,
             entry_price, outcome, exit_price, gross_return_pct, net_return_pct,
             round_trip_cost_pct, resolve_date.
    """
    import pandas as pd

    cutoff = (date.today() - timedelta(days=days)).isoformat()
    _ensure_db()
    with sqlite3.connect(str(TRADE_LOG_DB)) as con:
        df = pd.read_sql_query(
            """
            SELECT signal_date, ticker, p_alpha, p_calibrated,
                   pattern, confidence, entry_price,
                   outcome, exit_price,
                   gross_return_pct, net_return_pct, round_trip_cost_pct,
                   resolve_date, recorded_at
            FROM alpha_signals
            WHERE signal_date >= ?
            ORDER BY signal_date DESC, ticker
            """,
            con,
            params=(cutoff,),
        )
    return df


def get_weekly_win_rate() -> "pd.DataFrame":
    """
    Trả về DataFrame thống kê win rate theo tuần (chỉ tính WIN/LOSS, bỏ PENDING/EXPIRED).

    Columns: week_start, total_trades, wins, losses, win_rate_pct, avg_net_return_pct.
    """
    import pandas as pd

    _ensure_db()
    with sqlite3.connect(str(TRADE_LOG_DB)) as con:
        df = pd.read_sql_query(
            """
            SELECT
                -- SQLite: date(signal_date, 'weekday 0', '-6 days') = Monday of that week
                date(signal_date, 'weekday 1', '-7 days') AS week_start,
                COUNT(*) FILTER (WHERE outcome IN ('WIN','LOSS')) AS total_trades,
                COUNT(*) FILTER (WHERE outcome = 'WIN')  AS wins,
                COUNT(*) FILTER (WHERE outcome = 'LOSS') AS losses,
                ROUND(
                    100.0 * COUNT(*) FILTER (WHERE outcome = 'WIN')
                    / NULLIF(COUNT(*) FILTER (WHERE outcome IN ('WIN','LOSS')), 0),
                    1
                ) AS win_rate_pct,
                ROUND(AVG(net_return_pct) FILTER (WHERE outcome IN ('WIN','LOSS')), 2)
                    AS avg_net_return_pct
            FROM alpha_signals
            WHERE outcome IN ('WIN','LOSS','PENDING','EXPIRED')
            GROUP BY week_start
            ORDER BY week_start DESC
            """,
            con,
        )
    return df


def get_ticker_stats() -> "pd.DataFrame":
    """
    Thống kê theo ticker (chỉ WIN/LOSS).

    Columns: ticker, total_trades, wins, win_rate_pct, avg_net_return_pct,
             avg_p_alpha, last_signal_date.
    """
    import pandas as pd

    _ensure_db()
    with sqlite3.connect(str(TRADE_LOG_DB)) as con:
        df = pd.read_sql_query(
            """
            SELECT
                ticker,
                COUNT(*) FILTER (WHERE outcome IN ('WIN','LOSS')) AS total_trades,
                COUNT(*) FILTER (WHERE outcome = 'WIN')  AS wins,
                ROUND(
                    100.0 * COUNT(*) FILTER (WHERE outcome = 'WIN')
                    / NULLIF(COUNT(*) FILTER (WHERE outcome IN ('WIN','LOSS')), 0),
                    1
                ) AS win_rate_pct,
                ROUND(AVG(net_return_pct) FILTER (WHERE outcome IN ('WIN','LOSS')), 2)
                    AS avg_net_return_pct,
                ROUND(AVG(p_alpha), 3) AS avg_p_alpha,
                MAX(signal_date) AS last_signal_date
            FROM alpha_signals
            GROUP BY ticker
            HAVING COUNT(*) FILTER (WHERE outcome IN ('WIN','LOSS')) > 0
            ORDER BY win_rate_pct DESC, total_trades DESC
            """,
            con,
        )
    return df


def get_pending_count() -> int:
    """Số signal đang PENDING."""
    _ensure_db()
    with sqlite3.connect(str(TRADE_LOG_DB)) as con:
        row = con.execute(
            "SELECT COUNT(*) FROM alpha_signals WHERE outcome='PENDING'"
        ).fetchone()
    return int(row[0]) if row else 0


def get_summary_stats() -> dict:
    """
    Trả về dict tổng hợp: total, resolved, win_rate, avg_net_return, pending.
    """
    _ensure_db()
    with sqlite3.connect(str(TRADE_LOG_DB)) as con:
        row = con.execute(
            """
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE outcome IN ('WIN','LOSS')) AS resolved,
                COUNT(*) FILTER (WHERE outcome = 'WIN')  AS wins,
                COUNT(*) FILTER (WHERE outcome = 'PENDING') AS pending,
                ROUND(AVG(net_return_pct) FILTER (WHERE outcome IN ('WIN','LOSS')), 2)
                    AS avg_net_return_pct
            FROM alpha_signals
            """,
        ).fetchone()
    if row is None:
        return {"total": 0, "resolved": 0, "wins": 0, "pending": 0,
                "win_rate_pct": 0.0, "avg_net_return_pct": 0.0}
    total    = int(row[0])
    resolved = int(row[1])
    wins     = int(row[2])
    pending  = int(row[3])
    avg_net  = float(row[4] or 0)
    win_rate = round(wins / resolved * 100, 1) if resolved > 0 else 0.0
    return {
        "total":              total,
        "resolved":           resolved,
        "wins":               wins,
        "pending":            pending,
        "win_rate_pct":       win_rate,
        "avg_net_return_pct": avg_net,
    }
