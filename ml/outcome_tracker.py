"""
ml/outcome_tracker.py
E5 Outcome Tracker — Sprint 8.

Ghi toàn bộ signal phát ra vào SQLite signal_log.db.
Auto-check T+5 outcome lúc 15:15 hàng ngày.

DB: data/db/signal_log.db

Tables
------
signal_log
    id                INTEGER PRIMARY KEY AUTOINCREMENT
    signal_date       TEXT    YYYY-MM-DD (ngày phát signal)
    ticker            TEXT
    action            TEXT    "BUY"
    entry_price       REAL    (entry_lo từ AlertCard)
    sl_price          REAL
    target_price      REAL
    rr_ratio          REAL
    p_calibrated      REAL
    p_raw             REAL
    recommendation    TEXT    "STRONG_BUY" | "BUY" | ...
    position_size_pct REAL
    regime_state      TEXT
    reason            TEXT
    created_at        TEXT
    UNIQUE (signal_date, ticker)

signal_outcomes
    id           INTEGER PRIMARY KEY AUTOINCREMENT
    signal_id    INTEGER REFERENCES signal_log(id)
    outcome_date TEXT    YYYY-MM-DD (ngày resolve T+5)
    outcome      TEXT    "WIN" | "LOSS" | "FLAT" | "PENDING" | "EXPIRED"
    exit_price   REAL
    pnl_pct      REAL    (signed %)
    checked_at   TEXT
    UNIQUE (signal_id)

Hybrid Self-Learning (Bước 1–2):
    - E5 outcomes feed into E6 Rolling Calibrator (Sprint 9)
    - Drift Alert: rolling precision < 0.30 × 10 ngày → UI banner
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)

# ── DB setup ──────────────────────────────────────────────────────────────────

_DB_DIR  = Path(__file__).resolve().parent.parent / "data" / "db"
SIGNAL_LOG_DB = _DB_DIR / "signal_log.db"

_DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS signal_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_date       TEXT    NOT NULL,
    ticker            TEXT    NOT NULL,
    action            TEXT    NOT NULL DEFAULT 'BUY',
    entry_price       REAL    NOT NULL,
    sl_price          REAL    NOT NULL,
    target_price      REAL    NOT NULL,
    rr_ratio          REAL    NOT NULL DEFAULT 0,
    p_calibrated      REAL    NOT NULL DEFAULT 0,
    p_raw             REAL    NOT NULL DEFAULT 0,
    recommendation    TEXT    NOT NULL DEFAULT '',
    position_size_pct REAL    NOT NULL DEFAULT 0,
    regime_state      TEXT    NOT NULL DEFAULT '',
    reason            TEXT    NOT NULL DEFAULT '',
    created_at        TEXT    NOT NULL,
    UNIQUE (signal_date, ticker)
);

CREATE TABLE IF NOT EXISTS signal_outcomes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id    INTEGER NOT NULL REFERENCES signal_log(id) ON DELETE CASCADE,
    outcome_date TEXT    NOT NULL,
    outcome      TEXT    NOT NULL DEFAULT 'PENDING',
    exit_price   REAL,
    pnl_pct      REAL,
    checked_at   TEXT    NOT NULL,
    UNIQUE (signal_id)
);

CREATE INDEX IF NOT EXISTS idx_sl_date   ON signal_log (signal_date);
CREATE INDEX IF NOT EXISTS idx_sl_ticker ON signal_log (ticker, signal_date);
CREATE INDEX IF NOT EXISTS idx_so_outcome ON signal_outcomes (outcome);
"""


def _ensure_db() -> Path:
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(SIGNAL_LOG_DB)) as con:
        con.executescript(_DDL)
        con.commit()
    return SIGNAL_LOG_DB


@contextmanager
def _get_connection():
    _ensure_db()
    con = sqlite3.connect(str(SIGNAL_LOG_DB), check_same_thread=False)
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
    """Cộng n ngày giao dịch (T2-T6) vào ngày d."""
    current = d
    added = 0
    while added < n:
        current += timedelta(days=1)
        if current.weekday() < 5:
            added += 1
    return current


def _t5_date(signal_date: date) -> date:
    """T+5 trading date cho signal_date."""
    return _add_business_days(signal_date, 5)


# ── Log signals ───────────────────────────────────────────────────────────────

def log_signal(alert_card) -> int:
    """
    Ghi 1 AlertCard vào signal_log. Trả về signal_id.
    Bỏ qua (upsert) nếu (signal_date, ticker) đã tồn tại.
    """
    now_str = datetime.now().isoformat(timespec="seconds")
    today   = date.today().isoformat()
    with _get_connection() as con:
        cur = con.execute(
            """
            INSERT INTO signal_log
                (signal_date, ticker, action, entry_price, sl_price, target_price,
                 rr_ratio, p_calibrated, p_raw, recommendation, position_size_pct,
                 regime_state, reason, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(signal_date, ticker) DO UPDATE SET
                entry_price       = excluded.entry_price,
                sl_price          = excluded.sl_price,
                target_price      = excluded.target_price,
                rr_ratio          = excluded.rr_ratio,
                p_calibrated      = excluded.p_calibrated,
                p_raw             = excluded.p_raw,
                recommendation    = excluded.recommendation,
                position_size_pct = excluded.position_size_pct,
                regime_state      = excluded.regime_state,
                reason            = excluded.reason
            """,
            (
                getattr(alert_card, "date", today),
                alert_card.ticker,
                getattr(alert_card, "action", "BUY"),
                getattr(alert_card, "entry_lo", 0.0),
                alert_card.sl_price,
                alert_card.target_price,
                getattr(alert_card, "rr_ratio", 0.0),
                getattr(alert_card, "p_calibrated", 0.0),
                getattr(alert_card, "p_raw", 0.0),
                getattr(alert_card, "recommendation", ""),
                getattr(alert_card, "position_size_pct", 0.0),
                getattr(alert_card, "regime_state", ""),
                getattr(alert_card, "reason", ""),
                now_str,
            ),
        )
        signal_id = cur.lastrowid
        # Lấy id nếu là conflict (lastrowid = 0 khi DO UPDATE không trả về)
        if not signal_id:
            row = con.execute(
                "SELECT id FROM signal_log WHERE signal_date=? AND ticker=?",
                (getattr(alert_card, "date", today), alert_card.ticker),
            ).fetchone()
            signal_id = row["id"] if row else 0
        # Tạo outcome PENDING nếu chưa có
        con.execute(
            """
            INSERT OR IGNORE INTO signal_outcomes
                (signal_id, outcome_date, outcome, checked_at)
            VALUES (?, ?, 'PENDING', ?)
            """,
            (signal_id, _t5_date(date.today()).isoformat(), now_str),
        )
    log.info("Logged signal: %s %s (id=%s)", alert_card.ticker,
             getattr(alert_card, "date", today), signal_id)
    return signal_id


def log_signals_batch(alert_cards) -> list[int]:
    """Ghi nhiều AlertCard. Trả về list signal_ids."""
    return [log_signal(c) for c in alert_cards]


# ── Check T+5 outcomes ────────────────────────────────────────────────────────

def check_pending_outcomes(as_of_date: Optional[date] = None) -> list[dict]:
    """
    Kiểm tra tất cả signal PENDING có T+5 date <= as_of_date.
    Lấy OHLCV để xác định WIN / LOSS / FLAT.
    Trả về danh sách kết quả đã update.

    Logic:
        Duyệt từng ngày T+1 đến T+5:
            if low  <= sl_price    → LOSS (day i)
            if high >= target_price → WIN  (day i)
        Nếu T+5 qua mà chưa resolve: dùng close T+5 vs entry → WIN/LOSS/FLAT
    """
    if as_of_date is None:
        as_of_date = date.today()

    results = []
    with _get_connection() as con:
        rows = con.execute(
            """
            SELECT sl.id, sl.signal_date, sl.ticker, sl.entry_price,
                   sl.sl_price, sl.target_price, so.outcome_date, so.id as out_id
            FROM signal_log sl
            JOIN signal_outcomes so ON so.signal_id = sl.id
            WHERE so.outcome = 'PENDING'
              AND so.outcome_date <= ?
            ORDER BY sl.signal_date
            """,
            (as_of_date.isoformat(),),
        ).fetchall()

    for row in rows:
        signal_id  = row["id"]
        sig_date   = date.fromisoformat(row["signal_date"])
        ticker     = row["ticker"]
        entry      = row["entry_price"]
        sl         = row["sl_price"]
        target     = row["target_price"]

        outcome, exit_price, pnl_pct = _resolve_outcome(
            ticker, sig_date, entry, sl, target
        )
        _update_outcome(row["out_id"], outcome, exit_price, pnl_pct)
        results.append(
            dict(ticker=ticker, signal_date=sig_date.isoformat(),
                 outcome=outcome, exit_price=exit_price, pnl_pct=pnl_pct)
        )
        log.info("Outcome %s → %s (%.1f%%)", ticker, outcome, pnl_pct or 0)

    return results


def _resolve_outcome(
    ticker: str,
    sig_date: date,
    entry: float,
    sl: float,
    target: float,
) -> tuple[str, Optional[float], Optional[float]]:
    """
    Trả về (outcome, exit_price, pnl_pct).
    Dùng OHLCV từ T+1 đến T+5 để xác định.
    """
    try:
        from data.market_data import get_ohlcv
        t5 = _t5_date(sig_date)
        # lấy dữ liệu từ T+1 đến T+5 (dùng 1m period để cover 5 phiên giao dịch)
        df = get_ohlcv(ticker, period="1m")
        if df is None or df.empty:
            return "EXPIRED", None, None
        # Lọc ngày từ T+1 đến T+5
        t1 = _add_business_days(sig_date, 1)
        df["date"] = df.index if df.index.name == "time" else df.index
        df = df[(df.index >= str(t1)) & (df.index <= str(t5))].copy()
        if df.empty:
            return "PENDING", None, None
        # Duyệt từng ngày
        for _, row in df.iterrows():
            low  = row.get("low",  row.get("close", entry))
            high = row.get("high", row.get("close", entry))
            if low <= sl:
                pnl = round((sl - entry) / entry * 100, 2)
                return "LOSS", sl, pnl
            if high >= target:
                pnl = round((target - entry) / entry * 100, 2)
                return "WIN", target, pnl
        # T+5 qua, resolve theo close
        last_close = float(df.iloc[-1].get("close", entry))
        if last_close > entry:
            pnl = round((last_close - entry) / entry * 100, 2)
            return "WIN", last_close, pnl
        elif last_close < entry:
            pnl = round((last_close - entry) / entry * 100, 2)
            return "LOSS", last_close, pnl
        else:
            return "FLAT", last_close, 0.0
    except Exception as e:
        log.warning("_resolve_outcome %s: %s", ticker, e)
        return "EXPIRED", None, None


def _update_outcome(
    out_id: int,
    outcome: str,
    exit_price: Optional[float],
    pnl_pct: Optional[float],
) -> None:
    with _get_connection() as con:
        con.execute(
            """
            UPDATE signal_outcomes
            SET outcome=?, exit_price=?, pnl_pct=?, checked_at=?
            WHERE id=?
            """,
            (outcome, exit_price, pnl_pct,
             datetime.now().isoformat(timespec="seconds"), out_id),
        )


# ── Query helpers ─────────────────────────────────────────────────────────────

def get_recent_signals(n: int = 20) -> list[dict]:
    """Trả về n signal mới nhất kèm outcome."""
    with _get_connection() as con:
        rows = con.execute(
            """
            SELECT sl.*, so.outcome, so.exit_price, so.pnl_pct, so.outcome_date
            FROM signal_log sl
            LEFT JOIN signal_outcomes so ON so.signal_id = sl.id
            ORDER BY sl.signal_date DESC, sl.created_at DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_signal_stats() -> dict:
    """
    Tính precision, win rate, avg_pnl, rolling precision (30 ngày).
    Dùng cho Drift Alert.
    """
    with _get_connection() as con:
        total = con.execute(
            "SELECT COUNT(*) FROM signal_log"
        ).fetchone()[0]
        resolved = con.execute(
            "SELECT COUNT(*) FROM signal_outcomes WHERE outcome NOT IN ('PENDING','EXPIRED')"
        ).fetchone()[0]
        wins = con.execute(
            "SELECT COUNT(*) FROM signal_outcomes WHERE outcome='WIN'"
        ).fetchone()[0]
        avg_pnl_row = con.execute(
            "SELECT AVG(pnl_pct) FROM signal_outcomes "
            "WHERE outcome NOT IN ('PENDING','EXPIRED')"
        ).fetchone()[0]
        # Rolling 30 ngày
        cutoff = (date.today() - timedelta(days=30)).isoformat()
        roll_total = con.execute(
            "SELECT COUNT(*) FROM signal_log WHERE signal_date >= ?", (cutoff,)
        ).fetchone()[0]
        roll_wins = con.execute(
            """
            SELECT COUNT(*) FROM signal_outcomes so
            JOIN signal_log sl ON sl.id = so.signal_id
            WHERE so.outcome = 'WIN' AND sl.signal_date >= ?
            """,
            (cutoff,),
        ).fetchone()[0]

    precision  = wins / resolved if resolved > 0 else 0.0
    roll_prec  = roll_wins / roll_total if roll_total > 0 else 0.0
    drift_alert = roll_prec < 0.30 and roll_total >= 10

    return dict(
        total_signals   = total,
        resolved        = resolved,
        wins            = wins,
        precision       = round(precision, 4),
        avg_pnl_pct     = round(avg_pnl_row or 0.0, 4),
        roll_30d_total  = roll_total,
        roll_30d_prec   = round(roll_prec, 4),
        drift_alert     = drift_alert,
    )


def get_outcome_table() -> list[dict]:
    """Full outcome table (cho UI)."""
    with _get_connection() as con:
        rows = con.execute(
            """
            SELECT
                sl.signal_date, sl.ticker, sl.recommendation,
                sl.entry_price, sl.sl_price, sl.target_price,
                sl.p_calibrated, sl.regime_state, sl.reason,
                so.outcome, so.exit_price, so.pnl_pct, so.outcome_date
            FROM signal_log sl
            LEFT JOIN signal_outcomes so ON so.signal_id = sl.id
            ORDER BY sl.signal_date DESC, sl.ticker
            """
        ).fetchall()
    return [dict(r) for r in rows]
