"""
data/trade_logger.py
E2 Trade Logger — Sprint 8.

Nhập lệnh thủ công (paper + real) vào SQLite trades.db.

DB: data/db/trades.db

Tables
------
trades
    id           INTEGER PRIMARY KEY AUTOINCREMENT
    trade_type   TEXT    "paper" | "real"
    ticker       TEXT
    entry_date   TEXT    YYYY-MM-DD
    entry_price  REAL
    shares       INTEGER
    sl_price     REAL
    target_price REAL
    status       TEXT    "open" | "closed" | "stopped"
    exit_date    TEXT    (nullable)
    exit_price   REAL    (nullable)
    pnl_vnd      REAL    (nullable — realized PnL in VND)
    pnl_pct      REAL    (nullable — realized PnL %)
    signal_id    INTEGER (FK signal_log.id — optional)
    note         TEXT
    created_at   TEXT
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)

# ── DB setup ──────────────────────────────────────────────────────────────────

_DB_DIR    = Path(__file__).resolve().parent / "db"
TRADES_DB  = _DB_DIR / "trades.db"

_DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_type   TEXT    NOT NULL DEFAULT 'paper',
    ticker       TEXT    NOT NULL,
    entry_date   TEXT    NOT NULL,
    entry_price  REAL    NOT NULL,
    shares       INTEGER NOT NULL DEFAULT 0,
    sl_price     REAL    NOT NULL DEFAULT 0,
    target_price REAL    NOT NULL DEFAULT 0,
    status       TEXT    NOT NULL DEFAULT 'open',
    exit_date    TEXT,
    exit_price   REAL,
    pnl_vnd      REAL,
    pnl_pct      REAL,
    signal_id    INTEGER,
    note         TEXT    NOT NULL DEFAULT '',
    created_at   TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_ticker  ON trades (ticker, entry_date);
CREATE INDEX IF NOT EXISTS idx_trades_status  ON trades (status);
"""


def _ensure_db() -> Path:
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(TRADES_DB)) as con:
        con.executescript(_DDL)
        con.commit()
    return TRADES_DB


@contextmanager
def _get_connection():
    _ensure_db()
    con = sqlite3.connect(str(TRADES_DB), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ── Write helpers ─────────────────────────────────────────────────────────────

def add_trade(
    ticker:       str,
    entry_date:   str,          # YYYY-MM-DD
    entry_price:  float,
    shares:       int,
    sl_price:     float,
    target_price: float,
    trade_type:   str  = "paper",
    signal_id:    Optional[int]  = None,
    note:         str  = "",
) -> int:
    """
    Thêm lệnh mới. Trả về trade_id.
    trade_type: "paper" hoặc "real".
    """
    if entry_price <= 0:
        raise ValueError("entry_price phải > 0")
    if shares <= 0:
        raise ValueError("shares phải > 0")
    now = datetime.now().isoformat(timespec="seconds")
    with _get_connection() as con:
        cur = con.execute(
            """
            INSERT INTO trades
                (trade_type, ticker, entry_date, entry_price, shares,
                 sl_price, target_price, status, signal_id, note, created_at)
            VALUES (?,?,?,?,?,?,?,'open',?,?,?)
            """,
            (trade_type.lower(), ticker.upper(), entry_date,
             float(entry_price), int(shares),
             float(sl_price), float(target_price),
             signal_id, note, now),
        )
        trade_id = cur.lastrowid
    log.info("Trade added: %s %s ×%d @ %,.0f (id=%d)",
             ticker, entry_date, shares, entry_price, trade_id)
    return trade_id


def close_trade(
    trade_id:   int,
    exit_date:  str,   # YYYY-MM-DD
    exit_price: float,
    status:     str = "closed",   # "closed" | "stopped"
) -> dict:
    """
    Đóng lệnh. Tính realized PnL. Trả về dict trade đã cập nhật.
    """
    with _get_connection() as con:
        row = con.execute(
            "SELECT entry_price, shares FROM trades WHERE id=?", (trade_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Trade {trade_id} không tồn tại")
        entry_price = row["entry_price"]
        shares      = row["shares"]
        pnl_pct = round((exit_price - entry_price) / entry_price * 100, 4)
        pnl_vnd = round((exit_price - entry_price) * shares, 0)
        con.execute(
            """
            UPDATE trades
            SET status=?, exit_date=?, exit_price=?, pnl_vnd=?, pnl_pct=?
            WHERE id=?
            """,
            (status, exit_date, float(exit_price), pnl_vnd, pnl_pct, trade_id),
        )
    log.info("Trade closed: id=%d exit=%.0f pnl=%.2f%%", trade_id, exit_price, pnl_pct)
    return get_trade(trade_id)


# ── Read helpers ──────────────────────────────────────────────────────────────

def get_trade(trade_id: int) -> dict:
    with _get_connection() as con:
        row = con.execute(
            "SELECT * FROM trades WHERE id=?", (trade_id,)
        ).fetchone()
    if row is None:
        raise ValueError(f"Trade {trade_id} không tồn tại")
    return dict(row)


def get_open_trades(trade_type: Optional[str] = None) -> list[dict]:
    """Danh sách lệnh đang mở (status='open')."""
    with _get_connection() as con:
        if trade_type:
            rows = con.execute(
                "SELECT * FROM trades WHERE status='open' AND trade_type=? "
                "ORDER BY entry_date DESC",
                (trade_type.lower(),),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM trades WHERE status='open' ORDER BY entry_date DESC"
            ).fetchall()
    return [dict(r) for r in rows]


def get_all_trades(trade_type: Optional[str] = None) -> list[dict]:
    """Toàn bộ lịch sử lệnh."""
    with _get_connection() as con:
        if trade_type:
            rows = con.execute(
                "SELECT * FROM trades WHERE trade_type=? ORDER BY entry_date DESC",
                (trade_type.lower(),),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM trades ORDER BY entry_date DESC"
            ).fetchall()
    return [dict(r) for r in rows]


def update_unrealized_pnl(price_map: dict[str, float]) -> list[dict]:
    """
    Cập nhật unrealized PnL cho các lệnh đang mở dựa trên giá hiện tại.
    price_map: {ticker: current_price}
    Trả về list open trades đã cập nhật (pnl_pct, pnl_vnd không lưu DB — chỉ trả về).
    """
    open_trades = get_open_trades()
    result = []
    for t in open_trades:
        cur_price = price_map.get(t["ticker"].upper())
        if cur_price and cur_price > 0:
            unreal_pct = round((cur_price - t["entry_price"]) / t["entry_price"] * 100, 4)
            unreal_vnd = round((cur_price - t["entry_price"]) * t["shares"], 0)
            t["current_price"]   = cur_price
            t["unrealized_pct"]  = unreal_pct
            t["unrealized_vnd"]  = unreal_vnd
        result.append(t)
    return result


def get_trade_stats(trade_type: Optional[str] = None) -> dict:
    """
    Thống kê tổng hợp:
      total, open, closed, win_count, loss_count, win_rate,
      avg_win_pct, avg_loss_pct, profit_factor, total_pnl_vnd
    """
    all_t   = get_all_trades(trade_type)
    closed  = [t for t in all_t if t["status"] in ("closed", "stopped") and t.get("pnl_pct") is not None]
    open_t  = [t for t in all_t if t["status"] == "open"]
    wins    = [t for t in closed if t["pnl_pct"] > 0]
    losses  = [t for t in closed if t["pnl_pct"] <= 0]

    avg_win  = (sum(t["pnl_pct"] for t in wins)   / len(wins))   if wins   else 0.0
    avg_loss = (sum(t["pnl_pct"] for t in losses) / len(losses)) if losses else 0.0
    gross_win  = sum(t["pnl_vnd"] for t in wins   if t.get("pnl_vnd")) or 0.0
    gross_loss = abs(sum(t["pnl_vnd"] for t in losses if t.get("pnl_vnd")) or 0.0)
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    return dict(
        total          = len(all_t),
        open_count     = len(open_t),
        closed_count   = len(closed),
        win_count      = len(wins),
        loss_count     = len(losses),
        win_rate       = round(len(wins) / len(closed), 4) if closed else 0.0,
        avg_win_pct    = round(avg_win, 4),
        avg_loss_pct   = round(avg_loss, 4),
        profit_factor  = round(profit_factor, 4) if profit_factor != float("inf") else 999.0,
        total_pnl_vnd  = round(sum(t.get("pnl_vnd") or 0 for t in closed), 0),
    )
