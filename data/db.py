"""
data/db.py
D0.2 SQLite Database Manager — Daily Snapshot Logger schema & connection.

DB file: data/db/snapshots.db  (auto-created)

Tables
------
snapshots
    session_date       TEXT  YYYY-MM-DD
    ticker             TEXT
    foreign_buy        REAL  -- volume
    foreign_sell       REAL  -- volume
    foreign_net        REAL  -- buy - sell (positive = net buying)
    total_volume       REAL  -- OHLCV total volume (denominator for %)
    close              REAL
    proprietary_buy    REAL  -- Sprint 12: tự doanh mua (default 0)
    proprietary_sell   REAL  -- Sprint 12: tự doanh bán (default 0)
    proprietary_net    REAL  -- Sprint 12: tự doanh net = buy - sell
    PRIMARY KEY (session_date, ticker)

market_breadth
    session_date  TEXT  PRIMARY KEY
    advance       INTEGER  -- so ma tang
    decline       INTEGER  -- so ma giam
    unchanged     INTEGER  -- so ma di ngang
    total         INTEGER  -- tong so ma giao dich
    created_at    TEXT

logger_meta
    key           TEXT  PRIMARY KEY
    value         TEXT
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from utils.logger import get_logger

log = get_logger(__name__)

# DB path: <repo-root>/data/db/snapshots.db
_DB_DIR  = Path(__file__).resolve().parent / "db"
DB_PATH  = _DB_DIR / "snapshots.db"

_DDL = """
CREATE TABLE IF NOT EXISTS snapshots (
    session_date       TEXT NOT NULL,
    ticker             TEXT NOT NULL,
    foreign_buy        REAL NOT NULL DEFAULT 0,
    foreign_sell       REAL NOT NULL DEFAULT 0,
    foreign_net        REAL NOT NULL DEFAULT 0,
    total_volume       REAL NOT NULL DEFAULT 0,
    close              REAL NOT NULL DEFAULT 0,
    proprietary_buy    REAL NOT NULL DEFAULT 0,
    proprietary_sell   REAL NOT NULL DEFAULT 0,
    proprietary_net    REAL NOT NULL DEFAULT 0,
    created_at         TEXT NOT NULL,
    PRIMARY KEY (session_date, ticker)
);

CREATE TABLE IF NOT EXISTS market_breadth (
    session_date  TEXT NOT NULL PRIMARY KEY,
    advance       INTEGER NOT NULL DEFAULT 0,
    decline       INTEGER NOT NULL DEFAULT 0,
    unchanged     INTEGER NOT NULL DEFAULT 0,
    total         INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS logger_meta (
    key   TEXT NOT NULL PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_snapshots_ticker
    ON snapshots (ticker, session_date);
"""


def ensure_db() -> Path:
    """Khởi tạo DB và schema nếu chưa tồn tại. Trả về đường dẫn DB."""
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(DB_PATH)) as con:
        con.executescript(_DDL)
        # Sprint 12: migrate existing DB — add proprietary columns nếu chưa có
        _migrate_add_proprietary_columns(con)
        con.commit()
    log.debug("DB ready: %s", DB_PATH)
    return DB_PATH


def _migrate_add_proprietary_columns(con: sqlite3.Connection) -> None:
    """Idempotent migration: thêm proprietary columns nếu chưa có (SQLite không có IF NOT EXISTS cho ALTER TABLE)."""
    existing = {row[1] for row in con.execute("PRAGMA table_info(snapshots)")}
    for col, typedef in [
        ("proprietary_buy",  "REAL NOT NULL DEFAULT 0"),
        ("proprietary_sell", "REAL NOT NULL DEFAULT 0"),
        ("proprietary_net",  "REAL NOT NULL DEFAULT 0"),
    ]:
        if col not in existing:
            try:
                con.execute(f"ALTER TABLE snapshots ADD COLUMN {col} {typedef}")
                log.info("DB migration: added column snapshots.%s", col)
            except Exception as exc:
                log.warning("DB migration %s: %s", col, exc)


@contextmanager
def get_connection():
    """Context manager trả về sqlite3.Connection với WAL mode."""
    ensure_db()
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
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


# ── Public query helpers ───────────────────────────────────────────────────────

def get_snapshot_sessions(ticker: str | None = None) -> list[str]:
    """
    Danh sách session_date da log.
    Neu ticker=None: toan bo cac ngay co it nhat 1 ticker.
    """
    with get_connection() as con:
        if ticker:
            rows = con.execute(
                "SELECT DISTINCT session_date FROM snapshots "
                "WHERE ticker=? ORDER BY session_date",
                (ticker.upper(),),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT DISTINCT session_date FROM snapshots ORDER BY session_date"
            ).fetchall()
    return [r[0] for r in rows]


def get_session_count(ticker: str | None = None) -> int:
    """So phien da log cho ticker (hoac toan the truong)."""
    return len(get_snapshot_sessions(ticker))


def get_last_session_date() -> str | None:
    """Ngay log cuoi cung (hoac None neu chua co gi)."""
    sessions = get_snapshot_sessions()
    return sessions[-1] if sessions else None


def load_snapshots(ticker: str, last_n: int = 20) -> "pd.DataFrame":  # type: ignore[name-defined]
    """
    Load N phien gan nhat cho 1 ticker.
    Columns: session_date, ticker, foreign_buy, foreign_sell, foreign_net,
             total_volume, close, proprietary_buy, proprietary_sell, proprietary_net.
    """
    import pandas as pd
    with get_connection() as con:
        rows = con.execute(
            """
            SELECT session_date, ticker, foreign_buy, foreign_sell,
                   foreign_net, total_volume, close,
                   COALESCE(proprietary_buy,  0) AS proprietary_buy,
                   COALESCE(proprietary_sell, 0) AS proprietary_sell,
                   COALESCE(proprietary_net,  0) AS proprietary_net
            FROM   snapshots
            WHERE  ticker = ?
            ORDER  BY session_date DESC
            LIMIT  ?
            """,
            (ticker.upper(), last_n),
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=[
            "session_date", "ticker", "foreign_buy", "foreign_sell",
            "foreign_net", "total_volume", "close",
            "proprietary_buy", "proprietary_sell", "proprietary_net",
        ])
    df = pd.DataFrame([dict(r) for r in rows])
    df["session_date"] = pd.to_datetime(df["session_date"])
    return df.sort_values("session_date").reset_index(drop=True)


def upsert_snapshot(
    session_date: str,
    ticker: str,
    foreign_buy: float,
    foreign_sell: float,
    foreign_net: float,
    total_volume: float,
    close: float,
    proprietary_buy: float = 0.0,
    proprietary_sell: float = 0.0,
    proprietary_net: float = 0.0,
) -> bool:
    """
    Ghi 1 row vao snapshots. Idempotent (INSERT OR REPLACE).
    Sprint 12: thêm proprietary_buy/sell/net (default 0 cho backward compat).
    Returns True neu ghi moi, False neu ban ghi da ton tai va ghi de.
    """
    now = datetime.now().isoformat()
    with get_connection() as con:
        con.execute(
            """
            INSERT OR REPLACE INTO snapshots
                (session_date, ticker, foreign_buy, foreign_sell,
                 foreign_net, total_volume, close,
                 proprietary_buy, proprietary_sell, proprietary_net,
                 created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_date, ticker.upper(), foreign_buy, foreign_sell,
             foreign_net, total_volume, close,
             proprietary_buy, proprietary_sell, proprietary_net, now),
        )
    return True


def upsert_breadth(
    session_date: str,
    advance: int,
    decline: int,
    unchanged: int,
    total: int,
) -> None:
    """Ghi market breadth cho ngay."""
    now = datetime.now().isoformat()
    with get_connection() as con:
        con.execute(
            """
            INSERT OR REPLACE INTO market_breadth
                (session_date, advance, decline, unchanged, total, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_date, advance, decline, unchanged, total, now),
        )


def set_meta(key: str, value: str) -> None:
    with get_connection() as con:
        con.execute(
            "INSERT OR REPLACE INTO logger_meta (key, value) VALUES (?, ?)",
            (key, value),
        )


def get_meta(key: str, default: str = "") -> str:
    with get_connection() as con:
        row = con.execute(
            "SELECT value FROM logger_meta WHERE key=?", (key,)
        ).fetchone()
    return row[0] if row else default
