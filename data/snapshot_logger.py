"""
data/snapshot_logger.py
D0.2 Daily Snapshot Logger — ghi price board snapshot vao SQLite luc 15:05.

Thu thap moi phien:
  - Foreign buy/sell volume cho moi ticker VN30 (tu data.providers)
  - Market breadth toan san (advance/decline/unchanged) — uoc tinh tu vnstock
  - Close + total volume tu OHLCV cuoi phien

Su dung tu:
  - scripts/daily_snapshot.py (scheduled job)
  - Streamlit UI (nut "Log hom nay" trong sidebar)

Idempotent: goi nhieu lan trong 1 ngay -> chi ghi 1 lan (skip neu da co).
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Callable

import pandas as pd

from config.constants import VN30_TICKERS
from data.db import (
    get_session_count,
    get_snapshot_sessions,
    upsert_breadth,
    upsert_snapshot,
    set_meta,
)

log = logging.getLogger(__name__)

MIN_SESSIONS_FOR_S4 = 5   # so phien toi thieu de bat S4 features


# ── Internal helpers ──────────────────────────────────────────────────────────

def _today_str() -> str:
    return date.today().isoformat()


def _is_trading_day() -> bool:
    """True neu hom nay la ngay thuong (thu 2-6). Khong check lich nghi le."""
    return date.today().weekday() < 5


def _fetch_foreign_batch_for_session(session_date: str) -> dict[str, dict]:
    """
    Pre-fetch dữ liệu khối ngoại cho toàn bộ sàn từ FiinMarket trong 1-2 HTTP call.

    Returns
    -------
    dict: {ticker_upper -> {date, foreign_buy, foreign_sell, foreign_net}}
    Trả về dict rỗng nếu API không khả dụng (fallback về per-ticker sẽ xử lý).
    """
    try:
        from analytics.ssi_foreign import get_foreign_batch
        batch_vni  = get_foreign_batch("VNINDEX")   # toàn HOSE
        batch_vn30 = get_foreign_batch("VN30")       # bổ sung VN30 nếu thiếu
        # Gộp: VN30 trước để VNINDEX ghi đè (VNINDEX đầy đủ hơn)
        merged = {**batch_vn30, **batch_vni}
        log.info(
            "[D0.2] FiinMarket GetForeign batch: %d tickers, phiên %s",
            len(merged), session_date,
        )
        return merged
    except Exception as exc:
        log.warning(
            "[D0.2] FiinMarket GetForeign batch lỗi: %s — sẽ fallback về provider cũ",
            exc,
        )
        return {}


def _fetch_foreign_for_ticker(
    ticker: str,
    session_date: str,
    batch: dict | None = None,
) -> dict | None:
    """
    Lấy foreign flow hôm nay cho 1 ticker.

    Thứ tự ưu tiên:
      1. FiinMarket batch pre-fetched (batch param) — không tốn thêm HTTP call
      2. Provider cũ (KBS / VNDirect) — fallback
    Trả về None nếu không lấy được.
    """
    # ── Layer 1: FiinMarket batch (đã pre-fetch từ trước vòng lặp) ──────────
    if batch is not None:
        row = batch.get(ticker.upper())
        if row:
            log.debug("FiinMarket foreign OK: %s  net=%s", ticker, row.get("foreign_net"))
            return {
                "foreign_buy":  float(row.get("foreign_buy",  0) or 0),
                "foreign_sell": float(row.get("foreign_sell", 0) or 0),
                "foreign_net":  float(row.get("foreign_net",  0) or 0),
            }

    # ── Layer 2: Provider cũ (VNDirect / KBS) — fallback ────────────────────
    try:
        from data.foreign_flow import get_foreign_flow
        ff = get_foreign_flow(ticker, period="1w")
        if ff.empty:
            return None
        ff["date"] = pd.to_datetime(ff["date"]).dt.strftime("%Y-%m-%d")
        today_row = ff[ff["date"] == session_date]
        if today_row.empty:
            today_row = ff.iloc[[-1]]
        row = today_row.iloc[0]
        return {
            "foreign_buy":  float(row.get("buy_vol",  0) or 0),
            "foreign_sell": float(row.get("sell_vol", 0) or 0),
            "foreign_net":  float(row.get("net_vol",  0) or 0),
        }
    except Exception as exc:
        log.debug("_fetch_foreign_for_ticker(%s) provider fallback: %s", ticker, exc)
        return None


def _fetch_proprietary_for_ticker(ticker: str, session_date: str) -> dict | None:
    """
    Lay proprietary (tu doanh) flow cho 1 ticker.

    Fallback cascade:
      1. SSI iBoard API -> data ngay T
      2. SSI iBoard API -> data ngay T-1 (data T chua cap nhat sau phien)
      3. DB snapshot -> row gan nhat co proprietary != 0 (SSI khong kha dung)
      4. None -> 0.0 (hoan toan khong co data)
    """
    # --- Layer 1 & 2: SSI API ---
    try:
        from analytics.ssi_iboard import fetch_investor_flow
        df = fetch_investor_flow(ticker, limit=10)   # lay nhieu lich su hon
        if not df.empty:
            df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            row = df[df["date_str"] == session_date]
            if row.empty:
                # T chua co data -> dung row gan nhat (T-1)
                row = df.iloc[[-1]]
                log.debug(
                    "SSI %s: khong co data ngay T (%s), dung T-1 (%s)",
                    ticker, session_date, df["date_str"].iloc[-1],
                )
            r = row.iloc[0]
            return {
                "proprietary_buy":  float(r.get("proprietary_buy",  0) or 0),
                "proprietary_sell": float(r.get("proprietary_sell", 0) or 0),
                "proprietary_net":  float(r.get("proprietary_net",  0) or 0),
            }
    except Exception as exc:
        log.debug("_fetch_proprietary_for_ticker(%s) SSI error: %s", ticker, exc)

    # --- Layer 3: DB fallback (SSI tra data: null hoac loi mang) ---
    try:
        from data.db import load_snapshots
        db_df = load_snapshots(ticker, last_n=10)
        if not db_df.empty:
            # Bo qua row cua session_date hien tai (neu da co tu truoc)
            past = db_df[
                db_df["session_date"].dt.strftime("%Y-%m-%d") != session_date
            ]
            # Lay row gan nhat co proprietary != 0
            non_zero = past[
                (past["proprietary_buy"] != 0) | (past["proprietary_sell"] != 0)
            ]
            if not non_zero.empty:
                r = non_zero.iloc[-1]
                t1_date = r["session_date"].strftime("%Y-%m-%d")
                log.info(
                    "SSI %s: dung data T-1 tu DB (%s) vi SSI chua cap nhat",
                    ticker, t1_date,
                )
                return {
                    "proprietary_buy":  float(r["proprietary_buy"]),
                    "proprietary_sell": float(r["proprietary_sell"]),
                    "proprietary_net":  float(r["proprietary_net"]),
                }
    except Exception as exc:
        log.debug("_fetch_proprietary_for_ticker(%s) DB fallback error: %s", ticker, exc)

    return None


def _fetch_ohlcv_for_ticker(ticker: str, session_date: str) -> dict | None:
    """Lay close + total_volume tu OHLCV cuoi phien."""
    try:
        from data.market_data import get_ohlcv
        ohlcv = get_ohlcv(ticker, period="1w")
        if ohlcv.empty:
            return None
        ohlcv["date_str"] = pd.to_datetime(ohlcv["date"]).dt.strftime("%Y-%m-%d")
        row = ohlcv[ohlcv["date_str"] == session_date]
        if row.empty:
            row = ohlcv.iloc[[-1]]
        r = row.iloc[0]
        return {
            "close":        float(r.get("close",  0) or 0),
            "total_volume": float(r.get("volume", 0) or 0),
        }
    except Exception as exc:
        log.debug("_fetch_ohlcv_for_ticker(%s): %s", ticker, exc)
        return None


def _estimate_breadth(tickers: list[str], session_date: str) -> dict:
    """
    Uoc tinh market breadth tu danh sach tickers da log.
    (VN30 breadth, khong phai toan san)
    """
    advance = decline = unchanged = 0
    for ticker in tickers:
        try:
            from data.market_data import get_ohlcv
            ohlcv = get_ohlcv(ticker, period="1w")
            if ohlcv.empty:
                continue
            ohlcv["date_str"] = pd.to_datetime(ohlcv["date"]).dt.strftime("%Y-%m-%d")
            row = ohlcv[ohlcv["date_str"] == session_date]
            if row.empty:
                row = ohlcv.iloc[[-1]]
            r = row.iloc[0]
            chg = float(r.get("close", 0)) - float(r.get("open", 0))
            if chg > 0:
                advance += 1
            elif chg < 0:
                decline += 1
            else:
                unchanged += 1
        except Exception:
            pass
    return {
        "advance":   advance,
        "decline":   decline,
        "unchanged": unchanged,
        "total":     advance + decline + unchanged,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def log_session(
    session_date: str | None = None,
    tickers: list[str] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    force: bool = False,
) -> dict:
    """
    Ghi snapshot phien giao dich vao SQLite.

    Parameters
    ----------
    session_date      : YYYY-MM-DD. Mac dinh = hom nay.
    tickers           : Danh sach ma. Mac dinh = VN30_TICKERS.
    progress_callback : fn(pct, msg) cho UI.
    force             : True = ghi lai du da co (ghi de).

    Returns
    -------
    dict voi: session_date, tickers_logged, tickers_skipped, already_exists.
    """
    if session_date is None:
        session_date = _today_str()
    if tickers is None:
        tickers = list(VN30_TICKERS)

    log.info("[D0.2] Log session %s, %d tickers", session_date, len(tickers))

    # Kiem tra da log hom nay chua
    existing_sessions = get_snapshot_sessions()
    already_exists = session_date in existing_sessions
    if already_exists and not force:
        log.info("[D0.2] Session %s da ton tai, bo qua (dung force=True de ghi de)", session_date)
        if progress_callback:
            progress_callback(1.0, f"Session {session_date} da duoc log truoc do.")
        return {
            "session_date":    session_date,
            "tickers_logged":  0,
            "tickers_skipped": len(tickers),
            "already_exists":  True,
        }

    logged  = 0
    skipped = 0
    total   = len(tickers)

    # ── Pre-fetch toàn bộ foreign data trong 1-2 HTTP calls ──────────────────
    if progress_callback:
        progress_callback(0.03, "Pre-fetch dữ liệu khối ngoại (FiinMarket batch)...")
    foreign_batch = _fetch_foreign_batch_for_session(session_date)

    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(
                0.05 + 0.85 * (i / total),
                f"Thu thap {ticker} ({i+1}/{total})...",
            )

        ff    = _fetch_foreign_for_ticker(ticker, session_date, batch=foreign_batch)
        prop  = _fetch_proprietary_for_ticker(ticker, session_date)
        ohlcv = _fetch_ohlcv_for_ticker(ticker, session_date)

        if ohlcv is None:
            log.debug("[D0.2] Bo qua %s: khong co OHLCV", ticker)
            skipped += 1
            continue

        upsert_snapshot(
            session_date     = session_date,
            ticker           = ticker,
            foreign_buy      = ff["foreign_buy"]           if ff   else 0.0,
            foreign_sell     = ff["foreign_sell"]          if ff   else 0.0,
            foreign_net      = ff["foreign_net"]           if ff   else 0.0,
            total_volume     = ohlcv["total_volume"],
            close            = ohlcv["close"],
            proprietary_buy  = prop["proprietary_buy"]     if prop else 0.0,
            proprietary_sell = prop["proprietary_sell"]    if prop else 0.0,
            proprietary_net  = prop["proprietary_net"]     if prop else 0.0,
        )
        logged += 1

    # Market breadth
    if progress_callback:
        progress_callback(0.92, "Tinh market breadth VN30...")
    breadth = _estimate_breadth(tickers, session_date)
    upsert_breadth(
        session_date = session_date,
        advance      = breadth["advance"],
        decline      = breadth["decline"],
        unchanged    = breadth["unchanged"],
        total        = breadth["total"],
    )

    # Cap nhat meta
    set_meta("last_session_date", session_date)
    set_meta("last_run_at", datetime.now().isoformat())

    if progress_callback:
        progress_callback(1.0, f"Hoan tat: {logged} tickers logged ({session_date}).")

    log.info(
        "[D0.2] Session %s: %d logged, %d skipped. Breadth A%d/D%d/U%d",
        session_date, logged, skipped,
        breadth["advance"], breadth["decline"], breadth["unchanged"],
    )
    return {
        "session_date":    session_date,
        "tickers_logged":  logged,
        "tickers_skipped": skipped,
        "already_exists":  False,
    }


def get_logger_status() -> dict:
    """
    Tra ve trang thai logger: so phien, ngay cuoi, S4 da san sang chua.

    Returns
    -------
    dict: sessions, last_date, s4_ready, sessions_needed.
    """
    from data.db import get_session_count, get_last_session_date, get_meta
    sessions   = get_session_count()
    last_date  = get_last_session_date()
    s4_ready   = sessions >= MIN_SESSIONS_FOR_S4
    return {
        "sessions":        sessions,
        "last_date":       last_date,
        "s4_ready":        s4_ready,
        "sessions_needed": max(0, MIN_SESSIONS_FOR_S4 - sessions),
        "last_run_at":     get_meta("last_run_at"),
    }
