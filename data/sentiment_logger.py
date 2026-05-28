"""
data/sentiment_logger.py
Sprint 13: Log Fireant buzz + CafeF/Vietstock sentiment hàng ngày vào SQLite.

Thiết kế:
  - Chỉ lưu raw counts + scores, KHÔNG lưu nội dung bài viết (tránh storage bloat)
  - Fireant: đếm bài (raw count) + tổng likes — không parse NLP (buzz là signal, không phải content)
  - CafeF + Vietstock: sentiment score từ pipeline keyword/underthesea hiện có
  - Graceful degradation: nếu API lỗi → lưu 0, không raise exception
  - Idempotent: INSERT OR REPLACE — an toàn khi chạy lại cùng ngày

Được gọi từ:
  - scripts/daily_snapshot.py (sau 15:10 VNT hàng ngày)
  - scripts/backfill_sentiment.py (backfill lịch sử)
  - ml/dataset_builder.py (kiểm tra và backfill khi build dataset)
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from data.db import DB_PATH, ensure_db, get_connection
from utils.logger import get_logger

log = get_logger(__name__)


# ── Internal upsert ────────────────────────────────────────────────────────────

def _upsert_sentiment(row: dict) -> None:
    """INSERT OR REPLACE vào sentiment_snapshots."""
    ensure_db()
    sql = """
        INSERT OR REPLACE INTO sentiment_snapshots (
            session_date, ticker,
            fireant_buzz_count, fireant_buzz_likes,
            cafef_sent_score, vietstock_sent_score, combined_sent_score,
            article_count, bullish_count, bearish_count,
            created_at
        ) VALUES (
            :session_date, :ticker,
            :fireant_buzz_count, :fireant_buzz_likes,
            :cafef_sent_score, :vietstock_sent_score, :combined_sent_score,
            :article_count, :bullish_count, :bearish_count,
            datetime('now')
        )
    """
    with get_connection() as con:
        con.execute(sql, row)


# ── Fetch helpers ──────────────────────────────────────────────────────────────

def _fetch_fireant_buzz(ticker: str) -> dict:
    """
    Lấy số bài + tổng likes từ Fireant cho ticker hôm nay.
    Fireant API không hỗ trợ date filter → đây là snapshot thời điểm hiện tại.
    Trả về dict: {buzz_count, buzz_likes}
    Khi không có FIREANT_TOKEN: trả về {0, 0} và log ở mức DEBUG — 401 là expected.
    """
    try:
        from news.fireant_scraper import get_fireant_posts
        posts = get_fireant_posts(ticker, limit=50)
        return {
            "buzz_count": len(posts),
            "buzz_likes": sum(p.get("likes", 0) for p in posts),
        }
    except Exception as exc:
        log.debug("Fireant buzz fetch [%s]: %s", ticker, exc)
        return {"buzz_count": 0, "buzz_likes": 0}


def _fetch_cafef_sentiment(ticker: str, days: int = 1) -> dict:
    """
    Tính sentiment score từ CafeF cho ticker trong `days` ngày gần nhất.
    Trả về dict: {score, bullish_count, bearish_count, article_count}
    """
    try:
        from news.cafef_scraper import search_cafef_news
        from news.sentiment import aggregate_sentiment
        news = search_cafef_news(ticker, limit=30, days=days)
        agg = aggregate_sentiment(news)
        return {
            "score":         agg["avg_score"],
            "bullish_count": agg["bullish_count"],
            "bearish_count": agg["bearish_count"],
            "article_count": agg["article_count"],
        }
    except Exception as exc:
        log.warning("CafeF sentiment fetch lỗi [%s]: %s", ticker, exc)
        return {"score": 0.0, "bullish_count": 0, "bearish_count": 0, "article_count": 0}


def _fetch_vietstock_sentiment(ticker: str) -> dict:
    """
    Tính sentiment score từ Vietstock cho ticker.
    Trả về dict: {score, article_count}
    """
    try:
        from news.vietstock_scraper import search_vietstock_news
        from news.sentiment import aggregate_sentiment
        news = search_vietstock_news(ticker, limit=15)
        agg = aggregate_sentiment(news)
        return {"score": agg["avg_score"], "article_count": agg["article_count"]}
    except Exception as exc:
        log.warning("Vietstock sentiment fetch lỗi [%s]: %s", ticker, exc)
        return {"score": 0.0, "article_count": 0}


def _combined_score(cafef_score: float, cafef_n: int,
                    vietstock_score: float, vietstock_n: int) -> float:
    """
    Weighted average: trọng số tỷ lệ số lượng bài.
    Nếu không có bài nào → trả về 0.0.
    """
    total = cafef_n + vietstock_n
    if total == 0:
        return 0.0
    return round((cafef_score * cafef_n + vietstock_score * vietstock_n) / total, 4)


# ── Public API ─────────────────────────────────────────────────────────────────

def log_sentiment_snapshot(
    ticker: str,
    session_date: str | None = None,
    cafef_days: int = 1,
) -> dict:
    """
    Fetch và persist sentiment snapshot cho 1 ticker vào SQLite.

    Args:
        ticker:       Mã cổ phiếu (VD: "VIC")
        session_date: Ngày phiên YYYY-MM-DD. Mặc định = hôm nay.
        cafef_days:   Số ngày nhìn lại khi fetch CafeF (default 1 = chỉ hôm nay).
                      Dùng > 1 khi backfill lịch sử (VD: backfill tuần → cafef_days=7).

    Returns:
        dict với các field đã persist (để caller log hoặc verify).
    """
    session_date = session_date or date.today().isoformat()
    ticker = ticker.upper().strip()

    # Lấy song song (try/except từng nguồn)
    fireant  = _fetch_fireant_buzz(ticker)
    cafef    = _fetch_cafef_sentiment(ticker, days=cafef_days)
    vietstock = _fetch_vietstock_sentiment(ticker)

    combined = _combined_score(
        cafef["score"],    cafef["article_count"],
        vietstock["score"], vietstock["article_count"],
    )

    row = {
        "session_date":        session_date,
        "ticker":              ticker,
        "fireant_buzz_count":  fireant["buzz_count"],
        "fireant_buzz_likes":  fireant["buzz_likes"],
        "cafef_sent_score":    cafef["score"],
        "vietstock_sent_score": vietstock["score"],
        "combined_sent_score": combined,
        "article_count":       cafef["article_count"] + vietstock["article_count"],
        "bullish_count":       cafef["bullish_count"],
        "bearish_count":       cafef["bearish_count"],
    }

    _upsert_sentiment(row)
    log.info(
        "Sentiment logged [%s %s]: buzz=%d, sent=%.3f (%d articles)",
        session_date, ticker,
        fireant["buzz_count"], combined,
        row["article_count"],
    )
    return row


def log_sentiment_batch(
    tickers: list[str],
    session_date: str | None = None,
    cafef_days: int = 1,
    progress_callback: Callable[[float, str], None] | None = None,
) -> list[dict]:
    """
    Log sentiment cho nhiều tickers.
    Dùng trong daily_snapshot.py và backfill_sentiment.py.

    Args:
        tickers:           Danh sách mã cổ phiếu.
        session_date:      Ngày phiên. Mặc định = hôm nay.
        cafef_days:        Số ngày nhìn lại cho CafeF.
        progress_callback: fn(pct: float, msg: str) để cập nhật tiến trình.

    Returns:
        list[dict] các rows đã persist (kể cả rows bị lỗi → được điền 0).
    """
    results = []
    n = len(tickers)
    for i, ticker in enumerate(tickers):
        try:
            row = log_sentiment_snapshot(ticker, session_date, cafef_days)
            results.append(row)
        except Exception as exc:
            log.error("log_sentiment_snapshot lỗi [%s]: %s", ticker, exc)
            # Lưu row rỗng để không bị gap trong DB
            row = {
                "session_date": session_date or date.today().isoformat(),
                "ticker": ticker.upper(),
                "fireant_buzz_count": 0, "fireant_buzz_likes": 0,
                "cafef_sent_score": 0.0, "vietstock_sent_score": 0.0,
                "combined_sent_score": 0.0,
                "article_count": 0, "bullish_count": 0, "bearish_count": 0,
            }
            try:
                _upsert_sentiment(row)
            except Exception:
                pass
            results.append(row)

        if progress_callback:
            progress_callback((i + 1) / n, f"Sentiment {ticker} ({i+1}/{n})")

    log.info("Sentiment batch done: %d/%d tickers logged for %s",
             len(results), n, session_date or date.today().isoformat())
    return results


# ── Read API (dùng trong feature engineering) ──────────────────────────────────

def get_sentiment_history(ticker: str, days: int = 90) -> pd.DataFrame:
    """
    Đọc lịch sử sentiment từ SQLite cho 1 ticker.

    Args:
        ticker: Mã cổ phiếu.
        days:   Số ngày lịch sử cần lấy (tính từ hôm nay).

    Returns:
        DataFrame với cột:
          session_date, fireant_buzz_count, fireant_buzz_likes,
          cafef_sent_score, vietstock_sent_score, combined_sent_score,
          article_count, bullish_count, bearish_count
        Sắp xếp theo session_date ASC. Trả về DataFrame rỗng nếu không có data.
    """
    ensure_db()
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    sql = """
        SELECT
            session_date,
            fireant_buzz_count,
            fireant_buzz_likes,
            cafef_sent_score,
            vietstock_sent_score,
            combined_sent_score,
            article_count,
            bullish_count,
            bearish_count
        FROM sentiment_snapshots
        WHERE ticker = ?
          AND session_date >= ?
        ORDER BY session_date ASC
    """
    try:
        with get_connection() as con:
            rows = con.execute(sql, (ticker.upper(), cutoff)).fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=[
            "session_date", "fireant_buzz_count", "fireant_buzz_likes",
            "cafef_sent_score", "vietstock_sent_score", "combined_sent_score",
            "article_count", "bullish_count", "bearish_count",
        ])
        df["session_date"] = pd.to_datetime(df["session_date"])
        return df
    except Exception as exc:
        log.warning("get_sentiment_history lỗi [%s]: %s", ticker, exc)
        return pd.DataFrame()


def get_sentiment_coverage(tickers: list[str]) -> pd.DataFrame:
    """
    Tóm tắt số ngày đã có data sentiment cho mỗi ticker.
    Hữu ích để kiểm tra coverage trước khi train model.

    Returns:
        DataFrame: ticker, days_logged, first_date, last_date, avg_buzz, avg_sent
    """
    ensure_db()
    sql = """
        SELECT
            ticker,
            COUNT(*) as days_logged,
            MIN(session_date) as first_date,
            MAX(session_date) as last_date,
            ROUND(AVG(fireant_buzz_count), 1) as avg_buzz,
            ROUND(AVG(combined_sent_score), 4) as avg_sent
        FROM sentiment_snapshots
        WHERE ticker IN ({})
        GROUP BY ticker
        ORDER BY ticker
    """.format(",".join("?" * len(tickers)))
    try:
        with get_connection() as con:
            rows = con.execute(sql, [t.upper() for t in tickers]).fetchall()
        return pd.DataFrame(rows, columns=[
            "ticker", "days_logged", "first_date", "last_date", "avg_buzz", "avg_sent"
        ])
    except Exception as exc:
        log.warning("get_sentiment_coverage lỗi: %s", exc)
        return pd.DataFrame()


def has_sentiment_for_date(ticker: str, session_date: str) -> bool:
    """Kiểm tra nhanh xem đã có data cho (ticker, date) chưa."""
    ensure_db()
    with get_connection() as con:
        row = con.execute(
            "SELECT 1 FROM sentiment_snapshots WHERE ticker=? AND session_date=? LIMIT 1",
            (ticker.upper(), session_date),
        ).fetchone()
    return row is not None
