"""
scripts/backfill_sentiment.py
Sprint 13: Backfill lịch sử sentiment vào SQLite cho các ngày giao dịch trong quá khứ.

Chiến lược:
  - CafeF/Vietstock RSS: có 'published' date → có thể lọc theo ngày cụ thể
  - Fireant API: KHÔNG hỗ trợ date filter → buzz_count=0 cho ngày lịch sử (graceful degradation)
  - Idempotent: dùng has_sentiment_for_date() để skip ngày đã có data
  - Rate limiting: sleep giữa các request để tránh bị block

Cách chạy:
    # Backfill 30 ngày gần nhất cho toàn bộ VN30
    python scripts/backfill_sentiment.py

    # Backfill 60 ngày cho 1 mã cụ thể
    python scripts/backfill_sentiment.py --ticker VIC --days 60

    # Backfill toàn bộ VN30 trong 90 ngày, bỏ qua ngày đã có
    python scripts/backfill_sentiment.py --days 90 --skip-existing

Exit codes:
    0 = thành công
    1 = lỗi
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

# Thêm repo root vào sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _setup_logging() -> None:
    log_dir = _REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"backfill_sentiment_{date.today().isoformat()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(str(log_file), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _is_trading_day(d: date) -> bool:
    """Bỏ qua thứ 7 và chủ nhật."""
    return d.weekday() < 5


def _get_trading_days(start_date: date, end_date: date) -> list[date]:
    """Danh sách ngày giao dịch từ start đến end (inclusive), cũ nhất → mới nhất."""
    result = []
    current = start_date
    while current <= end_date:
        if _is_trading_day(current):
            result.append(current)
        current += timedelta(days=1)
    return result


def _backfill_ticker_for_date(
    ticker: str,
    target_date: date,
    skip_existing: bool,
    log: logging.Logger,
) -> bool:
    """
    Backfill sentiment cho 1 ticker × 1 ngày.
    Trả về True nếu thành công (kể cả skip), False nếu lỗi nghiêm trọng.
    """
    from data.sentiment_logger import (
        has_sentiment_for_date,
        log_sentiment_snapshot,
    )

    date_str = target_date.isoformat()

    # Skip nếu đã có data và được yêu cầu
    if skip_existing and has_sentiment_for_date(ticker, date_str):
        log.debug("Skip (đã có data): %s %s", date_str, ticker)
        return True

    try:
        # Tính cafef_days: số ngày cần nhìn lại từ target_date
        # CafeF RSS giữ bài khoảng 7-14 ngày — chỉ dùng window = 1 ngày
        # (bài trong đúng ngày target, ±12h)
        # Vì CafeF RSS không hỗ trợ date param, ta fetch toàn bộ và filter theo published
        # → cafef_days=1 là phù hợp; với backfill cũ hơn 14 ngày thì article_count sẽ = 0
        row = log_sentiment_snapshot(
            ticker=ticker,
            session_date=date_str,
            cafef_days=1,  # Fireant sẽ trả về posts hiện tại (không phải ngày lịch sử)
        )

        sent = row.get("combined_sent_score", 0.0)
        buzz = row.get("fireant_buzz_count", 0)
        arts = row.get("article_count", 0)
        log.info("  ✓ [%s %s] buzz=%d sent=%.3f articles=%d", date_str, ticker, buzz, sent, arts)
        return True

    except Exception as exc:
        log.error("  ✗ [%s %s] lỗi: %s", date_str, ticker, exc)
        return False


def main() -> int:
    _setup_logging()
    log = logging.getLogger("backfill_sentiment")

    parser = argparse.ArgumentParser(
        description="Backfill sentiment lịch sử vào SQLite"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Mã cụ thể (VD: VIC). Nếu không chỉ định → toàn bộ VN30.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Số ngày lịch sử cần backfill (default: 30).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Bỏ qua ngày đã có data trong DB (default: True).",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        default=False,
        help="Ghi đè toàn bộ, kể cả ngày đã có data.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.5,
        help="Giây nghỉ giữa mỗi ticker×ngày (default: 1.5s).",
    )
    args = parser.parse_args()

    skip_existing = not args.no_skip  # --no-skip override --skip-existing

    # Xác định danh sách tickers
    if args.ticker:
        tickers = [args.ticker.upper().strip()]
    else:
        from config.constants import VN30_TICKERS
        tickers = list(VN30_TICKERS)

    # Xác định khoảng ngày backfill
    end_date   = date.today()
    start_date = end_date - timedelta(days=args.days)
    trading_days = _get_trading_days(start_date, end_date)

    log.info("=== Backfill Sentiment START ===")
    log.info("Tickers: %d | Ngày: %d (%s → %s) | skip_existing=%s",
             len(tickers), len(trading_days),
             start_date.isoformat(), end_date.isoformat(), skip_existing)

    total = len(tickers) * len(trading_days)
    done = 0
    errors = 0

    # Duyệt theo ngày (outer) × ticker (inner)
    # Lý do: CafeF RSS cache 5 phút → cùng ngày sẽ tái dùng cache, tiết kiệm request
    for target_date in trading_days:
        log.info("--- Ngày: %s ---", target_date.isoformat())
        for ticker in tickers:
            ok = _backfill_ticker_for_date(ticker, target_date, skip_existing, log)
            if ok:
                done += 1
            else:
                errors += 1

            # Rate limiting giữa các requests
            time.sleep(args.sleep)

        # Nghỉ thêm giữa các ngày để tránh CafeF rate limit
        time.sleep(2.0)

    log.info("=== Backfill Sentiment DONE ===")
    log.info("Thành công: %d/%d | Lỗi: %d", done, total, errors)

    # In coverage summary
    try:
        from data.sentiment_logger import get_sentiment_coverage
        coverage = get_sentiment_coverage(tickers)
        if not coverage.empty:
            log.info("--- Coverage Summary ---")
            for _, row in coverage.iterrows():
                log.info(
                    "  %s: %d ngày | %s → %s | avg_buzz=%.1f avg_sent=%.3f",
                    row["ticker"], row["days_logged"],
                    row["first_date"], row["last_date"],
                    row["avg_buzz"], row["avg_sent"],
                )
    except Exception as exc:
        log.warning("Không thể in coverage summary: %s", exc)

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
