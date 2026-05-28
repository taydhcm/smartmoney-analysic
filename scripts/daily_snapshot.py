"""
scripts/daily_snapshot.py
D0.2 Daily Snapshot Job — chay luc 15:05 hang ngay.

Chay bang lenh:
    python scripts/daily_snapshot.py

Hoac tu Windows Task Scheduler (xem setup_scheduler.ps1).

Exit codes:
    0 = thanh cong
    1 = loi
    2 = khong phai ngay thuong (thu 7, chu nhat)
"""

from __future__ import annotations

import sys
import logging
from datetime import date
from pathlib import Path

# Them repo root vao sys.path de import duoc cac module
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _setup_logging() -> None:
    """Log ra file va console."""
    log_dir = _REPO_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"snapshot_{date.today().isoformat()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(str(log_file), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> int:
    _setup_logging()
    log = logging.getLogger("daily_snapshot")

    today = date.today()
    log.info("=== D0.2 Daily Snapshot Job START === %s", today.isoformat())

    # Kiem tra ngay giao dich
    if today.weekday() >= 5:
        log.info("Hom nay la %s (thu %d) — khong phai ngay giao dich, bo qua.",
                 today.strftime("%A"), today.weekday() + 1)
        return 2

    try:
        from data.snapshot_logger import log_session, get_logger_status

        def _progress(pct: float, msg: str) -> None:
            log.info("[%.0f%%] %s", pct * 100, msg)

        result = log_session(progress_callback=_progress)

        log.info(
            "Ket qua: session=%s, logged=%d, skipped=%d, already_existed=%s",
            result["session_date"],
            result["tickers_logged"],
            result["tickers_skipped"],
            result["already_exists"],
        )

        status = get_logger_status()
        log.info(
            "Trang thai logger: %d phien, s4_ready=%s, can them %d phien",
            status["sessions"],
            status["s4_ready"],
            status["sessions_needed"],
        )

        if status["s4_ready"]:
            log.info(">>> S4 Smart Money features DA san sang (>= 5 phien)!")
        else:
            log.info(">>> Can %d phien nua de bat S4 features.", status["sessions_needed"])

        # ── Sprint 13: Log sentiment snapshot cho toàn bộ VN30 ───────────────
        log.info("--- Sprint 13: Logging sentiment snapshots ---")
        try:
            from data.sentiment_logger import log_sentiment_batch
            from config.constants import VN30_TICKERS

            sent_results = log_sentiment_batch(
                tickers=list(VN30_TICKERS),
                session_date=result["session_date"],
                cafef_days=1,
                progress_callback=_progress,
            )
            logged_sent = sum(1 for r in sent_results if r.get("article_count", 0) > 0 or r.get("fireant_buzz_count", 0) > 0)
            log.info(
                "Sentiment logged: %d/%d tickers có data (buzz hoặc articles)",
                logged_sent, len(sent_results),
            )
        except Exception as sent_exc:
            # Sentiment log lỗi KHÔNG làm fail toàn bộ job
            log.warning("Sentiment logging lỗi (non-fatal): %s", sent_exc)

        log.info("=== D0.2 Daily Snapshot Job DONE ===")
        return 0

    except Exception as exc:
        log.exception("Loi trong daily snapshot job: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
