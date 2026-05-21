"""
utils/rate_limiter.py
Sliding-window rate limiter cho VNStock API calls.

VNStock free tier: 20 req/phút (hard cap)
  → throttle xuống 17 req/phút để có buffer an toàn
  → sleep tự động nếu sắp vượt quota

Khi có SSI API key: max_requests được set cao (vô giới hạn thực tế).
Singleton module-level → dùng chung across tất cả Streamlit sessions.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

log = logging.getLogger(__name__)

# ── Rate limits ────────────────────────────────────────────────────────────────
FREE_MAX_REQUESTS   = 17          # 17/60s → buffer dưới hard cap 20
FREE_WINDOW_SECONDS = 60.0
SSI_MAX_REQUESTS    = 10_000      # effectively unlimited
SSI_WINDOW_SECONDS  = 60.0


class SlidingWindowRateLimiter:
    """
    Thread-safe sliding-window rate limiter.

    Giữ log timestamp của các request trong window.
    Nếu đã đạt max_requests, sleep cho đến khi có slot.
    """

    def __init__(self, max_requests: int, window_seconds: float, name: str = "limiter") -> None:
        self.max_requests    = max_requests
        self.window_seconds  = window_seconds
        self.name            = name
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()
        self._enabled = max_requests < 9_000   # disabled khi unlimited

    @property
    def is_throttled(self) -> bool:
        """True nếu đang ở chế độ throttle (free tier)."""
        return self._enabled

    @property
    def requests_per_minute(self) -> int:
        return self.max_requests if self._enabled else 0

    def acquire(self) -> float:
        """
        Đợi cho đến khi có slot rồi ghi nhận 1 request.
        Trả về số giây đã sleep (0 nếu không cần đợi).
        """
        if not self._enabled:
            return 0.0

        with self._lock:
            now = time.monotonic()

            # Xóa timestamps đã ra ngoài window
            while self._timestamps and self._timestamps[0] < now - self.window_seconds:
                self._timestamps.popleft()

            slept = 0.0
            if len(self._timestamps) >= self.max_requests:
                # Phải đợi cho đến khi request cũ nhất ra khỏi window
                wait_until = self._timestamps[0] + self.window_seconds + 0.05
                sleep_time = wait_until - time.monotonic()
                if sleep_time > 0:
                    log.info(
                        "[RateLimiter:%s] Quota đang đầy (%d/%d), sleep %.1fs...",
                        self.name, len(self._timestamps), self.max_requests, sleep_time,
                    )
                    time.sleep(sleep_time)
                    slept = sleep_time

                # Xóa lại sau sleep
                now = time.monotonic()
                while self._timestamps and self._timestamps[0] < now - self.window_seconds:
                    self._timestamps.popleft()

            self._timestamps.append(time.monotonic())
            return slept

    def current_usage(self) -> tuple[int, int]:
        """Trả về (requests_used_in_window, max_requests)."""
        if not self._enabled:
            return (0, self.max_requests)
        with self._lock:
            now = time.monotonic()
            active = sum(1 for t in self._timestamps if t >= now - self.window_seconds)
            return (active, self.max_requests)

    def eta_seconds(self, remaining_calls: int) -> float:
        """
        Ước tính số giây còn lại để hoàn thành remaining_calls.
        Dùng để hiển thị ETA trên UI.
        """
        if not self._enabled:
            return 0.0
        used, max_r = self.current_usage()
        # Mỗi 'batch' max_r request mất window_seconds giây
        full_batches  = remaining_calls // max_r
        leftover      = remaining_calls % max_r
        # Slot hiện còn trống trong window này
        available_now = max(max_r - used, 0)
        first_batch   = max(remaining_calls - available_now, 0)
        eta = (first_batch / max_r) * self.window_seconds
        return eta


def _make_limiter() -> SlidingWindowRateLimiter:
    """Tạo singleton limiter dựa trên config (SSI vs free)."""
    try:
        from config.settings import SSI_API_KEY
        has_ssi = bool(SSI_API_KEY)
    except Exception:
        has_ssi = False

    if has_ssi:
        log.info("[RateLimiter] SSI API detected → unlimited mode")
        return SlidingWindowRateLimiter(
            SSI_MAX_REQUESTS, SSI_WINDOW_SECONDS, name="vnstock-ssi"
        )
    else:
        log.info(
            "[RateLimiter] VNStock free tier → throttle %d req/%ds",
            FREE_MAX_REQUESTS, int(FREE_WINDOW_SECONDS),
        )
        return SlidingWindowRateLimiter(
            FREE_MAX_REQUESTS, FREE_WINDOW_SECONDS, name="vnstock-free"
        )


# ── Module-level singleton ─────────────────────────────────────────────────────
# Khởi tạo 1 lần khi import; dùng chung toàn bộ ứng dụng
vnstock_limiter: SlidingWindowRateLimiter = _make_limiter()
