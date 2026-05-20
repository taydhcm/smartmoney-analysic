"""
utils/cache.py
TTL cache dùng diskcache. Tự động invalidate sau giờ giao dịch.
"""

from __future__ import annotations
import functools
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import diskcache
import pytz

from config.settings import CACHE_TTL
from config.constants import VN_TZ, TRADING_SESSIONS

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
_cache = diskcache.Cache(str(_CACHE_DIR))


def _make_key(func: Callable, args: tuple, kwargs: dict) -> str:
    raw = json.dumps(
        {"fn": f"{func.__module__}.{func.__qualname__}", "a": args, "k": kwargs},
        sort_keys=True,
        default=str,
    )
    return hashlib.md5(raw.encode()).hexdigest()


def _is_market_open() -> bool:
    now = datetime.now(VN_TZ)
    if now.weekday() >= 5:   # thứ 7, CN
        return False
    t = now.strftime("%H:%M")
    morning  = TRADING_SESSIONS["morning_open"]  <= t <= TRADING_SESSIONS["morning_close"]
    afternoon = TRADING_SESSIONS["afternoon_open"] <= t <= TRADING_SESSIONS["market_close"]
    return morning or afternoon


def ttl_cache(ttl: int = CACHE_TTL):
    """
    Decorator cache với TTL (giây).
    Trong giờ giao dịch: dùng TTL ngắn (15 phút mặc định).
    Ngoài giờ giao dịch: TTL × 4 (60 phút) để giảm API call.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            key = _make_key(func, args, kwargs)
            if key in _cache:
                return _cache[key]
            result = func(*args, **kwargs)
            effective_ttl = ttl if _is_market_open() else ttl * 4
            _cache.set(key, result, expire=effective_ttl)
            return result
        return wrapper
    return decorator


def clear_cache() -> None:
    """Xóa toàn bộ cache (dùng khi debug hoặc từ UI)."""
    _cache.clear()
