"""
analytics/accumulation_detection.py
Phát hiện mẫu tích lũy/phân phối theo lý thuyết Wyckoff và Volume Spread Analysis.
Tất cả logic dựa trên OHLCV – không cần API bổ sung.

Sprint 5: detect_accumulation_phase() nay delegate sang analytics.wyckoff.detect_wyckoff()
để dùng engine Wyckoff/VSA toàn diện.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import get_logger
from analytics.wyckoff import detect_wyckoff

log = get_logger(__name__)


def _rolling_std_ratio(series: pd.Series, window: int = 10) -> pd.Series:
    """Tỷ lệ độ lệch chuẩn gần đây / toàn kỳ."""
    recent_std = series.rolling(window).std()
    overall_std = series.std()
    return recent_std / (overall_std + 1e-9)


def detect_accumulation_phase(df: pd.DataFrame) -> str | None:
    """
    Phát hiện phase tích lũy Wyckoff — wrapper sang WyckoffResult.detect_wyckoff().
    Trả về: 'phase_b' | 'phase_c' | 'phase_d' | 'distribution' | 'none' | None

    Sprint 5: Đã dùng engine Wyckoff/VSA v2.0 toàn diện, bao gồm:
      Spring quality, LPS, effort-vs-result, no-supply bar, stopping volume.
    """
    if df is None or df.empty or len(df) < 10:
        return None
    return detect_wyckoff(df).phase


def detect_volume_climax(df: pd.DataFrame) -> list[int]:
    """
    Tìm các phiên có volume climax (đột biến cực lớn + đảo chiều giá).
    Trả về: list chỉ số phiên climax.
    """
    if df.empty or len(df) < 5:
        return []

    vol_mean = df["volume"].rolling(20).mean()
    vol_std  = df["volume"].rolling(20).std()
    threshold = vol_mean + 2.5 * vol_std

    climax_sessions = []
    for i in range(1, len(df)):
        is_surge = df["volume"].iloc[i] > threshold.iloc[i]
        # Đảo chiều: nến đỏ lớn sau khi đang tăng, hoặc ngược lại
        reversal = (
            (df["close"].iloc[i] < df["open"].iloc[i]) and
            (df["close"].iloc[i - 1] > df["open"].iloc[i - 1])
        ) or (
            (df["close"].iloc[i] > df["open"].iloc[i]) and
            (df["close"].iloc[i - 1] < df["open"].iloc[i - 1])
        )
        if is_surge and reversal:
            climax_sessions.append(i)

    return climax_sessions


def detect_stealth_accumulation(df: pd.DataFrame, window: int = 5) -> bool:
    """
    Tích lũy lặng lẽ: khối lượng tăng dần nhưng giá không tăng nhiều.
    Đây là dấu hiệu smart money đang âm thầm gom hàng.
    """
    if df.empty or len(df) < window * 2:
        return False

    recent = df.tail(window)
    previous = df.iloc[-window * 2:-window]

    vol_increasing = recent["volume"].mean() > previous["volume"].mean() * 1.2
    price_flat     = abs(recent["close"].mean() - previous["close"].mean()) / previous["close"].mean() < 0.03

    return bool(vol_increasing and price_flat)


def get_accumulation_summary(ticker: str, df: pd.DataFrame) -> str:
    """Tóm tắt pattern tích lũy cho LangGraph agent."""
    if df.empty:
        return f"Không đủ dữ liệu tích lũy cho {ticker}."

    phase = detect_accumulation_phase(df)
    climax = detect_volume_climax(df)
    stealth = detect_stealth_accumulation(df)

    phase_desc = {
        "phase_b":      "Phase B – Tích lũy lặng lẽ, volume thu hẹp",
        "phase_c":      "Phase C (Spring) – Test đáy, khả năng đảo chiều cao",
        "phase_d":      "Phase D – Markup bắt đầu, momentum tăng",
        "distribution": "Phân phối – Cảnh báo smart money đang xả hàng",
        "none":         "Chưa xác định mẫu rõ ràng",
        None:           "Không đủ dữ liệu",
    }

    parts = [f"[{ticker}] {phase_desc.get(phase, 'N/A')}"]
    if climax:
        parts.append(f"| Volume climax tại phiên: {climax}")
    if stealth:
        parts.append("| ⚡ Tín hiệu tích lũy lặng lẽ (stealth accumulation)")

    return " ".join(parts)
