"""
analytics/accumulation_detection.py
Phát hiện mẫu tích lũy/phân phối theo lý thuyết Wyckoff và Volume Spread Analysis.
Tất cả logic dựa trên OHLCV – không cần API bổ sung.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


def _rolling_std_ratio(series: pd.Series, window: int = 10) -> pd.Series:
    """Tỷ lệ độ lệch chuẩn gần đây / toàn kỳ."""
    recent_std = series.rolling(window).std()
    overall_std = series.std()
    return recent_std / (overall_std + 1e-9)


def detect_accumulation_phase(df: pd.DataFrame) -> str | None:
    """
    Phát hiện phase tích lũy Wyckoff đơn giản.
    Trả về: 'phase_b' | 'phase_c' | 'phase_d' | 'distribution' | 'none'

    Logic:
    - Phase B: Giá dao động trong range, volume giảm dần → tích lũy lặng lẽ
    - Phase C (Spring): Giá phá đáy ngắn rồi hồi phục ngay, volume thấp
    - Phase D: Giá bắt đầu bứt phá, volume tăng, các đợt pullback nhỏ
    - Distribution: Giá cao, volume tăng nhưng giá không tăng → phân phối
    """
    if df.empty or len(df) < 10:
        return None

    df = df.copy()
    close = df["close"]
    volume = df["volume"]
    high = df["high"]
    low  = df["low"]

    n = len(df)
    half = n // 2

    # Các chỉ số cơ bản
    price_range_ratio  = (close.max() - close.min()) / close.mean()  # range tương đối
    vol_trend          = np.polyfit(range(n), volume.fillna(0), 1)[0]  # slope volume
    price_trend        = np.polyfit(range(n), close.fillna(close.mean()), 1)[0]
    recent_vol_mean    = volume.tail(5).mean()
    overall_vol_mean   = volume.mean()
    vol_contraction    = recent_vol_mean < overall_vol_mean * 0.7

    last_close    = close.iloc[-1]
    period_low    = low.min()
    period_high   = high.max()
    near_low      = last_close < period_low * 1.05
    near_high     = last_close > period_high * 0.95

    # Phase D: giá đang tăng, volume tăng
    if price_trend > 0 and vol_trend > 0 and near_high:
        return "phase_d"

    # Phase C (Spring): giá test đáy với volume thấp rồi hồi
    if near_low and vol_contraction and price_range_ratio < 0.15:
        # Kiểm tra nếu có recovery sau khi chạm đáy
        recent_recovery = close.tail(3).mean() > close.iloc[-4] if n > 4 else False
        if recent_recovery:
            return "phase_c"

    # Phase B: range hẹp, volume thu hẹp, không trending mạnh
    if price_range_ratio < 0.12 and vol_contraction and abs(price_trend) < close.mean() * 0.001:
        return "phase_b"

    # Distribution: giá cao + volume cao nhưng giá không tiếp tục tăng
    if near_high and vol_trend > 0 and price_trend <= 0:
        return "distribution"

    return "none"


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
