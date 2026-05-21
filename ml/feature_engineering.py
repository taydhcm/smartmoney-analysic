"""
ml/feature_engineering.py
Tính toán feature vector cho mỗi (ticker, ngày) từ OHLCV.
Không có look-ahead bias: tất cả features chỉ dùng dữ liệu lịch sử.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.accumulation_detection import detect_accumulation_phase
from utils.logger import get_logger

log = get_logger(__name__)

# ─── Canonical feature list (train + predict dùng cùng thứ tự) ────────────────
FEATURE_COLS: list[str] = [
    # Return
    "return_1d",
    "return_3d",
    "return_intraday",
    # Volatility
    "atr_norm",
    # Momentum
    "rsi_14",
    "obv_trend",
    "price_vs_5sma",
    "price_vs_20sma",
    # Volume
    "volume_ratio_5d",
    "volume_ratio_20d",
    # Candle structure
    "body_ratio",
    "upper_shadow",
    "lower_shadow",
    # Smart money proxy
    "divergence_score",
    "pull_push_score",
    "accumulation_score",
    # Market / index
    "vn30_ret_1d",
    "relative_strength",
    "vn30_vs_20sma",
    # Derivatives proxy
    "basis_zscore",
    "basis_extreme_flag",
]


# ── Low-level indicators ───────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=window // 2).mean()


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI (Wilder's smoothing)."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window // 2).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window // 2).mean()
    rs = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def _obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df["close"].diff().fillna(0))
    return (direction * df["volume"]).cumsum()


# ── Smart money proxy signals ──────────────────────────────────────────────────

def _divergence_score(close: pd.Series, volume: pd.Series, window: int = 5) -> pd.Series:
    """
    Buying divergence: volume tăng mạnh nhưng giá không tăng tương ứng.
    Score > 0 → tích lũy ngầm.  Score < 0 → phân phối.
    """
    vol_chg   = volume.pct_change(window)
    price_chg = close.pct_change(window)

    score = pd.Series(0.0, index=close.index)
    buying  = (vol_chg > 0.20) & (price_chg < 0.02)
    selling = (vol_chg > 0.20) & (price_chg > 0.05)
    score[buying]  =  vol_chg[buying].clip(upper=1.0)
    score[selling] = -vol_chg[selling].clip(upper=1.0)
    return score.fillna(0.0)


def _pull_push_score(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    Pull/push proxy: tỷ lệ volume ngày tăng vs ngày giảm trong window.
    > 0 → smart money kéo (accumulation).  < 0 → đạp (distribution).
    """
    up_vol   = (df["close"] > df["open"]).astype(float) * df["volume"]
    down_vol = (df["close"] < df["open"]).astype(float) * df["volume"]
    up_roll   = up_vol.rolling(window, min_periods=2).sum()
    down_roll = down_vol.rolling(window, min_periods=2).sum()
    total = up_roll + down_roll + 1e-9
    return ((up_roll - down_roll) / total).fillna(0.0)


def _rolling_accumulation(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Rolling Wyckoff accumulation score (0–1) dùng cửa sổ trượt.
    Tránh look-ahead: cửa sổ kết thúc tại row hiện tại.
    """
    phase_scores = {
        "phase_d":     1.0,
        "phase_c":     0.8,
        "phase_b":     0.6,
        "none":        0.3,
        "distribution": 0.0,
        None:          0.3,
    }
    scores: list[float] = []
    for i in range(len(df)):
        if i < window - 1:
            scores.append(np.nan)
        else:
            win_df = df.iloc[i - window + 1 : i + 1].copy()
            phase  = detect_accumulation_phase(win_df)
            scores.append(phase_scores.get(phase, 0.3))
    return pd.Series(scores, index=df.index)


# ── Main feature builder ───────────────────────────────────────────────────────

def compute_stock_features(
    df: pd.DataFrame,
    vn30_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Tính tất cả features + label cho 1 ticker từ OHLCV.

    Parameters
    ----------
    df       : OHLCV của 1 ticker (cột: date, open, high, low, close, volume).
    vn30_df  : OHLCV chỉ số VN30 (optional, dùng để tính relative strength).

    Returns
    -------
    DataFrame với cột: date, [FEATURE_COLS], forward_return_2d, label.
    Label NaN với 2 rows cuối (không có T+2 trong tương lai).
    """
    if df.empty or len(df) < 20:
        return pd.DataFrame()

    df = df.copy().sort_values("date").reset_index(drop=True)
    close  = df["close"]
    volume = df["volume"]
    n      = len(df)

    out = pd.DataFrame({"date": df["date"]}, index=df.index)

    # ── Return ─────────────────────────────────────────────────────────────────
    out["return_1d"]       = close.pct_change(1)
    out["return_3d"]       = close.pct_change(3)
    out["return_intraday"] = (close - df["open"]) / (df["open"] + 1e-9)

    # ── Volatility ─────────────────────────────────────────────────────────────
    atr = _atr(df, 14)
    out["atr_norm"] = atr / (close + 1e-9)

    # ── Momentum ───────────────────────────────────────────────────────────────
    out["rsi_14"] = _rsi(close, 14)

    obv_series   = _obv(df)
    out["obv_trend"] = obv_series.pct_change(5).fillna(0.0)

    sma5  = close.rolling(5,  min_periods=3).mean()
    sma20 = close.rolling(20, min_periods=10).mean()
    out["price_vs_5sma"]  = close / (sma5  + 1e-9) - 1
    out["price_vs_20sma"] = close / (sma20 + 1e-9) - 1

    # ── Volume ─────────────────────────────────────────────────────────────────
    vol_ma5  = volume.rolling(5,  min_periods=3).mean()
    vol_ma20 = volume.rolling(20, min_periods=10).mean()
    out["volume_ratio_5d"]  = volume / (vol_ma5  + 1e-9)
    out["volume_ratio_20d"] = volume / (vol_ma20 + 1e-9)

    # ── Candle structure ───────────────────────────────────────────────────────
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    body         = (close - df["open"]).abs()
    upper        = df["high"] - df[["close", "open"]].max(axis=1)
    lower        = df[["close", "open"]].min(axis=1) - df["low"]

    out["body_ratio"]   = (body   / candle_range).clip(0, 1)
    out["upper_shadow"] = (upper  / candle_range).clip(0, 1)
    out["lower_shadow"] = (lower  / candle_range).clip(0, 1)

    # ── Smart money proxies ────────────────────────────────────────────────────
    out["divergence_score"]    = _divergence_score(close, volume)
    out["pull_push_score"]     = _pull_push_score(df)
    out["accumulation_score"]  = _rolling_accumulation(df, window=20)

    # ── Market / VN30 features ─────────────────────────────────────────────────
    if vn30_df is not None and not vn30_df.empty:
        vn30 = vn30_df.copy().sort_values("date")
        vn30_indexed = vn30.set_index("date")["close"]
        vn30_aligned = vn30_indexed.reindex(df["date"], method="ffill").values

        vn30_ret = pd.Series(vn30_aligned).pct_change(1).values
        vn30_sma20 = pd.Series(vn30_aligned).rolling(20, min_periods=10).mean().values

        out["vn30_ret_1d"]       = vn30_ret
        out["relative_strength"] = out["return_1d"] - pd.Series(vn30_ret, index=out.index)
        out["vn30_vs_20sma"]     = (vn30_aligned / (vn30_sma20 + 1e-9)) - 1

        # Basis proxy: relative premium của stock vs VN30 (zscore 10 phiên)
        rel   = pd.Series((close.values / (vn30_aligned + 1e-9)) - 1, index=df.index)
        mu    = rel.rolling(10, min_periods=5).mean()
        sigma = rel.rolling(10, min_periods=5).std()
        out["basis_zscore"]       = ((rel - mu) / (sigma + 1e-9)).clip(-4, 4)
        out["basis_extreme_flag"] = (out["basis_zscore"].abs() > 2.0).astype(float)
    else:
        out["vn30_ret_1d"]       = 0.0
        out["relative_strength"] = out["return_1d"]
        out["vn30_vs_20sma"]     = 0.0
        out["basis_zscore"]      = 0.0
        out["basis_extreme_flag"] = 0.0

    # ── Label: T+2 return >= 5% ────────────────────────────────────────────────
    # forward_return_2d = (close[t+2] - close[t]) / close[t]
    # shift(-2) → row t gets T+2 value (no look-ahead when label is dropped for last 2 rows)
    out["forward_return_2d"] = close.pct_change(2).shift(-2)
    out["label"]             = (out["forward_return_2d"] >= 0.05).astype("Int8")
    # Last 2 rows have no future close → label is NA (correct for prediction)
    out.loc[out.index[-2:], "label"] = pd.NA

    # ── Drop rows with too many missing features ───────────────────────────────
    required = ["return_1d", "rsi_14", "atr_norm"]
    out = out.dropna(subset=required).reset_index(drop=True)

    # Fill remaining NaN features with 0 (conservative)
    for col in FEATURE_COLS:
        if col in out.columns:
            out[col] = out[col].fillna(0.0)

    return out
