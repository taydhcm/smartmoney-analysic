"""
ml/feature_engineering.py
Tính toán feature vector cho mỗi (ticker, ngày) từ OHLCV.
Không có look-ahead bias: tất cả features chỉ dùng dữ liệu lịch sử.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.accumulation_detection import detect_accumulation_phase
from analytics.wyckoff import detect_wyckoff
from utils.logger import get_logger

log = get_logger(__name__)

# ─── Canonical feature list (train + predict dùng cùng thứ tự) ────────────────
# v2.0: Thêm 3 regime features từ S3 Market Regime Engine (adx_vn30, vn30_di_diff, regime_score)
# v3.0: Thêm 3 smart money features từ S4 D0.2 SQLite logger (foreign_net_pct, foreign_trend, smart_money_score)
# v5.0: Thêm 5 Wyckoff VSA features từ S1 engine v2.0 (Sprint 5) — accumulation_score nay dùng wyckoff_score
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
    # Smart money proxy (OHLCV-derived)
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
    # S3 Market Regime features (v2.0)
    "adx_vn30",           # ADX(14) của VN30, normalized [0,1]
    "vn30_di_diff",       # (DI+ - DI-)/100 — dương = bull trend, âm = bear trend
    "regime_score",       # Raw score [-4, +4]
    # S4 Smart Money Flow features from D0.2 SQLite (v3.0)
    # Giá trị = 0.0 khi chưa đủ 5 phiên dữ liệu (model vẫn hoạt động bình thường)
    "foreign_net_pct",    # TB 5 phiên: foreign net / total vol [-1, +1]
    "foreign_trend",      # Slope of foreign net pct [-1, +1]
    "smart_money_score",  # Composite D0.2 score [-1, +1]
    # S1 Wyckoff VSA features v2.0 (v5.0, Sprint 5) — rolling 20-bar windows
    "spring_quality",     # Chất lượng Spring pattern [0, 1]
    "lps_detected",       # Last Point of Support có hiện diện [0/1]
    "effort_vs_result",   # Effort (vol) vs Result (price) [-1, +1]
    "no_supply_count",    # Tỷ lệ no-supply bars trong 10 phiên [0, 1]
    "stopping_volume",    # Stopping Volume / Selling Climax [0/1]
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


def _rolling_wyckoff(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Rolling Wyckoff/VSA sub-signals (no look-ahead bias).
    Cửa sổ window bar kết thúc tại row hiện tại.

    Returns DataFrame với cột:
      accumulation_score, spring_quality, lps_detected,
      effort_vs_result, no_supply_count, stopping_volume
    """
    cols = [
        "accumulation_score",  # = wyckoff_score
        "spring_quality",
        "lps_detected",
        "effort_vs_result",
        "no_supply_count",
        "stopping_volume",
    ]
    rows: list[dict] = []
    for i in range(len(df)):
        if i < window - 1:
            rows.append({c: np.nan for c in cols})
        else:
            win_df = df.iloc[i - window + 1 : i + 1].copy()
            wr = detect_wyckoff(win_df)
            rows.append({
                "accumulation_score": wr.wyckoff_score,
                "spring_quality":     wr.spring_quality,
                "lps_detected":       float(wr.lps_detected),
                "effort_vs_result":   wr.effort_vs_result,
                # normalize no_supply_count [0,1]: 4 bars / 10-bar window = 40% threshold
                "no_supply_count":    min(1.0, wr.no_supply_count / max(1, 4)),
                "stopping_volume":    float(wr.stopping_volume),
            })
    return pd.DataFrame(rows, index=df.index)


# ── Main feature builder ───────────────────────────────────────────────────────

def compute_stock_features(
    df: pd.DataFrame,
    vn30_df: pd.DataFrame | None = None,
    sm_features: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Tính tất cả features + label cho 1 ticker từ OHLCV.

    Parameters
    ----------
    df          : OHLCV của 1 ticker (cột: date, open, high, low, close, volume).
    vn30_df     : OHLCV chỉ số VN30 (optional, dùng để tính relative strength).
    sm_features : Dict {"foreign_net_pct", "foreign_trend", "smart_money_score"}
                  từ S4 Smart Money Engine. None hoặc {} → set 0.0 (default khi
                  chưa đủ 5 phiên D0.2 data).

    Returns
    -------
    DataFrame với cột: date, [FEATURE_COLS], forward_return_2d, label.
    Label NaN với 5 rows cuối (không có T+5 trong tương lai).
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
    # pct_change từ 0 → inf khi OBV bắt đầu từ 0; replace và clip
    out["obv_trend"] = (
        obv_series.pct_change(5)
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .clip(-5.0, 5.0)
    )

    sma5  = close.rolling(5,  min_periods=3).mean()
    sma20 = close.rolling(20, min_periods=10).mean()
    out["price_vs_5sma"]  = (close / (sma5  + 1e-9) - 1).clip(-1.0, 1.0)
    out["price_vs_20sma"] = (close / (sma20 + 1e-9) - 1).clip(-1.0, 1.0)

    # ── Volume ─────────────────────────────────────────────────────────────────
    vol_ma5  = volume.rolling(5,  min_periods=3).mean()
    vol_ma20 = volume.rolling(20, min_periods=10).mean()
    out["volume_ratio_5d"]  = (volume / (vol_ma5  + 1e-9)).clip(0.0, 10.0)
    out["volume_ratio_20d"] = (volume / (vol_ma20 + 1e-9)).clip(0.0, 10.0)

    # ── Candle structure ───────────────────────────────────────────────────────
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    body         = (close - df["open"]).abs()
    upper        = df["high"] - df[["close", "open"]].max(axis=1)
    lower        = df[["close", "open"]].min(axis=1) - df["low"]

    out["body_ratio"]   = (body   / candle_range).clip(0, 1)
    out["upper_shadow"] = (upper  / candle_range).clip(0, 1)
    out["lower_shadow"] = (lower  / candle_range).clip(0, 1)

    # ── Smart money proxies ────────────────────────────────────────────────────
    out["divergence_score"] = _divergence_score(close, volume)
    out["pull_push_score"]  = _pull_push_score(df)

    # ── S1 Wyckoff VSA rolling signals (v5.0) ─────────────────────────────────
    _wyk = _rolling_wyckoff(df, window=20)
    out["accumulation_score"] = _wyk["accumulation_score"]
    out["spring_quality"]     = _wyk["spring_quality"]
    out["lps_detected"]       = _wyk["lps_detected"]
    out["effort_vs_result"]   = _wyk["effort_vs_result"]
    out["no_supply_count"]    = _wyk["no_supply_count"]
    out["stopping_volume"]    = _wyk["stopping_volume"]

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
        raw_zscore = (rel - mu) / (sigma + 1e-9)
        out["basis_zscore"]       = raw_zscore.replace([np.inf, -np.inf], 0.0).clip(-4.0, 4.0)
        out["basis_extreme_flag"] = (out["basis_zscore"].abs() > 2.0).astype(float)

        # ── S3 Regime features (v2.0) ─────────────────────────────────────────
        # Tính từ VN30 series — không look-ahead vì chỉ dùng data đến ngày T
        from .regime import compute_regime_series
        regime_ser = compute_regime_series(vn30)
        if not regime_ser.empty:
            reg_indexed = regime_ser.set_index("date")
            out["adx_vn30"]      = reg_indexed["adx_vn30"].reindex(df["date"], method="ffill").values
            out["vn30_di_diff"]  = reg_indexed["vn30_di_diff"].reindex(df["date"], method="ffill").values
            out["regime_score"]  = reg_indexed["regime_score"].reindex(df["date"], method="ffill").values
        else:
            out["adx_vn30"]     = 0.0
            out["vn30_di_diff"]  = 0.0
            out["regime_score"]  = 0.0
    else:
        out["vn30_ret_1d"]        = 0.0
        out["relative_strength"]  = out["return_1d"]
        out["vn30_vs_20sma"]      = 0.0
        out["basis_zscore"]       = 0.0
        out["basis_extreme_flag"] = 0.0
        out["adx_vn30"]           = 0.0
        out["vn30_di_diff"]       = 0.0
        out["regime_score"]       = 0.0

    # ── S4 Smart Money features (v3.0) ────────────────────────────────────────
    # Chuyen vao duoi dang scalar cho toan bo time series (gia tri ngay cuoi cung).
    # Khi chua du 5 phien D0.2 data: sm_features=None -> set 0.0 cho ca series.
    _sm = sm_features or {}
    out["foreign_net_pct"]   = float(_sm.get("foreign_net_pct",   0.0))
    out["foreign_trend"]     = float(_sm.get("foreign_trend",     0.0))
    out["smart_money_score"] = float(_sm.get("smart_money_score", 0.0))

    # ── M2 Label: Path-dependent (v2.0) ───────────────────────────────────────
    # y=1 khi: max(close[T+1..T+5])/close[T] >= 1.05  (đạt target +5%)
    #      VÀ: min(close[T+1..T+5])/close[T] >= 0.95  (không bị quét SL -5%)
    # Phản ánh đúng trading goal: hold tối đa 5 phiên, SL cứng -5%
    future_closes = pd.concat(
        [close.shift(-k) for k in range(1, 6)], axis=1
    )
    path_max = future_closes.max(axis=1) / (close + 1e-9) - 1   # max gain trong 5 phiên
    path_min = future_closes.min(axis=1) / (close + 1e-9) - 1   # max drawdown trong 5 phiên

    out["path_max_5d"]       = path_max   # auxiliary: upside tiềm năng
    out["path_min_5d"]       = path_min   # auxiliary: downside risk
    out["sl_hit"]            = (path_min <= -0.05).astype("Int8")  # SL bị quét
    out["forward_return_2d"] = close.pct_change(2).shift(-2)        # backward compat

    # Label chính: đạt target VÀ không chạm SL
    label_raw = ((path_max >= 0.05) & (path_min > -0.05))
    out["label"] = label_raw.astype("Int8")
    # 5 rows cuối không có đủ future data → label = NA
    out.loc[out.index[-5:], ["label", "sl_hit", "path_max_5d", "path_min_5d"]] = pd.NA

    # ── Drop rows with too many missing features ───────────────────────────────
    required = ["return_1d", "rsi_14", "atr_norm"]
    out = out.dropna(subset=required).reset_index(drop=True)

    # Fill remaining NaN + replace inf (safety net)
    for col in FEATURE_COLS:
        if col in out.columns:
            out[col] = (
                out[col]
                .replace([np.inf, -np.inf], 0.0)
                .fillna(0.0)
            )

    return out
