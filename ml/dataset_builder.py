"""
ml/dataset_builder.py
Xây dựng dataset train/test từ OHLCV lịch sử của nhiều tickers.
"""

from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

from config.constants import VN30_TICKERS
from data.market_data import get_ohlcv, get_index_data
from .feature_engineering import compute_stock_features, FEATURE_COLS

log = logging.getLogger(__name__)

# Cột output của dataset — v2.0: thêm M2 path-dependent label columns
DATASET_COLS = (
    ["ticker", "date"]
    + FEATURE_COLS
    + ["forward_return_2d", "path_max_5d", "path_min_5d", "sl_hit", "label"]
)


def build_dataset(
    tickers: list[str] | None = None,
    period: str = "6m",
    progress_callback: Callable[[float, str], None] | None = None,
) -> pd.DataFrame:
    """
    Xây dựng training dataset từ lịch sử OHLCV.

    Parameters
    ----------
    tickers           : Danh sách mã cổ phiếu. Mặc định = VN30.
    period            : Kỳ lấy dữ liệu ("3m", "6m", "12m").
    progress_callback : fn(pct: float, msg: str) để cập nhật progress bar.

    Returns
    -------
    DataFrame với cột: ticker, date, [FEATURE_COLS], forward_return_2d, label.
    Chỉ giữ rows có label hợp lệ (loại bỏ 2 phiên cuối mỗi ticker).
    """
    if tickers is None:
        tickers = VN30_TICKERS

    # VN30 index làm baseline tính relative strength
    if progress_callback:
        progress_callback(0.0, "Đang tải VN30 index...")
    vn30_df = get_index_data("VN30", period=period)

    all_frames: list[pd.DataFrame] = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(i / total, f"Đang xử lý {ticker} ({i+1}/{total})...")

        try:
            ohlcv = get_ohlcv(ticker, period=period)
            if ohlcv.empty or len(ohlcv) < 30:
                log.warning("Bỏ qua %s: chỉ có %d rows OHLCV", ticker, len(ohlcv))
                continue

            # S4 Smart Money features từ D0.2 SQLite (0.0 khi chưa đủ data)
            try:
                from ml.smart_money import compute_smart_money_features
                sm_feat = compute_smart_money_features(ticker)
            except Exception:
                sm_feat = None

            feat_df = compute_stock_features(ohlcv, vn30_df, sm_features=sm_feat)
            if feat_df.empty:
                continue

            # Chỉ giữ rows có label rõ ràng (không phải NA)
            labeled = feat_df.dropna(subset=["label"]).copy()
            labeled["label"] = labeled["label"].astype(int)
            # sl_hit có thể NA ở cuối — fill 0 sau khi đã drop label-NA rows
            if "sl_hit" in labeled.columns:
                labeled["sl_hit"] = labeled["sl_hit"].fillna(0).astype(int)
            labeled.insert(0, "ticker", ticker)

            # Giữ đúng các cột cần thiết
            keep_cols = [c for c in DATASET_COLS if c in labeled.columns]
            all_frames.append(labeled[keep_cols])

        except Exception as exc:
            log.warning("Lỗi khi xử lý %s: %s", ticker, exc)

    if progress_callback:
        progress_callback(1.0, "Hoàn tất build dataset.")

    if not all_frames:
        log.error("Dataset rỗng – không có dữ liệu hợp lệ")
        return pd.DataFrame()

    dataset = pd.concat(all_frames, ignore_index=True)
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset = dataset.sort_values(["date", "ticker"]).reset_index(drop=True)

    n_pos = int(dataset["label"].sum())
    n_total = len(dataset)
    log.info(
        "Dataset: %d rows | %d tickers | label_rate=%.1f%%",
        n_total,
        dataset["ticker"].nunique(),
        100 * n_pos / max(n_total, 1),
    )
    return dataset
