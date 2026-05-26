"""
scripts/retrain_v6.py
Sprint 12 — Retrain LightGBM model v6 với 38 features (thêm 3 proprietary flow).

Chạy sau khi:
  1. setup_ssi_credentials.py đã chạy (có .env với SSI credentials)
  2. D0.2 SQLite đã có ≥ 5 phiên dữ liệu (để có foreign flow features)
  3. (Optional) SSI iBoard đang hoạt động (để có proprietary features)

Model mới: v6_calibrated
  - 38 features (35 cũ + 3 proprietary: proprietary_net_pct, prop_trend, combined_institutional_score)
  - Nếu SSI không khả dụng: proprietary features = 0.0 (model vẫn train được)

Usage:
    py -3.12 scripts/retrain_v6.py
    py -3.12 scripts/retrain_v6.py --period 6m   # dữ liệu 6 tháng
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Load .env trước khi import bất cứ gì
from analytics.secret_manager import load_env
load_env()


def main(period: str = "3m", tickers: list | None = None) -> None:
    from config.constants import VN30_TICKERS
    from data.market_data import get_ohlcv, get_index_data
    from ml.feature_engineering import compute_stock_features, FEATURE_COLS
    from ml.smart_money import compute_institutional_flow_features
    from ml.model import train_model
    import pandas as pd

    if tickers is None:
        tickers = VN30_TICKERS

    print(f"Sprint 12 Retrain v6 — {len(tickers)} tickers, period={period}")
    print(f"Features: {len(FEATURE_COLS)} (38 total)")

    vn30_df = get_index_data("VN30", period=period)
    if vn30_df.empty:
        print("LỖI: Không tải được VN30 data")
        sys.exit(1)

    all_frames = []
    failed = []

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i:2d}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        try:
            ohlcv = get_ohlcv(ticker, period=period)
            if ohlcv.empty or len(ohlcv) < 30:
                print("skip (ít data)")
                continue

            # Lấy institutional flow features (foreign + proprietary)
            try:
                sm_feat = compute_institutional_flow_features(ticker)
            except Exception as e:
                print(f"  (flow fallback: {e})", end=" ")
                sm_feat = None

            feat_df = compute_stock_features(ohlcv, vn30_df, sm_features=sm_feat)
            if feat_df.empty:
                print("skip (features rỗng)")
                continue

            feat_df = feat_df.copy()
            feat_df.insert(0, "ticker", ticker)
            all_frames.append(feat_df)
            labeled = feat_df["label"].dropna()
            print(f"ok ({len(feat_df)} rows, {labeled.sum()}/{len(labeled)} positive)")

        except Exception as exc:
            print(f"LỖI: {exc}")
            failed.append(ticker)

    if not all_frames:
        print("\nLỖI: Không có dữ liệu để train!")
        sys.exit(1)

    full_df = pd.concat(all_frames, ignore_index=True)
    print(f"\nTotal rows: {len(full_df)}")
    print(f"Label distribution:\n{full_df['label'].value_counts()}")
    print(f"Failed tickers: {failed or 'none'}")

    print("\nBắt đầu train LightGBM v6...")
    metrics = train_model(full_df)
    print("\nKết quả train:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n✅ Retrain v6 hoàn thành! Model đã lưu vào ml/artifacts/")
    print("   Chạy lại Streamlit để dùng model mới.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="3m", help="Data period (e.g. 3m, 6m, 1y)")
    args = parser.parse_args()
    main(period=args.period)
