"""
data/sector_data.py
Phân tích dòng tiền theo ngành: rotation, heatmap, xếp hạng ngành.

Tối ưu: dùng 1 KBS price_board batch call cho toàn bộ tickers thay vì
gọi riêng lẻ từng mã (giảm từ ~50 xuống còn 1 API call).
"""

from __future__ import annotations

import pandas as pd

from config.constants import SECTOR_MAP, PERIOD_DAYS
from utils.cache import ttl_cache
from utils.logger import get_logger

log = get_logger(__name__)


def _batch_board(tickers: list[str]) -> pd.DataFrame:
    """
    Lấy KBS price_board cho toàn bộ danh sách tickers trong 1 lần gọi.
    Trả về DataFrame với cột phẳng: symbol, foreign_buy_volume,
    foreign_sell_volume, close_price, percent_change, volume_accumulated, ...
    """
    try:
        from vnstock.api.trading import Trading  # type: ignore
        return Trading(source="KBS").price_board(symbols_list=tickers)
    except Exception as exc:
        log.warning("_batch_board lỗi: %s", exc)
        return pd.DataFrame()


def _build_lookup(board: pd.DataFrame) -> tuple[dict, dict, dict]:
    """
    Từ price_board, xây dựng 3 dict:
    - foreign_map:  ticker → net_val (VND)
    - activity_map: ticker → proxy hoạt động giá (1.0 = bình thường)
    - price_map:    ticker → close_price (VND/share, dùng để quy đổi tự doanh)
    """
    foreign_map: dict[str, float] = {}
    activity_map: dict[str, float] = {}
    price_map: dict[str, float] = {}

    if board.empty:
        return foreign_map, activity_map, price_map

    ticker_col = "symbol" if "symbol" in board.columns else board.columns[0]
    buy_col    = "foreign_buy_volume"  if "foreign_buy_volume"  in board.columns else None
    sell_col   = "foreign_sell_volume" if "foreign_sell_volume" in board.columns else None
    close_col  = next((c for c in board.columns if c in ("close_price", "average_price")), None)
    pct_col    = "percent_change" if "percent_change" in board.columns else None

    for _, row in board.iterrows():
        ticker = str(row.get(ticker_col, "")).strip()
        if not ticker:
            continue

        px = float(pd.to_numeric(row.get(close_col, 0), errors="coerce") or 0) if close_col else 0
        price_map[ticker] = px

        # Foreign net value (VND)
        if buy_col and sell_col:
            bv = float(pd.to_numeric(row.get(buy_col,  0), errors="coerce") or 0)
            sv = float(pd.to_numeric(row.get(sell_col, 0), errors="coerce") or 0)
            foreign_map[ticker] = (bv - sv) * px
        else:
            foreign_map[ticker] = 0.0

        # Activity proxy: |%change| normalized → relative-vol-like metric
        if pct_col:
            pct = abs(float(pd.to_numeric(row.get(pct_col, 0), errors="coerce") or 0))
            # 3.5% change ~ 2× normal activity (rough heuristic)
            activity_map[ticker] = max(0.5, 1.0 + pct / 3.5)
        else:
            activity_map[ticker] = 1.0

    return foreign_map, activity_map, price_map


def _build_tu_doan_vol_map(exchange: str = "HOSE") -> dict[str, float]:
    """
    Lấy dữ liệu tự doanh từ SSI FiinMarket iBoard và trả về dict:
    ticker → proprietary_net (khối lượng cổ phiếu, có thể âm).
    """
    try:
        from data.proprietary_trading import get_top_tu_doan_net
        result = get_top_tu_doan_net(exchange, top_n=200)  # lấy nhiều để cover toàn ngành
        vol_map: dict[str, float] = {}
        for _, row in result["buy"].iterrows():
            vol_map[row["ticker"]] = float(row["net_vol"])
        for _, row in result["sell"].iterrows():
            vol_map[row["ticker"]] = float(row["net_vol"])
        return vol_map
    except Exception as exc:
        log.warning("_build_tu_doan_vol_map lỗi: %s", exc)
        return {}


@ttl_cache()
def get_sector_flow_summary(period: str = "1m") -> pd.DataFrame:
    """
    Tổng hợp dòng tiền ngoại theo từng ngành.
    Dùng 1 batch KBS price_board call cho toàn bộ tickers.
    Trả về DataFrame: sector, foreign_net_val, tu_doan_net_val, avg_rel_vol, score
    """
    # 1. Thu thập tất cả tickers từ tất cả ngành (bỏ trùng)
    all_tickers = sorted({t for tickers in SECTOR_MAP.values() for t in tickers})
    log.info("get_sector_flow_summary: batch fetch %d tickers từ KBS", len(all_tickers))

    # 2. MỘT lần gọi API duy nhất
    board = _batch_board(all_tickers)
    log.info("get_sector_flow_summary: KBS board shape=%s, cols=%s",
             board.shape, list(board.columns)[:8] if not board.empty else "EMPTY")
    foreign_map, activity_map, price_map = _build_lookup(board)

    # 2b. Lấy tự doanh từ SSI FiinMarket (batch toàn sàn)
    tu_doan_vol_map = _build_tu_doan_vol_map("HOSE")
    log.info("get_sector_flow_summary: SSI tự doanh %d tickers", len(tu_doan_vol_map))

    # 3. Tổng hợp theo ngành
    records = []
    for sector, tickers in SECTOR_MAP.items():
        f_net    = sum(foreign_map.get(t, 0.0) for t in tickers)
        acts     = [activity_map.get(t, 1.0) for t in tickers]
        avg_act  = round(sum(acts) / len(acts), 2) if acts else 1.0
        # Quy đổi tự doanh: vol × close_price → VND
        p_net    = sum(
            tu_doan_vol_map.get(t, 0.0) * price_map.get(t, 0.0)
            for t in tickers
        )

        records.append({
            "sector":          sector,
            "foreign_net_val": f_net,
            "tu_doan_net_val": p_net,
            "avg_rel_vol":     avg_act,
            "ticker_count":    len(tickers),
        })

    df = pd.DataFrame(records)

    # 4. Composite score (chuẩn hóa 0–1)
    if not df.empty:
        for col in ["foreign_net_val", "tu_doan_net_val", "avg_rel_vol"]:
            mn, mx = df[col].min(), df[col].max()
            df[f"{col}_norm"] = (df[col] - mn) / (mx - mn + 1e-9)
        has_tu_doan = df["tu_doan_net_val"].abs().sum() > 0
        if has_tu_doan:
            df["score"] = (
                df["foreign_net_val_norm"] * 0.55 +
                df["tu_doan_net_val_norm"] * 0.15 +
                df["avg_rel_vol_norm"]     * 0.30
            ).round(3)
        else:
            df["score"] = (
                df["foreign_net_val_norm"] * 0.70 +
                df["avg_rel_vol_norm"]     * 0.30
            ).round(3)
        df = df.sort_values("score", ascending=False)

    log.info("get_sector_flow_summary: xong, %d ngành", len(df))
    return df.reset_index(drop=True)


@ttl_cache()
def get_sector_top_picks(sector: str, period: str = "1m", top_n: int = 5) -> pd.DataFrame:
    """
    Top cổ phiếu trong ngành theo foreign flow hôm nay.
    Dùng 1 KBS batch call cho các tickers của ngành.
    Trả về: ticker, foreign_net_val, tu_doan_net_val, avg_rel_vol
    """
    tickers = SECTOR_MAP.get(sector, [])
    if not tickers:
        return pd.DataFrame()

    board = _batch_board(tickers)
    foreign_map, activity_map, price_map = _build_lookup(board)

    # Tự doanh từ SSI FiinMarket (dùng batch toàn sàn đã có trong cache)
    tu_doan_vol_map = _build_tu_doan_vol_map("HOSE")

    records = [
        {
            "ticker":          t,
            "foreign_net_val": foreign_map.get(t, 0.0),
            "tu_doan_net_val": tu_doan_vol_map.get(t, 0.0) * price_map.get(t, 0.0),
            "avg_rel_vol":     activity_map.get(t, 1.0),
        }
        for t in tickers
    ]

    df = pd.DataFrame(records)
    df = df.sort_values("foreign_net_val", ascending=False)
    return df.head(top_n).reset_index(drop=True)


def summarize_sector(sector: str, period: str = "1m") -> str:
    """Tóm tắt ngành cho LangGraph agent."""
    picks = get_sector_top_picks(sector, period)
    if picks.empty:
        return f"Không đủ dữ liệu cho ngành {sector}."
    top = picks.head(3)["ticker"].tolist()
    return (
        f"[{sector}] Top picks ({period}): {', '.join(top)} | "
        f"Ngoại net: {picks['foreign_net_val'].sum()/1e9:.1f} tỷ | "
        f"Tự doanh net: {picks['tu_doan_net_val'].sum()/1e9:.1f} tỷ"
    )
