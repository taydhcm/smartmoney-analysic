"""
data/sector_data.py
Phân tích dòng tiền theo ngành: rotation, heatmap, xếp hạng ngành.
"""

from __future__ import annotations

import pandas as pd

from config.constants import SECTOR_MAP, PERIOD_DAYS
from utils.cache import ttl_cache
from utils.logger import get_logger

log = get_logger(__name__)


@ttl_cache()
def get_sector_flow_summary(period: str = "1m") -> pd.DataFrame:
    """
    Tổng hợp dòng tiền ngoại + tự doanh theo từng ngành.
    Trả về DataFrame: sector, foreign_net_val, tu_doan_net_val, avg_rel_vol, score
    """
    from data.foreign_flow import get_foreign_flow
    from data.proprietary_trading import get_tu_doan_flow
    from data.volume_analysis import enrich_with_volume_indicators
    from data.market_data import get_ohlcv

    records = []
    for sector, tickers in SECTOR_MAP.items():
        f_net = 0.0
        p_net = 0.0
        rel_vols: list[float] = []

        for ticker in tickers:
            try:
                ff = get_foreign_flow(ticker, period)
                if not ff.empty and "net_val" in ff.columns:
                    f_net += ff["net_val"].sum()

                td = get_tu_doan_flow(ticker, period)
                if not td.empty and "net_val" in td.columns:
                    p_net += td["net_val"].sum()

                ohlcv = get_ohlcv(ticker, period)
                if not ohlcv.empty:
                    enriched = enrich_with_volume_indicators(ohlcv)
                    if "rel_vol" in enriched.columns:
                        rel_vols.append(enriched["rel_vol"].mean())
            except Exception as exc:
                log.debug("Lỗi lấy data ngành %s / %s: %s", sector, ticker, exc)

        records.append({
            "sector":          sector,
            "foreign_net_val": f_net,
            "tu_doan_net_val": p_net,
            "avg_rel_vol":     round(sum(rel_vols) / len(rel_vols), 2) if rel_vols else 1.0,
            "ticker_count":    len(tickers),
        })

    df = pd.DataFrame(records)
    # Composite score (chuẩn hóa 0–1 rồi gộp)
    if not df.empty:
        for col in ["foreign_net_val", "tu_doan_net_val", "avg_rel_vol"]:
            mn, mx = df[col].min(), df[col].max()
            df[f"{col}_norm"] = (df[col] - mn) / (mx - mn + 1e-9)
        df["score"] = (
            df["foreign_net_val_norm"] * 0.45 +
            df["tu_doan_net_val_norm"] * 0.30 +
            df["avg_rel_vol_norm"]     * 0.25
        ).round(3)
        df = df.sort_values("score", ascending=False)

    return df.reset_index(drop=True)


@ttl_cache()
def get_sector_top_picks(sector: str, period: str = "1m", top_n: int = 5) -> pd.DataFrame:
    """
    Top cổ phiếu trong ngành theo smart money score.
    Trả về: ticker, foreign_net_val, tu_doan_net_val, rel_vol
    """
    from data.foreign_flow import get_foreign_flow
    from data.proprietary_trading import get_tu_doan_flow
    from data.volume_analysis import enrich_with_volume_indicators
    from data.market_data import get_ohlcv

    tickers = SECTOR_MAP.get(sector, [])
    records = []

    for ticker in tickers:
        try:
            ff  = get_foreign_flow(ticker, period)
            td  = get_tu_doan_flow(ticker, period)
            ohlcv = enrich_with_volume_indicators(get_ohlcv(ticker, period))

            f_net = ff["net_val"].sum() if (not ff.empty and "net_val" in ff.columns) else 0
            p_net = td["net_val"].sum() if (not td.empty and "net_val" in td.columns) else 0
            rv    = ohlcv["rel_vol"].mean() if (not ohlcv.empty and "rel_vol" in ohlcv.columns) else 1.0

            records.append({
                "ticker":          ticker,
                "foreign_net_val": f_net,
                "tu_doan_net_val": p_net,
                "avg_rel_vol":     round(rv, 2),
            })
        except Exception as exc:
            log.debug("Lỗi sector_top_picks %s: %s", ticker, exc)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Simple rank: ưu tiên foreign net buy
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
