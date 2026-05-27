"""
data/foreign_flow.py
Phân tích dòng tiền khối ngoại (foreign investor flows).

Kiến trúc nguồn dữ liệu (theo thứ tự ưu tiên):
  1. FiinMarket GetForeign API (ssi_foreign.py) — real-time hôm nay, batch toàn sàn
  2. D0.2 SQLite (snapshots.db)  — lịch sử đa phiên, nguồn chính cho ML features
  3. VNDirect FINFO API          — lịch sử N ngày (fallback khi SQLite chưa đủ data)
  4. KBS Snapshot                — fallback cuối (tích lũy trên đĩa)

Để override provider: set biến môi trường FLOW_PROVIDER=ssi/kbs/vndirect/auto
"""

from __future__ import annotations

import os
import pandas as pd

from config.constants import PERIOD_DAYS, FOREIGN_ROOM_ALERT_PCT, VN30_TICKERS
from utils.cache import ttl_cache
from utils.logger import get_logger
from data.providers import get_provider

log = get_logger(__name__)

# Provider cũ (fallback cho lịch sử khi SQLite chưa đủ data)
_PROVIDER_NAME = os.getenv("FLOW_PROVIDER", "auto")
_provider = get_provider(_PROVIDER_NAME)
log.info("foreign_flow: provider fallback [%s]", _provider.name)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_from_fiinmarket(ticker: str) -> pd.DataFrame:
    """
    Lấy dữ liệu khối ngoại hôm nay từ FiinMarket GetForeign.
    Trả về DataFrame 1 row với schema: date, buy_vol, sell_vol, net_vol.
    Đơn vị: VND value (không phải khối lượng cổ phiếu).
    """
    try:
        from analytics.ssi_foreign import fetch_foreign_flow
        raw = fetch_foreign_flow(ticker)
        if raw.empty:
            return pd.DataFrame()
        row = raw.iloc[0]
        return pd.DataFrame([{
            "date":     row["date"],
            "buy_vol":  row["foreign_buy"],   # giá trị VND
            "sell_vol": row["foreign_sell"],  # giá trị VND
            "net_vol":  row["foreign_net"],   # giá trị VND
            "net_val":  row["foreign_net"],   # alias
        }])
    except Exception as exc:
        log.debug("_get_from_fiinmarket(%s): %s", ticker, exc)
        return pd.DataFrame()


def _get_from_sqlite(ticker: str, days: int) -> pd.DataFrame:
    """
    Lấy lịch sử dữ liệu khối ngoại từ D0.2 SQLite (snapshots.db).
    Trả về DataFrame nhiều row với schema: date, buy_vol, sell_vol, net_vol, net_val.
    """
    try:
        from data.db import load_snapshots
        db_df = load_snapshots(ticker, last_n=days)
        if db_df.empty:
            return pd.DataFrame()
        db_df = db_df.rename(columns={
            "session_date": "date",
            "foreign_buy":  "buy_vol",
            "foreign_sell": "sell_vol",
            "foreign_net":  "net_vol",
        })
        db_df["net_val"] = db_df["net_vol"]
        cols = ["date", "buy_vol", "sell_vol", "net_vol", "net_val"]
        return db_df[[c for c in cols if c in db_df.columns]].sort_values("date").reset_index(drop=True)
    except Exception as exc:
        log.debug("_get_from_sqlite(%s): %s", ticker, exc)
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

@ttl_cache()
def get_foreign_flow(ticker: str, period: str = "1w") -> pd.DataFrame:
    """
    Lịch sử dòng tiền khối ngoại cho 1 mã.

    Thứ tự ưu tiên:
      1. D0.2 SQLite (lịch sử đã tích lũy từ FiinMarket)
      2. FiinMarket GetForeign API (nếu SQLite chưa đủ → bổ sung hôm nay)
      3. Provider cũ VNDirect / KBS (fallback lịch sử)

    Args:
        ticker: Mã chứng khoán (VD: "VIC")
        period: "1w" | "2w" | "1m" | "3m"

    Returns:
        DataFrame: date, buy_vol, sell_vol, net_vol, net_val
        Sắp xếp tăng dần theo date.
    """
    days = PERIOD_DAYS.get(period, 7)

    # ── Layer 1: D0.2 SQLite (lịch sử đa phiên, tốt nhất cho ML trend) ─────
    df_sql = _get_from_sqlite(ticker, days)
    if len(df_sql) >= days:
        return df_sql

    # ── Layer 2: FiinMarket real-time (bổ sung hôm nay nếu SQLite thiếu) ───
    df_fiin = _get_from_fiinmarket(ticker)
    if not df_sql.empty and not df_fiin.empty:
        # Gộp: tránh trùng ngày
        combined = pd.concat([df_sql, df_fiin], ignore_index=True)
        # errors='coerce' để tránh OutOfBoundsDatetime khi date="0001-01-01"
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        combined = combined.dropna(subset=["date"])
        combined = combined[combined["date"] >= pd.Timestamp("2000-01-01")]
        combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
        combined = combined.drop_duplicates(subset=["date"], keep="last")
        return combined.sort_values("date").reset_index(drop=True)
    if not df_fiin.empty:
        return df_fiin

    # ── Layer 3: Provider cũ (VNDirect / KBS) — fallback lịch sử ───────────
    try:
        df = _provider.get_foreign_flow(ticker, days=days)
        return df
    except Exception as exc:
        log.warning("get_foreign_flow(%s, %s) lỗi toàn bộ: %s", ticker, period, exc)
        return pd.DataFrame()


@ttl_cache()
def get_foreign_flow_realtime(ticker: str) -> pd.DataFrame:
    """
    Lấy dữ liệu khối ngoại real-time hôm nay từ FiinMarket.

    Dùng cho dashboard phiên (không cần lịch sử).
    Nhanh hơn get_foreign_flow vì chỉ gọi FiinMarket (1 batch pre-cached).

    Returns:
        DataFrame 1 row: date, foreign_buy, foreign_sell, foreign_net
    """
    try:
        from analytics.ssi_foreign import fetch_foreign_flow
        return fetch_foreign_flow(ticker)
    except Exception as exc:
        log.warning("get_foreign_flow_realtime(%s): %s", ticker, exc)
        return pd.DataFrame()


@ttl_cache()
def get_top_foreign_net(exchange: str = "HOSE", top_n: int = 15) -> dict[str, pd.DataFrame]:
    """
    Top mã có khối ngoại mua ròng / bán ròng nhiều nhất hôm nay.

    Thứ tự ưu tiên: FiinMarket batch → provider cũ.

    Returns:
        {"buy": DataFrame, "sell": DataFrame}
        Mỗi DataFrame: ticker, net_vol (= net value VND từ FiinMarket)
    """
    try:
        from analytics.ssi_foreign import get_foreign_batch
        batch = get_foreign_batch("VNINDEX")
        if batch:
            rows = [
                {"ticker": t, "net_vol": d["foreign_net"], "net_val": d["foreign_net"]}
                for t, d in batch.items()
            ]
            all_df = pd.DataFrame(rows)
            buy_df  = all_df[all_df["net_vol"] > 0].nlargest(top_n, "net_vol").reset_index(drop=True)
            sell_df = all_df[all_df["net_vol"] < 0].nsmallest(top_n, "net_vol").reset_index(drop=True)
            return {"buy": buy_df, "sell": sell_df}
    except Exception as exc:
        log.warning("get_top_foreign_net FiinMarket lỗi: %s", exc)

    # Fallback provider cũ
    try:
        return _provider.get_top_foreign_net(VN30_TICKERS, top_n=top_n)
    except Exception as exc:
        log.warning("get_top_foreign_net provider lỗi: %s", exc)
        empty = pd.DataFrame(columns=["ticker", "net_vol", "net_val"])
        return {"buy": empty, "sell": empty}


# Cổ phiếu ngân hàng: trần sở hữu nước ngoài 30%; còn lại 49%
_BANK_TICKERS = frozenset({
    "ACB", "BID", "CTG", "HDB", "LPB", "MBB",
    "SHB", "SSB", "STB", "TCB", "TPB", "VCB", "VIB", "VPB",
})


@ttl_cache(ttl=3600)  # total shares thay đổi ít → cache 1 giờ
def _get_total_shares(ticker: str) -> float:
    """
    Lấy tổng cổ phiếu lưu hành từ vnstock Company overview (TCBS).
    Trả về 0.0 nếu không lấy được.
    TCBS trả về outstanding_share theo đơn vị triệu cp → ×1_000_000.
    """
    try:
        from vnstock.api.company import Company  # type: ignore
        df = Company(symbol=ticker, source="TCBS").overview()
        if df.empty:
            return 0.0
        col = next(
            (c for c in df.columns if "outstanding" in c.lower() or "issue" in c.lower()),
            None,
        )
        if col is None:
            return 0.0
        val = float(pd.to_numeric(df[col].iloc[0], errors="coerce") or 0)
        if val <= 0:
            return 0.0
        # TCBS trả về triệu cp (ví dụ ACB = 4400 → 4,400,000,000 cp)
        return val * 1_000_000 if val < 500_000 else val
    except Exception as exc:
        log.debug("_get_total_shares(%s): %s", ticker, exc)
        return 0.0


@ttl_cache()
def get_foreign_room(ticker: str) -> dict:
    """
    Lấy room nước ngoài còn lại của 1 mã từ KBS price_board.

    KBS trả về cột `foreign_room` (số cp còn được mua, đơn vị: cổ phiếu).
    Để tính %, cần tổng cp lưu hành → gọi thêm _get_total_shares().

    Returns:
        {ticker, max_room_pct, used_pct, remaining_pct, foreign_room_shares, alert}
        remaining_pct = None nếu không tính được % (thiếu total shares)
    """
    try:
        from vnstock.api.trading import Trading  # type: ignore
        board = Trading(source="KBS").price_board(symbols_list=[ticker])
        if board.empty:
            raise ValueError("price_board trống")

        row = board.iloc[0]
        # KBS: foreign_room = số cp khối ngoại còn được mua (không phải %)
        foreign_room_shares = int(
            pd.to_numeric(row.get("foreign_room", 0), errors="coerce") or 0
        )

        total_shares = _get_total_shares(ticker)
        max_pct = 30.0 if ticker.upper() in _BANK_TICKERS else 49.0

        if total_shares > 0:
            remaining_pct = round(foreign_room_shares / total_shares * 100, 2)
            used_pct      = round(max(max_pct - remaining_pct, 0.0), 2)
            return {
                "ticker":              ticker,
                "max_room_pct":        max_pct,
                "used_pct":            used_pct,
                "remaining_pct":       remaining_pct,
                "foreign_room_shares": foreign_room_shares,
                "alert":               remaining_pct < FOREIGN_ROOM_ALERT_PCT,
            }
        else:
            # Không lấy được total shares → trả về số cp thô, không có %
            log.info("get_foreign_room(%s): thiếu total_shares, trả về shares thô", ticker)
            return {
                "ticker":              ticker,
                "remaining_pct":       None,
                "foreign_room_shares": foreign_room_shares,
                "alert":               False,
            }
    except Exception as exc:
        log.warning("get_foreign_room(%s) lỗi: %s", ticker, exc)
        return {"ticker": ticker, "remaining_pct": None, "alert": False}


def summarize_foreign_flow(ticker: str, period: str = "1w") -> str:
    """Tóm tắt dòng tiền ngoại cho 1 mã (dùng bởi LangGraph tools)."""
    df = get_foreign_flow(ticker, period)
    room = get_foreign_room(ticker)

    if df.empty:
        return f"Không có dữ liệu khối ngoại cho {ticker}."

    net_total = df["net_val"].sum() if "net_val" in df.columns else 0
    trend = "MUA RÒNG" if net_total > 0 else "BÁN RÒNG"
    room_str = (
        f"Room còn lại: {room['remaining_pct']}%"
        + (" ⚠️ GẦN ĐẦY" if room.get("alert") else "")
        if room.get("remaining_pct") is not None else "N/A"
    )

    return (
        f"[{ticker}] Khối ngoại {period}: {trend} | "
        f"Net val: {net_total/1e9:.1f} tỷ ({len(df)} ngày) | {room_str}"
    )

