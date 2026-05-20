"""
data/foreign_flow.py
Phân tích dòng tiền khối ngoại (foreign investor flows).
Dữ liệu từ KBS price_board (thời gian thực tích lũy trong ngày).

Ghi chú: vnstock 4.x không còn endpoint lịch sử khối ngoại (TCBS đã bị xóa).
KBS price_board cung cấp dữ liệu HIỆN TẠI của ngày giao dịch (tích lũy từ đầu phiên).
Các cột:
    - foreign_buy_volume   : KL NN mua (shares)
    - foreign_sell_volume  : KL NN bán (shares)
    - foreign_room         : Room ngoại còn lại (raw, tỷ đồng hoặc %)
    - close_price          : Giá khớp hiện tại (VND)
"""

from __future__ import annotations

import pandas as pd
from datetime import date

from config.constants import PERIOD_DAYS, FOREIGN_ROOM_ALERT_PCT
from utils.cache import ttl_cache
from utils.logger import get_logger

log = get_logger(__name__)


def _get_kbs_board(symbols: list[str]) -> pd.DataFrame:
    """Lấy KBS price_board cho danh sách mã. Trả về DataFrame phẳng."""
    try:
        from vnstock.api.trading import Trading  # type: ignore
        board = Trading(source="KBS").price_board(symbols_list=symbols)
        return board
    except Exception as exc:
        log.warning("_get_kbs_board(%s) lỗi: %s", symbols, exc)
        return pd.DataFrame()


@ttl_cache()
def get_foreign_flow(ticker: str, period: str = "1m") -> pd.DataFrame:
    """
    Dữ liệu khối ngoại cho 1 mã (snapshot hiện tại của ngày giao dịch).
    Vì vnstock 4.x không có endpoint lịch sử ngoại, hàm này trả về
    1 dòng dữ liệu hôm nay với: date, buy_vol, sell_vol, net_vol, net_val.

    Dùng cho: sector_data (tính tổng net theo ngành), foreign_room.
    """
    try:
        board = _get_kbs_board([ticker])
        if board.empty:
            return pd.DataFrame()

        row = board.iloc[0]
        buy_vol  = float(pd.to_numeric(row.get("foreign_buy_volume",  0), errors="coerce") or 0)
        sell_vol = float(pd.to_numeric(row.get("foreign_sell_volume", 0), errors="coerce") or 0)
        net_vol  = buy_vol - sell_vol

        # Tính giá trị (VND) từ khối lượng × giá khớp
        close_px = float(pd.to_numeric(row.get("close_price", row.get("average_price", 0)), errors="coerce") or 0)
        net_val  = net_vol * close_px  # đơn vị VND

        df = pd.DataFrame([{
            "date":     date.today(),
            "buy_vol":  int(buy_vol),
            "sell_vol": int(sell_vol),
            "net_vol":  int(net_vol),
            "net_val":  net_val,
        }])
        df["date"] = pd.to_datetime(df["date"])
        return df

    except Exception as exc:
        log.warning("get_foreign_flow(%s) lỗi: %s", ticker, exc)
        return pd.DataFrame()


@ttl_cache()
def get_foreign_room(ticker: str) -> dict:
    """
    Lấy room nước ngoài còn lại của 1 mã (từ KBS price_board).
    Trả về: {ticker, max_room_pct, used_pct, remaining_pct, alert}
    """
    try:
        board = _get_kbs_board([ticker])
        if board.empty:
            raise ValueError("price_board trống")

        row = board.iloc[0]
        # KBS: foreign_room là room còn lại (dạng raw, có thể là tỷ đồng hoặc số cổ phần)
        # foreign_ownership_ratio là tỷ lệ sở hữu hiện tại (0–100)
        fp = float(pd.to_numeric(row.get("foreign_ownership_ratio", 0), errors="coerce") or 0)
        used_pct = fp if fp > 1 else fp * 100

        max_pct = 49.0  # Giới hạn mặc định HOSE
        remaining = max_pct - used_pct
        return {
            "ticker":        ticker,
            "max_room_pct":  round(max_pct, 2),
            "used_pct":      round(used_pct, 2),
            "remaining_pct": round(remaining, 2),
            "alert":         remaining < FOREIGN_ROOM_ALERT_PCT,
        }
    except Exception as exc:
        log.warning("get_foreign_room(%s) lỗi: %s", ticker, exc)
        return {"ticker": ticker, "remaining_pct": None, "alert": False}


@ttl_cache()
def get_top_foreign_net(exchange: str = "HOSE", top_n: int = 10) -> dict[str, pd.DataFrame]:
    """
    Top mã có khối ngoại mua ròng và bán ròng nhiều nhất hôm nay.
    Dùng KBS price_board với VN30 làm proxy (dữ liệu tích lũy trong phiên).
    Trả về: {"buy": DataFrame, "sell": DataFrame}
    """
    try:
        from config.constants import VN30_TICKERS

        board = _get_kbs_board(VN30_TICKERS)
        if board.empty:
            return {"buy": pd.DataFrame(), "sell": pd.DataFrame()}

        board = board.copy()

        # KBS cột phẳng: symbol, foreign_buy_volume, foreign_sell_volume, close_price
        if "foreign_buy_volume" not in board.columns or "foreign_sell_volume" not in board.columns:
            log.warning("get_top_foreign_net: không có cột foreign_buy/sell_volume. Columns: %s", list(board.columns))
            return {"buy": pd.DataFrame(), "sell": pd.DataFrame()}

        board["buy_vol"]  = pd.to_numeric(board["foreign_buy_volume"],  errors="coerce").fillna(0)
        board["sell_vol"] = pd.to_numeric(board["foreign_sell_volume"], errors="coerce").fillna(0)
        board["net_vol"]  = board["buy_vol"] - board["sell_vol"]

        # Tính giá trị VND (dùng để hiển thị tỷ đồng)
        close_col = next((c for c in board.columns if c in ("close_price", "average_price")), None)
        if close_col:
            board["price"] = pd.to_numeric(board[close_col], errors="coerce").fillna(0)
        else:
            board["price"] = 0
        board["net_val"] = board["net_vol"] * board["price"]

        # Lấy cột ticker
        ticker_col = next(
            (c for c in board.columns if c.lower() in ("symbol", "ticker", "code")),
            board.columns[0]
        )
        board["ticker"] = board[ticker_col]

        df = board[["ticker", "net_vol", "net_val"]].dropna(subset=["net_val"])
        return {
            "buy":  df.nlargest(top_n,  "net_val").reset_index(drop=True),
            "sell": df.nsmallest(top_n, "net_val").reset_index(drop=True),
        }
    except Exception as exc:
        log.warning("get_top_foreign_net lỗi: %s", exc)
        return {"buy": pd.DataFrame(), "sell": pd.DataFrame()}


def summarize_foreign_flow(ticker: str, period: str = "1m") -> str:
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
        f"[{ticker}] Khối ngoại hôm nay: {trend} | "
        f"Net vol: {net_total/1e9:.1f} tỷ | {room_str}"
    )
