"""data/providers/kbs_provider.py
KBS Real-time Snapshot Provider + Disk Persistence.

Chiến lược:
1. Mỗi lần gọi get_foreign_flow(), lấy snapshot hôm nay từ KBS price_board
2. Lưu snapshot vào file JSON theo ngày: .cache/ff_snapshots/{TICKER}/{YYYY-MM-DD}.json
3. Khi cần lịch sử N ngày, đọc các file đã lưu và ghép lại
4. Kết quả: ban đầu chỉ có 1 ngày, sau vài tuần có lịch sử đầy đủ

Đây là giải pháp thực dụng nhất khi không có API lịch sử.
Dữ liệu tích lũy tự động mỗi ngày ứng dụng được mở.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from utils.logger import get_logger
from .base import FlowProvider

log = get_logger(__name__)

_SNAPSHOT_DIR = Path(__file__).resolve().parent.parent.parent / ".cache" / "ff_snapshots"


def _snapshot_path(ticker: str, dt: date) -> Path:
    d = _SNAPSHOT_DIR / ticker.upper()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{dt.isoformat()}.json"


def _save_snapshot(ticker: str, dt: date, row: dict) -> None:
    """Lưu 1 ngày snapshot xuống đĩa (idempotent)."""
    p = _snapshot_path(ticker, dt)
    if p.exists():
        return  # đã lưu rồi, không ghi đè
    try:
        p.write_text(json.dumps(row, default=str), encoding="utf-8")
    except Exception as exc:
        log.warning("[KBS] save_snapshot %s %s: %s", ticker, dt, exc)


def _load_snapshot(ticker: str, dt: date) -> dict | None:
    p = _snapshot_path(ticker, dt)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _kbs_board(symbols: list[str]) -> pd.DataFrame:
    try:
        from vnstock.api.trading import Trading  # type: ignore
        return Trading(source="KBS").price_board(symbols_list=symbols)
    except Exception as exc:
        log.warning("[KBS] price_board(%s) lỗi: %s", symbols, exc)
        return pd.DataFrame()


class KBSProvider(FlowProvider):
    """
    Provider dùng KBS price_board:
    - Snapshot hôm nay (real-time tích lũy trong phiên)
    - Tích lũy lịch sử lên đĩa (.cache/ff_snapshots/) tự động
    - Lịch sử sẽ phong phú dần theo thời gian dùng
    """

    @property
    def name(self) -> str:
        return "kbs"

    @property
    def supports_history(self) -> bool:
        return True  # qua persistence layer

    def get_foreign_flow(self, ticker: str, days: int = 10) -> pd.DataFrame:
        today = date.today()

        # 1) Lấy snapshot hôm nay và lưu xuống đĩa
        board = _kbs_board([ticker])
        if not board.empty:
            try:
                row = board.iloc[0]
                buy_vol  = int(float(row.get("foreign_buy_volume",  0) or 0))
                sell_vol = int(float(row.get("foreign_sell_volume", 0) or 0))
                net_vol  = buy_vol - sell_vol
                close_px = float(row.get("close_price", row.get("average_price", 0)) or 0)
                net_val  = net_vol * close_px

                today_record = {
                    "date":     today.isoformat(),
                    "buy_vol":  buy_vol,
                    "sell_vol": sell_vol,
                    "net_vol":  net_vol,
                    "net_val":  net_val,
                }
                _save_snapshot(ticker, today, today_record)
                log.info("[KBS] saved snapshot %s %s (buy=%d sell=%d)", ticker, today, buy_vol, sell_vol)
            except Exception as exc:
                log.warning("[KBS] parse snapshot %s: %s", ticker, exc)

        # 2) Đọc tất cả snapshots trong range
        rows = []
        for i in range(days + 5):   # thêm buffer
            dt = today - timedelta(days=i)
            snap = _load_snapshot(ticker, dt)
            if snap:
                snap["date"] = pd.to_datetime(snap["date"])
                rows.append(snap)

        if not rows:
            # Chỉ có hôm nay nếu không có gì lưu trữ
            if not board.empty:
                try:
                    r = board.iloc[0]
                    bv = int(float(r.get("foreign_buy_volume",  0) or 0))
                    sv = int(float(r.get("foreign_sell_volume", 0) or 0))
                    nv = bv - sv
                    px = float(r.get("close_price", r.get("average_price", 0)) or 0)
                    rows.append({
                        "date": pd.to_datetime(today),
                        "buy_vol": bv, "sell_vol": sv,
                        "net_vol": nv, "net_val": nv * px,
                    })
                except Exception:
                    pass

        if not rows:
            return self._empty_flow()

        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        # Lấy `days` ngày giao dịch gần nhất (bỏ weekends nếu không có data)
        if len(df) > days:
            df = df.tail(days).reset_index(drop=True)
        return df

    def get_top_foreign_net(
        self, tickers: list[str], top_n: int = 10
    ) -> dict[str, pd.DataFrame]:
        board = _kbs_board(tickers)
        if board.empty:
            return self._empty_top()

        board = board.copy()
        board["buy_vol"]  = pd.to_numeric(board.get("foreign_buy_volume"),  errors="coerce").fillna(0)
        board["sell_vol"] = pd.to_numeric(board.get("foreign_sell_volume"), errors="coerce").fillna(0)
        board["net_vol"]  = board["buy_vol"] - board["sell_vol"]

        close_col = next((c for c in board.columns if c in ("close_price", "average_price")), None)
        board["price"]   = pd.to_numeric(board[close_col], errors="coerce").fillna(0) if close_col else 0
        board["net_val"] = board["net_vol"] * board["price"]

        ticker_col = next(
            (c for c in board.columns if c.lower() in ("symbol", "ticker", "code")),
            board.columns[0],
        )
        board["ticker"] = board[ticker_col]

        df = board[["ticker", "net_vol", "net_val"]].dropna(subset=["net_val"])
        return {
            "buy":  df.nlargest(top_n,  "net_val").reset_index(drop=True),
            "sell": df.nsmallest(top_n, "net_val").reset_index(drop=True),
        }
