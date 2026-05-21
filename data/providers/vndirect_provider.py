"""data/providers/vndirect_provider.py
VNDirect FINFO API — dữ liệu lịch sử khối ngoại miễn phí, không cần key.

Endpoint:
  GET https://finfo-api.vndirect.com.vn/v4/stock_prices
      ?q=code:{TICKER}~date:gte:{YYYY-MM-DD}~date:lte:{YYYY-MM-DD}
      &sort=date:asc&size=30
      &fields=code,date,foreignBuyVolume,foreignSellVolume,
               foreignNetVolume,foreignBuyValue,foreignSellValue,foreignNetValue

Response schema (mỗi phần tử trong data[]):
  {
    "code": "VIC",
    "date": "2026-05-20",
    "foreignBuyVolume":  123456,
    "foreignSellVolume": 98765,
    "foreignNetVolume":  24691,       # = buy - sell (cổ phần)
    "foreignBuyValue":   5617548000.0,
    "foreignSellValue":  4493817500.0,
    "foreignNetValue":   1123730500.0  # VND
  }

Lưu ý:
- API không cần authentication
- Có thể timeout nếu server chậm → dùng retry
- Nếu fail hoàn toàn → trả về DataFrame rỗng để composite provider dùng fallback
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import date, timedelta

import pandas as pd

from utils.logger import get_logger
from .base import FlowProvider

log = get_logger(__name__)

_BASE = "https://finfo-api.vndirect.com.vn"
_TIMEOUT  = 12      # giây, mỗi lần thử
_RETRIES  = 2       # số lần retry khi timeout
_FIELDS_FF = (
    "code,date"
    ",foreignBuyVolume,foreignSellVolume,foreignNetVolume"
    ",foreignBuyValue,foreignSellValue,foreignNetValue"
)
_FIELDS_PROP = (
    "code,date"
    ",propBuyVolume,propSellVolume,propNetVolume"
    ",propBuyValue,propSellValue,propNetValue"
)


def _http_get(url: str) -> dict | None:
    """HTTP GET với retry. Trả về dict JSON hoặc None nếu fail."""
    for attempt in range(1, _RETRIES + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "Accept": "application/json",
                    "Referer": "https://vndirect.com.vn/",
                },
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
                return json.loads(r.read())
        except Exception as exc:
            log.warning("[VNDirect] attempt %d/%d failed: %s", attempt, _RETRIES, exc)
            if attempt < _RETRIES:
                time.sleep(1.5)
    return None


def _build_url(ticker: str, from_date: date, to_date: date, fields: str) -> str:
    q = f"code:{ticker.upper()}~date:gte:{from_date}~date:lte:{to_date}"
    return (
        f"{_BASE}/v4/stock_prices"
        f"?q={q}"
        f"&sort=date:asc"
        f"&size=60"
        f"&fields={fields}"
    )


def _parse_flow(data_list: list[dict], buy_key: str, sell_key: str,
                val_key: str) -> pd.DataFrame:
    """Chuyển list record từ VNDirect → DataFrame chuẩn."""
    rows = []
    for item in data_list:
        try:
            buy_vol  = int(float(item.get(buy_key,  0) or 0))
            sell_vol = int(float(item.get(sell_key, 0) or 0))
            net_vol  = int(float(item.get("foreignNetVolume" if buy_key.startswith("foreign") else "propNetVolume", 0) or 0))
            net_val  = float(item.get(val_key, 0) or 0)
            rows.append({
                "date":     pd.to_datetime(item["date"]),
                "buy_vol":  buy_vol,
                "sell_vol": sell_vol,
                "net_vol":  net_vol,
                "net_val":  net_val,
            })
        except Exception as exc:
            log.debug("[VNDirect] skip row %s: %s", item, exc)
    if not rows:
        return FlowProvider._empty_flow()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


class VNDirectProvider(FlowProvider):
    """
    Provider sử dụng VNDirect FINFO API.
    Cung cấp lịch sử N ngày cho khối ngoại.
    Tự doanh: thử lấy nếu server trả về field; nếu không thì trả empty.
    """

    @property
    def name(self) -> str:
        return "vndirect"

    @property
    def supports_history(self) -> bool:
        return True

    @property
    def supports_prop_trading(self) -> bool:
        return True   # thử — VNDirect có thể có prop fields

    def get_foreign_flow(self, ticker: str, days: int = 10) -> pd.DataFrame:
        to_dt   = date.today()
        from_dt = to_dt - timedelta(days=days + 5)   # thêm buffer cho ngày nghỉ

        url = _build_url(ticker, from_dt, to_dt, _FIELDS_FF)
        log.info("[VNDirect] foreign_flow %s: %s", ticker, url)

        resp = _http_get(url)
        if not resp or not resp.get("data"):
            log.warning("[VNDirect] foreign_flow %s: không có dữ liệu", ticker)
            return self._empty_flow()

        df = _parse_flow(
            resp["data"],
            buy_key="foreignBuyVolume",
            sell_key="foreignSellVolume",
            val_key="foreignNetValue",
        )

        # Lấy đúng `days` ngày giao dịch gần nhất
        if len(df) > days:
            df = df.tail(days).reset_index(drop=True)
        return df

    def get_prop_trading(self, ticker: str, days: int = 10) -> pd.DataFrame:
        to_dt   = date.today()
        from_dt = to_dt - timedelta(days=days + 5)

        url = _build_url(ticker, from_dt, to_dt, _FIELDS_PROP)
        log.info("[VNDirect] prop_trading %s: %s", ticker, url)

        resp = _http_get(url)
        if not resp or not resp.get("data"):
            log.info("[VNDirect] prop_trading %s: không có dữ liệu (field có thể không tồn tại)", ticker)
            return self._empty_flow()

        # Kiểm tra xem field có thực sự tồn tại không
        first = resp["data"][0] if resp["data"] else {}
        if "propBuyVolume" not in first and "propNetVolume" not in first:
            log.info("[VNDirect] prop_trading: VNDirect không có field propBuyVolume")
            return self._empty_flow()

        return _parse_flow(
            resp["data"],
            buy_key="propBuyVolume",
            sell_key="propSellVolume",
            val_key="propNetValue",
        )

    def get_top_foreign_net(
        self, tickers: list[str], top_n: int = 10
    ) -> dict[str, pd.DataFrame]:
        """
        Lấy top mua/bán ròng hôm nay từ VNDirect.
        Batch: 1 request per ticker — dùng hôm nay only để tránh quá nhiều request.
        """
        today = date.today()
        rows = []
        for tkr in tickers:
            url = _build_url(tkr, today, today, _FIELDS_FF)
            resp = _http_get(url)
            if not resp or not resp.get("data"):
                continue
            item = resp["data"][-1]  # lấy record mới nhất
            try:
                net_vol = int(float(item.get("foreignNetVolume", 0) or 0))
                net_val = float(item.get("foreignNetValue", 0) or 0)
                rows.append({"ticker": tkr, "net_vol": net_vol, "net_val": net_val})
            except Exception:
                pass

        if not rows:
            return self._empty_top()

        df = pd.DataFrame(rows)
        return {
            "buy":  df.nlargest(top_n,  "net_val").reset_index(drop=True),
            "sell": df.nsmallest(top_n, "net_val").reset_index(drop=True),
        }
