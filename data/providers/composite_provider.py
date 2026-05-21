"""data/providers/composite_provider.py
Composite Provider — thử từng provider theo thứ tự ưu tiên.

Nếu provider đầu tiên thất bại (exception hoặc DataFrame rỗng),
tự động chuyển sang provider tiếp theo.

Ví dụ: [VNDirectProvider, KBSProvider]
  → thử VNDirect trước, nếu timeout/fail → dùng KBS
"""

from __future__ import annotations

import pandas as pd

from utils.logger import get_logger
from .base import FlowProvider

log = get_logger(__name__)


class CompositeProvider(FlowProvider):
    """
    Chain nhiều providers theo thứ tự ưu tiên.
    Provider đầu tiên có kết quả hợp lệ (non-empty DataFrame) được dùng.
    """

    def __init__(self, providers: list[FlowProvider]) -> None:
        if not providers:
            raise ValueError("Cần ít nhất 1 provider")
        self._providers = providers

    @property
    def name(self) -> str:
        return " → ".join(p.name for p in self._providers)

    @property
    def supports_history(self) -> bool:
        return any(p.supports_history for p in self._providers)

    @property
    def supports_prop_trading(self) -> bool:
        return any(p.supports_prop_trading for p in self._providers)

    def get_foreign_flow(self, ticker: str, days: int = 10) -> pd.DataFrame:
        for provider in self._providers:
            try:
                df = provider.get_foreign_flow(ticker, days)
                if df is not None and not df.empty:
                    log.info("[Composite] foreign_flow %s: dùng %s (%d rows)",
                             ticker, provider.name, len(df))
                    return df
                log.info("[Composite] foreign_flow %s: %s trả rỗng, thử tiếp",
                         ticker, provider.name)
            except Exception as exc:
                log.warning("[Composite] foreign_flow %s: %s lỗi (%s), thử tiếp",
                            ticker, provider.name, exc)

        log.warning("[Composite] foreign_flow %s: tất cả providers thất bại", ticker)
        return self._empty_flow()

    def get_prop_trading(self, ticker: str, days: int = 10) -> pd.DataFrame:
        for provider in self._providers:
            if not provider.supports_prop_trading:
                continue
            try:
                df = provider.get_prop_trading(ticker, days)
                if df is not None and not df.empty:
                    log.info("[Composite] prop_trading %s: dùng %s", ticker, provider.name)
                    return df
            except Exception as exc:
                log.warning("[Composite] prop_trading %s: %s lỗi (%s)", ticker, provider.name, exc)

        return self._empty_flow()

    def get_top_foreign_net(
        self, tickers: list[str], top_n: int = 10
    ) -> dict[str, pd.DataFrame]:
        for provider in self._providers:
            try:
                result = provider.get_top_foreign_net(tickers, top_n)
                if result and not result["buy"].empty:
                    log.info("[Composite] top_foreign_net: dùng %s", provider.name)
                    return result
            except Exception as exc:
                log.warning("[Composite] top_foreign_net: %s lỗi (%s)", provider.name, exc)

        return self._empty_top()
