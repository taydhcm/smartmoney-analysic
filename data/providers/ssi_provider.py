"""data/providers/ssi_provider.py
SSI Fast Connect Provider — STUB (chưa kích hoạt).

Khi nào dùng:
  Sau khi lấy được consumerID + consumerSecret từ:
  https://iboard.ssi.com.vn/support/api-service/management

Cài đặt:
  pip install ssi-fc-data

Kích hoạt:
  1. Thêm vào .streamlit/secrets.toml:
       SSI_CONSUMER_ID     = "your_id"
       SSI_CONSUMER_SECRET = "your_secret"
  2. Trong config/settings.py hoặc env:
       FLOW_PROVIDER = "ssi"
  3. Uncomment code bên dưới

Tham khảo:
  https://github.com/SSI-Securities-Corporation/python-fcdata
"""

from __future__ import annotations

import pandas as pd

from utils.logger import get_logger
from .base import FlowProvider

log = get_logger(__name__)


class SSIProvider(FlowProvider):
    """
    SSI Fast Connect provider.
    Hiện tại là stub — raise NotImplementedError để composite provider
    tự động fallback sang VNDirect hoặc KBS.

    Sau khi có credentials, implement các phương thức bên dưới.
    """

    def __init__(self) -> None:
        self._client = None
        self._config = None
        self._ready  = False
        self._init_client()

    def _init_client(self) -> None:
        """Khởi tạo SSI Fast Connect client từ secrets."""
        try:
            import streamlit as st
            consumer_id     = st.secrets.get("SSI_CONSUMER_ID", "")
            consumer_secret = st.secrets.get("SSI_CONSUMER_SECRET", "")
        except Exception:
            import os
            consumer_id     = os.getenv("SSI_CONSUMER_ID", "")
            consumer_secret = os.getenv("SSI_CONSUMER_SECRET", "")

        if not consumer_id or not consumer_secret:
            log.info("[SSI] Chưa có credentials — provider ở chế độ stub")
            return

        try:
            from ssi_fc_data import fc_md_client, model  # type: ignore

            class _Cfg:
                auth_type      = "Bearer"
                consumerID     = consumer_id
                consumerSecret = consumer_secret
                url            = "https://fc-data.ssi.com.vn/"
                stream_url     = "https://fc-data.ssi.com.vn/"

            self._config = _Cfg()
            self._client = fc_md_client.MarketDataClient(self._config)
            self._ready  = True
            log.info("[SSI] Fast Connect client khởi tạo thành công")
        except ImportError:
            log.warning("[SSI] ssi-fc-data chưa được cài: pip install ssi-fc-data")
        except Exception as exc:
            log.warning("[SSI] Lỗi khởi tạo client: %s", exc)

    @property
    def name(self) -> str:
        return "ssi"

    @property
    def supports_history(self) -> bool:
        return self._ready

    @property
    def supports_prop_trading(self) -> bool:
        return self._ready

    def get_foreign_flow(self, ticker: str, days: int = 10) -> pd.DataFrame:
        if not self._ready:
            raise NotImplementedError("SSI credentials chưa được cấu hình")

        # ── TODO: implement sau khi có credentials ──────────────────────────────
        # from datetime import date, timedelta
        # from ssi_fc_data import model
        #
        # to_dt   = date.today()
        # from_dt = to_dt - timedelta(days=days + 5)
        # fmt = "%d/%m/%Y"
        # req = model.daily_stock_price(
        #     ticker,
        #     from_dt.strftime(fmt), to_dt.strftime(fmt),
        #     1, 100, "hose"
        # )
        # raw = self._client.daily_stock_price(self._config, req)
        # # Parse raw response → DataFrame với schema chuẩn
        # # (cần kiểm tra response format thực tế từ SSI)
        # ─────────────────────────────────────────────────────────────────────────
        raise NotImplementedError("SSI get_foreign_flow chưa được implement")

    def get_prop_trading(self, ticker: str, days: int = 10) -> pd.DataFrame:
        if not self._ready:
            raise NotImplementedError("SSI credentials chưa được cấu hình")
        raise NotImplementedError("SSI get_prop_trading chưa được implement")

    def get_top_foreign_net(
        self, tickers: list[str], top_n: int = 10
    ) -> dict[str, pd.DataFrame]:
        if not self._ready:
            raise NotImplementedError("SSI credentials chưa được cấu hình")
        raise NotImplementedError("SSI get_top_foreign_net chưa được implement")
