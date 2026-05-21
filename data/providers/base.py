"""data/providers/base.py
Abstract base class cho tất cả flow providers.

Mọi provider (VNDirect, KBS, SSI, ...) phải implement interface này.
Đảm bảo có thể swap provider mà không cần sửa code ở tầng trên.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class FlowProvider(ABC):
    """
    Interface chuẩn cho dữ liệu dòng tiền.

    DataFrame schema cho foreign_flow và prop_trading:
        date      : datetime64[ns]  (ngày giao dịch)
        buy_vol   : int64           (KL mua, cổ phần)
        sell_vol  : int64           (KL bán, cổ phần)
        net_vol   : int64           (KL mua ròng = buy - sell)
        net_val   : float64         (Giá trị mua ròng, VND)

    DataFrame schema cho top_foreign_net:
        ticker    : str
        net_vol   : int64
        net_val   : float64
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tên provider (để log và debug)."""

    @property
    def supports_history(self) -> bool:
        """True nếu provider có dữ liệu lịch sử (>1 ngày)."""
        return False

    @property
    def supports_prop_trading(self) -> bool:
        """True nếu provider có dữ liệu tự doanh."""
        return False

    @abstractmethod
    def get_foreign_flow(self, ticker: str, days: int = 10) -> pd.DataFrame:
        """
        Trả về lịch sử dòng tiền khối ngoại cho `ticker` trong `days` ngày gần nhất.
        Kết quả sắp xếp tăng dần theo date.
        """

    def get_prop_trading(self, ticker: str, days: int = 10) -> pd.DataFrame:
        """
        Trả về lịch sử dòng tiền tự doanh cho `ticker` trong `days` ngày gần nhất.
        Mặc định trả về DataFrame rỗng — override trong provider hỗ trợ.
        """
        return pd.DataFrame()

    @abstractmethod
    def get_top_foreign_net(
        self, tickers: list[str], top_n: int = 10
    ) -> dict[str, pd.DataFrame]:
        """
        Top mã có khối ngoại mua ròng / bán ròng hôm nay.
        Trả về {"buy": DataFrame, "sell": DataFrame}.
        """

    @staticmethod
    def _empty_flow() -> pd.DataFrame:
        """DataFrame rỗng với đúng schema."""
        return pd.DataFrame(columns=["date", "buy_vol", "sell_vol", "net_vol", "net_val"])

    @staticmethod
    def _empty_top() -> dict[str, pd.DataFrame]:
        empty = pd.DataFrame(columns=["ticker", "net_vol", "net_val"])
        return {"buy": empty, "sell": empty}
