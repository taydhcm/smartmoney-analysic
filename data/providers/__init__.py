"""data/providers/__init__.py
Provider registry cho dòng tiền khối ngoại & tự doanh.

Thứ tự ưu tiên:
  1. VNDirect FINFO API  — lịch sử N ngày, miễn phí, không cần key
  2. KBS Snapshot        — dữ liệu hôm nay + tích lũy lên đĩa
  3. SSI Fast Connect    — (stub) cài khi có credentials từ iBoard SSI

Để chuyển provider, thay đổi ACTIVE_PROVIDER hoặc để mặc định "auto"
(auto = thử VNDirect trước, nếu fail thì dùng KBS).
"""

from __future__ import annotations
from .base import FlowProvider

def get_provider(name: str = "auto") -> FlowProvider:
    """
    Factory trả về FlowProvider theo tên.

    Tên hợp lệ: "auto", "vndirect", "kbs", "ssi"
    """
    if name == "ssi":
        from .ssi_provider import SSIProvider
        return SSIProvider()

    if name == "kbs":
        from .kbs_provider import KBSProvider
        return KBSProvider()

    if name == "vndirect":
        from .vndirect_provider import VNDirectProvider
        return VNDirectProvider()

    # "auto": VNDirect với KBS làm fallback
    from .composite_provider import CompositeProvider
    from .vndirect_provider import VNDirectProvider
    from .kbs_provider import KBSProvider
    return CompositeProvider([VNDirectProvider(), KBSProvider()])
