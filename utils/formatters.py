"""
utils/formatters.py
Format số, tiền tệ VND, phần trăm cho display trên Streamlit.
"""

from __future__ import annotations


def fmt_volume(vol: float | int) -> str:
    """1_500_000 → '1.5M cổ phiếu'"""
    if vol >= 1_000_000:
        return f"{vol / 1_000_000:.1f}M CP"
    if vol >= 1_000:
        return f"{vol / 1_000:.0f}K CP"
    return str(int(vol))


def fmt_value_vnd(val: float) -> str:
    """15_000_000_000 → '15.0 tỷ'"""
    if abs(val) >= 1_000_000_000_000:
        return f"{val / 1_000_000_000_000:.2f} nghìn tỷ"
    if abs(val) >= 1_000_000_000:
        return f"{val / 1_000_000_000:.1f} tỷ"
    if abs(val) >= 1_000_000:
        return f"{val / 1_000_000:.0f} triệu"
    return f"{val:,.0f} đ"


def fmt_pct(val: float, decimals: int = 2) -> str:
    """0.0725 → '+7.25%'"""
    sign = "+" if val > 0 else ""
    return f"{sign}{val * 100:.{decimals}f}%"


def fmt_price(price: float) -> str:
    """Giá cổ phiếu VN (nghìn đồng). 45_200 → '45,200'"""
    return f"{price:,.0f}"


def color_sign(val: float) -> str:
    """Trả về màu Streamlit markdown theo dấu số."""
    if val > 0:
        return "green"
    if val < 0:
        return "red"
    return "gray"


def signed_str(val: float, fmt_fn=fmt_value_vnd) -> str:
    """Thêm dấu + cho số dương."""
    prefix = "+" if val > 0 else ""
    return f"{prefix}{fmt_fn(val)}"
