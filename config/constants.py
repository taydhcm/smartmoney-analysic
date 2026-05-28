"""
config/constants.py
Hằng số đặc thù thị trường chứng khoán Việt Nam.
"""

from __future__ import annotations
import pytz

# ── Timezone ─────────────────────────────────────────────────────────────────
VN_TZ = pytz.timezone("Asia/Ho_Chi_Minh")

# ── Trading Hours (local time) ────────────────────────────────────────────────
TRADING_SESSIONS = {
    "morning_open":  "09:15",
    "morning_close": "11:30",
    "afternoon_open": "13:00",
    "atc_start":     "14:30",   # ATC – lệnh khớp cuối phiên
    "market_close":  "15:00",
}

# ── Settlement ────────────────────────────────────────────────────────────────
SETTLEMENT_CYCLE = 2   # T+2

# ── Exchanges ─────────────────────────────────────────────────────────────────
EXCHANGES = ["HOSE", "HNX", "UPCOM"]

PRICE_LIMIT_PCT: dict[str, float] = {
    "HOSE":  0.07,   # ±7%
    "HNX":   0.10,   # ±10%
    "UPCOM": 0.15,   # ±15%
}

# ── Indices ───────────────────────────────────────────────────────────────────
INDICES = {
    "HOSE":  "VNINDEX",
    "HNX":   "HNXIndex",
    "HNX30": "HNX30",
    "UPCOM": "UPCoMIndex",
    "VN30":  "VN30",
}

# ── VN30 Basket (cập nhật kỳ cơ cấu 02/2026) ────────────────────────────────
# Đã loại: BCM (01/2026), BVH (07/2025), POW (01/2026)
# Thêm mới: LPB (LienVietPostBank), BSR (Bình Sơn Refinery), VPL
VN30_TICKERS = [
    "ACB", "BID", "BSR", "CTG", "FPT", "GAS", "GVR",
    "HDB", "HPG", "LPB", "MBB", "MSN", "MWG", "PLX", "SAB",
    "SHB", "SSB", "SSI", "STB", "TCB", "TPB", "VCB", "VHM",
    "VIB", "VIC", "VJC", "VNM", "VPB", "VPL", "VRE",
]

# ── Sector Map (nhóm ngành GICS-like cho HOSE/HNX) ────────────────────────────
SECTOR_MAP: dict[str, list[str]] = {
    "Ngân hàng":         ["ACB", "BID", "CTG", "HDB", "LPB", "MBB", "SHB", "STB",
                          "TCB", "TPB", "VCB", "VIB", "VPB", "SSB"],
    "Bất động sản":      ["VHM", "VIC", "VPL", "VRE", "BCM", "NVL", "DXG", "KDH",
                          "PDR", "DIG"],
    "Thép – Vật liệu":   ["HPG", "HSG", "NKG", "POM"],
    "Dầu khí – Năng lượng": ["GAS", "PLX", "POW", "PVD", "PVS", "BSR"],
    "Công nghệ":         ["FPT", "CMG", "VGI"],
    "Tiêu dùng":         ["MWG", "VNM", "SAB", "MSN", "PNJ"],
    "Bảo hiểm – Chứng khoán": ["BVH", "SSI", "VND", "HCM", "MBS"],
    "Cao su – Nông nghiệp":   ["GVR", "HAG", "HNG"],
    "Hàng không – Vận tải":   ["VJC", "HVN", "GMD", "PVT"],
    "Xây dựng – Hạ tầng":    ["VCG", "CTD", "HHV", "FCN"],
}

# ── Foreign Ownership Room Alert Threshold ────────────────────────────────────
FOREIGN_ROOM_ALERT_PCT = 5.0   # cảnh báo khi room còn < 5%

# ── Volume Anomaly Threshold ──────────────────────────────────────────────────
VOLUME_SURGE_RATIO = 2.0       # volume gấp đôi TB 20 phiên → anomaly

# ── Smart Money Score Weights ─────────────────────────────────────────────────
SIGNAL_WEIGHTS = {
    "foreign_net_buy":     0.35,
    "tu_doan_net_buy":     0.20,
    "volume_surge":        0.20,
    "news_sentiment":      0.15,
    "accumulation_pattern": 0.10,
}

# ── Period shortcuts ──────────────────────────────────────────────────────────
PERIOD_LABELS: dict[str, str] = {
    "1w":  "1 tuần",
    "2w":  "2 tuần",
    "1m":  "1 tháng",
    "3m":  "3 tháng",
    "6m":  "6 tháng",
    "12m": "12 tháng",
}

PERIOD_DAYS: dict[str, int] = {
    "1w": 5,
    "2w": 10,
    "1m": 22,
    "3m": 66,
    "6m": 130,
    "12m": 260,
}
