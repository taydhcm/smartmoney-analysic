# SmartMoney Analysis — Progress & Architecture

> Cập nhật lần cuối: 2026-05-20  
> Deployed: https://tayd-smartmoney.streamlit.app  
> Repo: https://github.com/taydhcm/smartmoney-analysic (branch `main`)

---

## 1. Tech Stack

| Layer | Công nghệ |
|---|---|
| UI Framework | Streamlit 1.57.0 |
| Charting | Plotly |
| Market Data | vnstock 4.0.4 (Quote/Trading/Listing) |
| News / RSS | feedparser + CafeF RSS feeds |
| Sentiment | underthesea (Vietnamese NLP) |
| AI Agent | LangChain + LangGraph + Groq (free LLM) |
| Cache | diskcache (TTL-based persistent cache) |
| Python | 3.12.10 local / 3.12.13 Streamlit Cloud |
| Deployment | Streamlit Cloud (auto-deploy on git push to main) |

---

## 2. Project Structure

```
smartmoney-analysic/
├── app.py                        # Streamlit entry point
├── requirements.txt
├── config/
│   ├── constants.py              # VN30_TICKERS, SECTOR_MAP, PERIOD_DAYS, trading hours
│   └── settings.py               # Env vars, secrets
├── data/                         # Data layer (tất cả đã rewrite cho vnstock 4.x)
│   ├── market_data.py            # OHLCV, index, breadth, ticker listing
│   ├── foreign_flow.py           # Khối ngoại buy/sell/net (KBS snapshot)
│   ├── proprietary_trading.py    # Tự doanh — hiện tại trả empty (không có nguồn)
│   ├── sector_data.py            # Sector rotation, heatmap
│   ├── volume_analysis.py        # Volume indicators
│   └── block_deals.py            # Giao dịch thoả thuận
├── news/
│   ├── cafef_scraper.py          # CafeF RSS → filter ticker + 7-day date filter
│   ├── sentiment.py              # aggregate_sentiment() dùng underthesea
│   ├── f319_scraper.py           # F319 forum (chưa active)
│   ├── fireant_scraper.py        # FireAnt (chưa active)
│   ├── hose_announcements.py     # HOSE công bố chính thức
│   └── vietstock_scraper.py      # Vietstock (chưa active)
├── pages/
│   ├── 01_market_overview.py     # VNINDEX chart + sector heatmap + breadth
│   ├── 02_foreign_flow.py        # Top foreign net mua/bán toàn thị trường
│   ├── 03_sector_analysis.py     # Sector rotation (batch API, fixed timeout)
│   ├── 04_stock_detail.py        # Chi tiết 1 mã: OHLCV + foreign flow + news
│   └── 05_ai_analysis.py         # AI agent phân tích (LangGraph)
├── ui/
│   └── components/
│       ├── charts.py             # Plotly chart components
│       └── tables.py             # Streamlit table components
├── agents/                       # LangGraph AI agents
├── analytics/                    # Technical indicators
└── utils/                        # Helpers
```

---

## 3. Data Sources & Limitations

### vnstock 4.0.4 — Breaking Changes
**TCBS bị xoá hoàn toàn** (HTTP 404) trong vnstock 4.0.4. Chỉ còn 2 provider hoạt động:

| Provider | Dùng cho | Class |
|---|---|---|
| **VCI (VCSC)** | OHLCV lịch sử (reliable) | `Quote(symbol, source='VCI').history()` |
| **KBS** | Real-time price board, foreign flow snapshot | `Trading(source='KBS').price_board(symbols_list=[...])` |
| **KBS** | Danh sách mã niêm yết | `Listing(source='KBS').symbols_by_exchange()` |

### KBS Price Board Columns (flat)
```
symbol, price_change, percent_change, close_price, ceiling_price, floor_price,
foreign_buy_volume, foreign_sell_volume, foreign_room, foreign_ownership_ratio,
volume_accumulated, total_value, open_price, high_price, low_price, average_price,
bid/ask prices
```

### Data Availability

| Data | Có | Nguồn | Ghi chú |
|---|---|---|---|
| OHLCV lịch sử | ✅ | VCI | Dùng `Quote(source='VCI').history()` |
| Index lịch sử | ✅ | VCI → KBS fallback | |
| Market breadth | ✅ | KBS price_board (VN30) | up/down/ceil/floor count |
| Foreign flow real-time | ✅ | KBS price_board | Chỉ có snapshot hôm nay, KHÔNG có lịch sử |
| Foreign flow lịch sử | ❌ | Không có | TCBS gone, KBS chỉ có hôm nay |
| Tự doanh (prop trading) | ❌ | Không có | `proprietary_trading.py` trả empty DataFrame |
| Danh sách ticker | ✅ | KBS Listing | `symbols_by_exchange()` |
| Tin tức | ✅ | CafeF RSS | 5 feeds, filter ticker + 7 ngày |

---

## 4. Commit History (tóm tắt)

| Commit | Nội dung |
|---|---|
| `763eca4` | **[LATEST]** Stock detail: foreign flow + news 1 tuần |
| `bf2e40b` | Sector analysis: batch API 1 lần thay vì ~50 lần (fix timeout) |
| `a8b0ae9` | Migrate toàn bộ data layer từ TCBS → VCI/KBS (vnstock 4.x) |
| `2187848` | Security: xoá secret khỏi .env.example |
| `c234cf9` | Initial commit |

---

## 5. Các vấn đề đã giải quyết

### ✅ TCBS API removed (vnstock 4.0.4)
- **Vấn đề**: `from vnstock import Vnstock` + TCBS → HTTP 404
- **Giải pháp**: Dùng `Quote(source='VCI')`, `Trading(source='KBS')`, `Listing(source='KBS')`
- **Files**: `data/market_data.py`, `data/foreign_flow.py`, `data/proprietary_trading.py`

### ✅ Sector analysis timeout (5+ phút)
- **Vấn đề**: ~50 mã × 1 API call = quá chậm, Streamlit timeout
- **Giải pháp**: Gom tất cả ticker → 1 batch KBS `price_board()` call → lookup dict
- **File**: `data/sector_data.py`
- **Score formula**: `foreign_net_val_norm × 0.70 + avg_rel_vol_norm × 0.30`

### ✅ Foreign flow chart chỉ hiện 1 ngày
- **Vấn đề**: Chart bar hiện dữ liệu hôm nay nhưng x-axis không rõ context
- **Giải pháp**: `foreign_flow_bar_chart(df, ticker, days=7)` — force x-axis range = 7 ngày, thêm buy/sell overlay bars
- **File**: `ui/components/charts.py`

### ✅ News section chỉ hiện 5 bài, không giới hạn ngày
- **Vấn đề**: `news[:5]` hard-cap, không lọc theo ngày
- **Giải pháp**: `search_cafef_news(ticker, limit=100, days=7)` — parse RSS `published` date, filter 7 ngày, hiện tất cả
- **File**: `news/cafef_scraper.py`, `pages/04_stock_detail.py`

---

## 6. Trạng thái từng trang

| Trang | Status | Ghi chú |
|---|---|---|
| 01 Market Overview | ✅ Hoạt động | VNINDEX chart, sector heatmap, market breadth từ KBS |
| 02 Foreign Flow | ✅ Hoạt động | Top mua/bán KBS, chỉ có snapshot hôm nay |
| 03 Sector Analysis | ✅ Hoạt động | Fixed timeout, batch API |
| 04 Stock Detail | ✅ Hoạt động | OHLCV + foreign flow 7 ngày + news 7 ngày |
| 05 AI Analysis | ⚠️ Chưa test đầy đủ | LangGraph + Groq |

---

## 7. Hướng phát triển tiếp theo

### Cần làm ngay
- [ ] **SSI Fast Connect API** — tích hợp để lấy foreign flow lịch sử thực sự
  - Lấy `consumerID` + `consumerSecret` từ: https://iboard.ssi.com.vn/support/api-service/management
  - `pip install ssi-fc-data`
  - Viết lại `data/foreign_flow.py` để dùng `client.daily_stock_price()` từ SSI FC
  - Lưu credentials vào `.streamlit/secrets.toml` (Streamlit Cloud)

### Nice to have
- [ ] Tự doanh (prop trading) — cần nguồn data (Simplize.vn ~500k/tháng hoặc SSI FC nếu có)
- [ ] TCBS REST API trực tiếp (bypass vnstock) — thử `https://apipubaws.tcbs.com.vn/` vẫn còn sống
- [ ] DNSE Entrade API — miễn phí, có foreign flow lịch sử

---

## 8. SSI Fast Connect — Hướng dẫn tích hợp

### Lấy credentials
1. Đăng nhập: https://iboard.ssi.com.vn
2. Support → API Service → Management → Tạo API key
3. Nhận `consumerID` + `consumerSecret`

### Cài đặt
```bash
pip install ssi-fc-data
```

### Thêm vào secrets.toml
```toml
# .streamlit/secrets.toml
SSI_CONSUMER_ID = "your_consumer_id"
SSI_CONSUMER_SECRET = "your_consumer_secret"
```

### Code mẫu
```python
# data/ssi_client.py
import streamlit as st
from ssi_fc_data import fc_md_client, model

class _SSIConfig:
    auth_type      = 'Bearer'
    consumerID     = st.secrets["SSI_CONSUMER_ID"]
    consumerSecret = st.secrets["SSI_CONSUMER_SECRET"]
    url            = 'https://fc-data.ssi.com.vn/'
    stream_url     = 'https://fc-data.ssi.com.vn/'

_config = _SSIConfig()
_client = fc_md_client.MarketDataClient(_config)

def get_daily_stock_price(ticker: str, from_date: str, to_date: str):
    """from_date, to_date format: 'DD/MM/YYYY'"""
    req = model.daily_stock_price(ticker, from_date, to_date, 1, 100, 'hose')
    return _client.daily_stock_price(_config, req)
```

---

## 9. Môi trường & Secrets

```
.env (local)                    → GROQ_API_KEY, OPENAI_API_KEY (optional)
.streamlit/secrets.toml (local) → GROQ_API_KEY (dùng cho Streamlit)
Streamlit Cloud Secrets         → Cùng các key trên, thêm SSI khi có
```

**Không bao giờ commit**: `.env`, `secrets.toml`, `ssi_config.py`
