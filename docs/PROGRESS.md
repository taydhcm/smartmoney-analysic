# SmartMoney Analysis — Progress & Architecture

> Cập nhật lần cuối: 2026-05-21  
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
│   ├── foreign_flow.py           # Khối ngoại — dùng provider layer (VNDirect → KBS)
│   ├── proprietary_trading.py    # Tự doanh — dùng provider layer (VNDirect → empty)
│   ├── sector_data.py            # Sector rotation, heatmap
│   ├── volume_analysis.py        # Volume indicators
│   ├── block_deals.py            # Giao dịch thoả thuận
│   └── providers/                # Provider pattern cho flow data
│       ├── __init__.py           # get_provider(name) factory
│       ├── base.py               # FlowProvider ABC (interface chuẩn)
│       ├── vndirect_provider.py  # VNDirect FINFO API (lịch sử N ngày, miễn phí)
│       ├── kbs_provider.py       # KBS snapshot + tích lũy lên đĩa tự động
│       ├── composite_provider.py # Chain providers: thử lần lượt, fallback tự động
│       └── ssi_provider.py       # SSI Fast Connect stub (sẵn sàng khi có credentials)
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
| Foreign flow real-time | ✅ | KBS price_board | Snapshot hôm nay (tích lũy phiên) |
| Foreign flow lịch sử | ⚠️ | VNDirect → KBS | VNDirect nếu available; KBS tích lũy đĩa |
| Tự doanh (prop trading) | ⚠️ | VNDirect (thử) | Cần SSI FC để chắc chắn có data |
| Danh sách ticker | ✅ | KBS Listing | `symbols_by_exchange()` |
| Tin tức | ✅ | CafeF RSS | 5 feeds, filter ticker + 7 ngày |

### Provider Priority Chain (auto mode)
```
get_foreign_flow(ticker, period)
  └─ CompositeProvider
       ├─ 1. VNDirectProvider  → GET finfo-api.vndirect.com.vn/v4/stock_prices (lịch sử thực)
       └─ 2. KBSProvider       → price_board snapshot hôm nay + đọc .cache/ff_snapshots/
```

Chuyển sang SSI sau khi có credentials:
```bash
# .env hoặc Streamlit secrets
FLOW_PROVIDER=ssi
SSI_CONSUMER_ID=your_id
SSI_CONSUMER_SECRET=your_secret
```

---

## 4. Commit History (tóm tắt)

| Commit | Nội dung |
|---|---|
| `3b754e0` | **[LATEST]** Provider architecture: VNDirect → KBS fallback + SSI stub |
| `916d3b3` | docs: PROGRESS.md |
| `763eca4` | Stock detail: foreign flow + news 1 tuần |
| `bf2e40b` | Sector analysis: batch API 1 lần thay vì ~50 lần (fix timeout) |
| `a8b0ae9` | Migrate toàn bộ data layer từ TCBS → VCI/KBS (vnstock 4.x) |
| `c234cf9` | Initial commit |

---

## 5. Các vấn đề đã giải quyết

### ✅ TCBS API removed (vnstock 4.0.4)
- **Vấn đề**: `from vnstock import Vnstock` + TCBS → HTTP 404. TCBS REST API cũng dead hoàn toàn.
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

### ✅ Foreign flow & tự doanh — Provider Architecture
- **Vấn đề**: KBS chỉ có snapshot hôm nay, không có lịch sử. Không có nguồn tự doanh.
- **Giải pháp**: `data/providers/` — provider pattern với chain VNDirect → KBS + SSI stub
  - `VNDirectProvider`: VNDirect FINFO API, lịch sử N ngày, miễn phí, không cần key
  - `KBSProvider`: real-time snapshot + tự động tích lũy lên `.cache/ff_snapshots/` theo ngày
  - `CompositeProvider`: thử VNDirect trước, fallback KBS nếu timeout/fail
  - `SSIProvider`: stub sẵn sàng, chỉ cần thêm credentials vào `secrets.toml`
- **Chuyển provider**: set `FLOW_PROVIDER=ssi` trong env, không cần sửa code

---

## 6. Trạng thái từng trang

| Trang | Status | Ghi chú |
|---|---|---|
| 01 Market Overview | ✅ Hoạt động | VNINDEX chart, sector heatmap, market breadth từ KBS |
| 02 Foreign Flow | ✅ Hoạt động | Top mua/bán qua provider (VNDirect → KBS) |
| 03 Sector Analysis | ✅ Hoạt động | Fixed timeout, batch API |
| 04 Stock Detail | ✅ Hoạt động | OHLCV + foreign flow 7 ngày + news 7 ngày |
| 05 AI Analysis | ⚠️ Chưa test đầy đủ | LangGraph + Groq |

---

## 7. Hướng phát triển tiếp theo

### Cần làm ngay
- [ ] **Kích hoạt SSI Fast Connect** — provider stub đã sẵn sàng, chỉ cần credentials
  - Đăng ký nộp giấy tờ tại quầy SSI (mất ~2 tuần)
  - Sau khi có: thêm `SSI_CONSUMER_ID` + `SSI_CONSUMER_SECRET` vào `secrets.toml`
  - Set `FLOW_PROVIDER=ssi` trong Streamlit Cloud secrets
  - Implement `get_foreign_flow()` và `get_prop_trading()` trong `data/providers/ssi_provider.py`
- [ ] **Implement SSI `get_foreign_flow()`** trong `ssi_provider.py` (TODO đã có trong file)
- [ ] **Implement SSI `get_prop_trading()`** trong `ssi_provider.py` — lấy được tự doanh

### Nice to have
- [ ] Kiểm tra VNDirect finfo API trên Streamlit Cloud (có thể bị block local, OK trên cloud)
- [ ] Test page 05 AI Analysis (LangGraph + Groq) end-to-end
- [ ] Simplize.vn (~500k/tháng) nếu muốn tự doanh ngay mà không đợi SSI

---

## 8. Provider Architecture (data/providers/)

### Sơ đồ ưu tiên
```
auto mode (mặc định)
  └─ CompositeProvider
       ├─ [1] VNDirectProvider
       │     GET finfo-api.vndirect.com.vn/v4/stock_prices
       │     fields: foreignBuyVolume, foreignSellVolume, foreignNetValue
       │     Retry: 2 lần, timeout 12s mỗi lần
       │     → Lịch sử N ngày thực sự (nếu server phản hồi)
       └─ [2] KBSProvider (fallback)
             KBS price_board snapshot hôm nay
             + đọc .cache/ff_snapshots/{TICKER}/{YYYY-MM-DD}.json
             → Tích lũy lịch sử dần theo thời gian dùng
```

### Chuyển sang SSI (khi có credentials)
```toml
# .streamlit/secrets.toml
FLOW_PROVIDER       = "ssi"
SSI_CONSUMER_ID     = "your_id"
SSI_CONSUMER_SECRET = "your_secret"
```
Không cần sửa bất kỳ code nào khác. Chỉ implement phần TODO trong `data/providers/ssi_provider.py`.

### SSI Fast Connect — Bước đăng ký
1. Đăng nhập: https://iboard.ssi.com.vn
2. Support → API Service → Management → Tạo API key
3. Nhận `consumerID` + `consumerSecret`
4. `pip install ssi-fc-data` (thêm vào `requirements.txt`)

### KBS Snapshot Persistence
Mỗi lần app được mở, snapshot hôm nay tự động được ghi:
```
.cache/ff_snapshots/
  VIC/
    2026-05-21.json  → {date, buy_vol, sell_vol, net_vol, net_val}
    2026-05-22.json
    ...
```
Sau 2 tuần sử dụng sẽ có ~10 ngày lịch sử. Không cần API bên ngoài.

---

## 9. Môi trường & Secrets

```
.env (local)                    → GROQ_API_KEY, OPENAI_API_KEY (optional)
.streamlit/secrets.toml (local) → GROQ_API_KEY (dùng cho Streamlit)
Streamlit Cloud Secrets         → Cùng các key trên, thêm SSI khi có
```

**Không bao giờ commit**: `.env`, `secrets.toml`, `ssi_config.py`

---

## 10. Ghi chú API đã kiểm tra (2026-05-21)

| API | Status | Ghi chú |
|---|---|---|
| TCBS `apipubaws.tcbs.com.vn/stock-insight/` | ❌ Dead | HTTP 404 toàn bộ endpoint |
| DNSE `services.entrade.com.vn/chart-api/v2/ohlcs/stock` | ✅ Hoạt động | Chỉ có OHLCV, không có foreign flow |
| DNSE foreign flow endpoints | ❌ Dead | HTTP 401/404 |
| VNDirect `finfo-api.vndirect.com.vn/v4/stock_prices` | ⚠️ Timeout local | Có thể OK trên Streamlit Cloud |
| CafeF AJAX `s.cafef.vn/Ajax/.../GetForeignStatistic` | ❌ Dead | HTTP 404 |
| FireAnt REST v2 | ❌ Auth | HTTP 401, cần token |
| Simplize API | ❌ Dead | HTTP 404 |
| SSI Fast Connect | ⏳ Chờ credentials | Stub đã implement sẵn |
