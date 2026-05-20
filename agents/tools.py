"""
agents/tools.py
Tất cả @tool functions cho LangGraph agents.
Mỗi tool gọi vào data/news/analytics layer – không chứa business logic.
"""

from __future__ import annotations
from typing import Optional

from langchain_core.tools import tool


# ══════════════════════════════════════════════════════════════════════════════
#  DATA AGENT TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def tool_get_foreign_flow(ticker: str, period: str = "1m") -> str:
    """
    Lấy và tóm tắt dòng tiền khối ngoại (mua/bán/net) cho 1 mã cổ phiếu.
    period: '1w' | '2w' | '1m' | '3m'
    """
    from data.foreign_flow import summarize_foreign_flow
    return summarize_foreign_flow(ticker, period)


@tool
def tool_get_foreign_room(ticker: str) -> str:
    """
    Kiểm tra room nước ngoài còn lại của 1 mã.
    Cảnh báo khi room < 5%.
    """
    from data.foreign_flow import get_foreign_room
    room = get_foreign_room(ticker)
    if room.get("remaining_pct") is None:
        return f"Không có dữ liệu room ngoại cho {ticker}."
    alert = " ⚠️ GẦN ĐẦY ROOM!" if room.get("alert") else ""
    return (
        f"[{ticker}] Foreign room: max={room['max_room_pct']}% | "
        f"đang dùng={room['used_pct']}% | còn lại={room['remaining_pct']}%{alert}"
    )


@tool
def tool_get_tu_doan_flow(ticker: str, period: str = "1m") -> str:
    """
    Lấy và tóm tắt dòng tiền tự doanh (proprietary trading) cho 1 mã.
    Tự doanh mua ròng liên tục là tín hiệu smart money quan trọng.
    """
    from data.proprietary_trading import summarize_tu_doan
    return summarize_tu_doan(ticker, period)


@tool
def tool_get_volume_analysis(ticker: str, period: str = "1m") -> str:
    """
    Phân tích volume: OBV trend, MFI, relative volume, phiên đột biến.
    Dùng để xác nhận tín hiệu accumulation/distribution.
    """
    from data.market_data import get_ohlcv
    from data.volume_analysis import summarize_volume
    df = get_ohlcv(ticker, period)
    return summarize_volume(ticker, df)


@tool
def tool_get_top_foreign_net(exchange: str = "HOSE", top_n: int = 10) -> str:
    """
    Lấy top mã có khối ngoại mua ròng / bán ròng mạnh nhất toàn thị trường hôm nay.
    exchange: 'HOSE' | 'HNX' | 'UPCOM'
    """
    from data.foreign_flow import get_top_foreign_net
    result = get_top_foreign_net(exchange, top_n)
    buy_tickers  = result["buy"]["ticker"].tolist()  if not result["buy"].empty  else []
    sell_tickers = result["sell"]["ticker"].tolist() if not result["sell"].empty else []
    return (
        f"[{exchange}] Khối ngoại MUA RÒNG mạnh nhất: {', '.join(buy_tickers[:5])}\n"
        f"[{exchange}] Khối ngoại BÁN RÒNG mạnh nhất: {', '.join(sell_tickers[:5])}"
    )


@tool
def tool_detect_accumulation(ticker: str, period: str = "1m") -> str:
    """
    Phát hiện mẫu tích lũy/phân phối Wyckoff: Phase B, C (Spring), D, Distribution.
    Dùng kết hợp với volume và dòng tiền để xác nhận.
    """
    from data.market_data import get_ohlcv
    from analytics.accumulation_detection import get_accumulation_summary
    df = get_ohlcv(ticker, period)
    return get_accumulation_summary(ticker, df)


@tool
def tool_get_smart_money_score(ticker: str, period: str = "1m") -> str:
    """
    Tính Smart Money Score tổng hợp (0–100) cho 1 mã.
    Kết hợp: foreign flow, tự doanh, volume, sentiment, accumulation pattern.
    """
    from analytics.smart_money_signals import batch_score_tickers
    df = batch_score_tickers([ticker], period)
    if df.empty:
        return f"Không tính được score cho {ticker}."
    row = df.iloc[0]
    return (
        f"[{ticker}] Smart Money Score: {row['score']}/100 — {row['grade']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  NEWS AGENT TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def tool_get_news_sentiment(ticker: str) -> str:
    """
    Tổng hợp tin tức và sentiment từ CafeF, Vietstock cho 1 mã cổ phiếu.
    Trả về: tóm tắt sentiment (tích cực/tiêu cực/trung lập) + bài viết nổi bật.
    """
    from news.cafef_scraper import search_cafef_news
    from news.vietstock_scraper import search_vietstock_news
    from news.sentiment import aggregate_sentiment

    news = search_cafef_news(ticker, 10) + search_vietstock_news(ticker, 5)
    agg  = aggregate_sentiment(news)
    label_map = {"positive": "TÍCH CỰC 🟢", "negative": "TIÊU CỰC 🔴", "neutral": "TRUNG TÍNH ⚪"}
    return (
        f"[{ticker}] Tin tức sentiment: {label_map.get(agg['label'], agg['label'])} "
        f"(score={agg['avg_score']:+.2f}) | "
        f"{agg['bullish_count']} bài tích cực, {agg['bearish_count']} bài tiêu cực "
        f"/ {agg['article_count']} bài phân tích"
    )


@tool
def tool_get_forum_buzz(ticker: str) -> str:
    """
    Kiểm tra mức độ quan tâm và sentiment từ diễn đàn chứng khoán:
    f319, Fireant, XamVN. Bài nhiều lượt xem/like được tính trọng số cao hơn.
    """
    from news.f319_scraper import search_f319
    from news.fireant_scraper import get_fireant_posts
    from news.xamvn_scraper import search_xamvn
    from news.sentiment import aggregate_sentiment

    forum_news = (
        search_f319(ticker, 5) +
        get_fireant_posts(ticker, 10) +
        search_xamvn(ticker, 5)
    )

    if not forum_news:
        return f"[{ticker}] Không tìm thấy bài đăng trên diễn đàn."

    agg = aggregate_sentiment(forum_news)
    total_posts = len(forum_news)
    fireant_posts = [p for p in forum_news if p.get("source") == "fireant"]
    top_likes = max((p.get("likes", 0) for p in fireant_posts), default=0)

    return (
        f"[{ticker}] Diễn đàn buzz: {total_posts} bài | "
        f"Sentiment: {agg['label'].upper()} (score={agg['avg_score']:+.2f}) | "
        f"Fireant top likes: {top_likes}"
    )


@tool
def tool_get_corporate_announcements(ticker: str) -> str:
    """
    Lấy thông báo chính thức từ HOSE/HNX: cổ tức, phát hành thêm, BCTC, ĐHCĐ.
    Đây là catalyst quan trọng ảnh hưởng đến giá cổ phiếu.
    """
    from news.hose_announcements import get_announcements_by_ticker
    anns = get_announcements_by_ticker(ticker)
    if not anns:
        return f"[{ticker}] Không có thông báo chính thức gần đây."

    lines = [f"  - [{a['event_type'].upper()}] {a['title'][:80]}" for a in anns[:5]]
    return f"[{ticker}] Thông báo HOSE/HNX ({len(anns)} thông báo):\n" + "\n".join(lines)


@tool
def tool_get_market_news() -> str:
    """
    Lấy tin tức thị trường tổng quát (không theo ticker cụ thể).
    Bao gồm tin vĩ mô, chính sách, biến động chỉ số.
    """
    from news.cafef_scraper import fetch_cafef_rss
    from news.sentiment import analyze_news_list

    news = fetch_cafef_rss("thi_truong", limit=10) + fetch_cafef_rss("vi_mo", limit=5)
    analyzed = analyze_news_list(news)
    positive = sum(1 for n in analyzed if n.get("sentiment", {}).get("label") == "positive")
    negative = sum(1 for n in analyzed if n.get("sentiment", {}).get("label") == "negative")

    headlines = [n["title"][:70] for n in news[:5]]
    return (
        f"Tin tức thị trường ({len(news)} bài): {positive} tích cực / {negative} tiêu cực\n"
        + "\n".join(f"  • {h}" for h in headlines)
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SECTOR AGENT TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def tool_get_sector_flow(sector: str, period: str = "1m") -> str:
    """
    Phân tích dòng tiền cho 1 ngành: foreign net, tự doanh net, volume.
    Tên ngành: 'Ngân hàng' | 'Bất động sản' | 'Thép – Vật liệu' | 'Công nghệ' | v.v.
    """
    from data.sector_data import summarize_sector
    return summarize_sector(sector, period)


@tool
def tool_get_sector_top_picks(sector: str, period: str = "1m") -> str:
    """
    Top cổ phiếu trong ngành theo smart money score.
    """
    from data.sector_data import get_sector_top_picks
    df = get_sector_top_picks(sector, period, top_n=5)
    if df.empty:
        return f"Không đủ dữ liệu cho ngành {sector}."
    lines = [
        f"  {i+1}. {row['ticker']} – Ngoại: {row['foreign_net_val']/1e9:.1f}tỷ | "
        f"Tự doanh: {row['tu_doan_net_val']/1e9:.1f}tỷ | RelVol: {row['avg_rel_vol']}x"
        for i, row in df.iterrows()
    ]
    return f"Top picks ngành {sector} ({period}):\n" + "\n".join(lines)


@tool
def tool_get_sector_rotation(period: str = "1m") -> str:
    """
    Phân tích rotation dòng tiền giữa các ngành: ngành nào đang hút tiền / bị xả.
    Trả về bảng xếp hạng ngành theo Smart Money composite score.
    """
    from data.sector_data import get_sector_flow_summary
    df = get_sector_flow_summary(period)
    if df.empty:
        return "Không đủ dữ liệu sector rotation."

    top3    = df.head(3)["sector"].tolist()
    bottom3 = df.tail(3)["sector"].tolist()
    return (
        f"Sector Rotation ({period}):\n"
        f"  🟢 Dòng tiền VÀO: {' > '.join(top3)}\n"
        f"  🔴 Dòng tiền RA:  {' > '.join(bottom3)}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL REGISTRIES (dùng khi tạo agents)
# ══════════════════════════════════════════════════════════════════════════════

DATA_AGENT_TOOLS = [
    tool_get_foreign_flow,
    tool_get_foreign_room,
    tool_get_tu_doan_flow,
    tool_get_volume_analysis,
    tool_get_top_foreign_net,
    tool_detect_accumulation,
    tool_get_smart_money_score,
]

NEWS_AGENT_TOOLS = [
    tool_get_news_sentiment,
    tool_get_forum_buzz,
    tool_get_corporate_announcements,
    tool_get_market_news,
]

SECTOR_AGENT_TOOLS = [
    tool_get_sector_flow,
    tool_get_sector_top_picks,
    tool_get_sector_rotation,
]
