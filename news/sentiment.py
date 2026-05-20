"""
news/sentiment.py
Phân tích sentiment tiếng Việt cho tin tức & bài đăng diễn đàn.
Sử dụng underthesea (VN NLP) + keyword weighting cho context chứng khoán.
"""

from __future__ import annotations
import re
from functools import lru_cache

from utils.logger import get_logger

log = get_logger(__name__)

# ── Keyword dictionaries (domain-specific stock market VN) ────────────────────
BULLISH_KEYWORDS = [
    "tăng mạnh", "bứt phá", "đột biến", "tăng trưởng", "lợi nhuận tăng",
    "kết quả tốt", "vượt kế hoạch", "cổ tức cao", "mua ròng", "tích lũy",
    "dòng tiền vào", "khối ngoại mua", "room còn", "uptrend", "hỗ trợ mạnh",
    "triển vọng tích cực", "kết quả khả quan", "tăng đột biến", "phá đỉnh",
]

BEARISH_KEYWORDS = [
    "giảm mạnh", "lao dốc", "thua lỗ", "lợi nhuận giảm", "bán ròng",
    "phân phối", "dòng tiền ra", "khối ngoại bán", "căng thẳng", "rủi ro",
    "áp lực bán", "downtrend", "cảnh báo", "lo ngại", "kinh doanh xấu",
    "nợ xấu", "vi phạm", "điều tra", "phạt", "vỡ nợ", "margin call",
]

NEUTRAL_KEYWORDS = [
    "đi ngang", "ổn định", "giao dịch cầm chừng", "chờ đợi", "sideways",
]


@lru_cache(maxsize=1)
def _load_underthesea():
    """Lazy load underthesea để tránh slow import khi không cần."""
    try:
        from underthesea import sentiment as _sentiment  # type: ignore
        return _sentiment
    except ImportError:
        log.warning("underthesea chưa cài. Dùng keyword fallback.")
        return None


def _keyword_score(text: str) -> float:
    """
    Keyword-based score: [-1.0, +1.0].
    Bullish keywords → dương, bearish → âm.
    """
    text_lower = text.lower()
    bull = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
    bear = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total


def analyze_sentiment(text: str) -> dict:
    """
    Phân tích sentiment 1 văn bản.
    Trả về:
        {score: float [-1,1], label: 'positive'|'negative'|'neutral', method: str}
    """
    if not text or not text.strip():
        return {"score": 0.0, "label": "neutral", "method": "empty"}

    # Thử underthesea
    sentiment_fn = _load_underthesea()
    if sentiment_fn:
        try:
            result = sentiment_fn(text)
            # underthesea trả về "positive"/"negative"/"neutral"
            label = result if isinstance(result, str) else str(result)
            score_map = {"positive": 0.7, "negative": -0.7, "neutral": 0.0}
            kw_score = _keyword_score(text)
            # Kết hợp: 60% underthesea + 40% keyword
            base_score = score_map.get(label, 0.0)
            final_score = round(0.6 * base_score + 0.4 * kw_score, 3)
            return {
                "score":  max(-1.0, min(1.0, final_score)),
                "label":  label,
                "method": "underthesea+keyword",
            }
        except Exception as exc:
            log.debug("underthesea lỗi: %s – dùng keyword fallback", exc)

    # Fallback: keyword only
    score = _keyword_score(text)
    label = "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral")
    return {"score": round(score, 3), "label": label, "method": "keyword"}


def analyze_news_list(news_items: list[dict]) -> list[dict]:
    """
    Thêm sentiment vào danh sách tin tức.
    Input: list of dicts với key 'title' và 'snippet'.
    Output: same list với thêm key 'sentiment'.
    """
    for item in news_items:
        text = f"{item.get('title', '')} {item.get('snippet', '')}"
        item["sentiment"] = analyze_sentiment(text)
    return news_items


def aggregate_sentiment(news_items: list[dict]) -> dict:
    """
    Tổng hợp sentiment của nhiều bài viết thành 1 điểm tổng.
    Dùng interaction weight (views/likes) để tăng trọng số bài viral.
    Trả về: {avg_score, label, article_count, bullish_count, bearish_count}
    """
    if not news_items:
        return {"avg_score": 0.0, "label": "neutral", "article_count": 0,
                "bullish_count": 0, "bearish_count": 0}

    items_with_sentiment = analyze_news_list(
        [i.copy() for i in news_items if "sentiment" not in i]
    ) + [i for i in news_items if "sentiment" in i]

    total_weight = 0.0
    weighted_sum = 0.0
    bullish = 0
    bearish = 0

    for item in items_with_sentiment:
        sent  = item.get("sentiment", {})
        score = sent.get("score", 0.0)
        weight = max(float(item.get("weight", 1.0)), 0.1)

        weighted_sum += score * weight
        total_weight += weight

        if score > 0.1:
            bullish += 1
        elif score < -0.1:
            bearish += 1

    avg = weighted_sum / total_weight if total_weight > 0 else 0.0
    avg = round(avg, 3)
    label = "positive" if avg > 0.1 else ("negative" if avg < -0.1 else "neutral")

    return {
        "avg_score":     avg,
        "label":         label,
        "article_count": len(items_with_sentiment),
        "bullish_count": bullish,
        "bearish_count": bearish,
    }
