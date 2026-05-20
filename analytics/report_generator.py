"""
analytics/report_generator.py
Tạo báo cáo Smart Money cuối cùng từ kết quả của tất cả agents.
Output: structured markdown report với action items.
"""

from __future__ import annotations
from datetime import datetime

from config.constants import VN_TZ


def _now_str() -> str:
    return datetime.now(VN_TZ).strftime("%d/%m/%Y %H:%M")


def build_stock_report(
    ticker: str,
    score_result: dict,
    foreign_summary: str,
    tu_doan_summary: str,
    volume_summary:  str,
    accumulation_summary: str,
    news_summary:    str,
    sentiment_result: dict,
    ai_insight:      str = "",
) -> str:
    """
    Tạo báo cáo chi tiết cho 1 mã cổ phiếu.
    Trả về: markdown string.
    """
    score = score_result.get("score", 0)
    grade = score_result.get("grade", "N/A")
    components = score_result.get("component_scores", {})

    action = (
        "**✅ KHUYẾN NGHỊ: THEO DÕI MUA / MUA TÍCH LŨY**"
        if score >= 70 else
        "**⚠️ KHUYẾN NGHỊ: THEO DÕI – Chờ thêm tín hiệu xác nhận**"
        if score >= 50 else
        "**❌ KHUYẾN NGHỊ: TRÁNH / CHỜ ĐỢI**"
    )

    sent_label = sentiment_result.get("label", "neutral")
    sent_score = sentiment_result.get("avg_score", 0)
    sent_emoji = "🟢" if sent_label == "positive" else ("🔴" if sent_label == "negative" else "⚪")

    comp_lines = "\n".join(
        f"  - {k.replace('_', ' ').title()}: {v}/100"
        for k, v in components.items()
    )

    report = f"""# 📊 Báo Cáo Smart Money: {ticker}
*Cập nhật: {_now_str()}*

---

## 🎯 Smart Money Score: {score}/100 — {grade}

{action}

### Điểm thành phần:
{comp_lines}

---

## 💰 Dòng Tiền Khối Ngoại
{foreign_summary}

## 🏦 Dòng Tiền Tự Doanh
{tu_doan_summary}

## 📊 Phân Tích Volume
{volume_summary}

## 🔍 Mẫu Tích Lũy / Phân Phối
{accumulation_summary}

---

## 📰 Tin Tức & Sentiment
**Sentiment thị trường:** {sent_emoji} {sent_label.upper()} (điểm: {sent_score:+.2f})

**Bài viết phân tích ({sentiment_result.get('article_count', 0)} bài):**
> {news_summary}

---

## 🤖 AI Insight
{ai_insight if ai_insight else '_Chưa có phân tích AI_'}

---
*Smart Money Detector – Dữ liệu từ TCBS/vnstock3, CafeF, Fireant, f319*
"""
    return report


def build_market_overview_report(
    market_breadth: dict,
    top_foreign_buy: list[dict],
    top_foreign_sell: list[dict],
    top_tu_doan_buy:  list[dict],
    sector_summary: str,
    ai_insight: str = "",
) -> str:
    """Báo cáo tổng quan thị trường."""

    def _fmt_list(items: list[dict], val_key: str = "net_val") -> str:
        if not items:
            return "_Không có dữ liệu_"
        return "\n".join(
            f"  {i+1}. **{r.get('ticker', '?')}** – {r.get(val_key, 0)/1e9:.1f} tỷ"
            for i, r in enumerate(items[:5])
        )

    adv = market_breadth.get("advance", 0)
    dec = market_breadth.get("decline", 0)
    unc = market_breadth.get("unchanged", 0)
    ceil = market_breadth.get("ceiling", 0)
    flr  = market_breadth.get("floor", 0)

    breadth_signal = (
        "🟢 Thị trường TÍCH CỰC"  if adv > dec * 1.5 else
        "🔴 Thị trường TIÊU CỰC" if dec > adv * 1.5 else
        "⚪ Thị trường TRUNG TÍNH"
    )

    report = f"""# 🏛️ Tổng Quan Thị Trường
*Cập nhật: {_now_str()}*

---

## 📈 Độ Rộng Thị Trường
{breadth_signal}
- Tăng: **{adv}** | Giảm: **{dec}** | Đứng: **{unc}**
- Trần: **{ceil}** | Sàn: **{flr}**

---

## 💰 Khối Ngoại Mua Ròng Mạnh Nhất
{_fmt_list(top_foreign_buy)}

## 📤 Khối Ngoại Bán Ròng Mạnh Nhất
{_fmt_list(top_foreign_sell)}

## 🏦 Tự Doanh Mua Ròng Mạnh Nhất
{_fmt_list(top_tu_doan_buy)}

---

## 🔄 Phân Tích Dòng Tiền Ngành
{sector_summary}

---

## 🤖 AI Insight
{ai_insight if ai_insight else '_Chưa có phân tích AI_'}

---
*Smart Money Detector – dữ liệu TCBS/vnstock3*
"""
    return report
