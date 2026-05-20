"""
agents/state.py
AgentState – shared state cho toàn bộ LangGraph workflow.
Thiết kế cho thị trường chứng khoán Việt Nam.
"""

from __future__ import annotations
from typing import Annotated, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ── Conversation ──────────────────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Input parameters ──────────────────────────────────────────────────────
    tickers:  list[str]          # Danh sách mã cần phân tích
    period:   str                # "1w" | "2w" | "1m" | "3m"
    exchange: str                # "HOSE" | "HNX" | "UPCOM" | "ALL"
    sector:   str | None         # Tên ngành (nếu phân tích theo ngành)

    # ── Agent outputs (được điền bởi worker agents) ───────────────────────────
    volume_data:       dict | None   # Kết quả từ data_agent
    foreign_room_data: dict | None   # Foreign room per ticker
    tu_doan_data:      dict | None   # Tự doanh summary
    news_data:         list | None   # Tin tức + sentiment đã aggregate
    sector_data:       dict | None   # Sector rotation results
    signal_scores:     dict | None   # Smart money scores per ticker

    # ── Supervisor routing ────────────────────────────────────────────────────
    next:          str | None        # Agent tiếp theo cần gọi (supervisor dùng)
    completed:     list[str]         # Agents đã chạy xong

    # ── Final output ──────────────────────────────────────────────────────────
    final_report:  str | None        # Markdown report cuối cùng


def initial_state(
    query: str,
    tickers: list[str],
    period:   str = "1m",
    exchange: str = "HOSE",
    sector:   str | None = None,
) -> AgentState:
    """Khởi tạo state ban đầu từ input người dùng."""
    from langchain_core.messages import HumanMessage
    return AgentState(
        messages=[HumanMessage(content=query)],
        tickers=tickers,
        period=period,
        exchange=exchange,
        sector=sector,
        volume_data=None,
        foreign_room_data=None,
        tu_doan_data=None,
        news_data=None,
        sector_data=None,
        signal_scores=None,
        next=None,
        completed=[],
        final_report=None,
    )
