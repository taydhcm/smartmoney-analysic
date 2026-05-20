"""
agents/workflow.py
LangGraph StateGraph – kết nối Supervisor + Workers thành pipeline hoàn chỉnh.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents.state import AgentState
from agents.supervisor import supervisor_node, make_worker_node
from agents.data_agent import create_data_agent
from agents.news_agent import create_news_agent
from agents.sector_agent import create_sector_agent
from utils.logger import get_logger

log = get_logger(__name__)

# Singleton agents (tạo 1 lần, tái sử dụng)
_data_agent   = None
_news_agent   = None
_sector_agent = None


def _get_agents():
    global _data_agent, _news_agent, _sector_agent
    if _data_agent is None:
        log.info("Khởi tạo agents...")
        _data_agent   = create_data_agent()
        _news_agent   = create_news_agent()
        _sector_agent = create_sector_agent()
    return _data_agent, _news_agent, _sector_agent


def build_smart_money_graph(use_memory: bool = True):
    """
    Build và compile LangGraph workflow.

    Graph structure:
        START → supervisor → [data_agent | news_agent | sector_agent] → supervisor → END

    Parallel execution: supervisor có thể route đến nhiều agents cùng lúc.

    Args:
        use_memory: Bật MemorySaver để lưu lịch sử conversation (cho chat UI).
    """
    data_agent, news_agent, sector_agent = _get_agents()

    # Tạo worker nodes
    data_node   = make_worker_node("data_agent",   data_agent)
    news_node   = make_worker_node("news_agent",   news_agent)
    sector_node = make_worker_node("sector_agent", sector_agent)

    # Build graph
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("supervisor",    supervisor_node)
    builder.add_node("data_agent",    data_node)
    builder.add_node("news_agent",    news_node)
    builder.add_node("sector_agent",  sector_node)

    # Edges: START → supervisor
    builder.add_edge(START, "supervisor")

    # Workers quay về supervisor sau khi xong (handled bởi Command trong make_worker_node)
    # Supervisor → END được handled bởi Command(goto="__end__") trong _finalize

    checkpointer = MemorySaver() if use_memory else None
    graph = builder.compile(checkpointer=checkpointer)

    log.info("Smart Money Graph compiled thành công.")
    return graph


def run_analysis(
    query:    str,
    tickers:  list[str],
    period:   str = "1m",
    exchange: str = "HOSE",
    sector:   str | None = None,
    thread_id: str = "default",
) -> str:
    """
    Entry point đơn giản để chạy full analysis.
    Trả về final_report (markdown string).
    """
    from agents.state import initial_state
    graph = build_smart_money_graph()

    state = initial_state(query, tickers, period, exchange, sector)
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(state, config=config)
    return result.get("final_report") or result["messages"][-1].content


def stream_analysis(
    query:    str,
    tickers:  list[str],
    period:   str = "1m",
    exchange: str = "HOSE",
    sector:   str | None = None,
    thread_id: str = "default",
):
    """
    Generator: stream events từ graph về Streamlit.
    Yield: (node_name: str, content: str)
    """
    from agents.state import initial_state
    graph = build_smart_money_graph()

    state  = initial_state(query, tickers, period, exchange, sector)
    config = {"configurable": {"thread_id": thread_id}}

    for event in graph.stream(state, config=config, stream_mode="updates"):
        for node_name, node_update in event.items():
            if node_name == "__end__":
                continue
            messages = node_update.get("messages", [])
            for msg in messages:
                content = msg.content if hasattr(msg, "content") else str(msg)
                if content:
                    yield node_name, content
