"""
agents/supervisor.py
Manual Supervisor Agent – điều phối các worker agents.
Dùng Command(goto=...) để routing parallel/sequential.
Pattern: LangGraph Command-based (recommended từ LangChain 2025).
"""

from __future__ import annotations
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.types import Command

from config.settings import get_llm
from agents.state import AgentState
from utils.logger import get_logger

log = get_logger(__name__)

SUPERVISOR_SYSTEM_PROMPT = """
Bạn là Supervisor điều phối đội ngũ Smart Money Detector cho thị trường chứng khoán Việt Nam.

Các worker agents có sẵn:
1. **data_agent**: Phân tích dòng tiền (foreign flow, tự doanh, volume, OBV, MFI, accumulation).
2. **news_agent**: Phân tích tin tức CafeF/Vietstock, sentiment diễn đàn f319/Fireant/XamVN, 
   thông báo chính thức HOSE/HNX.
3. **sector_agent**: Phân tích dòng tiền ngành, sector rotation, top picks theo ngành.

Quy tắc routing:
- Nếu người dùng hỏi về DÒ TÌM CỔ PHIẾU cụ thể → chạy data_agent + news_agent (song song).
- Nếu hỏi về NGÀNH cụ thể → chạy sector_agent + data_agent.
- Nếu hỏi TỔNG QUAN THỊ TRƯỜNG → chạy cả 3 agents.
- Nếu hỏi về TIN TỨC / SENTIMENT → chỉ cần news_agent.
- Nếu đã có đủ thông tin → FINISH để tổng hợp báo cáo.

Sau khi các agents hoàn thành, tổng hợp kết quả thành:
1. Kết luận SMART MONEY đang vào/ra cổ phiếu/ngành nào.
2. Top cổ phiếu ĐÁNG THEO DÕI với lý do cụ thể.
3. Cảnh báo rủi ro nếu có.
4. Khuyến nghị hành động: MUA TÍCH LŨY / THEO DÕI / TRÁNH.

Trả lời bằng tiếng Việt, súc tích nhưng đầy đủ số liệu.
"""

# Workers có thể route đến — phải dùng "__end__" (không phải "FINISH")
# LangGraph 1.2+ đọc type annotation này để tạo edges, nên phải khớp đúng tên node
WorkerName = Literal["data_agent", "news_agent", "sector_agent", "__end__"]


def supervisor_node(state: AgentState) -> Command[WorkerName]:
    """
    Supervisor node: đọc state, quyết định route đến worker nào tiếp theo.
    Dùng LLM để phân tích yêu cầu và chọn agents phù hợp.
    Sau khi workers xong, tổng hợp final report.
    """
    llm = get_llm()
    completed = state.get("completed") or []
    tickers = state.get("tickers", [])
    sector  = state.get("sector")
    messages = state.get("messages", [])

    # Tất cả workers đã chạy → tổng hợp
    all_workers = {"data_agent", "news_agent", "sector_agent"}
    if all_workers.issubset(set(completed)):
        log.info("Supervisor: tất cả agents xong → tổng hợp report")
        return _finalize(state, llm, messages)

    # Xây dựng context cho LLM supervisor
    system = SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)
    routing_prompt = HumanMessage(content=(
        f"Yêu cầu người dùng: {messages[-1].content if messages else 'Phân tích thị trường'}\n"
        f"Tickers: {tickers}\n"
        f"Ngành: {sector or 'không chỉ định'}\n"
        f"Agents đã chạy: {completed or 'chưa có'}\n\n"
        "Bạn cần gọi agent nào tiếp theo? Chỉ trả lời TÊN AGENT: "
        "'data_agent', 'news_agent', 'sector_agent', hoặc 'FINISH'.\n"
        "Nếu cần nhiều agents song song, liệt kê cách nhau dấu phẩy."
    ))

    response = llm.invoke([system, routing_prompt])
    response_text = response.content.strip().lower()

    # Parse routing decision
    chosen: list[WorkerName] = []
    for agent in ["data_agent", "news_agent", "sector_agent"]:
        if agent in response_text and agent not in completed:
            chosen.append(agent)  # type: ignore

    if not chosen or "finish" in response_text:
        return _finalize(state, llm, messages)

    log.info("Supervisor routing → %s", chosen)

    if len(chosen) == 1:
        return Command(goto=chosen[0])

    # Parallel: gửi đến nhiều agents cùng lúc
    from langgraph.types import Send
    return Command(goto=[Send(agent, state) for agent in chosen])  # type: ignore


def _finalize(state: AgentState, llm, messages: list) -> Command:
    """Tổng hợp kết quả từ tất cả agents thành final report."""
    system = SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)

    # Thu thập kết quả từ state
    context_parts = []
    if state.get("volume_data"):
        context_parts.append(f"DATA AGENT:\n{state['volume_data']}")
    if state.get("news_data"):
        context_parts.append(f"NEWS AGENT:\n{state['news_data']}")
    if state.get("sector_data"):
        context_parts.append(f"SECTOR AGENT:\n{state['sector_data']}")

    # Tổng hợp từ messages (agent outputs)
    agent_outputs = "\n\n".join(
        f"{m.content}" for m in messages
        if isinstance(m, AIMessage) and m.content
    )

    synthesis_prompt = HumanMessage(content=(
        f"Kết quả từ các agents:\n{agent_outputs}\n\n"
        f"Tickers phân tích: {state.get('tickers', [])}\n"
        f"Thời gian: {state.get('period', '1m')}\n\n"
        "Hãy tổng hợp thành báo cáo Smart Money cuối cùng bằng tiếng Việt, "
        "bao gồm: kết luận dòng tiền, top picks, cảnh báo rủi ro, khuyến nghị hành động."
    ))

    final_response = llm.invoke([system, synthesis_prompt])
    final_report   = final_response.content

    from langchain_core.messages import AIMessage as AI
    return Command(
        goto="__end__",
        update={
            "final_report": final_report,
            "messages": [AI(content=final_report)],
        },
    )


def make_worker_node(agent_name: str, agent):
    """
    Wrapper tạo node function cho worker agent.
    Sau khi chạy xong → cập nhật 'completed' và quay về supervisor.
    """
    def worker_node(state: AgentState) -> Command:
        log.info("%s bắt đầu chạy...", agent_name)
        try:
            result = agent.invoke(state)
            output_content = ""
            if result.get("messages"):
                last = result["messages"][-1]
                output_content = last.content if hasattr(last, "content") else str(last)
        except Exception as exc:
            log.error("%s lỗi: %s", agent_name, exc)
            output_content = f"[{agent_name}] Lỗi: {exc}"

        completed = list(state.get("completed") or [])
        if agent_name not in completed:
            completed.append(agent_name)

        from langchain_core.messages import AIMessage as AI
        update: dict = {
            "completed": completed,
            "messages": [AI(content=output_content, name=agent_name)],
        }

        # Lưu kết quả vào field tương ứng
        if agent_name == "data_agent":
            update["volume_data"] = output_content
        elif agent_name == "news_agent":
            update["news_data"] = output_content
        elif agent_name == "sector_agent":
            update["sector_data"] = output_content

        return Command(goto="supervisor", update=update)

    worker_node.__name__ = agent_name
    return worker_node
