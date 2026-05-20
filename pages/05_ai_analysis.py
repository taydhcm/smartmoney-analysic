"""
pages/05_ai_analysis.py
Giao diện chat với LangGraph Multi-Agent Smart Money Detector.
Stream real-time agent thinking → Streamlit.
"""

import streamlit as st
from config.constants import SECTOR_MAP, VN30_TICKERS, PERIOD_LABELS
from config.settings import active_llm_provider

st.set_page_config(page_title="AI Smart Money", layout="wide")
st.title("🤖 AI Smart Money Detector")
st.caption(f"LLM Provider: **{active_llm_provider()}**")

# ── Sidebar: Input parameters ─────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Cấu Hình Phân Tích")

    analysis_mode = st.radio(
        "Chế độ phân tích",
        ["Cổ phiếu cụ thể", "Ngành", "Toàn thị trường"],
        index=0,
    )

    tickers: list[str] = []
    sector: str | None = None
    exchange = "HOSE"

    if analysis_mode == "Cổ phiếu cụ thể":
        ticker_input = st.text_input(
            "Mã cổ phiếu (nhiều mã cách nhau bằng dấu phẩy)",
            value="VIC, HPG, FPT",
        )
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        exchange = st.selectbox("Sàn", ["HOSE", "HNX", "UPCOM"])

    elif analysis_mode == "Ngành":
        sector = st.selectbox("Chọn ngành", list(SECTOR_MAP.keys()))
        tickers = SECTOR_MAP.get(sector, [])[:5]

    else:  # Toàn thị trường
        tickers = VN30_TICKERS[:10]  # Limit để tiết kiệm API calls
        exchange = "HOSE"

    period = st.selectbox(
        "Kỳ phân tích",
        list(PERIOD_LABELS.keys()),
        index=2,
        format_func=lambda x: PERIOD_LABELS[x],
    )

    st.divider()
    st.caption("💡 **Tips:**")
    st.caption("• 'Cổ phiếu cụ thể': phân tích sâu 1-3 mã")
    st.caption("• 'Ngành': tìm top picks trong ngành")
    st.caption("• 'Toàn thị trường': bức tranh dòng tiền tổng thể")

# ── Chat Interface ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Suggested queries
if not st.session_state.messages:
    st.info("💬 Gợi ý câu hỏi:")
    suggestions = [
        "Smart money đang vào cổ phiếu nào mạnh nhất?",
        "Phân tích dòng tiền khối ngoại và tự doanh",
        "Ngành nào đang được tích lũy?",
        "Có tín hiệu bứt phá không?",
    ]
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        if cols[i % 2].button(suggestion, use_container_width=True):
            st.session_state.pending_query = suggestion

# Handle suggested query click
query = st.session_state.pop("pending_query", None) or st.chat_input(
    f"Hỏi về {', '.join(tickers[:3])}{'...' if len(tickers) > 3 else ''}..."
)

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run agents with streaming
    with st.chat_message("assistant"):
        status_container = st.empty()
        report_container = st.empty()

        full_response = ""
        agent_logs: list[str] = []

        try:
            from agents.workflow import stream_analysis

            with st.status("🔄 Đang phân tích...", expanded=True) as status:
                for node_name, content in stream_analysis(
                    query=query,
                    tickers=tickers,
                    period=period,
                    exchange=exchange,
                    sector=sector,
                    thread_id=f"session_{id(st.session_state)}",
                ):
                    agent_display = {
                        "supervisor":   "🎯 Supervisor",
                        "data_agent":   "📊 Data Agent",
                        "news_agent":   "📰 News Agent",
                        "sector_agent": "🔄 Sector Agent",
                    }.get(node_name, node_name)

                    st.write(f"**{agent_display}** đang phân tích...")
                    agent_logs.append(f"**{agent_display}:** {content[:200]}...")
                    full_response = content  # Giữ output cuối nhất

                status.update(label="✅ Phân tích hoàn tất!", state="complete")

            # Hiển thị báo cáo cuối
            if full_response:
                report_container.markdown(full_response)
            else:
                report_container.warning("Không nhận được kết quả từ agents.")

        except EnvironmentError as e:
            report_container.error(f"⚠️ Lỗi cấu hình LLM: {e}")
            full_response = str(e)
        except Exception as e:
            report_container.error(f"❌ Lỗi: {e}")
            full_response = f"Lỗi: {e}"

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# ── Clear history ─────────────────────────────────────────────────────────────
if st.session_state.messages:
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
