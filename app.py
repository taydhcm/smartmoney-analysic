"""
app.py
Smart Money Detector – Entry point Streamlit.
"""

import streamlit as st
from config.settings import APP_TITLE, active_llm_provider

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar global info ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## 💹 {APP_TITLE}")
    st.caption(f"LLM: **{active_llm_provider()}**")
    st.divider()
    st.markdown("""
    **Điều hướng:**
    - 🏛️ Tổng Quan Thị Trường
    - 💰 Khối Ngoại
    - 🔄 Phân Tích Ngành
    - 🔍 Chi Tiết Cổ Phiếu
    - 🤖 AI Smart Money
    """)

# ── Home page ─────────────────────────────────────────────────────────────────
st.title(f"💹 {APP_TITLE}")
st.markdown("""
### Hệ thống phát hiện dòng tiền thông minh (Smart Money) trên TTCK Việt Nam

Sử dụng **LangGraph Multi-Agent AI** + dữ liệu real-time từ TCBS/vnstock3.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
    **📊 Dòng Tiền Khối Ngoại**
    - Foreign buy/sell/net theo ngày
    - Cảnh báo room nước ngoài gần đầy
    - Top mã khối ngoại mua/bán ròng mạnh nhất
    """)

with col2:
    st.info("""
    **🏦 Dòng Tiền Tự Doanh**
    - Proprietary trading của các CTCK
    - Phát hiện tích lũy/phân phối
    - Volume profile & OBV analysis
    """)

with col3:
    st.info("""
    **🤖 AI Smart Money Agent**
    - LangGraph Supervisor + 3 Workers
    - Phân tích đa nguồn: data + news + sector
    - Groq (free) hoặc OpenAI (nếu có key)
    """)

st.divider()
st.markdown("""
**Nguồn dữ liệu:**
`vnstock3` (TCBS API) · `CafeF RSS` · `Vietstock RSS` · `HOSE/HNX` · `Fireant API` · `f319` · `XamVN`

**Chú ý:** Dữ liệu có thể bị delay 15 phút trong giờ giao dịch. 
Ứng dụng này chỉ mang tính chất tham khảo, không phải khuyến nghị đầu tư.
""")

# ── Quick status check ────────────────────────────────────────────────────────
st.subheader("⚡ Kiểm Tra Hệ Thống")
col_a, col_b, col_c = st.columns(3)

with col_a:
    try:
        provider = active_llm_provider()
        st.success(f"✅ LLM: {provider}")
    except Exception as e:
        st.error(f"❌ LLM: {e}")

with col_b:
    try:
        import vnstock  # type: ignore  # noqa: F401
        st.success("✅ vnstock3 đã cài")
    except ImportError:
        st.warning("⚠️ vnstock3 chưa cài – chạy: `pip install vnstock`")

with col_c:
    try:
        import diskcache  # noqa: F401
        st.success("✅ Cache (diskcache) sẵn sàng")
    except ImportError:
        st.warning("⚠️ diskcache chưa cài")
