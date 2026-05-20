"""
config/settings.py
Load environment variables + LLM factory.

Nguồn secrets theo thứ tự ưu tiên:
  1. Streamlit Cloud (st.secrets) – khi deploy lên streamlit.io
  2. File .env local – khi chạy trên máy cá nhân
  3. OS environment variables
"""

from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env local (chỉ có tác dụng khi chạy local, bị ignore khi deploy)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _get_secret(key: str, default: str = "") -> str:
    """
    Lấy secret theo thứ tự ưu tiên:
    1. st.secrets (Streamlit Cloud)
    2. os.environ / .env
    """
    # Thử Streamlit secrets trước (chỉ available khi chạy trong Streamlit)
    try:
        import streamlit as st  # type: ignore
        val = st.secrets.get(key, "")
        if val:
            return str(val)
    except Exception:
        pass  # Không chạy trong Streamlit → bỏ qua

    return os.getenv(key, default)


# ── Raw env values ─────────────────────────────────────────────────────────────
GROQ_API_KEY:    str | None = _get_secret("GROQ_API_KEY") or None
OPENAI_API_KEY:  str | None = _get_secret("OPENAI_API_KEY") or None

LLM_MODEL_GROQ:   str = _get_secret("LLM_MODEL_GROQ",   "llama-3.3-70b-versatile")
LLM_MODEL_OPENAI: str = _get_secret("LLM_MODEL_OPENAI", "gpt-4o-mini")
LLM_TEMPERATURE:  float = float(_get_secret("LLM_TEMPERATURE", "0.3"))

APP_TITLE:        str = _get_secret("APP_TITLE",         "Smart Money Detector")
CACHE_TTL:        int = int(_get_secret("CACHE_TTL_SECONDS", "900"))
LOG_LEVEL:        str = _get_secret("LOG_LEVEL",         "INFO")


def get_llm():
    """
    LLM factory – trả về LangChain ChatModel.
    Ưu tiên: OpenAI (nếu có key) → Groq (default miễn phí).
    """
    if OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=LLM_MODEL_OPENAI,
            temperature=LLM_TEMPERATURE,
            api_key=OPENAI_API_KEY,
        )

    if GROQ_API_KEY:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=LLM_MODEL_GROQ,
            temperature=LLM_TEMPERATURE,
            api_key=GROQ_API_KEY,
        )

    raise EnvironmentError(
        "Chưa cấu hình LLM API key.\n"
        "Tạo file .env từ .env.example và điền GROQ_API_KEY (miễn phí tại console.groq.com) "
        "hoặc OPENAI_API_KEY."
    )


def active_llm_provider() -> str:
    """Trả về tên provider đang dùng (để hiển thị trên UI)."""
    if OPENAI_API_KEY:
        return f"OpenAI / {LLM_MODEL_OPENAI}"
    if GROQ_API_KEY:
        return f"Groq / {LLM_MODEL_GROQ}"
    return "Chưa cấu hình"
