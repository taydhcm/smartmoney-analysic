"""
pages/10_go_live.py
Sprint 10 — Go-Live Validator UI.

Pre-flight checklist trước khi deploy real capital.
    - 13 checks: model files, E5/E6 systems, drift, capital, config, pipeline
    - GO/NO-GO banner (xanh / đỏ)
    - Conservative params config (p_min=0.70, SL buffer=2%, max pos=20%)
    - Readiness score (X/13 checks)
    - Quick-fix links cho từng check fail
    - "First Real Trade" readiness summary
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Go-Live Validator (S10)", layout="wide")
st.title("🔴 Go-Live Validator  (Sprint 10)")
st.caption(
    "Pre-flight checklist · Conservative params · First Real Trade readiness"
)

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from ml.go_live_checker import (
        CheckResult,
        GoLiveConfig,
        GoLiveStatus,
        run_go_live_checks,
    )
    _BACKEND_OK = True
except Exception as _e:
    _BACKEND_OK = False
    st.error(f"Lỗi import backend: {_e}")
    st.stop()

# ── Sidebar: Config ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Conservative Params")
    st.caption("Tham số bảo thủ cho lệnh real đầu tiên:")

    capital_m = st.number_input(
        "Portfolio Capital (Triệu VND)",
        min_value=0, max_value=10_000, value=500, step=50,
    )
    portfolio_capital = capital_m * 1_000_000

    p_min = st.slider("P_min (xác suất tối thiểu)", 0.60, 0.90, 0.70, 0.05)
    max_pos_pct = st.slider("Max position / lệnh (%)", 5, 30, 20, 5)
    sl_buffer = st.slider("SL buffer tối thiểu (%)", 1.0, 5.0, 2.0, 0.5)
    rr_min = st.slider("R/R tối thiểu", 1.0, 4.0, 2.0, 0.5)
    max_positions = st.slider("Max positions mở", 1, 10, 5, 1)

    st.divider()
    if st.button("🔄 Re-run Checks", type="primary", use_container_width=True):
        st.rerun()

config = GoLiveConfig(
    p_min            = p_min,
    max_position_pct = max_pos_pct / 100.0,
    sl_buffer_pct    = sl_buffer,
    rr_min           = rr_min,
    max_positions    = max_positions,
    min_capital      = 100_000_000.0,
)

# ── Config validation ─────────────────────────────────────────────────────────
_config_errors = config.validate()
if _config_errors:
    st.error("❌ Config không hợp lệ: " + " | ".join(_config_errors))
    st.stop()

# ── Run all checks ────────────────────────────────────────────────────────────
with st.spinner("Đang chạy pre-flight checks..."):
    status = run_go_live_checks(config=config, portfolio_capital=portfolio_capital)

# ── GO / NO-GO Banner ─────────────────────────────────────────────────────────
if status.all_passed:
    st.success(
        f"## 🟢 GO-LIVE READY\n"
        f"**{status.score}/{status.total}** checks passed · "
        f"{status.warnings} warnings · 0 hard failures\n\n"
        f"✅ Tất cả điều kiện bắt buộc đã đáp ứng. Có thể thực hiện lệnh real đầu tiên!"
    )
else:
    st.error(
        f"## 🔴 NOT READY\n"
        f"**{status.score}/{status.total}** checks passed · "
        f"**{status.hard_fails} hard failures** · {status.warnings} warnings\n\n"
        f"❌ Khắc phục các lỗi bên dưới trước khi Go-Live."
    )

# ── Score & KPI row ───────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Score", f"{status.score}/{status.total}")
c2.metric("Readiness", f"{status.score_pct:.0%}")
c3.metric("Hard Failures", status.hard_fails,
          delta=f"-{status.hard_fails}" if status.hard_fails else "0",
          delta_color="inverse")
c4.metric("Warnings", status.warnings)
c5.metric("Status", "✅ READY" if status.all_passed else "❌ NOT READY")

st.divider()

# ── Checklist Table ───────────────────────────────────────────────────────────
st.subheader("📋 Pre-flight Checklist")

# Group checks by category
_CATEGORIES = [
    ("🤖 Model Artifacts",  slice(0, 3)),
    ("📊 E5/E6 Systems",    slice(3, 6)),
    ("💰 Config & Capital", slice(6, 11)),
    ("⚙️ Pipeline",        slice(11, 13)),
]

for cat_name, cat_slice in _CATEGORIES:
    cat_checks = status.checks[cat_slice]
    with st.expander(cat_name, expanded=True):
        for chk in cat_checks:
            if chk.passed and not chk.is_warning:
                icon = "✅"
                color = "green"
            elif chk.is_warning:
                icon = "⚠️"
                color = "orange"
            else:
                icon = "❌"
                color = "red"

            col_icon, col_name, col_detail = st.columns([0.05, 0.35, 0.60])
            col_icon.markdown(icon)
            col_name.markdown(f"**{chk.name}**")
            col_detail.caption(chk.detail)

st.divider()

# ── Hard failures detail ──────────────────────────────────────────────────────
_hard_fails = status.failed_checks()
if _hard_fails:
    st.subheader("❌ Các lỗi cần khắc phục")
    for chk in _hard_fails:
        st.error(f"**{chk.name}**: {chk.detail}")

_warn_checks = status.warning_checks()
if _warn_checks:
    st.subheader("⚠️ Cảnh báo (không bắt buộc nhưng nên xử lý)")
    for chk in _warn_checks:
        st.warning(f"**{chk.name}**: {chk.detail}")

# ── First Real Trade Readiness ────────────────────────────────────────────────
st.subheader("🚀 First Real Trade — Tham số")

col_params1, col_params2 = st.columns(2)

with col_params1:
    st.markdown("**Điều kiện vào lệnh:**")
    st.markdown(f"- P_calibrated >= **{config.p_min:.2f}** ({config.p_min:.0%})")
    st.markdown(f"- R/R ratio >= **{config.rr_min:.1f}:1**")
    st.markdown(f"- SL buffer >= **{config.sl_buffer_pct:.1f}%**")
    st.markdown(f"- Tối đa **{config.max_positions}** lệnh mở cùng lúc")

with col_params2:
    st.markdown("**Sizing:**")
    st.markdown(f"- Max **{config.max_position_pct:.0%}** vốn / lệnh")
    cap_per_trade = portfolio_capital * config.max_position_pct
    st.markdown(f"- = **{cap_per_trade/1_000_000:.0f}M VND** / lệnh")
    total_deployed = portfolio_capital * config.max_position_pct * config.max_positions
    max_deployed_pct = config.max_position_pct * config.max_positions
    st.markdown(f"- Tối đa **{min(max_deployed_pct, 1.0):.0%}** vốn deploy")
    st.markdown(f"  = **{min(total_deployed, portfolio_capital)/1_000_000:.0f}M VND**")

# ── Go-Live Checklist (manual) ────────────────────────────────────────────────
st.divider()
with st.expander("📋 Manual Go-Live Checklist (trước khi vào lệnh đầu tiên)"):
    st.markdown("""
**Trước khi bấm "Mua" lần đầu:**

- [ ] Đã chạy **retrain_v5.py** với dữ liệu mới nhất
- [ ] **Regime** đang là BULL hoặc BULL_WEAK (không trade khi BEAR)
- [ ] Signal có **P_cal >= 0.70** và **R/R >= 2.0**
- [ ] **SL** đặt rõ ràng, không phải mental stop
- [ ] **Size** không vượt 20% portfolio
- [ ] **E5 Outcome Tracker** đang hoạt động (signal sẽ được log tự động)
- [ ] Đã mở lệnh **paper trade** tương ứng trong Trade Journal (07)
- [ ] Không có **Drift Alert** đỏ trên trang 08 / 09
- [ ] Đã đọc lý do signal (**reason**) và đồng ý với phân tích
- [ ] **Stop-loss** đã đặt trên sàn (không chờ review EOD)

**Sau khi vào lệnh:**

- [ ] Log vào **Trade Journal** (07) tab Nhập Lệnh — trade_type="real"
- [ ] Set **SL alert** trên app giao dịch
- [ ] Review sau **T+3** theo Outcome Tracker
    """)

# ── Quick links ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔗 Quick Links để khắc phục")

lc1, lc2, lc3, lc4 = st.columns(4)
lc1.info("**Model chưa có?**\n\nChạy:\n```\npython retrain_v5.py\n```")
lc2.info("**E6 chưa fit?**\n\nVào trang:\n**09 Self-Learning** → Re-fit")
lc3.info("**Chưa có signal?**\n\nVào trang:\n**06 Alpha Signals** → Generate")
lc4.info("**Drift Alert?**\n\nVào:\n**08 Outcome Tracker** → Kiểm tra T+5")
