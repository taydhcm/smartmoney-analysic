"""
pages/09_self_learning.py
E6 Hybrid Self-Learning UI — Sprint 9.

Dashboard cho Rolling IsotonicRegression Calibrator.
    - Trạng thái calibrator (last fit, n_samples, precision, stale?)
    - Drift Alert banner (precision < 30%)
    - Re-fit button → fit_rolling_calibrator()
    - Calibration Curve (scatter p_raw vs actual label + isotonic curve)
    - Training data table (resolved outcomes dùng để fit)
    - Workflow guide (Hybrid 4-bước)
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Self-Learning (E6)", layout="wide")
st.title("🤖 Hybrid Self-Learning  (E6)")
st.caption(
    "Rolling IsotonicRegression Calibrator · Weekly re-fit · Drift Alert · Sprint 9"
)

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from ml.rolling_calibrator import (
        apply_rolling_calibration,
        fit_rolling_calibrator,
        get_calibration_data,
        get_calibrator_status,
        load_rolling_calibrator,
        needs_weekly_refit,
    )
    _BACKEND_OK = True
except Exception as _e:
    _BACKEND_OK = False
    st.error(f"Lỗi import backend: {_e}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cấu hình")
    n_outcomes  = st.slider("Số outcomes cho re-fit", 20, 200, 90, 10)
    min_samples = st.slider("Minimum samples", 5, 50, 10, 5)
    stale_days  = st.slider("Stale threshold (ngày)", 1, 30, 7, 1)
    st.divider()
    auto_refit  = st.toggle("Auto re-fit khi stale", value=True)
    if st.button("🔄 Refresh trang"):
        st.rerun()
    st.divider()
    st.caption("**Hybrid 4-bước**")
    st.caption("1️⃣ LightGBM batch retrain (2w)")
    st.caption("2️⃣ E5 Outcome Tracker (T+5)")
    st.caption("3️⃣ **E6 Rolling Calibrator** ← đây")
    st.caption("4️⃣ Drift Alert → manual retrain")

# ── Load trạng thái ──────────────────────────────────────────────────────────
status = get_calibrator_status()

# ── Drift Alert Banner ────────────────────────────────────────────────────────
if status.get("drift_alert"):
    st.error(
        "🚨 **DRIFT ALERT**: Rolling precision < 30%  —  "
        "Cần re-fit calibrator ngay hoặc manual batch retrain!"
    )
elif status.get("needs_refit"):
    st.warning(
        "⚠️ Rolling calibrator chưa có hoặc đã cũ (> 7 ngày). "
        "Nhấn **Re-fit ngay** bên dưới."
    )

# ── Section 1: Trạng thái Calibrator ─────────────────────────────────────────
st.subheader("📊 Trạng thái Calibrator")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "Trạng thái",
    "✅ Fitted" if status["is_fitted"] else "❌ Chưa fit",
)
c2.metric("Số outcomes", status["n_samples"])
c3.metric("Precision (training)", f"{status['precision']:.1%}" if status["n_samples"] else "—")
_last = status["last_fit_at"]
c4.metric("Last fit", _last[:10] if _last else "N/A")
c5.metric("Stale?", "⚠️ Có" if status["is_stale"] else "✅ Không")

col_info1, col_info2 = st.columns(2)
with col_info1:
    _roll_label = "✅ Tồn tại" if status["rolling_path_exists"] else "❌ Chưa có"
    st.info(f"**Rolling calibrator pkl**: {_roll_label}")
with col_info2:
    _batch_label = "✅ Tồn tại" if status["batch_path_exists"] else "❌ Chưa có"
    st.info(f"**Batch calibrator pkl**: {_batch_label}")

# ── Section 2: Re-fit ─────────────────────────────────────────────────────────
st.subheader("🔄 Re-fit Rolling Calibrator")

col_btn, col_info = st.columns([1, 2])
with col_btn:
    do_refit = st.button("🔄 Re-fit ngay", type="primary", width='stretch')
with col_info:
    _stale = needs_weekly_refit(stale_days=stale_days)
    if _stale:
        st.caption(f"⏰ Calibrator cần re-fit (stale threshold: {stale_days} ngày).")
    else:
        st.caption(f"✅ Calibrator đang fresh (threshold: {stale_days} ngày).")

_should_refit = do_refit or (auto_refit and status["needs_refit"] and not do_refit)

if do_refit:
    with st.spinner(f"Đang fit IsotonicRegression trên {n_outcomes} outcomes..."):
        _result = fit_rolling_calibrator(n=n_outcomes, min_samples=min_samples)

    if _result["status"] == "ok":
        st.success(
            f"✅ Re-fit thành công!  "
            f"{_result['n_samples']} outcomes · "
            f"precision = {_result['precision']:.1%} · "
            f"fit lúc {_result['fit_at']}"
        )
        st.rerun()
    elif _result["status"] == "insufficient_data":
        st.warning(
            f"⚠️ Không đủ dữ liệu: chỉ có **{_result['n_samples']}** resolved outcomes "
            f"(cần >= {min_samples}). "
            "Hãy chạy E5 T+5 check để có thêm outcomes."
        )
    else:
        st.error(
            f"❌ Lỗi: `{_result['status']}`. "
            "Kiểm tra `signal_log.db` có tồn tại không (E5 Outcome Tracker)."
        )

# ── Section 3: Calibration Curve ─────────────────────────────────────────────
st.subheader("📈 Calibration Curve")

df_cal = get_calibration_data(n=n_outcomes)

if df_cal.empty:
    st.info(
        "Chưa có resolved outcomes. "
        "Vào trang **08_outcome_tracker** → *Kiểm tra T+5* trước."
    )
else:
    cal = load_rolling_calibrator()

    p_range = np.linspace(0.0, 1.0, 200)

    fig = go.Figure()

    # Scatter: actual outcomes (jittered vertically để dễ đọc)
    rng    = np.random.default_rng(42)
    jitter = rng.uniform(-0.025, 0.025, len(df_cal))
    colors = df_cal["label"].map({1: "#22c55e", 0: "#ef4444"}).tolist()

    fig.add_trace(go.Scatter(
        x=df_cal["p_raw"].tolist(),
        y=(df_cal["label"].values + jitter).tolist(),
        mode="markers",
        name="Actual outcome (WIN=1 / non-WIN=0)",
        marker=dict(color=colors, size=7, opacity=0.65),
    ))

    # Isotonic fitted curve
    if cal is not None:
        y_iso = np.asarray(cal.transform(p_range.reshape(-1))).clip(0, 1)
        fig.add_trace(go.Scatter(
            x=p_range.tolist(), y=y_iso.tolist(),
            mode="lines",
            name="Rolling Isotonic Calibration",
            line=dict(color="#3b82f6", width=3),
        ))

    # Perfect calibration diagonal
    fig.add_trace(go.Scatter(
        x=[0.0, 1.0], y=[0.0, 1.0],
        mode="lines",
        name="Perfect Calibration",
        line=dict(color="#94a3b8", dash="dash", width=1),
    ))

    # Drift threshold line at 0.30
    fig.add_hline(
        y=0.30,
        line_dash="dot",
        line_color="#f97316",
        annotation_text="Drift threshold 30%",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title="Calibration Curve: p_raw (LightGBM) vs Actual Win Rate",
        xaxis_title="Raw Probability (p_raw)",
        yaxis_title="Actual Win Rate / Outcome",
        yaxis=dict(range=[-0.1, 1.1]),
        height=420,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified",
    )
    st.plotly_chart(fig)

    # Stats row
    cs1, cs2, cs3, cs4 = st.columns(4)
    cs1.metric("Total resolved", len(df_cal))
    cs2.metric("WIN count",  int(df_cal["label"].sum()))
    cs3.metric("LOSS / FLAT", int((df_cal["label"] == 0).sum()))
    cs4.metric("Precision", f"{df_cal['label'].mean():.1%}")

# ── Section 4: Training Data Table ───────────────────────────────────────────
st.subheader("📋 Dữ liệu Training (Resolved Outcomes)")

if df_cal.empty:
    st.info("Không có dữ liệu. Hãy chạy E5 T+5 check trước.")
else:
    _display = df_cal[
        ["signal_date", "ticker", "p_raw", "p_calibrated", "outcome", "pnl_pct", "label"]
    ].copy()
    _display["p_raw"]         = _display["p_raw"].map("{:.4f}".format)
    _display["p_calibrated"]  = _display["p_calibrated"].map("{:.4f}".format)
    _display["pnl_pct"]       = _display["pnl_pct"].apply(
        lambda x: f"{x:+.2f}%" if x is not None and not pd.isna(x) else "—"
    )
    _OUTCOME_ICONS = {"WIN": "✅ WIN", "LOSS": "❌ LOSS", "FLAT": "➖ FLAT"}
    _display["outcome"] = _display["outcome"].map(
        lambda v: _OUTCOME_ICONS.get(v, v)
    )
    st.dataframe(_display, hide_index=True)

# ── Section 5: Demo apply_rolling_calibration ─────────────────────────────────
with st.expander("🔬 Demo: apply_rolling_calibration()"):
    st.caption(
        "Áp dụng rolling calibrator lên một dải p_raw để xem sự hiệu chỉnh."
    )
    demo_arr = np.round(np.linspace(0.1, 0.9, 9), 2)
    calibrated = apply_rolling_calibration(demo_arr)
    demo_df = pd.DataFrame({
        "p_raw":       demo_arr,
        "p_calibrated": np.round(calibrated, 4),
        "delta":       np.round(calibrated - demo_arr, 4),
    })
    st.dataframe(demo_df, hide_index=True)

# ── Section 6: Workflow Guide ─────────────────────────────────────────────────
with st.expander("📖 Hướng dẫn Hybrid Self-Learning 4-bước"):
    st.markdown("""
**Bước 1 — LightGBM Batch Retrain** *(mỗi 2 tuần)*
> Chạy `retrain_v5.py` để retrain LightGBM trên toàn bộ dữ liệu lịch sử.
> Tạo ra `alpha_model.pkl` + `alpha_calibrator.pkl` (batch isotonic).

**Bước 2 — E5 Outcome Tracker** *(hàng ngày 15:15)*
> Trang `08_outcome_tracker` → **Kiểm tra T+5** → resolve WIN / LOSS / FLAT.
> Dữ liệu outcome được lưu vào `signal_log.db` với `p_raw` của mỗi signal.

**Bước 3 — E6 Rolling Calibrator** *(mỗi tuần — trang này)*
> Nhấn **🔄 Re-fit ngay** → IsotonicRegression re-fit trên 90 outcomes gần nhất.
> `alpha_calibrator_rolling.pkl` được cập nhật.
> Khi dự báo: `apply_rolling_calibration(p_raw)` → p_cal chính xác hơn theo market hiện tại.

**Bước 4 — Drift Alert** *(tự động)*
> Banner đỏ khi rolling precision < 30% trên ≥ 10 outcomes.
> **Khi có Drift Alert**: re-fit calibrator → nếu vẫn drifting → batch retrain thủ công.

---

**Lưu ý về sự khác biệt Batch vs Rolling**:
| | Batch Calibrator | Rolling Calibrator (E6) |
|---|---|---|
| Dữ liệu | Toàn bộ historical | 90 outcomes gần nhất |
| Cập nhật | Mỗi 2 tuần (retrain) | Mỗi tuần (thích nghi) |
| File | `alpha_calibrator.pkl` | `alpha_calibrator_rolling.pkl` |
| Dùng cho | Baseline ổn định | Thích nghi market conditions |
    """)
