"""
pages/06_alpha_signals.py
Alpha Signal System — ML-based prediction engine cho VN30 T+2 returns.

Pipeline:
  1. Build dataset từ OHLCV lịch sử (VCI source)
  2. Train LightGBM/XGBoost model (TimeSeriesSplit CV)
  3. Predict P(return_T+2 >= 5%) cho từng ticker hôm nay
  4. Hiển thị Top Alpha Picks với explanation
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.constants import VN30_TICKERS, SECTOR_MAP
from ml.dataset_builder import build_dataset
from ml.model import MODEL_LABEL_VERSION, META_PATH, MODEL_PATH, is_model_compatible, model_exists, train_model
from ml.backtest import run_backtest
from ml.predictor import get_current_regime, get_feature_importance, predict_all, predict_today
from ml.regime import RegimeState
from ml.alert_generator import generate_morning_report, compute_portfolio_usage

st.set_page_config(page_title="Alpha Signals", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cấu hình")

    training_period = st.selectbox(
        "Kỳ train dataset",
        options=["3m", "6m", "12m"],
        index=1,
        format_func=lambda x: {"3m": "3 tháng", "6m": "6 tháng", "12m": "12 tháng"}[x],
        help="Dữ liệu OHLCV lịch sử dùng để train model",
    )

    min_prob = st.slider(
        "Ngưỡng P_alpha tối thiểu",
        min_value=0.55,
        max_value=0.90,
        value=0.65,
        step=0.05,
        help=(
            "Sàn cứng 55% — cổ phiếu có P < 55% bị loại dù VN_score cao. "
            "Slider này điều chỉnh ngưỡng cao hơn (regime gate có thể nâng thêm)."
        ),
    )

    max_rsi = st.slider(
        "RSI tối đa (loại overbought)",
        min_value=60,
        max_value=90,
        value=80,
        step=5,
    )

    st.divider()
    st.caption("**Tickers:** VN30 (30 mã)")
    st.caption("**Model:** LightGBM → XGBoost → RF")
    st.caption("**Label:** Path-dependent (target +5%, SL -5%, T+5)")

    st.divider()
    # ── Regime Gate toggle ─────────────────────────────────────────────
    enable_regime_gate = st.checkbox(
        "🌍 Bật Regime Gate (D3.1)",
        value=True,
        help=(
            "Khi bật: BEAR → không vào lệnh long, các state khác tự động "
            "nâng ngưỡng P_min theo regime thị trường."
        ),
    )

    # ── Portfolio Capital (D3.4) ───────────────────────────────────────
    st.divider()
    portfolio_capital = st.number_input(
        "Vốn đầu tư (triệu VND)",
        min_value=0,
        max_value=10_000,
        value=0,
        step=50,
        help="Nhập vốn để xem khuyến nghị số tiền mỗi vị thế (0 = chỉ xem %).",
    )
    portfolio_capital_vnd = portfolio_capital * 1_000_000   # convert to VND

    # ── Rate limit mode badge ──────────────────────────────────────────
    st.divider()
    from utils.rate_limiter import vnstock_limiter
    if vnstock_limiter.is_throttled:
        st.warning(
            "⏱ **Free tier: 17 req/phút**\n\n"
            "API sẽ tự sleep giữa các request.\n"
            "Có SSI API key? Thêm vào `.env`:\n"
            "`SSI_API_KEY=...`\n`SSI_SECRET_KEY=...`"
        )
    else:
        st.success("⚡ **SSI API** — Không giới hạn request")

    # ── D0.2 Logger Status ────────────────────────────────────────────
    st.divider()
    st.markdown("**📦 D0.2 Data Logger**")
    try:
        from data.snapshot_logger import get_logger_status
        _ls = get_logger_status()
        _sessions = _ls["sessions"]
        _s4_ready = _ls["s4_ready"]
        _needed   = _ls["sessions_needed"]
        _last_dt  = _ls.get("last_date") or "—"
        if _s4_ready:
            st.success(f"✅ **S4 sẵn sàng** — {_sessions} phiên ({_last_dt})")
        else:
            st.info(f"⏳ {_sessions}/{5} phiên — cần {_needed} phiên nữa để bật S4")
        if st.button("📥 Log hôm nay", key="btn_log_today",
                     help="Ghi snapshot phiên hôm nay vào SQLite (15:05 tự động)"):
            with st.spinner("Đang thu thập dữ liệu..."):
                from data.snapshot_logger import log_session
                _res = log_session()
                if _res["already_exists"]:
                    st.info(f"Đã log phiên {_res['session_date']} rồi.")
                else:
                    st.success(
                        f"✅ Logged {_res['tickers_logged']} tickers "
                        f"({_res['session_date']})"
                    )
            st.rerun()
    except Exception as _e:
        st.caption(f"Logger: {_e}")

# ── Header ────────────────────────────────────────────────────────────────────
from utils.rate_limiter import vnstock_limiter as _limiter
_throttled = _limiter.is_throttled
_n_tickers = len(VN30_TICKERS) + 1   # +1 cho VN30 index
_eta_train  = _limiter.eta_seconds(_n_tickers * 5) if _throttled else 0  # 6m = ~5 batch calls
_eta_predict = _limiter.eta_seconds(_n_tickers) if _throttled else 0

st.title("🎯 Alpha Signal System")
st.caption(
    "Dự đoán xác suất đạt **+5% trong 5 phiên không chạm SL -5%** dựa trên ML (Gradient Boosting). "
    f"Cập nhật: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
)

# Rate limit banner toàn trang
if _throttled:
    st.info(
        f"⏱ **Chế độ Free Tier** — VNStock giới hạn 20 req/phút. "
        f"Hệ thống tự động sleep giữa các API call. "
        f"ETA dự đoán: **~{int(_eta_predict//60)}ph{int(_eta_predict%60):02d}s** (lần đầu, sẽ nhanh sau khi cache). "
        "Muốn nhanh hơn? Thêm `SSI_API_KEY` vào `.env`.",
        icon="⚠️",
    )

# ── S3 Market Regime Banner ────────────────────────────────────────────────────
_REGIME_COLORS = {
    RegimeState.BULL:      "green",
    RegimeState.BULL_WEAK: "orange",
    RegimeState.NEUTRAL:   "blue",
    RegimeState.BEAR_WEAK: "orange",
    RegimeState.BEAR:      "red",
}
_REGIME_ICONS = {
    RegimeState.BULL:      "🟢",
    RegimeState.BULL_WEAK: "🟡",
    RegimeState.NEUTRAL:   "⚪",
    RegimeState.BEAR_WEAK: "🟠",
    RegimeState.BEAR:      "🔴",
}

with st.expander("🌍 Market Regime (S3)", expanded=True):
    _regime_cols = st.columns([2, 1, 1, 1, 2])
    with st.spinner("Đang xác định regime thị trường..."):
        try:
            _regime = get_current_regime(period="3m")
            _ri = _regime.as_dict()
            _icon = _REGIME_ICONS[_regime.state]
            _color = _REGIME_COLORS[_regime.state]
        except Exception as _e:
            _regime = None
            _ri = {}
            _icon = "⚪"
            _color = "blue"

    if _regime is not None:
        with _regime_cols[0]:
            _regime_msg = f"{_icon} **{_ri['regime']}** — {_ri['regime_label']}"
            if _regime.state == RegimeState.BEAR:
                st.error(_regime_msg + "\n\n🚫 Tắt toàn bộ long signal hôm nay!")
            elif _regime.state == RegimeState.BULL:
                st.success(_regime_msg)
            elif _regime.state == RegimeState.BEAR_WEAK:
                st.warning(_regime_msg)
            else:
                st.info(_regime_msg)
        with _regime_cols[1]:
            st.metric(
                "VN30 vs SMA20",
                f"{_ri['vn30_vs_sma20']:+.2f}%",
                delta=None,
            )
        with _regime_cols[2]:
            st.metric("ADX(14)", f"{_ri['adx']:.1f}")
        with _regime_cols[3]:
            st.metric("P_min gate", f"{_ri['min_prob']:.2f}")
        with _regime_cols[4]:
            st.caption(
                f"**DI+** {_ri['di_plus']:.1f} · **DI-** {_ri['di_minus']:.1f} · "
                f"Max vị thế: **{_ri['max_positions']}** · "
                f"Confirmed: **{_ri['confirmed_days']}** phiên\n\n"
                f"_{_ri['reason']}_"
            )
# SECTION 1 — Model Status
# ─────────────────────────────────────────────────────────────────────────────
st.header("📦 Trạng Thái Model")

col_status, col_train = st.columns([3, 2])

with col_status:
    _compatible, _compat_reason = is_model_compatible()
    _artifact_exists = MODEL_PATH.exists()

    if _compatible:
        try:
            import pickle
            with open(META_PATH, "rb") as _f:
                _meta = pickle.load(_f)
            trained_at = _meta.get("trained_at", "unknown")
            st.success(
                f"✅ Model sẵn sàng — **{_meta.get('model_type', 'GBM')}**\n\n"
                f"- Train: {_meta.get('n_samples', '?')} rows | "
                f"{_meta.get('n_tickers', '?')} tickers\n"
                f"- Features: **{_meta.get('n_features', '?')}** | "
                f"Label: `{_meta.get('label_version', '?')}`\n"
                f"- CV AUC: **{_meta.get('cv_auc_mean', 0):.3f}** "
                f"± {_meta.get('cv_auc_std', 0):.3f}\n"
                f"- Precision@0.65: **{_meta.get('cv_prec_mean', 0):.2f}**\n"
                f"- Trained at: {trained_at[:19] if trained_at else '?'}"
            )
        except Exception:
            st.success("✅ Model sẵn sàng (metadata không đọc được)")

    elif not _artifact_exists:
        st.info(
            "ℹ️ Chưa có model. Nhấn **Train Model** để bắt đầu.\n\n"
            "Yêu cầu: kết nối VCI OHLCV (chạy local)."
        )
    else:
        # Artifact tồn tại nhưng không tương thích
        st.warning(
            f"⚠️ **Model lỗi thời — cần retrain!**\n\n"
            f"🔍 Lý do: _{_compat_reason}_\n\n"
            f"Feature hiện tại: **{len(FEATURE_COLS)} cols** — "
            f"Label: `{MODEL_LABEL_VERSION}`\n\n"
            "➡️ Nhấn **Retrain Model** bên cạnh để xây lại."
        )

with col_train:
    _btn_label = "🏋️ Retrain Model" if (not _compatible and _artifact_exists) else "🏋️ Train Model"
    do_train = st.button(
        _btn_label,
        type="primary",
        width='stretch',
        help="Build dataset → train GBM → lưu model",
    )

# ── Training flow ──────────────────────────────────────────────────────────────
if do_train:
    st.divider()
    st.subheader("🔄 Đang train model...")
    if _throttled:
        _train_eta_min = int(_limiter.eta_seconds(_n_tickers * 5) // 60)
        _train_eta_sec = int(_limiter.eta_seconds(_n_tickers * 5) % 60)
        st.warning(
            f"⏱ Free tier: ETA train khoảng **{_train_eta_min}ph {_train_eta_sec:02d}s** "
            "(do thờ́t API 17 req/phút). Lần tiếp theo sẽ nhanh hơn nhờ cache."
        )
    prog_bar  = st.progress(0.0)
    prog_text = st.empty()

    def _update_progress(pct: float, msg: str) -> None:
        prog_bar.progress(min(pct, 1.0))
        if _throttled:
            used, cap = _limiter.current_usage()
            prog_text.caption(f"{msg}  |  ⏱ API: {used}/{cap} req/phút")
        else:
            prog_text.caption(msg)

    with st.spinner("Đang build dataset..."):
        try:
            dataset = build_dataset(
                tickers=VN30_TICKERS,
                period=training_period,
                progress_callback=_update_progress,
            )
        except Exception as exc:
            st.error(f"Lỗi build dataset: {exc}")
            st.stop()

    if dataset.empty:
        st.error("Dataset rỗng — không thể train. Kiểm tra kết nối VCI OHLCV.")
        st.stop()

    # Thống kê dataset
    n_pos = int(dataset["label"].sum())
    n_tot = len(dataset)
    st.info(
        f"Dataset: **{n_tot:,}** rows | **{dataset['ticker'].nunique()}** tickers | "
        f"Label rate: **{100*n_pos/max(n_tot,1):.1f}%** (y=1 = T+2 ≥ +5%)"
    )

    prog_text.caption("Đang train model...")
    prog_bar.progress(0.7)

    with st.spinner("Training GBM model..."):
        try:
            result = train_model(dataset, save=True)
        except Exception as exc:
            st.error(f"Lỗi train model: {exc}")
            st.stop()

    prog_bar.progress(1.0)
    prog_text.caption("✅ Hoàn tất!")

    m = result["metrics"]
    st.success(
        f"✅ **{m['model_type']}** trained thành công!\n\n"
        f"- AUC trung bình: **{m['cv_auc_mean']:.3f}** ± {m['cv_auc_std']:.3f}\n"
        f"- Precision@0.65: **{m['cv_prec_mean']:.2f}**\n"
        f"- Samples: {m['n_samples']:,} | pos_weight: {m['pos_weight']:.1f}x"
    )

    # Fold details table
    fold_df = pd.DataFrame(m["fold_details"])
    st.dataframe(
        fold_df.style.format({"auc": "{:.3f}", "precision": "{:.2f}", "recall": "{:.2f}"}),
    )

    st.session_state["_model_just_trained"] = True
    st.session_state["alpha_dataset"] = dataset   # cache for backtest
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Run Prediction
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.header("🔮 Dự Đoán Hôm Nay")

col_run, col_all = st.columns([2, 3])
with col_run:
    do_predict = st.button(
        "▶️ Chạy Prediction",
        type="primary",
        width='stretch',
        disabled=not model_exists(),
    )
with col_all:
    st.caption("📊 P_alpha Toàn VN30 luôn hiện bên dưới")

if not model_exists():
    if not _compatible and _artifact_exists:
        st.warning(
            f"⚠️ Model lỗi thời: **{_compat_reason}**\n\n"
            "↩️ Nhấn **Retrain Model** ở trên rồi chạy prediction lại."
        )
    else:
        st.info("ℹ️ Chưa có model. Train trước để sử dụng prediction.")
    st.stop()

# ETA notice ngay trước khi predict
if _throttled and do_predict:
    _peta = int(_limiter.eta_seconds(_n_tickers))
    st.info(
        f"⏳ Ước tính ~**{_peta//60}ph {_peta%60:02d}s** để dự đoán {_n_tickers-1} tickers "
        "(hệ thống tự ngủ giữa mỗi API call). "
        "Các lần sau sẽ nhanh hơn do cache 15 phút."
    )

# ── Prediction ─────────────────────────────────────────────────────────────────
if do_predict or "alpha_picks" in st.session_state:
    if do_predict:
        _pred_prog  = st.progress(0.0)
        _pred_text  = st.empty()

        def _pred_update(pct: float, msg: str) -> None:
            _pred_prog.progress(min(pct, 1.0))
            if _throttled:
                used, cap = _limiter.current_usage()
                _pred_text.caption(f"{msg}  |  ⏱ API quota: {used}/{cap} req/phút")
            else:
                _pred_text.caption(msg)

        with st.spinner("Đang tính P_alpha cho VN30..."):
            try:
                # Luôn lấy toàn bộ VN30 (OHLCV cache → không tốn thêm API call)
                _all_df = predict_all(
                    tickers=VN30_TICKERS,
                    period="3m",
                    progress_callback=_pred_update,
                )
                st.session_state["alpha_all_df"] = _all_df

                _picks = predict_today(
                    tickers=VN30_TICKERS,
                    period="3m",
                    min_probability=min_prob,
                    max_rsi=float(max_rsi),
                    progress_callback=_pred_update,
                    enable_regime_gate=enable_regime_gate,
                )
                st.session_state["alpha_picks"] = _picks
                st.session_state["alpha_min_prob"] = min_prob
            except Exception as exc:
                st.error(f"Lỗi prediction: {exc}")
                st.stop()

        _pred_prog.progress(1.0)
        _pred_text.empty()

    picks      = st.session_state.get("alpha_picks", [])
    all_df     = st.session_state.get("alpha_all_df")
    saved_prob = st.session_state.get("alpha_min_prob", min_prob)
    _TOP_N     = 5
    top_picks  = picks[:_TOP_N]   # Top 5 theo VN_score (sắp xếp trong predict_today)

    # ── 4b: Portfolio Usage Tracker sidebar (Sprint 7) ────────────────────────
    # Dùng top_picks (top 5 VN_score) để sizing portfolio
    _alerts_sidebar = generate_morning_report(top_picks, capital=portfolio_capital_vnd)
    _regime_max_pos = (_regime.max_positions if _regime is not None else 5)
    _usage = compute_portfolio_usage(_alerts_sidebar, _regime_max_pos, portfolio_capital_vnd)
    with st.sidebar:
        st.divider()
        st.markdown("**📊 Portfolio Usage (Sprint 7)**")
        _u_c1, _u_c2 = st.columns(2)
        _u_c1.metric("Deployed", f"{_usage.total_allocated_pct:.0%}")
        _u_c2.metric("Remaining", f"{_usage.remaining_pct:.0%}")
        st.progress(
            min(_usage.total_allocated_pct, 1.0),
            text=f"{_usage.picks_count}/{_usage.max_positions} vị thế",
        )
        if portfolio_capital_vnd > 0:
            st.caption(f"Deployed: {_usage.capital_deployed_vnd:,.0f} VND")
            st.caption(f"Còn lại:  {_usage.capital_remaining_vnd:,.0f} VND")
        if _usage.is_full:
            st.warning("⚠️ Đã đủ vị thế theo regime")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — Top Alpha Picks
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    _hard_floor = max(0.55, saved_prob)
    st.header(f"🏆 Top Alpha Picks  ·  VN_score↓ · P≥55% (sàn cứng) · P≥{saved_prob:.0%} (slider)")
    st.caption(
        "📌 Chiến lược chọn: **VN_score** sắp xếp (pattern › volume › RS › RSI) · "
        "**P_alpha ≥ 55%** làm filter loại trừ · **Regime Gate** an toàn (BEAR = dừng)"
    )

    if not picks:
        st.warning(
            f"Không có cổ phiếu nào vượt sàn P_alpha ≥ {_hard_floor:.0%} hôm nay. "
            "Thử giảm slider ngưỡng hoặc kiểm tra lại model."
        )
    else:
        st.caption(f"🏅 **Top {_TOP_N} picks** theo VN_score (từ **{len(picks)}** mã vượt ngưỡng P≥{_hard_floor:.0%})")

        # ── 4a: Regime Banner (full-width, Sprint 7) ────────────────────────────
        if _regime is not None:
            _ri2 = _regime.as_dict()
            _banner_msg = (
                f"{_regime.emoji} **Regime: {_ri2['regime']} — {_ri2['regime_label']}**  |  "
                f"Max vị thế: **{_ri2['max_positions']}**  |  P_min: **{_ri2['min_prob']}**\n\n"
                f"_{_ri2['reason']}_"
            )
            if _regime.state == RegimeState.BEAR:
                st.error(_banner_msg)
            elif _regime.state == RegimeState.BULL:
                st.success(_banner_msg)
            elif _regime.state in (RegimeState.BEAR_WEAK,):
                st.warning(_banner_msg)
            else:
                st.info(_banner_msg)

        # ── VN_score bar chart — toàn bộ picks vượt ngưỡng, highlight top 5 ───────────
        picks_df = pd.DataFrame(picks)
        _top_tickers = {p["ticker"] for p in top_picks}

        color_map = {"high": "#00cc66", "medium": "#ffaa00", "low": "#ff6666"}
        fig_bar = go.Figure()
        for _, row in picks_df.iterrows():
            _vn_sc   = float(row.get("vn_score", 0.0))
            _rs_lbl  = (row.get("rs") or {}).get("rs_label", "Neutral")
            _rs_icon = {"Outperform": "🟢", "Neutral": "⚪", "Underperform": "🔴"}.get(_rs_lbl, "⚪")
            _is_top  = row["ticker"] in _top_tickers
            fig_bar.add_trace(
                go.Bar(
                    x=[_vn_sc],
                    y=[row["ticker"]],
                    orientation="h",
                    marker_color=(color_map.get(row["confidence"], "#888")
                                  if _is_top else "#555"),
                    name=row["confidence"],
                    showlegend=False,
                    text=f"P={row['probability']:.0%} {_rs_icon}",
                    textposition="inside",
                    hovertemplate=(
                        f"<b>{row['ticker']}</b><br>"
                        f"VN_score: {_vn_sc:.3f}<br>"
                        f"P_alpha: {row['probability']:.1%}<br>"
                        f"RS: {_rs_lbl}<br>"
                        f"Pattern: {row['pattern']}<br>"
                        f"RSI: {row['rsi']:.0f} | Vol×: {row['volume_ratio_5d']:.1f}<extra></extra>"
                    ),
                )
            )

        fig_bar.update_layout(
            title=f"VN_score — {len(picks)} mã vượt ngưỡng  ·  ▌màu = top {_TOP_N} ·  label = P_alpha 🟢/⚪/🔴 RS",
            xaxis=dict(tickformat=".2f", range=[0, 1], title="VN_score (40%×pattern + 25%×vol + 20%×RS + 15%×RSI)"),
            yaxis=dict(categoryorder="total ascending"),
            height=max(200, len(picks) * 45 + 80),
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar)

        # ── 4c: Morning Report Cards (D3.5, Sprint 7) — top 5 theo VN_score ─────────────────────────
        if _alerts_sidebar:
            st.subheader("🌅 Morning Report — Alert Cards")
            st.caption(
                f"Conviction sizing: P≥0.70 → 30% | P<0.70 → 20% | "
                f"Regime cap: 1/{_regime_max_pos}"
            )
            for _card in _alerts_sidebar:
                with st.container(border=True):
                    _hdr_col, _size_col = st.columns([3, 1])
                    with _hdr_col:
                        st.markdown(f"### {_card.ticker}  —  {_card.recommendation}")
                        st.caption(
                            f"P_cal: **{_card.p_calibrated:.1%}**  |  "
                            f"CI [{_card.ci_lo:.1%}–{_card.ci_hi:.1%}]  |  "
                            f"{_card.reason}"
                        )
                    with _size_col:
                        _conviction_label = "🔥 HIGH" if _card.position_size_pct >= 0.30 else "📌 STD"
                        st.metric(
                            "Conviction Size",
                            f"{_card.position_size_pct:.0%}",
                            delta=_conviction_label,
                            delta_color="normal",
                        )
                        if portfolio_capital_vnd > 0 and _card.position_size_vnd > 0:
                            st.caption(f"{_card.position_size_vnd:,.0f} VND")
                            if _card.shares > 0:
                                st.caption(f"≈ {_card.shares:,} cổ phiếu")
                    # Price levels
                    _ec1, _ec2, _ec3 = st.columns(3)
                    _ec1.metric(
                        "Entry Zone",
                        f"{_card.entry_lo:,.0f}–{_card.entry_hi:,.0f}",
                    )
                    _ec2.metric(
                        "Stop-Loss",
                        f"{_card.sl_price:,.0f}",
                        delta=f"−{_card.sl_pct:.1%}",
                        delta_color="inverse",
                    )
                    _ec3.metric(
                        "Target",
                        f"{_card.target_price:,.0f}",
                        delta=f"R:R 1:{_card.rr_ratio:.1f}",
                    )
                    # Copyable alert text
                    _alert_text = (
                        f"[ALERT] {_card.ticker} {_card.date}\n"
                        f"Entry: {_card.entry_lo:,.0f}–{_card.entry_hi:,.0f}\n"
                        f"SL: {_card.sl_price:,.0f} (−{_card.sl_pct:.1%})\n"
                        f"Target: {_card.target_price:,.0f}  R:R 1:{_card.rr_ratio:.1f}\n"
                        f"Size: {_card.position_size_pct:.0%}"
                        + (
                            f" ({_card.position_size_vnd:,.0f} VND, {_card.shares} cp)"
                            if _card.shares > 0 else ""
                        ) + "\n"
                        f"P_cal: {_card.p_calibrated:.1%}  [{_card.ci_lo:.1%}–{_card.ci_hi:.1%}]\n"
                        f"Lý do: {_card.reason}"
                    )
                    with st.expander("📋 Copy Alert Text"):
                        st.code(_alert_text, language=None)

        # ── Picks cards ────────────────────────────────────────────────────────
        st.subheader(f"📋 Chi Tiết Top {_TOP_N} Picks (VN_score ↓)")

        conf_colors = {"high": "🟢", "medium": "🟡", "low": "🔴"}
        _RS_BADGE  = {"Outperform": "🟢 Outperform", "Neutral": "⚪ Neutral", "Underperform": "🔴 Underperform"}
        _VOL_BADGE = {"Strong": "💧💧💧 Strong", "Moderate": "💧💧 Moderate", "Weak": "💧 Weak", "Bearish": "🔻 Bearish"}
        _SM_BADGE  = {"Accumulating": "🏦 Acc", "Distributing": "🔻 Dist", "Neutral": "⚪ Neutral", "Insufficient data": "❓ —"}

        for pick in top_picks:
            conf_icon = conf_colors.get(pick["confidence"], "⚪")
            _rs       = pick.get("rs") or {}
            _vc       = pick.get("vc") or {}
            _sm       = pick.get("sm") or {}
            _rs_label = _RS_BADGE.get(_rs.get("rs_label", "Neutral"), "⚪ Neutral")
            _vc_label = _VOL_BADGE.get(_vc.get("label", "Weak"), "💧 Weak")
            _sm_label = _SM_BADGE.get(_sm.get("label", "Insufficient data"), "❓ —")
            _comp     = pick.get("composite_score", pick["probability"])

            with st.expander(
                f"{conf_icon} **{pick['ticker']}** — P={pick['probability']:.1%}  |  "
                f"Score={_comp:.2f}  |  RS: {_rs_label}  |  Vol: {_vc_label}  |  SM: {_sm_label}",
                expanded=(pick["confidence"] == "high"),
            ):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("P_alpha", f"{pick['probability']:.1%}")
                    st.metric("Composite Score", f"{_comp:.3f}",
                              help="0.6×P_alpha + 0.4×RS_rank (D3.2)")
                with c2:
                    st.metric("RSI (14)", f"{pick['rsi']:.0f}")
                    st.metric("Volume ×5d avg", f"{pick['volume_ratio_5d']:.2f}×")
                with c3:
                    st.metric("Return hôm qua", f"{pick['return_1d_pct']:+.2f}%")
                    st.metric("vs VN30 (1d)", f"{pick['relative_strength']:+.2f}%")

                # ── M4 Probability Calibration card (Sprint 6) ─────────────────
                _p_cal = pick.get("p_calibrated", pick["probability"])
                _ci_lo = pick.get("ci_lo", _p_cal - 0.05)
                _ci_hi = pick.get("ci_hi", _p_cal + 0.05)
                _rec   = pick.get("recommendation", "WATCH")
                _REC_STYLE = {
                    "STRONG_BUY": ("🚀 STRONG BUY",  "success"),
                    "BUY":        ("✅ BUY",          "success"),
                    "WATCH":      ("👀 WATCH",        "info"),
                    "HOLD":       ("⏸️ HOLD",         "warning"),
                    "AVOID":      ("🚫 AVOID",        "error"),
                }
                _rec_text, _rec_style = _REC_STYLE.get(_rec, ("👀 WATCH", "info"))
                st.markdown("---")
                _cr1, _cr2 = st.columns([1, 2])
                with _cr1:
                    getattr(st, _rec_style)(f"**{_rec_text}**")
                    st.metric("P_calibrated", f"{_p_cal:.1%}",
                              help="Isotonic-calibrated probability (Sprint 6)")
                with _cr2:
                    st.caption(f"90% CI: [{_ci_lo:.1%} — {_ci_hi:.1%}]")
                    st.progress(
                        float(min(max(_p_cal, 0.0), 1.0)),
                        text=f"P_cal={_p_cal:.1%}  |  CI [{_ci_lo:.1%} – {_ci_hi:.1%}]"
                    )

                st.markdown(f"**Pattern:** `{pick['pattern']}`")

                col_a, col_b, col_c = st.columns(3)
                col_a.progress(pick["accumulation_score"], text=f"Accumulation: {pick['accumulation_score']:.2f}")
                col_b.progress(max(pick["divergence_score"], 0), text=f"Divergence: {pick['divergence_score']:.2f}")
                _pp_norm = (pick["pull_push_score"] + 1) / 2
                col_c.progress(_pp_norm, text=f"Pull/Push: {pick['pull_push_score']:+.2f}")

                # ── S2 Relative Strength card ───────────────────────────────
                if _rs:
                    st.markdown("---")
                    st.markdown("**📈 Relative Strength (S2)**")
                    _rs_cols = st.columns(5)
                    _rs_cols[0].metric("RS 1 ngày",  f"{_rs.get('rs_1d', 0):+.2f}%")
                    _rs_cols[1].metric("RS 5 ngày",  f"{_rs.get('rs_5d', 0):+.2f}%")
                    _rs_cols[2].metric("RS 20 ngày", f"{_rs.get('rs_20d', 0):+.2f}%")
                    _rs_cols[3].metric(
                        "RS Rank",
                        f"{_rs.get('rs_rank', 0.5):.0%}",
                        help="Percentile trong VN30 universe (100% = mạnh nhất)"
                    )
                    _rs_cols[4].metric(
                        "RS Score",
                        f"{_rs.get('rs_score', 0):+.2f}",
                        help="Composite RS score [-1, +1]"
                    )

                # ── S5 Entry Timing card ────────────────────────────────────
                _entry = pick.get("entry")
                if _entry:
                    st.markdown("---")
                    st.markdown("**🎯 Entry Timing (S5)**")
                    _e_cols = st.columns(4)
                    _e_cols[0].metric(
                        "Entry Zone",
                        f"{_entry['entry_low']:,.0f}–{_entry['entry_high']:,.0f}",
                        help="Vùng giá nên mua, không chase cao hơn entry_high"
                    )
                    _e_cols[1].metric(
                        "Stop-Loss",
                        f"{_entry['sl_price']:,.0f}",
                        delta=f"−{_entry['sl_pct']:.1%}",
                        delta_color="inverse",
                    )
                    _e_cols[2].metric(
                        "Target (+5%)",
                        f"{_entry['target_price']:,.0f}",
                    )
                    _e_cols[3].metric(
                        "Risk:Reward",
                        f"1 : {_entry['rr_ratio']:.1f}",
                        help="Tốt nhất ≥ 1:2 (target gấp đôi SL)"
                    )
                    _entry_note = (
                        f"📌 Support: **{_entry['support']:,.0f}** | "
                        f"Resistance: **{_entry['resistance']:,.0f}** | "
                        f"ATR(14): **{_entry['atr']:,.0f}** điểm"
                    )
                    if _entry.get("in_entry_zone"):
                        st.success(f"✅ Giá đang trong entry zone. {_entry_note}")
                    else:
                        st.info(f"ℹ️ Chờ giá về entry zone. {_entry_note}")

                # ── S4 Volume Confirmation card ─────────────────────────────
                _vc = pick.get("vc")
                if _vc:
                    st.markdown("---")
                    st.markdown("**📊 Volume Confirmation (S4)**")
                    _v_cols = st.columns(4)
                    _v_cols[0].metric(
                        "Vol Surge",
                        f"{_vc.get('vol_surge', 0):.2f}×",
                        help="Volume hiện tại / MA20. >1.15 là surge"
                    )
                    _v_cols[1].metric(
                        "Vol Quality",
                        f"{_vc.get('vol_quality', 0):.1%}",
                        help="Tỷ lệ volume ngày tăng trong 10 phiên gần nhất"
                    )
                    _v_cols[2].metric(
                        "OBV Score",
                        f"{_vc.get('obv_score', 0):+.2f}",
                        help="On-Balance Volume trend [-1, +1]"
                    )
                    _v_cols[3].metric(
                        "Vol Score",
                        f"{_vc.get('volume_score', 0):+.2f}",
                        help="Composite volume score [-1, +1]"
                    )
                    _vc_confirmed = _vc.get("is_confirmed", False)
                    _vc_lbl = _vc.get("label", "Weak")
                    if _vc_confirmed:
                        st.success(f"✅ Volume xác nhận xu hướng: **{_vc_lbl}**")
                    else:
                        st.warning(f"⚠️ Volume chưa xác nhận: **{_vc_lbl}**")

                # ── D3.4 Portfolio Sizing card ──────────────────────────────
                _sz = pick.get("sizing")
                if _sz:
                    st.markdown("---")
                    st.markdown("**💼 Portfolio Sizing — Kelly (D3.4)**")
                    _s_cols = st.columns(4)
                    _kelly_pct = _sz.get("kelly_pct", 0.0)
                    _half_kelly = _sz.get("half_kelly_pct", 0.0)
                    _rec_pct    = _sz.get("recommended_pct", 0.0)
                    _s_cols[0].metric("Kelly %",      f"{_kelly_pct:.1%}")
                    _s_cols[1].metric("Half-Kelly %", f"{_half_kelly:.1%}")
                    _s_cols[2].metric("Khuyến nghị",  f"{_rec_pct:.1%}",
                                      help="Đã giới hạn: min 2%, max 20%, ≤ 1/max_positions")
                    if portfolio_capital_vnd > 0:
                        _cap_vnd = _rec_pct * portfolio_capital_vnd
                        _s_cols[3].metric("Số tiền (VND)", f"{_cap_vnd:,.0f}")
                    else:
                        _s_cols[3].metric("Số tiền", "—", help="Nhập vốn ở sidebar để tính")
                    if _sz.get("reasoning"):
                        st.caption(_sz["reasoning"])

                # ── S4 Smart Money Flow card ─────────────────────────────────
                _sm = pick.get("sm") or {}
                _sm_sessions = _sm.get("sessions", 0)
                st.markdown("---")
                st.markdown("**🏦 Smart Money Flow (S4 D0.2)**")
                if _sm_sessions < 5:
                    st.caption(
                        f"⏳ Cần thêm **{5 - _sm_sessions} phiên** dữ liệu D0.2. "
                        f"Bấm 'Log hôm nay' ở sidebar mỗi ngày sau 15:05."
                    )
                else:
                    _sm_cols = st.columns(4)
                    _sm_cols[0].metric(
                        "Net% hôm nay",
                        f"{_sm.get('foreign_net_pct', 0):+.2%}",
                        help="Foreign net vol / total vol hôm nay"
                    )
                    _sm_cols[1].metric(
                        "Net% TB5d",
                        f"{_sm.get('foreign_net_5d', 0):+.2%}",
                        help="TB 5 phiên gần nhất"
                    )
                    _sm_cols[2].metric(
                        "Trend",
                        f"{_sm.get('foreign_trend', 0):+.2f}",
                        help="Slope của foreign net [-1, +1]"
                    )
                    _sm_cols[3].metric(
                        "SM Score",
                        f"{_sm.get('smart_money_score', 0):+.2f}",
                        help="Composite score [-1, +1]. >0.15 = Accumulating"
                    )
                    _sm_label = _sm.get("label", "Neutral")
                    if _sm_label == "Accumulating":
                        st.success(f"🏦 Ngoại tệ đang **tích lũy** — {_sm_sessions} phiên data")
                    elif _sm_label == "Distributing":
                        st.error(f"🔻 Ngoại tệ đang **phân phối** — {_sm_sessions} phiên data")
                    else:
                        st.info(f"⚪ Dòng tiền ngoại **trung tính** — {_sm_sessions} phiên data")

                # ── S1 Wyckoff VSA card (v2.0, Sprint 5) ─────────────────────
                _wyk = pick.get("wyckoff") or {}
                if _wyk:
                    _PHASE_BADGE = {
                        "phase_d": "📈 Phase D — Markup",
                        "phase_c": "🌀 Phase C — Spring",
                        "phase_b": "🔲 Phase B — Accumulation",
                        "distribution": "🔴 Distribution",
                        "none": "⬜ Chưa xác định",
                    }
                    st.markdown("---")
                    st.markdown("**🔬 Wyckoff VSA (S1 v2.0)**")
                    _w_cols = st.columns(4)
                    _wyk_score   = _wyk.get("wyckoff_score", 0.0)
                    _spring_q    = _wyk.get("spring_quality", 0.0)
                    _evr         = _wyk.get("effort_vs_result", 0.0)
                    _no_sup      = _wyk.get("no_supply_count", 0.0)
                    _lps_ok      = _wyk.get("lps_detected", False)
                    _stop_v      = _wyk.get("stopping_volume", False)
                    _w_cols[0].metric(
                        "Wyckoff Score", f"{_wyk_score:.2f}",
                        help="Composite [0,1]: phase + spring + LPS + EVR + no-supply"
                    )
                    _w_cols[1].metric(
                        "Spring Quality", f"{_spring_q:.2f}",
                        help="Chất lượng Spring [0,1]: penetration nhỏ, vol thấp, hồi nhanh"
                    )
                    _w_cols[2].metric(
                        "Effort/Result", f"{_evr:+.2f}",
                        help="Dương = demand mạnh; Âm = supply hấp thụ"
                    )
                    _w_cols[3].metric(
                        "No-Supply", f"{_no_sup:.0%}",
                        help="Tỷ lệ no-supply bar trong 10 phiên gần nhất"
                    )
                    _wyk_badges = []
                    if _lps_ok:
                        _wyk_badges.append("✅ LPS confirmed")
                    if _stop_v:
                        _wyk_badges.append("🛑 Stopping Volume")
                    if _wyk_score >= 0.75:
                        st.success(
                            "🔬 Wyckoff mạnh: " + (" | ".join(_wyk_badges) if _wyk_badges else "Phase tích lũy rõ ràng")
                        )
                    elif _wyk_score >= 0.55:
                        st.info(
                            "🔬 Wyckoff trung bình: " + (" | ".join(_wyk_badges) if _wyk_badges else "Đang trong vùng tích lũy")
                        )
                    else:
                        st.caption(
                            "🔬 Wyckoff yếu / chưa rõ. " + (" | ".join(_wyk_badges) if _wyk_badges else "")
                        )

                # Output JSON (compact)
                with st.expander("🔎 Raw JSON", expanded=False):
                    st.code(
                        __import__("json").dumps(
                            {
                                "ticker":          pick["ticker"],
                                "probability":     pick["probability"],
                                "composite_score": pick.get("composite_score"),
                                "expected_return": pick["expected_return"],
                                "pattern":         pick["pattern"],
                                "confidence":      pick["confidence"],
                                "rs":              _rs,
                                "entry":           _entry,
                                "vc":              _vc,
                                "sizing":          _sz,
                                "sm":              _sm,
                                "wyckoff":         _wyk,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        language="json",
                    )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — Full VN30 chart (luôn hiện)
    # ─────────────────────────────────────────────────────────────────────────
    if all_df is not None and not all_df.empty:
        st.divider()
        st.header("🗺️ P_alpha Toàn VN30  *(xếp theo VN Score: pattern › volume › RS › RSI)*")

        fig_heat = px.bar(
            all_df.sort_values("vn_score", ascending=True),
            x="probability",
            y="ticker",
            orientation="h",
            color="vn_score",
            color_continuous_scale=["#cc0000", "#ff8800", "#ffdd00", "#00cc66"],
            range_color=[0.2, 0.8],
            text=all_df.sort_values("vn_score", ascending=True)["probability"].apply(
                lambda x: f"{x:.0%}"
            ),
            labels={"probability": "P_alpha", "ticker": "", "vn_score": "VN Score"},
        )
        fig_heat.add_vline(
            x=saved_prob, line_dash="dot", line_color="white",
            annotation_text=f"threshold {saved_prob:.0%}", annotation_position="top right",
        )
        fig_heat.update_layout(
            height=800,
            xaxis=dict(tickformat=".0%"),
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=20, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_heat)

        # Summary table
        st.dataframe(
            all_df.sort_values("vn_score", ascending=False)[[
                "ticker", "vn_score", "probability", "confidence", "rsi",
                "volume_ratio_5d", "return_1d_pct", "relative_strength", "pattern",
            ]].style.format({
                "vn_score":          "{:.2f}",
                "probability":       "{:.1%}",
                "rsi":               "{:.0f}",
                "volume_ratio_5d":   "{:.2f}×",
                "return_1d_pct":     "{:+.2f}%",
                "relative_strength": "{:+.2f}%",
            }).background_gradient(subset=["vn_score"], cmap="RdYlGn"),
            height=600,
        )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
if model_exists():
    st.divider()
    st.header("📊 Feature Importance")

    feat_imp = get_feature_importance()
    if feat_imp:
        fi_df = pd.DataFrame(
            list(feat_imp.items()), columns=["feature", "importance"]
        ).sort_values("importance", ascending=True).tail(15)

        # Feature name → readable label
        _labels = {
            "return_1d":           "Return 1 ngày",
            "return_3d":           "Return 3 ngày",
            "return_intraday":     "Return intraday",
            "atr_norm":            "Volatility ATR",
            "rsi_14":              "RSI (14)",
            "obv_trend":           "OBV Trend",
            "price_vs_5sma":       "Giá vs SMA5",
            "price_vs_20sma":      "Giá vs SMA20",
            "volume_ratio_5d":     "Volume ×5d",
            "volume_ratio_20d":    "Volume ×20d",
            "body_ratio":          "Candle body",
            "upper_shadow":        "Bóng trên",
            "lower_shadow":        "Bóng dưới",
            "divergence_score":    "Volume Divergence",
            "pull_push_score":     "Pull/Push Score",
            "accumulation_score":  "Wyckoff Phase",
            "vn30_ret_1d":         "VN30 Return",
            "relative_strength":   "Relative Strength",
            "vn30_vs_20sma":       "VN30 vs SMA20",
            "basis_zscore":        "Basis Z-score",
            "basis_extreme_flag":  "Basis Extreme",
        }
        fi_df["label"] = fi_df["feature"].map(_labels).fillna(fi_df["feature"])

        fig_fi = px.bar(
            fi_df,
            x="importance",
            y="label",
            orientation="h",
            color="importance",
            color_continuous_scale="Blues",
            text=fi_df["importance"].apply(lambda x: f"{x:.1%}"),
            labels={"importance": "Importance", "label": ""},
        )
        fig_fi.update_layout(
            height=500,
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=20, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_fi)

        # Top 5 insights
        top5 = list(reversed(fi_df.tail(5)["label"].tolist()))
        st.info(
            "**Top features ảnh hưởng nhất:**\n" +
            "\n".join(f"{i+1}. {f}" for i, f in enumerate(top5))
        )
    else:
        st.caption("Không có feature importance data")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5.5 — In-sample Backtest
# ─────────────────────────────────────────────────────────────────────────────
if model_exists() and "alpha_dataset" in st.session_state:
    st.divider()
    st.header("🔄 In-sample Backtest")
    st.caption(
        "⚠️ **In-sample** — kết quả có overfitting bias. "
        "Dùng để kiểm tra tính nhất quán của tín hiệu, không phải để dự báo hiệu suất thực tế."
    )

    _bt_prob = st.slider(
        "Min probability cho backtest",
        min_value=0.50, max_value=0.90, value=0.65, step=0.05,
        key="bt_min_prob",
    )

    if st.button("▶ Chạy Backtest", key="run_bt"):
        with st.spinner("Đang chạy backtest…"):
            try:
                from ml.model import load_model, load_calibrator
                from ml.feature_engineering import FEATURE_COLS
                _bt_model, _bt_scaler, _bt_meta = load_model()
                _bt_cal = load_calibrator()
                _bt_dataset = st.session_state["alpha_dataset"]
                _bt_result = run_backtest(
                    dataset=_bt_dataset,
                    model=_bt_model,
                    scaler=_bt_scaler,
                    feature_cols=FEATURE_COLS,
                    min_prob=_bt_prob,
                    calibrator=_bt_cal,
                    precision_target=0.35,
                )
                st.session_state["_bt_result"] = _bt_result
            except Exception as _e:
                st.error(f"Lỗi backtest: {_e}")

    _bt = st.session_state.get("_bt_result")
    if _bt is not None:
        # Summary metrics
        _bm_cols = st.columns(6)
        _bm_cols[0].metric("Tổng tín hiệu", _bt.total_signals)
        _bm_cols[1].metric("Tổng trades",   _bt.total_trades)
        _bm_cols[2].metric("Win Rate",       f"{_bt.win_rate:.1%}")
        _bm_cols[3].metric("Avg Return",     f"{_bt.avg_return_pct:+.2f}%")
        _bm_cols[4].metric("Sharpe",         f"{_bt.sharpe:.2f}")
        _bm_cols[5].metric("Max Drawdown",   f"{_bt.max_drawdown_pct:.1f}%")

        _bm_cols2 = st.columns(4)
        _bm_cols2[0].metric("Avg Win",  f"{_bt.avg_win_pct:+.2f}%")
        _bm_cols2[1].metric("Avg Loss", f"{_bt.avg_loss_pct:+.2f}%")
        _bm_cols2[2].metric("Precision", f"{_bt.precision:.1%}",
                            delta=f"target ≥{_bt.precision_target:.0%}",
                            delta_color="normal" if _bt.meets_target else "inverse")
        _bm_cols2[3].metric("Calmar",   f"{_bt.calmar:.2f}")
        if _bt.meets_target:
            st.success(f"✅ Precision {_bt.precision:.1%} ≥ target {_bt.precision_target:.0%}")
        else:
            st.warning(f"⚠️ Precision {_bt.precision:.1%} < target {_bt.precision_target:.0%} — cân nhắc hạ ngưỡng hoặc retrain")

        if _bt.note:
            st.caption(f"📝 {_bt.note}")

        # Equity curve
        if _bt.equity_curve is not None and len(_bt.equity_curve) > 1:
            st.markdown("#### Equity Curve")
            _eq = _bt.equity_curve.reset_index()
            _eq.columns = ["date", "equity"]
            _fig_eq = px.line(
                _eq, x="date", y="equity",
                labels={"equity": "Portfolio (khởi đầu=100)", "date": "Ngày"},
                color_discrete_sequence=["#00cc66"],
            )
            _fig_eq.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=20, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(_fig_eq)

        # By-ticker breakdown
        if _bt.by_ticker is not None and not _bt.by_ticker.empty:
            with st.expander("📋 Kết quả theo ticker", expanded=False):
                st.dataframe(
                    _bt.by_ticker.style.format({
                        "win_rate":    "{:.1%}",
                        "avg_return":  "{:+.2f}%",
                        "total_trades":"{:.0f}",
                    }).background_gradient(subset=["win_rate"], cmap="RdYlGn"),
                )

        # By-confidence breakdown
        if _bt.by_confidence is not None and len(_bt.by_confidence) > 0:
            with st.expander("📋 Kết quả theo ngưỡng xác suất", expanded=False):
                st.dataframe(_bt.by_confidence)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("📖 Methodology & Disclaimer", expanded=False):
    st.markdown("""
    ### Cách hệ thống hoạt động

    **Dataset:**
    - OHLCV lịch sử từ VCI (vnstock 4.x)
    - Tính 21 features cho mỗi (ticker, ngày): return, volatility, momentum, volume, candle, smart money proxy, market context

    **Label:**
    - `y = 1` nếu `(close[T+2] - close[T]) / close[T] >= 5%`
    - `y = 0` ngược lại
    - Class imbalance thường ~10–20% positive → dùng `scale_pos_weight`

    **Model:**
    - LightGBM (preferred) → XGBoost → RandomForest (fallback)
    - Validation: `TimeSeriesSplit(n_splits=3)` → tránh look-ahead bias
    - Metric: ROC-AUC, Precision@threshold=0.65

    **Prediction:**
    - Tính features từ OHLCV hôm nay
    - Xuất `P_alpha = P(return_T+2 ≥ 5%)`
    - Lọc: P ≥ threshold + RSI ≤ 80

    **Giới hạn:**
    - Model train trên pattern lịch sử → không dự đoán được sự kiện đột biến (tin tức, macro)
    - VNDirect API hiện không khả dụng → không có foreign flow thực trong features
    - **KHÔNG phải khuyến nghị đầu tư.** Chỉ mang tính nghiên cứu định lượng.
    """)
