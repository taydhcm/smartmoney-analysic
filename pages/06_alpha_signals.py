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
from ml.model import model_exists, train_model
from ml.predictor import get_current_regime, get_feature_importance, predict_all, predict_today
from ml.regime import RegimeState

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
        "Ngưỡng xác suất tối thiểu",
        min_value=0.50,
        max_value=0.90,
        value=0.65,
        step=0.05,
        help="Chỉ hiện cổ phiếu có P_alpha >= ngưỡng này",
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
    if model_exists():
        try:
            from ml.model import META_PATH
            import pickle
            with open(META_PATH, "rb") as _f:
                _meta = pickle.load(_f)
            trained_at = _meta.get("trained_at", "unknown")
            st.success(
                f"✅ Model đã sẵn sàng — **{_meta.get('model_type', 'GBM')}**\n\n"
                f"- Train: {_meta.get('n_samples', '?')} rows | "
                f"{_meta.get('n_tickers', '?')} tickers\n"
                f"- CV AUC: **{_meta.get('cv_auc_mean', 0):.3f}** "
                f"± {_meta.get('cv_auc_std', 0):.3f}\n"
                f"- Precision@0.65: **{_meta.get('cv_prec_mean', 0):.2f}**\n"
                f"- Trained at: {trained_at[:19] if trained_at else '?'}"
            )
        except Exception:
            st.success("✅ Model đã sẵn sàng (metadata không đọc được)")
    else:
        st.warning(
            "⚠️ Chưa có model. Nhấn **Train Model** để bắt đầu.\n\n"
            "Yêu cầu: kết nối VCI OHLCV (chạy local)."
        )

with col_train:
    do_train = st.button(
        "🏋️ Train Model",
        type="primary",
        use_container_width=True,
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
        use_container_width=True,
    )

    st.session_state["_model_just_trained"] = True
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
        use_container_width=True,
        disabled=not model_exists(),
    )
with col_all:
    show_all = st.checkbox(
        "Hiện tất cả tickers (không lọc threshold)",
        value=False,
        help="Xem P_alpha của toàn bộ VN30",
    )

if not model_exists():
    st.info("Chưa có model. Train trước để sử dụng prediction.")
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
                if show_all:
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

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — Top Alpha Picks
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.header(f"🏆 Top Alpha Picks  (P ≥ {saved_prob:.0%})")

    if not picks:
        st.warning(
            f"Không có cổ phiếu nào đạt ngưỡng P_alpha ≥ {saved_prob:.0%} hôm nay. "
            "Thử giảm ngưỡng hoặc kiểm tra lại model."
        )
    else:
        st.caption(f"Tìm thấy **{len(picks)}** alpha picks hôm nay")

        # ── Probability bar chart ──────────────────────────────────────────────
        picks_df = pd.DataFrame(picks)

        color_map = {"high": "#00cc66", "medium": "#ffaa00", "low": "#ff6666"}
        fig_bar = go.Figure()
        for _, row in picks_df.iterrows():
            fig_bar.add_trace(
                go.Bar(
                    x=[row["probability"]],
                    y=[row["ticker"]],
                    orientation="h",
                    marker_color=color_map.get(row["confidence"], "#888"),
                    name=row["confidence"],
                    showlegend=False,
                    text=f"{row['probability']:.0%}",
                    textposition="inside",
                    hovertemplate=(
                        f"<b>{row['ticker']}</b><br>"
                        f"P_alpha: {row['probability']:.1%}<br>"
                        f"Pattern: {row['pattern']}<br>"
                        f"Expected: {row['expected_return']}<br>"
                        f"RSI: {row['rsi']:.0f} | Vol×: {row['volume_ratio_5d']:.1f}<extra></extra>"
                    ),
                )
            )

        # Add threshold line
        fig_bar.add_vline(x=saved_prob, line_dash="dash", line_color="white", opacity=0.5)

        fig_bar.update_layout(
            title="P(return T+2 ≥ +5%) per ticker",
            xaxis=dict(tickformat=".0%", range=[0, 1], title="Probability"),
            yaxis=dict(categoryorder="total ascending"),
            height=max(200, len(picks) * 45 + 80),
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Picks cards ────────────────────────────────────────────────────────
        st.subheader("📋 Chi Tiết Alpha Picks")

        conf_colors = {"high": "🟢", "medium": "🟡", "low": "🔴"}

        for pick in picks:
            conf_icon = conf_colors.get(pick["confidence"], "⚪")
            with st.expander(
                f"{conf_icon} **{pick['ticker']}** — P={pick['probability']:.1%}  |  "
                f"{pick['expected_return']}  |  {pick['confidence'].upper()}",
                expanded=(pick["confidence"] == "high"),
            ):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("P_alpha", f"{pick['probability']:.1%}")
                    st.metric("Expected Return", pick["expected_return"])
                with c2:
                    st.metric("RSI (14)", f"{pick['rsi']:.0f}")
                    st.metric("Volume ×5d avg", f"{pick['volume_ratio_5d']:.2f}×")
                with c3:
                    st.metric("Return hôm qua", f"{pick['return_1d_pct']:+.2f}%")
                    st.metric("vs VN30", f"{pick['relative_strength']:+.2f}%")

                st.markdown(f"**Pattern:** `{pick['pattern']}`")

                col_a, col_b, col_c = st.columns(3)
                col_a.progress(pick["accumulation_score"], text=f"Accumulation: {pick['accumulation_score']:.2f}")
                col_b.progress(max(pick["divergence_score"], 0), text=f"Divergence: {pick['divergence_score']:.2f}")
                _pp_norm = (pick["pull_push_score"] + 1) / 2
                col_c.progress(_pp_norm, text=f"Pull/Push: {pick['pull_push_score']:+.2f}")

                # Output JSON
                st.code(
                    json.dumps(
                        {
                            "ticker":          pick["ticker"],
                            "probability":     pick["probability"],
                            "expected_return": pick["expected_return"],
                            "pattern":         pick["pattern"],
                            "confidence":      pick["confidence"],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    language="json",
                )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — Full VN30 heatmap (nếu show_all)
    # ─────────────────────────────────────────────────────────────────────────
    if show_all and all_df is not None and not all_df.empty:
        st.divider()
        st.header("🗺️ P_alpha Toàn VN30")

        fig_heat = px.bar(
            all_df.sort_values("probability", ascending=True),
            x="probability",
            y="ticker",
            orientation="h",
            color="probability",
            color_continuous_scale=["#cc0000", "#ff8800", "#ffdd00", "#00cc66"],
            range_color=[0.3, 0.9],
            text=all_df.sort_values("probability", ascending=True)["probability"].apply(
                lambda x: f"{x:.0%}"
            ),
            labels={"probability": "P_alpha", "ticker": ""},
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
        st.plotly_chart(fig_heat, use_container_width=True)

        # Summary table
        st.dataframe(
            all_df[[
                "ticker", "probability", "confidence", "rsi",
                "volume_ratio_5d", "return_1d_pct", "relative_strength", "pattern",
            ]].style.format({
                "probability":       "{:.1%}",
                "rsi":               "{:.0f}",
                "volume_ratio_5d":   "{:.2f}×",
                "return_1d_pct":     "{:+.2f}%",
                "relative_strength": "{:+.2f}%",
            }).background_gradient(subset=["probability"], cmap="RdYlGn"),
            use_container_width=True,
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
        st.plotly_chart(fig_fi, use_container_width=True)

        # Top 5 insights
        top5 = list(reversed(fi_df.tail(5)["label"].tolist()))
        st.info(
            "**Top features ảnh hưởng nhất:**\n" +
            "\n".join(f"{i+1}. {f}" for i, f in enumerate(top5))
        )
    else:
        st.caption("Không có feature importance data")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Methodology Note
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
