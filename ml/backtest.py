"""
ml/backtest.py
Backtest module — in-sample và walk-forward simulation trên historical dataset.

Hai chế độ:
  1. In-sample  : Model train trên toàn bộ dataset rồi test lại chính dataset đó.
                  Win rate ~95% là overfitting — chỉ dùng debug/consistency check.
  2. Walk-Forward: Train 70% đầu theo thời gian, test 30% cuối (out-of-sample).
                  Win rate phản ánh thực tế hơn, sát với CV metrics (≈ 20-35%).

v3 (Module A): Walk-forward backtest + BacktestResult.mode field.
v2 (Sprint 6): Calibrated probability support + precision target reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from utils.logger import get_logger

log = get_logger(__name__)

# ── Trade simulation constants ────────────────────────────────────────────────
TARGET_PCT    = 0.05    # M2 target +5%
SL_PCT        = -0.05   # M2 stop-loss -5%
POSITION_SIZE = 0.10    # 10% vốn mỗi trade (fixed, đơn giản hoá)

# ── Transaction cost (Module C) ───────────────────────────────────────────────
# Thị trường VN thực tế: mua 0.15% + bán 0.15% + thuế GTGT 0.10% + slippage 0.0%
ROUND_TRIP_COST = 0.004   # 0.40% mỗi round-trip (mua + bán)


@dataclass
class BacktestResult:
    """Kết quả backtest đầy đủ."""
    # ── Tổng quan ─────────────────────────────────────────────────────────────
    total_signals:    int            # Số tín hiệu phát ra (P >= min_prob)
    total_trades:     int            # Trades có đủ future data (label không NA)
    win_rate:         float          # % trades đạt target (label=1)
    avg_return_pct:   float          # Return trung bình mỗi trade (%)
    avg_win_pct:      float          # Return trung bình khi thắng (%)
    avg_loss_pct:     float          # Return trung bình khi thua (%)
    precision:        float          # Precision (= win_rate cho binary)
    precision_target: float          # Target precision (default 0.35, Sprint 6)
    meets_target:     bool           # precision >= precision_target (Sprint 6)
    # ── Risk metrics ──────────────────────────────────────────────────────────
    max_drawdown_pct: float          # Max drawdown equity curve (%)
    sharpe:           float          # Annualized Sharpe (simplified)
    calmar:           float          # Annualized return / Max drawdown
    # ── Chi tiết ──────────────────────────────────────────────────────────────
    trades_df:        pd.DataFrame   # Trade log: date, ticker, p_alpha, label, return
    equity_curve:     pd.Series      # Equity curve theo ngày (normalized)
    by_ticker:        pd.DataFrame   # Win rate + avg return theo ticker
    by_confidence:    dict           # Breakdown theo confidence level (high/medium/low)
    note:             str = "In-sample — có overfitting bias. Dùng walk-forward để đánh giá thực."
    # ── Walk-forward metadata (None khi in-sample) ─────────────────────────────
    mode:             str             = "in-sample"   # "in-sample" | "walk-forward"
    split_date:       str | None      = None          # Ngày phân tách train/test
    train_rows:       int             = 0             # Số rows dùng train
    test_rows:        int             = 0             # Số rows dùng test
    # ── Transaction cost (Module C) ────────────────────────────────────────────
    gross_avg_return_pct: float       = 0.0           # Avg return trước phí (%)
    round_trip_cost_pct:  float       = 0.0           # Phí round-trip đã áp dụng (%)

    def summary(self) -> dict:
        return {
            "mode":                  self.mode,
            "total_trades":          self.total_trades,
            "win_rate":              f"{self.win_rate:.1%}",
            "precision":             f"{self.precision:.3f}",
            "precision_target":      f"{self.precision_target:.2f}",
            "meets_target":          self.meets_target,
            "gross_avg_return":      f"{self.gross_avg_return_pct:+.2f}%",
            "avg_return (net)":      f"{self.avg_return_pct:+.2f}%",
            "round_trip_cost":       f"{self.round_trip_cost_pct:.2f}%",
            "avg_win":               f"{self.avg_win_pct:+.2f}%",
            "avg_loss":              f"{self.avg_loss_pct:+.2f}%",
            "max_drawdown":          f"{self.max_drawdown_pct:.2f}%",
            "sharpe":                f"{self.sharpe:.2f}",
            "calmar":                f"{self.calmar:.2f}",
            "split_date":            self.split_date,
            "train_rows":            self.train_rows,
            "test_rows":             self.test_rows,
        }


def run_backtest(
    dataset:              pd.DataFrame,
    model,
    scaler,
    feature_cols:         list[str],
    min_prob:             float = 0.65,
    initial_equity:       float = 100.0,
    position_size_pct:    float = POSITION_SIZE,
    precision_target:     float = 0.35,
    calibrator=None,
    round_trip_cost:      float = 0.0,            # Module C: phí round-trip (0.0 = không tính)
) -> BacktestResult:
    """
    Chạy in-sample backtest trên dataset lịch sử.

    Parameters
    ----------
    dataset          : Output của build_dataset() — cần có path_max_5d, path_min_5d, label.
    model            : Trained model (có predict_proba).
    scaler           : Trained scaler (transform).
    feature_cols     : Danh sách feature names theo đúng thứ tự model.
    min_prob         : Ngưỡng signal (P >= min_prob → vào lệnh).
    initial_equity   : Vốn khởi đầu (normalized, e.g. 100 = 100 đơn vị).
    position_size_pct: % vốn mỗi trade.
    precision_target : Target precision để đánh giá meets_target (Sprint 6, default 0.35).
    calibrator       : Isotonic calibrator (Sprint 6). Nếu có, dùng p_calibrated cho signal.
    round_trip_cost  : Phí round-trip mỗi trade (Module C). 0.004 = 0.40% VN thực tế.
                       0.0 (default) = không tính phí (backward compat).

    Returns
    -------
    BacktestResult
    """
    # ── Early exit on empty input ─────────────────────────────────────────────
    if dataset is None or dataset.empty:
        return _empty_result("Dataset rỗng")

    # ── Validate ─────────────────────────────────────────────────────────────
    required_cols = set(feature_cols) | {"path_max_5d", "path_min_5d", "label", "ticker", "date"}
    missing = required_cols - set(dataset.columns)
    if missing:
        log.error("Backtest: dataset thieu columns: %s", missing)
        raise ValueError(f"Dataset thieu: {sorted(missing)}")

    # Drop rows NA label (5 rows cuoi moi ticker)
    df = dataset.dropna(subset=["label", "path_max_5d", "path_min_5d"]).copy()
    df["label"] = df["label"].astype(int)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    if len(df) < 10:
        log.warning("Backtest: qua it rows (%d) sau khi drop NA", len(df))
        return _empty_result()

    # ── Predict P_alpha cho toan bo dataset ──────────────────────────────────
    X_raw = df[feature_cols].values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=3.0, neginf=-3.0)

    try:
        X_scaled = scaler.transform(X_raw)
        # Wrap DataFrame để preserve feature names cho LightGBM
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
        probs = model.predict_proba(X_scaled_df)[:, 1]
    except Exception as exc:
        log.error("Backtest predict_proba loi: %s", exc)
        raise

    # Sprint 6: dùng calibrated probs nếu có calibrator
    if calibrator is not None:
        try:
            raw_probs = model.predict_proba(X_scaled_df)[:, 1]
            probs_cal = np.clip(calibrator.predict(raw_probs), 0.0, 1.0)
        except Exception:
            probs_cal = probs
    else:
        probs_cal = probs

    df["p_alpha"]     = probs
    df["p_calibrated"] = probs_cal
    df["confidence"]  = pd.cut(
        probs_cal,
        bins=[0, 0.70, 0.80, 1.0],
        labels=["low", "medium", "high"],
        right=True,
    ).astype(str)
    df["signal"]      = probs_cal >= min_prob

    # ── Loc lay trades (signal=True) ─────────────────────────────────────────
    signals_df = df[df["signal"]].copy()
    total_signals = int(df["signal"].sum())

    if signals_df.empty:
        log.warning("Backtest: khong co signal voi min_prob=%.2f", min_prob)
        return _empty_result()

    # ── Estimate trade return tu path_max/path_min ────────────────────────────
    # Win (label=1): actual path_max_5d (cap 2x target de tranh outlier)
    # Loss (label=0): actual path_min_5d (floor 2x SL)
    win_mask  = signals_df["label"] == 1
    loss_mask = ~win_mask

    gross_return = np.where(
        win_mask,
        signals_df["path_max_5d"].clip(upper=TARGET_PCT * 2),
        signals_df["path_min_5d"].clip(lower=SL_PCT * 2),
    )
    # Module C: trừ phí round-trip mỗi trade
    net_return = gross_return - round_trip_cost

    signals_df["gross_return"] = gross_return
    signals_df["trade_return"] = net_return    # net dùng cho equity curve + metrics

    total_trades = len(signals_df)
    win_rate     = float(signals_df["label"].mean())

    win_returns  = signals_df.loc[win_mask,  "trade_return"]
    loss_returns = signals_df.loc[loss_mask, "trade_return"]

    gross_avg   = float(gross_return.mean())
    avg_return  = float(net_return.mean())
    avg_win     = float(win_returns.mean())  if not win_returns.empty  else 0.0
    avg_loss    = float(loss_returns.mean()) if not loss_returns.empty else 0.0

    # ── Equity curve ─────────────────────────────────────────────────────────
    # Group theo date: neu co nhieu trade 1 ngay → average return cua ngay do
    date_returns = (
        signals_df
        .groupby("date")["trade_return"]
        .mean()
        .sort_index()
    )

    equity = [float(initial_equity)]
    eq_dates: list[str] = []
    for dt, ret in date_returns.items():
        equity.append(equity[-1] * (1.0 + ret * position_size_pct))
        eq_dates.append(str(dt))

    equity_series = pd.Series(equity[1:], index=eq_dates, name="equity")

    # Max drawdown
    eq_arr    = np.array(equity)
    roll_max  = np.maximum.accumulate(eq_arr)
    drawdowns = (eq_arr - roll_max) / (roll_max + 1e-9)
    max_dd    = float(drawdowns.min() * 100)  # in %

    # Sharpe (annualized, 250 days, on position-sized returns)
    pos_sized_rets = date_returns * position_size_pct
    sharpe = (
        float(pos_sized_rets.mean() / (pos_sized_rets.std() + 1e-9) * np.sqrt(250))
        if len(pos_sized_rets) > 1
        else 0.0
    )

    # Calmar = annualized return / |max_drawdown|
    total_return_pct = (equity[-1] - initial_equity) / initial_equity * 100
    calmar = total_return_pct / abs(max_dd) if abs(max_dd) > 0.01 else 0.0

    # ── By ticker ─────────────────────────────────────────────────────────────
    by_ticker = (
        signals_df
        .groupby("ticker")
        .agg(
            trades      =("trade_return", "count"),
            win_rate    =("label",        "mean"),
            avg_return  =("trade_return", "mean"),
            avg_p_alpha =("p_alpha",      "mean"),
        )
        .reset_index()
        .sort_values("avg_return", ascending=False)
    )
    by_ticker["win_rate"]   = by_ticker["win_rate"].round(3)
    by_ticker["avg_return"] = (by_ticker["avg_return"] * 100).round(2)
    by_ticker["avg_p_alpha"] = by_ticker["avg_p_alpha"].round(3)

    # ── By confidence ─────────────────────────────────────────────────────────
    by_confidence: dict[str, dict] = {}
    for conf in ["high", "medium", "low"]:
        sub = signals_df[signals_df["confidence"] == conf]
        if not sub.empty:
            by_confidence[conf] = {
                "trades":      len(sub),
                "win_rate":    round(float(sub["label"].mean()), 3),
                "avg_return":  round(float(sub["trade_return"].mean()) * 100, 2),
            }

    # ── Trade log ────────────────────────────────────────────────────────────
    trades_out = signals_df[[
        "date", "ticker", "p_alpha", "p_calibrated", "confidence",
        "label", "gross_return", "trade_return", "path_max_5d", "path_min_5d",
    ]].copy()
    trades_out["gross_return_pct"] = (trades_out["gross_return"] * 100).round(2)
    trades_out["net_return_pct"]   = (trades_out["trade_return"] * 100).round(2)
    trades_out["p_alpha"]          = trades_out["p_alpha"].round(4)
    trades_out["p_calibrated"]     = trades_out["p_calibrated"].round(4)
    trades_out = trades_out.drop(columns=["gross_return", "trade_return"])

    return BacktestResult(
        total_signals=total_signals,
        total_trades=total_trades,
        win_rate=round(win_rate, 4),
        avg_return_pct=round(avg_return * 100, 3),
        avg_win_pct=round(avg_win * 100, 3),
        avg_loss_pct=round(avg_loss * 100, 3),
        precision=round(win_rate, 4),
        precision_target=float(precision_target),
        meets_target=bool(win_rate >= precision_target),
        max_drawdown_pct=round(max_dd, 3),
        sharpe=round(sharpe, 3),
        calmar=round(calmar, 3),
        trades_df=trades_out,
        equity_curve=equity_series,
        by_ticker=by_ticker,
        by_confidence=by_confidence,
        gross_avg_return_pct=round(gross_avg * 100, 3),
        round_trip_cost_pct=round(round_trip_cost * 100, 3),
    )


def _empty_result(note: str = "", mode: str = "in-sample") -> BacktestResult:
    """Trả về BacktestResult rỗng khi không đủ dữ liệu."""
    return BacktestResult(
        total_signals=0, total_trades=0, win_rate=0.0,
        avg_return_pct=0.0, avg_win_pct=0.0, avg_loss_pct=0.0,
        precision=0.0, precision_target=0.35, meets_target=False,
        max_drawdown_pct=0.0, sharpe=0.0, calmar=0.0,
        trades_df=pd.DataFrame(),
        equity_curve=pd.Series(dtype=float),
        by_ticker=pd.DataFrame(),
        by_confidence={},
        note=note or "Không đủ dữ liệu để backtest.",
        mode=mode,
    )


# ── Walk-Forward Backtest (Module A) ─────────────────────────────────────────

def run_walk_forward_backtest(
    dataset:           pd.DataFrame,
    feature_cols:      list[str],
    min_prob:          float = 0.65,
    train_ratio:       float = 0.70,
    initial_equity:    float = 100.0,
    position_size_pct: float = POSITION_SIZE,
    precision_target:  float = 0.35,
    round_trip_cost:   float = 0.0,   # Module C: phí round-trip (0.0 = không tính)
) -> BacktestResult:
    """
    Walk-forward backtest thực sự (out-of-sample).

    Pipeline:
      1. Sort dataset theo thời gian
      2. Train 70% đầu → model tạm (không ghi đè model production)
      3. Test 30% cuối → predict, không có data leakage
      4. Trả về metrics trên tập test

    Win rate kết quả (~20-35%) phản ánh thực tế, sát với CV Precision@0.65.

    Parameters
    ----------
    dataset         : Output của build_dataset().
    feature_cols    : Danh sách feature names.
    min_prob        : Ngưỡng signal.
    train_ratio     : Tỷ lệ data train (default 0.70 = 70% đầu theo thời gian).
    initial_equity  : Vốn khởi đầu.
    position_size_pct: % vốn mỗi trade.
    precision_target: Target precision.
    round_trip_cost : Phí round-trip mỗi trade (Module C). 0.004 = 0.40% VN thực tế.

    Returns
    -------
    BacktestResult với mode="walk-forward", split_date, train_rows, test_rows.
    """
    from sklearn.preprocessing import StandardScaler as _SS

    MODE = "walk-forward"

    if dataset is None or dataset.empty:
        return _empty_result("Dataset rỗng", mode=MODE)

    required_cols = set(feature_cols) | {"path_max_5d", "path_min_5d", "label", "ticker", "date"}
    missing = required_cols - set(dataset.columns)
    if missing:
        raise ValueError(f"Dataset thiếu columns: {sorted(missing)}")

    # ── Sort + drop NA ────────────────────────────────────────────────────────
    df = dataset.dropna(subset=["label", "path_max_5d", "path_min_5d"]).copy()
    df["label"] = df["label"].astype(int)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    if len(df) < 60:
        return _empty_result(
            f"Chỉ có {len(df)} rows sau khi drop NA — cần ít nhất 60 để walk-forward.",
            mode=MODE,
        )

    # ── Tách train / test theo thời gian ─────────────────────────────────────
    split_idx  = int(len(df) * train_ratio)
    split_idx  = max(split_idx, 30)   # đảm bảo train có ít nhất 30 rows
    train_df   = df.iloc[:split_idx].copy()
    test_df    = df.iloc[split_idx:].copy()
    split_date = str(test_df.iloc[0]["date"].date() if hasattr(test_df.iloc[0]["date"], "date") else test_df.iloc[0]["date"])

    log.info(
        "[WF] Train: %d rows (%s → %s) | Test: %d rows (%s → %s)",
        len(train_df),
        str(train_df.iloc[0]["date"])[:10],
        str(train_df.iloc[-1]["date"])[:10],
        len(test_df),
        str(test_df.iloc[0]["date"])[:10],
        str(test_df.iloc[-1]["date"])[:10],
    )

    if len(test_df) < 20:
        return _empty_result(
            f"Tập test chỉ có {len(test_df)} rows — quá nhỏ. Thử dùng period dài hơn (12m).",
            mode=MODE,
        )

    # ── Train model tạm trên train_df ────────────────────────────────────────
    feat_train = train_df[feature_cols].copy()
    feat_train = feat_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_train    = np.clip(feat_train.values.astype(np.float32), -1e6, 1e6)
    y_train    = train_df["label"].values.astype(int)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = max(n_neg / max(n_pos, 1), 1.0)

    try:
        import lightgbm as lgb
        wf_model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            num_leaves=20, min_child_samples=10, subsample=0.8,
            colsample_bytree=0.7, scale_pos_weight=pos_weight,
            random_state=42, verbose=-1, n_jobs=-1,
        )
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        wf_model = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )

    wf_scaler = _SS()
    X_train_scaled = wf_scaler.fit_transform(X_train)
    X_train_df     = pd.DataFrame(X_train_scaled, columns=feature_cols)
    wf_model.fit(X_train_df, y_train)

    # ── Predict trên test_df (out-of-sample) ─────────────────────────────────
    feat_test  = test_df[feature_cols].copy()
    feat_test  = feat_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test     = np.clip(feat_test.values.astype(np.float32), -1e6, 1e6)
    X_test_scaled = wf_scaler.transform(X_test)
    X_test_df     = pd.DataFrame(X_test_scaled, columns=feature_cols)

    probs = wf_model.predict_proba(X_test_df)[:, 1]
    test_df = test_df.copy()
    test_df["p_alpha"]     = probs
    test_df["p_calibrated"] = probs   # không calibrate riêng để tránh bias
    test_df["confidence"]  = pd.cut(
        probs,
        bins=[0, 0.70, 0.80, 1.0],
        labels=["low", "medium", "high"],
        right=True,
    ).astype(str)
    test_df["signal"] = probs >= min_prob

    total_signals = int(test_df["signal"].sum())
    signals_df    = test_df[test_df["signal"]].copy()

    if signals_df.empty:
        return _empty_result(
            f"Không có tín hiệu nào trong tập test với min_prob={min_prob:.2f}. "
            "Thử hạ min_prob hoặc dùng period dài hơn.",
            mode=MODE,
        )

    # ── Tính trade returns ────────────────────────────────────────────────────
    win_mask  = signals_df["label"] == 1
    loss_mask = ~win_mask

    gross_return = np.where(
        win_mask,
        signals_df["path_max_5d"].clip(upper=TARGET_PCT * 2),
        signals_df["path_min_5d"].clip(lower=SL_PCT * 2),
    )
    # Module C: trừ phí round-trip
    net_return = gross_return - round_trip_cost

    signals_df = signals_df.copy()
    signals_df["gross_return"] = gross_return
    signals_df["trade_return"] = net_return

    total_trades = len(signals_df)
    win_rate     = float(signals_df["label"].mean())

    win_returns  = signals_df.loc[win_mask,  "trade_return"]
    loss_returns = signals_df.loc[loss_mask, "trade_return"]

    gross_avg  = float(gross_return.mean())
    avg_return = float(net_return.mean())
    avg_win    = float(win_returns.mean())  if not win_returns.empty  else 0.0
    avg_loss   = float(loss_returns.mean()) if not loss_returns.empty else 0.0

    # ── Equity curve ─────────────────────────────────────────────────────────
    date_returns = (
        signals_df
        .groupby("date")["trade_return"]
        .mean()
        .sort_index()
    )
    equity:     list[float] = [float(initial_equity)]
    eq_dates:   list[str]   = []
    for dt, ret in date_returns.items():
        equity.append(equity[-1] * (1.0 + ret * position_size_pct))
        eq_dates.append(str(dt))
    equity_series = pd.Series(equity[1:], index=eq_dates, name="equity")

    eq_arr    = np.array(equity)
    roll_max  = np.maximum.accumulate(eq_arr)
    drawdowns = (eq_arr - roll_max) / (roll_max + 1e-9)
    max_dd    = float(drawdowns.min() * 100)

    pos_sized_rets = date_returns * position_size_pct
    sharpe = (
        float(pos_sized_rets.mean() / (pos_sized_rets.std() + 1e-9) * np.sqrt(250))
        if len(pos_sized_rets) > 1 else 0.0
    )

    total_return_pct = (equity[-1] - initial_equity) / initial_equity * 100
    calmar = total_return_pct / abs(max_dd) if abs(max_dd) > 0.01 else 0.0

    # ── By ticker ─────────────────────────────────────────────────────────────
    by_ticker = (
        signals_df
        .groupby("ticker")
        .agg(
            trades      =("trade_return", "count"),
            win_rate    =("label",        "mean"),
            avg_return  =("trade_return", "mean"),
            avg_p_alpha =("p_alpha",      "mean"),
        )
        .reset_index()
        .sort_values("avg_return", ascending=False)
    )
    by_ticker["win_rate"]    = by_ticker["win_rate"].round(3)
    by_ticker["avg_return"]  = (by_ticker["avg_return"] * 100).round(2)
    by_ticker["avg_p_alpha"] = by_ticker["avg_p_alpha"].round(3)

    # ── By confidence ─────────────────────────────────────────────────────────
    by_confidence: dict[str, dict] = {}
    for conf in ["high", "medium", "low"]:
        sub = signals_df[signals_df["confidence"] == conf]
        if not sub.empty:
            by_confidence[conf] = {
                "trades":     len(sub),
                "win_rate":   round(float(sub["label"].mean()), 3),
                "avg_return": round(float(sub["trade_return"].mean()) * 100, 2),
            }

    # ── Trade log ─────────────────────────────────────────────────────────────
    trades_out = signals_df[[
        "date", "ticker", "p_alpha", "confidence",
        "label", "gross_return", "trade_return", "path_max_5d", "path_min_5d",
    ]].copy()
    trades_out["gross_return_pct"] = (trades_out["gross_return"] * 100).round(2)
    trades_out["net_return_pct"]   = (trades_out["trade_return"] * 100).round(2)
    trades_out["p_alpha"]          = trades_out["p_alpha"].round(4)
    trades_out = trades_out.drop(columns=["gross_return", "trade_return"])

    wf_note = (
        f"Walk-forward: train {len(train_df)} rows → test {len(test_df)} rows "
        f"(split {split_date}). Không có data leakage."
        + (f" Phí: {round_trip_cost*100:.2f}%/round-trip." if round_trip_cost > 0 else "")
    )

    return BacktestResult(
        total_signals=total_signals,
        total_trades=total_trades,
        win_rate=round(win_rate, 4),
        avg_return_pct=round(avg_return * 100, 3),
        avg_win_pct=round(avg_win * 100, 3),
        avg_loss_pct=round(avg_loss * 100, 3),
        precision=round(win_rate, 4),
        precision_target=float(precision_target),
        meets_target=bool(win_rate >= precision_target),
        max_drawdown_pct=round(max_dd, 3),
        sharpe=round(sharpe, 3),
        calmar=round(calmar, 3),
        trades_df=trades_out,
        equity_curve=equity_series,
        by_ticker=by_ticker,
        by_confidence=by_confidence,
        note=wf_note,
        mode=MODE,
        split_date=split_date,
        train_rows=len(train_df),
        test_rows=len(test_df),
        gross_avg_return_pct=round(gross_avg * 100, 3),
        round_trip_cost_pct=round(round_trip_cost * 100, 3),
    )
