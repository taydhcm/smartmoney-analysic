"""
ml/backtest.py
Backtest module — in-sample simulation trên historical dataset.

Flow:
  1. Nhận dataset (output của build_dataset) + trained model
  2. Predict P_alpha cho MỌI row trong dataset
  3. Signal = P >= min_prob (in-sample, có overfitting bias — ghi chú rõ ở UI)
  4. Trade outcome = actual label (path_max_5d >= 5% AND path_min_5d > -5%)
  5. Ước tính return mỗi trade từ path_max_5d / path_min_5d thực tế
  6. Tính equity curve, Sharpe, drawdown, win rate, ...

v2 (Sprint 6): Calibrated probability support + precision target reporting.

Lưu ý: Đây là in-sample backtest. Để có kết quả unbiased, dùng walk-forward CV
(train trên 70% earliest data, test trên 30% remaining). Xem comment bên dưới.
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
    note:             str = "In-sample — có overfitting bias. Dùng walk-forward CV để đánh giá thực."

    def summary(self) -> dict:
        return {
            "total_trades":    self.total_trades,
            "win_rate":        f"{self.win_rate:.1%}",
            "precision":       f"{self.precision:.3f}",
            "precision_target": f"{self.precision_target:.2f}",
            "meets_target":    self.meets_target,
            "avg_return":      f"{self.avg_return_pct:+.2f}%",
            "avg_win":         f"{self.avg_win_pct:+.2f}%",
            "avg_loss":        f"{self.avg_loss_pct:+.2f}%",
            "max_drawdown":    f"{self.max_drawdown_pct:.2f}%",
            "sharpe":          f"{self.sharpe:.2f}",
            "calmar":          f"{self.calmar:.2f}",
        }


def run_backtest(
    dataset:              pd.DataFrame,
    model,
    scaler,
    feature_cols:         list[str],
    min_prob:             float = 0.65,
    initial_equity:       float = 100.0,
    position_size_pct:    float = POSITION_SIZE,
    precision_target:     float = 0.35,          # Sprint 6: target precision
    calibrator=None,                              # Sprint 6: isotonic calibrator (optional)
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
        probs = model.predict_proba(X_scaled)[:, 1]
    except Exception as exc:
        log.error("Backtest predict_proba loi: %s", exc)
        raise

    # Sprint 6: dùng calibrated probs nếu có calibrator
    if calibrator is not None:
        try:
            raw_probs = model.predict_proba(X_scaled)[:, 1]
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

    trade_return = np.where(
        win_mask,
        signals_df["path_max_5d"].clip(upper=TARGET_PCT * 2),     # win: capped at 10%
        signals_df["path_min_5d"].clip(lower=SL_PCT * 2),         # loss: floored at -10%
    )
    signals_df["trade_return"] = trade_return

    total_trades = len(signals_df)
    win_rate     = float(signals_df["label"].mean())

    win_returns  = signals_df.loc[win_mask,  "trade_return"]
    loss_returns = signals_df.loc[loss_mask, "trade_return"]

    avg_return  = float(signals_df["trade_return"].mean())
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
        "label", "trade_return", "path_max_5d", "path_min_5d",
    ]].copy()
    trades_out["trade_return_pct"] = (trades_out["trade_return"] * 100).round(2)
    trades_out["p_alpha"]          = trades_out["p_alpha"].round(4)
    trades_out["p_calibrated"]     = trades_out["p_calibrated"].round(4)
    trades_out = trades_out.drop(columns=["trade_return"])

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
    )


def _empty_result(note: str = "") -> BacktestResult:
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
        note=note or "Khong du du lieu de backtest.",
    )
