"""
ml/pnl_tracker.py
E3 P&L Tracker — Sprint 8.

Pure-computation module (không DB).
Tính toán equity curve, win/loss streak, max drawdown từ danh sách trades.

Functions
---------
compute_trade_pnl(trade)             → dict  (realized + unrealized)
compute_equity_curve(trades, cap)    → pd.DataFrame (date, equity, pnl_pct, dd_pct)
compute_stats(trades)                → dict  (win_rate, avg_win, streak, max_dd, ...)
compute_win_streak(trades)           → tuple[int, int]  (cur_streak, max_streak)
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd


# ── Per-trade PnL ─────────────────────────────────────────────────────────────

def compute_trade_pnl(trade: dict[str, Any]) -> dict[str, Any]:
    """
    Tính PnL cho 1 trade dict (từ trade_logger.get_all_trades()).
    Returns dict mở rộng với:
        pnl_pct         : realized nếu closed, None nếu open
        pnl_vnd         : realized VND
        unrealized_pct  : nếu có current_price (từ update_unrealized_pnl)
        r_multiple      : pnl / initial_risk  (nếu đủ data)
        risk_reward     : target - entry / entry - sl
    """
    entry  = trade.get("entry_price", 0) or 0
    sl     = trade.get("sl_price", 0) or 0
    target = trade.get("target_price", 0) or 0
    shares = trade.get("shares", 0) or 0

    # Risk metrics
    risk_per_share   = entry - sl if entry > sl else 0.0
    reward_per_share = target - entry if target > entry else 0.0
    risk_reward      = round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else 0.0
    max_loss_vnd     = round(risk_per_share * shares, 0)

    result = dict(trade)
    result["risk_reward"]   = risk_reward
    result["max_loss_vnd"]  = max_loss_vnd

    # Realized PnL
    exit_price = trade.get("exit_price")
    if exit_price and entry > 0:
        pnl_pct = round((exit_price - entry) / entry * 100, 4)
        pnl_vnd = round((exit_price - entry) * shares, 0)
        r_mult  = round((exit_price - entry) / risk_per_share, 2) if risk_per_share > 0 else 0.0
        result["pnl_pct"]   = pnl_pct
        result["pnl_vnd"]   = pnl_vnd
        result["r_multiple"] = r_mult

    # Unrealized (if current_price provided by update_unrealized_pnl)
    cur = trade.get("current_price")
    if cur and entry > 0 and not exit_price:
        unreal_pct = round((cur - entry) / entry * 100, 4)
        unreal_vnd = round((cur - entry) * shares, 0)
        result["unrealized_pct"] = unreal_pct
        result["unrealized_vnd"] = unreal_vnd

    return result


# ── Equity curve ──────────────────────────────────────────────────────────────

def compute_equity_curve(
    trades: list[dict[str, Any]],
    initial_capital: float = 1_000_000_000,
) -> pd.DataFrame:
    """
    Xây dựng equity curve từ danh sách trades (đã closed).

    Returns DataFrame với cột:
        date        : exit_date (YYYY-MM-DD)
        pnl_vnd     : realized PnL của trade đó
        equity      : tích lũy equity từ initial_capital
        equity_pct  : equity / initial_capital - 1
        drawdown_pct: max drawdown từ peak (negative value)

    Chỉ tính trade đã closed.
    """
    closed = [
        t for t in trades
        if t.get("status") in ("closed", "stopped")
        and t.get("exit_date")
        and t.get("pnl_vnd") is not None
    ]

    if not closed:
        return pd.DataFrame(columns=["date", "pnl_vnd", "equity", "equity_pct", "drawdown_pct"])

    df = pd.DataFrame(closed)[["exit_date", "pnl_vnd"]].copy()
    df.columns = ["date", "pnl_vnd"]
    df = df.sort_values("date").reset_index(drop=True)

    df["equity"]      = initial_capital + df["pnl_vnd"].cumsum()
    df["equity_pct"]  = (df["equity"] / initial_capital - 1).round(6)
    df["peak"]        = df["equity"].cummax()
    df["drawdown_pct"] = ((df["equity"] - df["peak"]) / df["peak"]).round(6)

    return df[["date", "pnl_vnd", "equity", "equity_pct", "drawdown_pct"]]


# ── Win / loss streak ─────────────────────────────────────────────────────────

def compute_win_streak(trades: list[dict[str, Any]]) -> tuple[int, int]:
    """
    Trả về (current_streak, max_streak).
    Positive streak = wins, negative streak = losses.
    current_streak: giá trị hiện tại (ví dụ +3 = đang thắng 3 lệnh liên tiếp,
                    -2 = đang thua 2 lệnh liên tiếp).
    max_streak: chuỗi thắng liên tiếp dài nhất.
    """
    closed = sorted(
        [t for t in trades
         if t.get("status") in ("closed", "stopped")
         and t.get("pnl_pct") is not None],
        key=lambda t: t.get("exit_date", ""),
    )
    if not closed:
        return 0, 0

    wins = [t["pnl_pct"] > 0 for t in closed]

    cur_streak = 0
    max_streak = 0
    cur_cnt    = 0
    last_win   = None

    for w in wins:
        if last_win is None or w == last_win:
            cur_cnt += 1
        else:
            cur_cnt = 1
        last_win = w
        if w:
            max_streak = max(max_streak, cur_cnt)

    # current streak: positive = đang win, negative = đang lose
    cur_cnt = 0
    last_win = None
    for w in wins:
        if last_win is None or w == last_win:
            cur_cnt += 1
        else:
            cur_cnt = 1
        last_win = w
    cur_streak = cur_cnt if (last_win is True) else -cur_cnt

    return cur_streak, max_streak


# ── Aggregate stats ───────────────────────────────────────────────────────────

def compute_stats(
    trades: list[dict[str, Any]],
    initial_capital: float = 1_000_000_000,
) -> dict[str, Any]:
    """
    Tổng hợp thống kê cho portfolio.

    Returns dict:
        total, open_count, closed_count
        win_count, loss_count, win_rate
        avg_win_pct, avg_loss_pct, avg_pnl_pct
        profit_factor
        max_drawdown_pct
        total_pnl_vnd, total_pnl_pct
        current_streak, max_win_streak
        sharpe_ratio  (annualized, approx — nếu >= 5 trades)
    """
    closed = [
        t for t in trades
        if t.get("status") in ("closed", "stopped")
        and t.get("pnl_pct") is not None
    ]
    open_t = [t for t in trades if t.get("status") == "open"]
    wins   = [t for t in closed if t["pnl_pct"] > 0]
    losses = [t for t in closed if t["pnl_pct"] <= 0]

    avg_win  = np.mean([t["pnl_pct"] for t in wins])   if wins   else 0.0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0.0
    avg_pnl  = np.mean([t["pnl_pct"] for t in closed]) if closed else 0.0

    gross_win  = sum(t.get("pnl_vnd") or 0 for t in wins)
    gross_loss = abs(sum(t.get("pnl_vnd") or 0 for t in losses))
    profit_factor = round(gross_win / gross_loss, 4) if gross_loss > 0 else 999.0
    total_pnl_vnd = sum(t.get("pnl_vnd") or 0 for t in closed)

    # Equity curve for max drawdown
    eq_df = compute_equity_curve(closed, initial_capital)
    max_dd = float(eq_df["drawdown_pct"].min()) if not eq_df.empty else 0.0

    # Total return %
    total_pnl_pct = (total_pnl_vnd / initial_capital * 100) if initial_capital > 0 else 0.0

    # Streak
    cur_streak, max_win_streak = compute_win_streak(trades)

    # Sharpe (crude, trade-level)
    sharpe = 0.0
    if len(closed) >= 5:
        pnls = [t["pnl_pct"] for t in closed]
        mu, sigma = np.mean(pnls), np.std(pnls, ddof=1)
        sharpe = round((mu / sigma) * np.sqrt(252 / max(len(closed), 1)), 4) if sigma > 0 else 0.0

    return dict(
        total            = len(trades),
        open_count       = len(open_t),
        closed_count     = len(closed),
        win_count        = len(wins),
        loss_count       = len(losses),
        win_rate         = round(len(wins) / len(closed), 4) if closed else 0.0,
        avg_win_pct      = round(float(avg_win), 4),
        avg_loss_pct     = round(float(avg_loss), 4),
        avg_pnl_pct      = round(float(avg_pnl), 4),
        profit_factor    = profit_factor,
        max_drawdown_pct = round(max_dd * 100, 4),
        total_pnl_vnd    = round(total_pnl_vnd, 0),
        total_pnl_pct    = round(total_pnl_pct, 4),
        current_streak   = cur_streak,
        max_win_streak   = max_win_streak,
        sharpe_ratio     = sharpe,
    )
