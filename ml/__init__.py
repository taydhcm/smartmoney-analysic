"""
ml/__init__.py
Alpha Signal ML package — v12.0 (Sprint 13: Sentiment Integration + 43 features)
"""

from .feature_engineering import (
    FEATURE_COLS, SENTIMENT_FEATURE_COLS,
    compute_stock_features, compute_sentiment_features,
)
from .model import (
    MODEL_LABEL_VERSION, CALIBRATOR_PATH,
    is_model_compatible, load_model, load_calibrator, model_exists,
    train_model, wilson_ci,
)
from .predictor import (
    Recommendation,
    get_current_regime, get_feature_importance,
    predict_all, predict_today,
)
from .regime import RegimeInfo, RegimeState, get_market_regime
from .relative_strength import RSInfo, compute_stock_rs, rank_by_rs
from .entry_timing import EntryZone, compute_entry_zone
from .volume_confirmation import VolumeConfirmation, compute_volume_confirmation
from .portfolio_sizing import PositionSize, compute_position_size, compute_conviction_size
from .alert_generator import AlertCard, PortfolioUsage, generate_alert, generate_morning_report, compute_portfolio_usage, _build_reason
from .outcome_tracker import (
    log_signal, log_signals_batch,
    check_pending_outcomes,
    get_recent_signals, get_signal_stats, get_outcome_table,
    SIGNAL_LOG_DB,
)
from .pnl_tracker import (
    compute_trade_pnl, compute_equity_curve, compute_stats, compute_win_streak,
)
from .rolling_calibrator import (
    get_calibration_data,
    fit_rolling_calibrator,
    load_rolling_calibrator,
    apply_rolling_calibration,
    get_calibrator_status,
    needs_weekly_refit,
    ROLLING_CALIBRATOR_PATH,
    ROLLING_META_PATH,
)
from .go_live_checker import (
    GoLiveConfig,
    CheckResult,
    GoLiveStatus,
    run_go_live_checks,
)
from .scale_advisor import (
    ScaleDecision,
    ConditionResult,
    SCALE_STAGES,
    compute_scale_recommendation,
    check_scale_conditions,
)
from .monitor import (
    HealthStatus,
    run_daily_health_check,
    check_d02_history_depth,
    check_retrain_schedule,
    check_milestone_progress,
)
from .backtest import BacktestResult, run_backtest
from .smart_money import (
    SmartMoneySignal, compute_smart_money, compute_smart_money_features,
    # Sprint 12
    ProprietarySignal, InstitutionalFlowSignal,
    compute_institutional_flow, compute_institutional_flow_features,
)
from .trade_log import (
    log_alpha_signals,
    resolve_pending_outcomes,
    get_track_record,
    get_weekly_win_rate,
    get_ticker_stats,
    get_pending_count,
    get_summary_stats,
    TRADE_LOG_DB,
)
from .drift_detector import (
    DriftStatus,
    get_drift_status,
    get_rolling_series,
    DRIFT_WINDOW,
    DRIFT_THRESHOLD_PCT,
    MIN_SAMPLES,
)
from analytics.wyckoff import WyckoffResult, detect_wyckoff

__all__ = [
    # feature engineering
    "FEATURE_COLS",
    "compute_stock_features",
    # model
    "train_model",
    "load_model",
    "load_calibrator",
    "model_exists",
    "is_model_compatible",
    "MODEL_LABEL_VERSION",
    "CALIBRATOR_PATH",
    "wilson_ci",
    # predictor
    "predict_today",
    "predict_all",
    "get_feature_importance",
    "get_current_regime",
    "Recommendation",
    # regime
    "get_market_regime",
    "RegimeState",
    "RegimeInfo",
    # S2 relative strength
    "RSInfo",
    "compute_stock_rs",
    "rank_by_rs",
    # S5 entry timing
    "EntryZone",
    "compute_entry_zone",
    # volume confirmation (Sprint 3)
    "VolumeConfirmation",
    "compute_volume_confirmation",
    # D3.4 portfolio sizing
    "PositionSize",
    "compute_position_size",
    "compute_conviction_size",       # Sprint 7
    # D3.5 alert generator (Sprint 7)
    "AlertCard",
    "PortfolioUsage",
    "generate_alert",
    "generate_morning_report",
    "compute_portfolio_usage",
    "_build_reason",
    # backtest (v2, Sprint 6)
    "BacktestResult",
    "run_backtest",
    # S4 Smart Money Flow (Sprint 4)
    "SmartMoneySignal",
    "compute_smart_money",
    "compute_smart_money_features",
    # Sprint 12: SSI iBoard Tự Doanh
    "ProprietarySignal",
    "InstitutionalFlowSignal",
    "compute_institutional_flow",
    "compute_institutional_flow_features",
    # S1 Wyckoff VSA v2.0 (Sprint 5)
    "WyckoffResult",
    "detect_wyckoff",
    # E5 Outcome Tracker (Sprint 8)
    "log_signal",
    "log_signals_batch",
    "check_pending_outcomes",
    "get_recent_signals",
    "get_signal_stats",
    "get_outcome_table",
    "SIGNAL_LOG_DB",
    # E3 P&L Tracker (Sprint 8)
    "compute_trade_pnl",
    "compute_equity_curve",
    "compute_stats",
    "compute_win_streak",
    # E6 Rolling Calibrator (Sprint 9)
    "get_calibration_data",
    "fit_rolling_calibrator",
    "load_rolling_calibrator",
    "apply_rolling_calibration",
    "get_calibrator_status",
    "needs_weekly_refit",
    "ROLLING_CALIBRATOR_PATH",
    "ROLLING_META_PATH",
    # Sprint 10: Go-Live Validator
    "GoLiveConfig",
    "CheckResult",
    "GoLiveStatus",
    "run_go_live_checks",
    # Sprint 11: Scale Advisor
    "ScaleDecision",
    "ConditionResult",
    "SCALE_STAGES",
    "compute_scale_recommendation",
    "check_scale_conditions",
    # Sprint 11: System Monitor
    "HealthStatus",
    "run_daily_health_check",
    "check_d02_history_depth",
    "check_retrain_schedule",
    "check_milestone_progress",
]
