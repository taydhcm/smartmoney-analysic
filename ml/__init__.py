"""
ml/__init__.py
Alpha Signal ML package — v6.0 (Sprint 7: D3.4/D3.5 conviction sizing + alert generator)
"""

from .feature_engineering import FEATURE_COLS, compute_stock_features
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
from .backtest import BacktestResult, run_backtest
from .smart_money import SmartMoneySignal, compute_smart_money, compute_smart_money_features
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
    # S1 Wyckoff VSA v2.0 (Sprint 5)
    "WyckoffResult",
    "detect_wyckoff",
]
