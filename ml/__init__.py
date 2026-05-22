"""
ml/__init__.py
Alpha Signal ML package — v2.2
"""

from .feature_engineering import FEATURE_COLS, compute_stock_features
from .model import MODEL_LABEL_VERSION, is_model_compatible, load_model, model_exists, train_model
from .predictor import get_current_regime, get_feature_importance, predict_all, predict_today
from .regime import RegimeInfo, RegimeState, get_market_regime
from .relative_strength import RSInfo, compute_stock_rs, rank_by_rs
from .entry_timing import EntryZone, compute_entry_zone
from .volume_confirmation import VolumeConfirmation, compute_volume_confirmation
from .portfolio_sizing import PositionSize, compute_position_size
from .backtest import BacktestResult, run_backtest

__all__ = [
    # feature engineering
    "FEATURE_COLS",
    "compute_stock_features",
    # model
    "train_model",
    "load_model",
    "model_exists",
    "is_model_compatible",
    "MODEL_LABEL_VERSION",
    # predictor
    "predict_today",
    "predict_all",
    "get_feature_importance",
    "get_current_regime",
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
    # S4 volume confirmation
    "VolumeConfirmation",
    "compute_volume_confirmation",
    # D3.4 portfolio sizing
    "PositionSize",
    "compute_position_size",
    # backtest
    "BacktestResult",
    "run_backtest",
]
