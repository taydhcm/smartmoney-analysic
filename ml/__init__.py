"""
ml/__init__.py
Alpha Signal ML package — v2.0
"""

from .feature_engineering import FEATURE_COLS, compute_stock_features
from .model import MODEL_LABEL_VERSION, is_model_compatible, load_model, model_exists, train_model
from .predictor import get_current_regime, get_feature_importance, predict_all, predict_today
from .regime import RegimeInfo, RegimeState, get_market_regime

__all__ = [
    "FEATURE_COLS",
    "compute_stock_features",
    "train_model",
    "load_model",
    "model_exists",
    "is_model_compatible",
    "MODEL_LABEL_VERSION",
    "predict_today",
    "predict_all",
    "get_feature_importance",
    "get_current_regime",
    "get_market_regime",
    "RegimeState",
    "RegimeInfo",
]
