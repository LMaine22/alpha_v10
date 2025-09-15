"""
Predictive mini-engine configuration.
Edit ONLY the INTEGRATION_* dotted paths if your registry/split utilities
live under different modules. Everything else should run unchanged.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any

# ---- INTEGRATION HOOKS (update if your project uses different entrypoints) ----
# These dotted paths are imported dynamically by pipeline.py
# 1) Feature builder: must return a DataFrame with index including 'date' and 'ticker'
#    or a MultiIndex (date, ticker). Columns = engineered features.
INTEGRATION_FEATURE_BUILDER = "alpha_discovery.features.core:build_feature_matrix"

# 2) Split builder: must return an iterable of (train_index, test_index) over rows of the
#    (date, ticker) long-frame. If you don't have one, our cv.PurgedWalkForward is used.
INTEGRATION_SPLIT_BUILDER = "alpha_discovery.utils.splits:make_walkforward_splits"

# 3) Optional: path to your processed panel data (if your feature builder needs it)
DEFAULT_PANEL_DATA_PATH = "data_store/processed/bb_data.parquet"

# ---- Horizons & targets --------------------------------------------------------
HORIZONS_DAYS: List[int] = [1, 3, 5]  # 1d / 3d / 5d

# Quantile levels for intervals (used by quantile regressors and WIS/CRPS)
QUANTILES: List[float] = [0.10, 0.50, 0.90]

# ---- Metrics & GA objectives (if you later wire your GA here) ------------------
# Primary scoring used internally during model selection
PRIMARY_REGRESSION_METRIC = "MAE"   # on forward returns
PRIMARY_CLASSIFICATION_METRIC = "AUC"  # on up/down direction

# ---- Model configs -------------------------------------------------------------
# Lightweight defaults with no extra dependencies
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "clf_dir": {  # direction classifier
        "type": "logistic",
        "params": {
            "C": 1.0,
            "max_iter": 200,
            "class_weight": "balanced",
        },
        "calibrate": True,  # Platt scaling via CalibratedClassifierCV
        "cv_folds": 4,
    },
    "reg_median": {  # median return (q50)
        "type": "gbr",
        "params": {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "loss": "squared_error",
            "random_state": 42,
        },
    },
    "reg_q_low": {  # q10
        "type": "gbr_quantile",
        "params": {
            "alpha": 0.10,
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "random_state": 42,
        },
    },
    "reg_q_high": {  # q90
        "type": "gbr_quantile",
        "params": {
            "alpha": 0.90,
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "random_state": 42,
        },
    },
}

# ---- Data / IO -----------------------------------------------------------------
OUTDIR = "artifacts/predict"  # models, forecasts, reports (created if not exist)
PREDICTION_DATE_COL = "date"
TICKER_COL = "ticker"

# ---- Purging / Embargo for time-series CV -------------------------------------
# Used by our internal PurgedWalkForward if you don't provide your own split fn.
EMBARGO_DAYS = 5
N_SPLITS = 5  # walk-forward folds if fallback is used

# ---- Feature hygiene -----------------------------------------------------------
# Columns to drop if they slip into the features frame
LEAKY_COLUMNS = {
    "PX_LAST_FWD_1D", "PX_LAST_FWD_3D", "PX_LAST_FWD_5D",
    "ret_fwd_1d", "ret_fwd_3d", "ret_fwd_5d",
    "target_dir_1d", "target_dir_3d", "target_dir_5d",
}

# ---- Reporting knobs -----------------------------------------------------------
TOP_N_BULLISH = 25  # per horizon, in the daily outlook sheet
CONFIDENCE_MIN = 0.55  # minimum calibrated directional probability to surface

@dataclass
class PredictConfig:
    horizons: List[int] = field(default_factory=lambda: HORIZONS_DAYS)
    quantiles: List[float] = field(default_factory=lambda: QUANTILES)
    outdir: str = OUTDIR
    panel_path: str = DEFAULT_PANEL_DATA_PATH
    feature_builder_path: str = INTEGRATION_FEATURE_BUILDER
    split_builder_path: str = INTEGRATION_SPLIT_BUILDER
    pred_date_col: str = PREDICTION_DATE_COL
    ticker_col: str = TICKER_COL
    embargo_days: int = EMBARGO_DAYS
    n_splits: int = N_SPLITS
    model_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: MODEL_CONFIGS)
    leak_drop: set = field(default_factory=lambda: LEAKY_COLUMNS)
    top_n_bullish: int = TOP_N_BULLISH
    confidence_min: float = CONFIDENCE_MIN
