from __future__ import annotations
import importlib
from typing import Callable, List, Tuple
import os
import pandas as pd
import numpy as np

from .config import PredictConfig
from .targets import add_forward_returns, add_direction_labels
from .cv import purged_walk_forward_indices

def _import_callable(dotted: str) -> Callable:
    """
    Import a function from a dotted path like 'pkg.mod:func'
    """
    if ":" not in dotted:
        raise ValueError(f"Expected 'module:callable', got {dotted}")
    mod, fn = dotted.split(":")
    m = importlib.import_module(mod)
    return getattr(m, fn)

def _to_long_index(df: pd.DataFrame, date_col: str, ticker_col: str) -> pd.DataFrame:
    # Ensure long format with [date, ticker, ...]
    if date_col in df.columns and ticker_col in df.columns:
        return df
    if isinstance(df.index, pd.MultiIndex) and df.index.nlevels == 2:
        df = df.reset_index()
        df.columns = [date_col, ticker_col] + list(df.columns[2:])
        return df
    # Heuristic: if columns contain tickers wide-form, stack them (not common in your pipeline)
    raise ValueError("Feature frame must contain date/ticker columns or MultiIndex (date,ticker).")

def build_features(cfg: PredictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calls your project's feature builder to get the (date,ticker)-indexed features.
    Returns both the feature matrix and the original panel data.
    """
    builder = _import_callable(cfg.feature_builder_path)
    
    # Load panel data
    panel_data = pd.read_parquet(cfg.panel_path)
    
    # Build features from panel data
    feat = builder(panel_data)
    
    # Drop any leak columns that might appear
    leak = [c for c in cfg.leak_drop if c in feat.columns]
    if leak:
        feat = feat.drop(columns=leak)
    
    return feat, panel_data

def attach_targets(feat: pd.DataFrame, panel_data: pd.DataFrame, cfg: PredictConfig) -> pd.DataFrame:
    """
    Adds forward return and direction targets for configured horizons, per ticker.
    Uses panel_data for PX_LAST prices and merges with feature matrix.
    """
    # Extract PX_LAST columns from panel_data
    price_cols = [col for col in panel_data.columns if col.endswith('_PX_LAST')]
    if not price_cols:
        raise ValueError("No PX_LAST columns found in panel_data")
    
    # Create long format data for target calculation
    long_data_list = []
    for col in price_cols:
        ticker = col.replace('_PX_LAST', '')
        price_series = panel_data[col].dropna()
        if len(price_series) > 0:
            df_ticker = pd.DataFrame({
                cfg.pred_date_col: price_series.index,
                cfg.ticker_col: ticker,
                'PX_LAST': price_series.values
            })
            long_data_list.append(df_ticker)
    
    if not long_data_list:
        raise ValueError("No valid price data found")
    
    long_data = pd.concat(long_data_list, ignore_index=True)
    
    # Calculate targets
    df = add_forward_returns(long_data, cfg.horizons, price_col='PX_LAST',
                             date_col=cfg.pred_date_col, ticker_col=cfg.ticker_col)
    df = add_direction_labels(df, cfg.horizons)
    
    # Convert feature matrix to long format for merging
    feat_reset = feat.reset_index()
    feat_reset = feat_reset.rename(columns={feat_reset.columns[0]: cfg.pred_date_col})
    
    # Merge features with targets
    result = df.merge(feat_reset, on=cfg.pred_date_col, how='left')
    
    return result

def make_splits(df: pd.DataFrame, cfg: PredictConfig):
    """
    Uses your project's split builder if available, else falls back to purged walk-forward.
    """
    try:
        splitter = _import_callable(cfg.split_builder_path)
        return list(splitter(df))
    except Exception:
        dates = df[cfg.pred_date_col]
        return list(purged_walk_forward_indices(dates, n_splits=cfg.n_splits, embargo_days=cfg.embargo_days))

def feature_target_columns(df: pd.DataFrame, cfg: PredictConfig, horizon: int) -> Tuple[List[str], List[str]]:
    """
    Infer feature vs target columns for a given horizon.
    """
    y_cols = [f"ret_fwd_{horizon}d", f"target_dir_{horizon}d"]
    missing = [c for c in y_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing targets {missing}. Did attach_targets() run?")
    # features are everything else except date/ticker and any target cols
    drop = set([cfg.pred_date_col, cfg.ticker_col] + [c for c in df.columns if c.startswith("ret_fwd_") or c.startswith("target_dir_")])
    X_cols = [c for c in df.columns if c not in drop]
    return X_cols, y_cols
