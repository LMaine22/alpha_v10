from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List

def _ensure_sorted(df: pd.DataFrame, date_col: str, ticker_col: str) -> pd.DataFrame:
    return df.sort_values([ticker_col, date_col]).reset_index(drop=True)

def add_forward_returns(long_df: pd.DataFrame,
                        horizons: List[int],
                        price_col: str = "PX_LAST",
                        date_col: str = "date",
                        ticker_col: str = "ticker") -> pd.DataFrame:
    """
    Adds forward log returns for each horizon per (ticker, date).
    Expects a long frame with columns [date, ticker, PX_LAST, ...].
    """
    df = _ensure_sorted(long_df, date_col, ticker_col).copy()
    for h in horizons:
        fwd = df.groupby(ticker_col)[price_col].shift(-h)
        cur = df[price_col]
        ret = np.log(fwd / cur)
        df[f"ret_fwd_{h}d"] = ret
    return df

def add_direction_labels(df: pd.DataFrame,
                         horizons: List[int]) -> pd.DataFrame:
    """
    Adds binary direction labels: 1 if ret_fwd_h > 0 else 0.
    """
    out = df.copy()
    for h in horizons:
        out[f"target_dir_{h}d"] = (out[f"ret_fwd_{h}d"] > 0).astype("float32")
    return out

def slice_xy(df: pd.DataFrame,
             y_cols: List[str],
             feature_cols: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = df[feature_cols].copy()
    y = df[y_cols].copy()
    return X, y
