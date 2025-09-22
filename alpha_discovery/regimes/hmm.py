from __future__ import annotations
import numpy as np
import pandas as pd
from ..config import settings

__all__ = ["label_regimes"]


def _rolling_vol(r: pd.Series, w: int) -> pd.Series:
    return r.rolling(w, min_periods=max(5, w//2)).std()


def _rolling_trend(px: pd.Series, w: int) -> pd.Series:
    # simple normalized momentum over window w
    ret = px.pct_change(w)
    return ret


def label_regimes(r_k: pd.Series) -> pd.Series:
    """
    Labels market regimes based on the volatility and trend of the given returns.

    The function computes volatility and trend for a given return series `r_k`,
    then determines thresholds for median values. It classifies regimes into four
    categories {0, 1, 2, 3} based on high/low volatility and upward/downward trend.

    :param r_k: Series of returns for which regimes will be labelled.
    :type r_k: pd.Series
    :return: A series where each value is the regime classification {0, 1, 2, 3}
             corresponding to the indices of the input series.
    :rtype: pd.Series"""
    # build price from returns (relative)
    px = (1.0 + r_k.fillna(0.0)).cumprod()

    vol_w = int(getattr(settings.regimes, 'vol_window', 20))
    tr_w = int(getattr(settings.regimes, 'trend_window', 20))

    vol = _rolling_vol(r_k, vol_w)
    tr = _rolling_trend(px, tr_w)

    vol_med = float(np.nanmedian(vol)) if np.isfinite(vol).any() else np.nan
    tr_med = float(np.nanmedian(tr)) if np.isfinite(tr).any() else np.nan

    vol_hi = (vol > vol_med).astype(int)
    tr_up = (tr > tr_med).astype(int)

    reg = vol_hi * 2 + tr_up  # {0,1,2,3}
    reg = reg.reindex(r_k.index).ffill().fillna(0).astype(int)
    return reg



#    reg = reg.reindex(r_k.index).ffill().fillna(0).astype(int)

