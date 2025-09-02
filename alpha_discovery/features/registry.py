# alpha_discovery/features/registry.py

import pandas as pd
import numpy as np
import itertools
from typing import Dict, Callable, List, Tuple

from ..config import settings
from . import core as fcore  # fcore for "feature core"
from ..data.events import build_event_features


def _get_series(df: pd.DataFrame, ticker: str, column: str) -> pd.Series:
    """Extract a single ticker/column series from the main dataframe."""
    col_name = f"{ticker}_{column}"
    if col_name in df.columns:
        return pd.to_numeric(df[col_name], errors="coerce")
    # Empty series with the same index if the column is not found
    return pd.Series(index=df.index, dtype=float)


# ===================================================================
# FEATURE DEFINITION REGISTRY
# Each entry defines HOW to build a feature.
# - key: A descriptive feature name.
# - value: lambda(df, t) or lambda(df, t1, t2) -> pd.Series
# All features are lagged by 1 in build_feature_matrix to avoid lookahead.
# ===================================================================

FEATURE_SPECS: Dict[str, Callable] = {
    # =========================
    # A) CORE (Kept)
    # =========================
    "px_zscore_90d": lambda df, t: fcore.zscore_rolling(
        _get_series(df, t, "PX_LAST"), window=90
    ),
    "realized_vol_21d": lambda df, t: fcore.get_realized_vol(
        _get_series(df, t, "PX_LAST"), window=21
    ),
    "news_pub_z_30d": lambda df, t: fcore.spike_z(
        _get_series(df, t, "NEWS_PUBLICATION_COUNT"), 30
    ),
    "news_heat_z_30d": lambda df, t: fcore.spike_z(
        _get_series(df, t, "NEWS_HEAT_READ_DMAX"), 30
    ),
    "tw_pub_z_30d": lambda df, t: fcore.spike_z(
        _get_series(df, t, "TWITTER_PUBLICATION_COUNT"), 30
    ),
    "tw_sent_avg_dev_7d": lambda df, t: fcore.deviation_from_mean(
        _get_series(df, t, "TWITTER_SENTIMENT_DAILY_AVG"), 7
    ),
    "vol_turnover_zscore_60d": lambda df, t: fcore.zscore_rolling(
        _get_series(df, t, "TURNOVER"), window=60
    ),

    # =========================
    # B) PRICE / MICROSTRUCTURE (New)
    # =========================
    "px_trend_5_20_z_60d": lambda df, t: fcore.zscore_rolling(
        fcore.momentum_trend(_get_series(df, t, "PX_LAST"), 5, 20), window=60
    ),
    "px_range_polarity_z_60d": lambda df, t: fcore.zscore_rolling(
        fcore.range_polarity(
            _get_series(df, t, "PX_LAST"),
            _get_series(df, t, "PX_LOW"),
            _get_series(df, t, "PX_HIGH"),
        ),
        window=60,
    ),
    "turnover_minus_volume_z_60d": lambda df, t: (
        fcore.zscore_rolling(_get_series(df, t, "TURNOVER"), 60)
        - fcore.zscore_rolling(_get_series(df, t, "PX_VOLUME"), 60)
    ),

    # =========================
    # C) DERIVATIVES / VOL (IVOL-FREE)
    # =========================
    "iv_call3m_shock_1d_z_30d": lambda df, t: fcore.zscore_rolling(
        fcore.diff_n(_get_series(df, t, "3MO_CALL_IMP_VOL"), 1), window=30
    ),
    "iv_term_call3m_minus_rv21_z_60d": lambda df, t: fcore.zscore_rolling(
        _get_series(df, t, "3MO_CALL_IMP_VOL") - fcore.get_realized_vol(_get_series(df, t, "PX_LAST"), 21),
        window=60,
    ),
    "iv_skew_put_minus_call_3mo_z_60d": lambda df, t: fcore.zscore_rolling(
        _get_series(df, t, "3MO_PUT_IMP_VOL") - _get_series(df, t, "3MO_CALL_IMP_VOL"),
        window=60,
    ),
    "moneyness_tilt_call3m_minus_moneyness_z_60d": lambda df, t: fcore.zscore_rolling(
        _get_series(df, t, "3MO_CALL_IMP_VOL") - _get_series(df, t, "IVOL_MONEYNESS"),
        window=60,
    ),
    "iv_vs_newsheat_divergence_call3m_z_60d": lambda df, t: (
        fcore.zscore_rolling(_get_series(df, t, "3MO_CALL_IMP_VOL"), 60)
        - fcore.zscore_rolling(_get_series(df, t, "NEWS_HEAT_READ_DMAX"), 60)
    ),
    "options_total_volume_spike_z_30d": lambda df, t: fcore.zscore_rolling(
        _get_series(df, t, "TOT_OPT_VOLUME_CUR_DAY"), window=30
    ),
    "options_vs_equity_flow_z_60d": lambda df, t: fcore.zscore_rolling(
        fcore.safe_divide(
            _get_series(df, t, "TOT_OPT_VOLUME_CUR_DAY"),
            _get_series(df, t, "PX_VOLUME"),
        ),
        window=60,
    ),
    "oi_callput_skew_z_60d": lambda df, t: fcore.zscore_rolling(
        fcore.safe_divide(
            (_get_series(df, t, "OPEN_INT_TOTAL_CALL") - _get_series(df, t, "OPEN_INT_TOTAL_PUT")),
            (_get_series(df, t, "OPEN_INT_TOTAL_CALL") + _get_series(df, t, "OPEN_INT_TOTAL_PUT")),
        ),
        window=60,
    ),
    "open_interest_call_pctchg_3d_z_60d": lambda df, t: fcore.zscore_rolling(
        fcore.pct_change_n(_get_series(df, t, "OPEN_INT_TOTAL_CALL"), 3), window=60
    ),

    # =========================
    # D) SENTIMENT / NEWS COMPOSITES (New)
    # =========================
    "tw_posneg_balance_z_60d": lambda df, t: fcore.zscore_rolling(
        fcore.safe_divide(
            (_get_series(df, t, "TWITTER_POS_SENTIMENT_COUNT") - _get_series(df, t, "TWITTER_NEG_SENTIMENT_COUNT")),
            (_get_series(df, t, "TWITTER_POS_SENTIMENT_COUNT") + _get_series(df, t, "TWITTER_NEG_SENTIMENT_COUNT")),
        ),
        window=60,
    ),
    "px_vs_sentiment_divergence_z_60d": lambda df, t: fcore.zscore_rolling(
        (
            fcore.pct_change_n(_get_series(df, t, "PX_LAST"), 7)
            - fcore.diff_n(_get_series(df, t, "TWITTER_SENTIMENT_DAILY_AVG"), 7)
        ),
        window=60,
    ),

    # =========================
    # E) FLOW / PCR (Replacement)
    # =========================
    "pcr_ema5_z_30d": lambda df, t: fcore.zscore_rolling(
        _get_series(df, t, "PUT_CALL_VOLUME_RATIO_CUR_DAY").ewm(span=5, min_periods=2).mean(),
        window=30,
    ),

    # =========================
    # F) REGIME
    # =========================
    "vol90d_regime_z_120d": lambda df, t: fcore.zscore_rolling(
        _get_series(df, t, "VOLATILITY_90D"), window=120
    ),

    # =========================
    # G) CROSS-ASSET (vs SPY): Fisher-z corr + deltas, and z-scored beta
    # =========================
    "corr_px_fisher20_z_60d": lambda df, t1, t2: fcore.zscore_rolling(
        fcore.rolling_corr_fisher(
            _get_series(df, t1, "PX_LAST").pct_change(),
            _get_series(df, t2, "PX_LAST").pct_change(),
            window=20,
        )[1],
        window=60,
    ),
    "corr_delta_fisher_20_60_z_60d": lambda df, t1, t2: fcore.zscore_rolling(
        (
            fcore.rolling_corr_fisher(
                _get_series(df, t1, "PX_LAST").pct_change(),
                _get_series(df, t2, "PX_LAST").pct_change(),
                window=20,
            )[1]
            - fcore.rolling_corr_fisher(
                _get_series(df, t1, "PX_LAST").pct_change(),
                _get_series(df, t2, "PX_LAST").pct_change(),
                window=60,
            )[1]
        ),
        window=60,
    ),
    "beta_px_60d_z_120d": lambda df, t1, t2: fcore.zscore_rolling(
        fcore.rolling_beta(
            _get_series(df, t1, "PX_LAST").pct_change(),
            _get_series(df, t2, "PX_LAST").pct_change(),
            window=60,
        ),
        window=120,
    ),
}


# ---------- Helpers for Option A correlation block ----------

def _ret_1d(df: pd.DataFrame, t: str) -> pd.Series:
    return fcore.pct_change_n(_get_series(df, t, "PX_LAST"), 1)

def _oi_skew_ratio(df: pd.DataFrame, t: str) -> pd.Series:
    c = _get_series(df, t, "OPEN_INT_TOTAL_CALL")
    p = _get_series(df, t, "OPEN_INT_TOTAL_PUT")
    return fcore.safe_divide((c - p), (c + p))

def _pcr_ema5(df: pd.DataFrame, t: str) -> pd.Series:
    return _get_series(df, t, "PUT_CALL_VOLUME_RATIO_CUR_DAY").ewm(span=5, min_periods=2).mean()

def _name_safe(s: str) -> str:
    return s.replace(" ", "_")


# ===================================================================
# THE FEATURE BUILDER
# ===================================================================

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs a complete matrix of lagged features for all tickers.

    Iterates through all defined feature specs and tickers, calculates each
    feature, and applies shift(1) to prevent lookahead bias.
    Also appends EV_* daily features (same for all tickers) and the
    Option A correlation block.
    """
    print("Starting feature matrix construction...")
    all_features: Dict[str, pd.Series] = {}
    all_tickers = settings.data.tradable_tickers + settings.data.macro_tickers

    # --- Build Single-Asset Features ---
    for ticker in all_tickers:
        for spec_name, spec_func in FEATURE_SPECS.items():
            if spec_func.__code__.co_argcount == 2:  # (df, t)
                feature_name = f"{ticker}_{spec_name}"
                try:
                    raw_feature = spec_func(df, ticker)
                    all_features[feature_name] = pd.to_numeric(raw_feature, errors="coerce").shift(1)
                except Exception as e:
                    print(f" Could not build feature '{feature_name}': {e}")

    print(f" Built {len(all_features)} single-asset features.")

    # --- Build Cross-Asset Features (vs benchmark) ---
    benchmark_ticker = 'SPY US Equity'
    cross_asset_features: Dict[str, pd.Series] = {}

    for ticker in all_tickers:
        if ticker == benchmark_ticker:
            continue
        for spec_name, spec_func in FEATURE_SPECS.items():
            if spec_func.__code__.co_argcount == 3:  # (df, t1, t2)
                feature_name = f"{ticker}_vs_{benchmark_ticker}_{spec_name}"
                try:
                    raw_feature = spec_func(df, ticker, benchmark_ticker)
                    cross_asset_features[feature_name] = pd.to_numeric(raw_feature, errors="coerce").shift(1)
                except Exception as e:
                    print(f" Could not build feature '{feature_name}': {e}")

    print(f" Built {len(cross_asset_features)} cross-asset features.")
    all_features.update(cross_asset_features)

    # --- EV_* Event Features (added once, then replicated per ticker) ---
    print("Building event features from economic calendar...")
    try:
        ev_df = build_event_features(df.index)
        if isinstance(ev_df, pd.DataFrame) and not ev_df.empty:
            ev_cols = ev_df.columns.tolist()
            # replicate per tradable ticker only (flow vars exist there)
            for t in settings.data.tradable_tickers:
                for col in ev_cols:
                    all_features[f"{t}_EV__{col}"] = pd.to_numeric(ev_df[col], errors="coerce").shift(1)
            print(f"  Successfully built {len(ev_cols)} event features.")
            print(f" Added {len(ev_cols) * len(settings.data.tradable_tickers)} EV_* per-ticker columns.")
        else:
            print(" No EV_* features added (empty calendar or load issue).")
    except Exception as e:
        print(f" Could not build EV_* features: {e}")

    # --- Option A: Correlation features (within-ticker + SPY driver) ---
    corr_features: Dict[str, pd.Series] = {}
    windows = [20, 60]
    min_support = 126  # require enough overlap for stability

    def fisher_corr(s1: pd.Series, s2: pd.Series, win: int) -> pd.Series:
        # Corr on aligned series
        try:
            _, fisher = fcore.rolling_corr_fisher(s1, s2, window=win)
            return pd.to_numeric(fisher, errors="coerce")
        except Exception:
            return pd.Series(index=df.index, dtype=float)

    for t in settings.data.tradable_tickers:
        # Base series
        RET_1D = _ret_1d(df, t)
        TOT_OPT = _get_series(df, t, "TOT_OPT_VOLUME_CUR_DAY")
        OI_SKEW = _oi_skew_ratio(df, t)
        CALLIV3M = _get_series(df, t, "3MO_CALL_IMP_VOL")
        NEWS_HEAT = _get_series(df, t, "NEWS_HEAT_READ_DMAX")
        PCR5 = _pcr_ema5(df, t)

        # SPY driver
        SPY_RET_1D = _ret_1d(df, "SPY US Equity")

        # Pair list (7 per ticker)
        pairs: List[Tuple[str, pd.Series, str, pd.Series]] = [
            ("RET_1D", RET_1D, "TOT_OPT_VOLUME", TOT_OPT),
            ("RET_1D", RET_1D, "OI_SKEW", OI_SKEW),
            ("RET_1D", RET_1D, "CALLIV3M", CALLIV3M),
            ("RET_1D", RET_1D, "NEWS_HEAT", NEWS_HEAT),
            ("RET_1D", RET_1D, "PCR_EMA5", PCR5),
            ("SPY_RET_1D", SPY_RET_1D, "TOT_OPT_VOLUME", TOT_OPT),
            ("SPY_RET_1D", SPY_RET_1D, "OI_SKEW", OI_SKEW),
        ]

        for lhs_name, lhs, rhs_name, rhs in pairs:
            # Windows
            fishers = {}
            for w in windows:
                fz = fisher_corr(lhs, rhs, w).shift(1)  # anti-leakage
                # support gate
                if fz.count() < min_support:
                    continue
                fishers[w] = fz

                base_name = f"{t}_corrF{w}__{_name_safe(lhs_name)}__{_name_safe(rhs_name)}"
                corr_features[base_name] = fz
                corr_features[f"{t}_invcorrF{w}__{_name_safe(lhs_name)}__{_name_safe(rhs_name)}"] = -fz

            # Delta(20–60)
            if 20 in fishers and 60 in fishers:
                delta = (fishers[20] - fishers[60]).shift(0)  # already shifted above
                if delta.count() >= min_support:
                    base = f"{t}_corrDelta20_60__{_name_safe(lhs_name)}__{_name_safe(rhs_name)}"
                    corr_features[base] = delta
                    corr_features[f"{t}_invcorrDelta20_60__{_name_safe(lhs_name)}__{_name_safe(rhs_name)}"] = -delta

    print(f" Built {len(corr_features)} correlation features (Option A).")
    all_features.update(corr_features)

    # Combine, prune all-NaN columns
    feature_matrix = pd.DataFrame(all_features).dropna(how="all", axis=1)

    print(f" Feature matrix construction complete. Shape: {feature_matrix.shape}")
    return feature_matrix
