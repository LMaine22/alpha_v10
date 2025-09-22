from __future__ import annotations
from typing import List, Dict, Optional, Any, NamedTuple
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import warnings
from datetime import date

# Filter out convergence warnings from hmmlearn, which are common and expected
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from ..config import settings

class RegimeModel(NamedTuple):
    """
    A wrapper for a trained HMM model, its scaler, and metadata.
    """
    model: hmm.GaussianHMM
    scaler: StandardScaler
    n_regimes: int
    features_used: List[str]
    # Store original (pre-alignment) means for diagnostics
    original_means: np.ndarray
    # ID of the fold this model was trained on (e.g., "cv_fold_4")
    anchor_model_id: Optional[str] = None
    # Store the permutation map used to align labels to an anchor
    alignment_map: Optional[Dict[int, int]] = None


def _calculate_features(df: pd.DataFrame, price_field: str) -> pd.DataFrame:
    """Calculates volatility and trend features for regime detection."""
    cfg = settings.regimes
    features = pd.DataFrame(index=df.index)
    
    px = df.get(price_field, df.iloc[:, 0] if not df.columns.empty else df)
    returns = px.pct_change()

    # Ensure we use a consistent feature order
    feature_list = []
    if "volatility" in cfg.features:
        features["volatility"] = returns.rolling(window=cfg.vol_window).std()
        feature_list.append("volatility")
    if "trend" in cfg.features:
        features["trend"] = returns.rolling(window=cfg.trend_window).mean()
        feature_list.append("trend")

    return features[feature_list].dropna()


def fit_regimes(train_df: pd.DataFrame, price_field: str) -> tuple[Optional[RegimeModel], pd.DataFrame]:
    """
    Fits a Gaussian HMM, selecting the best number of regimes (K) via BIC.
    """
    if not settings.regimes.enabled:
        return None, pd.DataFrame()

    features = _calculate_features(train_df, price_field)
    if features.empty or features.shape[1] == 0:
        return None, pd.DataFrame()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(features)

    best_model, best_bic, best_k = None, np.inf, -1
    bic_scores = {}

    k_range = settings.regimes.n_regimes_range
    for k in range(min(k_range), max(k_range) + 1):
        try:
            model = hmm.GaussianHMM(
                n_components=k, covariance_type="full", n_iter=150,
                random_state=settings.ga.seed, tol=1e-3
            )
            model.fit(X_train)
            if not model.monitor_.converged:
                continue
            
            bic = model.bic(X_train)
            bic_scores[k] = bic
            
            if bic < best_bic:
                best_bic, best_model, best_k = bic, model, k
        except (ValueError, ConvergenceWarning):
            continue

    if best_model:
        model_result = RegimeModel(
            model=best_model, scaler=scaler, n_regimes=best_k,
            features_used=list(features.columns),
            original_means=best_model.means_
        )
        return model_result, pd.DataFrame.from_dict(bic_scores, orient='index', columns=['BIC'])
    
    return None, pd.DataFrame.from_dict(bic_scores, orient='index', columns=['BIC'])


def align_and_map_regimes(anchor_model: RegimeModel, other_model: RegimeModel) -> RegimeModel:
    """
    Aligns the labels of `other_model` to `anchor_model` using the Hungarian algorithm
    on the state means and returns a new, aligned model.
    """
    # Use cdist to compute the Euclidean distance between all pairs of state means
    cost_matrix = cdist(anchor_model.original_means, other_model.original_means, 'euclidean')
    
    # The Hungarian algorithm finds the minimum weight matching in a bipartite graph
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # The result is the optimal mapping: anchor_label -> other_label
    # We want the inverse: other_label -> anchor_label
    alignment_map = {original_label: anchor_label for anchor_label, original_label in zip(row_ind, col_ind)}
    
    return other_model._replace(alignment_map=alignment_map)


def assign_regimes(df: pd.DataFrame, price_field: str, regime_model: RegimeModel) -> pd.Series:
    """
    Assigns (and aligns) regime labels to new data using a pre-trained model.
    """
    if not settings.regimes.enabled or not regime_model:
        return pd.Series(0, index=df.index, name="regime")

    features = _calculate_features(df, price_field)
    if features.empty:
        return pd.Series(np.nan, index=df.index, name="regime")

    # Re-align features to the order used in training
    features = features[regime_model.features_used]
    X = regime_model.scaler.transform(features)
    raw_labels = regime_model.model.predict(X)
    
    # Apply the alignment map if it exists
    if regime_model.alignment_map:
        aligned_labels = pd.Series(raw_labels).map(regime_model.alignment_map).values
    else:
        aligned_labels = raw_labels # Anchor model has no map
    
    return pd.Series(aligned_labels, index=features.index, name="regime").reindex(df.index).ffill()


def regime_breadth(edge_by_regime: Dict[int, float], cohort_median_edge: float, n_total_regimes: int) -> float:
    """
    Calculates the fraction of regimes where a setup's edge exceeds the cohort median.
    """
    if not edge_by_regime:
        return 0.0
    
    above_median_count = sum(1 for edge in edge_by_regime.values() if edge >= cohort_median_edge)
    
    return above_median_count / n_total_regimes if n_total_regimes > 0 else 0.0


def regime_overlap(window_regimes: pd.Series, favorable_regimes: List[int]) -> float:
    """
    Calculates the fraction of time a given window spends in favorable regimes.
    """
    if window_regimes.empty or not favorable_regimes:
        return 0.0
    
    return window_regimes.isin(favorable_regimes).mean()


def regime_today(regime_assignments: pd.Series, as_of_date: Optional[date] = None) -> Optional[int]:
    """
    Returns the aligned regime for the most recent date available in the series.
    """
    if regime_assignments.empty:
        return None
    
    target_date = pd.to_datetime(as_of_date) if as_of_date else regime_assignments.index.max()
    
    try:
        # Get the most recent regime label at or before the target date
        return int(regime_assignments.asof(target_date))
    except (KeyError, TypeError, ValueError):
        return None


def tau_cv_reg_weighted(
    trigger_rate_by_regime: Dict[int, float], 
    regime_occupancy_train: Dict[int, float]
) -> float:
    """
    Computes the regime-weighted CV trigger rate for the ELV prior.
    τ̂_CV,reg = Σ (trigger_rate_in_regime_i * occupancy_of_regime_i)
    """
    weighted_rates = [
        trigger_rate_by_regime.get(regime, 0.0) * occupancy
        for regime, occupancy in regime_occupancy_train.items()
    ]
    return sum(weighted_rates)
