"""Regime detection and assignment using GaussianMixture only (no HMM).

Public API:
- RegimeModel(model, scaler, n_regimes, features_used, regime_version, model_type="gmm")
- fit_regimes(df, price_col, K=5, vol_window=20, trend_window=20, version="R1")
- assign_regime(df, price_col, regime_model, vol_window=20, trend_window=20)
- similarity(today_vector, regime_centroid)
"""

from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture


class RegimeModel:
    """Wrapper around GaussianMixture with a stable interface."""
    
    def __init__(
        self,
        model: GaussianMixture,
        scaler: StandardScaler,
        n_regimes: int,
        features_used: List[str],
        regime_version: str,
        model_type: str = "gmm",
    ):
        self.model = model
        self.scaler = scaler
        self.n_regimes = n_regimes
        self.features_used = features_used
        self.regime_version = regime_version
        self.model_type = model_type  # always "gmm"
        # Expose means_ for centroid similarity
        self.means_ = getattr(model, "means_", None)
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict regime labels for feature matrix."""
        Xs = self.scaler.transform(features)
        return self.model.predict(Xs)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict regime probabilities."""
        Xs = self.scaler.transform(features)
        return self.model.predict_proba(Xs)


def _build_regime_features(
    df: pd.DataFrame,
    price_col: str,
    vol_window: int,
    trend_window: int,
) -> pd.DataFrame:
    """Create volatility & trend features from a price series."""
    if price_col not in df.columns:
        return pd.DataFrame()
    
    prices = pd.to_numeric(df[price_col], errors="coerce").dropna()
    if prices.empty:
        return pd.DataFrame()
    
    rets = prices.pct_change()
    
    vol = rets.rolling(vol_window, min_periods=max(2, vol_window // 2)).std()
    tr = rets.rolling(trend_window, min_periods=max(2, trend_window // 2)).sum()
    
    feats = pd.DataFrame({"volatility": vol, "trend": tr}, index=prices.index).dropna()
    return feats


def fit_regimes(
    df: pd.DataFrame,
    price_col: str,
    K: int = 5,
    vol_window: int = 20,
    trend_window: int = 20,
    version: str = "R1",
) -> Tuple[Optional[RegimeModel], pd.DataFrame]:
    """
    Fit a K-regime GaussianMixture on volatility & trend features.
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price series
        K: Number of regimes (default 5)
        vol_window: Window for volatility calculation (default 20)
        trend_window: Window for trend calculation (default 20)
        version: Regime model version tag (default "R1")
        
    Returns:
        (RegimeModel, features_df) or (None, empty_df) if fitting fails
    """
    feats = _build_regime_features(df, price_col, vol_window, trend_window)
    
    # Basic sample sufficiency guards
    if feats.empty or len(feats) < max(K * 10, 50):
        return None, pd.DataFrame()
    
    X = feats.values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    
    try:
        gmm = GaussianMixture(
            n_components=K,
            covariance_type="full",
            random_state=42,
            max_iter=500,
            n_init=5,
        )
        gmm.fit(Xs)
        
        rm = RegimeModel(
            model=gmm,
            scaler=scaler,
            n_regimes=K,
            features_used=["volatility", "trend"],
            regime_version=version,
            model_type="gmm",
        )
        return rm, feats
        
    except Exception as e:
        print(f"[regime] GMM fit failed: {e}")
        return None, pd.DataFrame()


def assign_regime(
    df: pd.DataFrame,
    price_col: str,
    regime_model: RegimeModel,
    vol_window: int = 20,
    trend_window: int = 20,
) -> pd.Series:
    """
    Assign regime labels ('R0'..'R{K-1}') aligned to df.index via forward fill.
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price series
        regime_model: Trained RegimeModel
        vol_window: Window for volatility (must match training)
        trend_window: Window for trend (must match training)
        
    Returns:
        Series of regime labels (strings "R0" to "R{K-1}") indexed to df
    """
    feats = _build_regime_features(df, price_col, vol_window, trend_window)
    if feats.empty:
        return pd.Series(index=df.index, dtype=str)
    
    labels = regime_model.predict(feats.values)
    regime_series = pd.Series([f"R{int(i)}" for i in labels], index=feats.index)
    
    # Reindex to original df index, forward-fill gaps
    return regime_series.reindex(df.index, method="ffill")


def similarity(today_vector: np.ndarray, regime_centroid: np.ndarray) -> float:
    """
    Cosine similarity in [0, 1] between today's vector and a regime centroid.
    
    Args:
        today_vector: Current feature vector (shape: n_features)
        regime_centroid: Regime mean vector (shape: n_features)
        
    Returns:
        Cosine similarity in [0, 1] (1 = identical direction, 0 = orthogonal)
    """
    if today_vector.size == 0 or regime_centroid.size == 0:
        return 0.0
    if today_vector.shape != regime_centroid.shape:
        return 0.0
    
    num = float(np.dot(today_vector, regime_centroid))
    den = float(np.linalg.norm(today_vector) * np.linalg.norm(regime_centroid))
    
    if den == 0.0:
        return 0.0
    
    cos = num / den
    return float(np.clip(cos, 0.0, 1.0))
