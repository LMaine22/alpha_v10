# alpha_discovery/meta_labeling/features.py
"""
Meta-Feature Extraction for Meta-Labeling

This module extracts entry-time features for meta-labeling, including
options flow, underlying context, sentiment, events, and setup internals.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd

from ..config import settings


class MetaFeatureExtractor:
    """Extracts meta-features from entry-time data for meta-labeling."""
    
    def __init__(self, config: Any):
        """Initialize the feature extractor."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def extract_features(self, setup_id: str, setup_info: Dict, 
                        oos_ledger: pd.DataFrame, master_df: pd.DataFrame,
                        signals_df: pd.DataFrame, signals_metadata: List[Dict]) -> pd.DataFrame:
        """
        Extract meta-features for a setup.
        
        Args:
            setup_id: Setup identifier
            setup_info: Setup information (ticker, direction, etc.)
            oos_ledger: OOS trade ledger
            master_df: Master data frame with all features
            signals_df: Signals data frame
            signals_metadata: Signals metadata
            
        Returns:
            DataFrame with meta-features for each trade
        """
        self.logger.info(f"Extracting meta-features for setup {setup_id}")
        
        # Initialize feature matrix
        features = pd.DataFrame(index=oos_ledger.index)
        
        # Extract different feature categories
        features = self._extract_options_flow_features(features, oos_ledger, master_df)
        features = self._extract_underlying_context_features(features, oos_ledger, master_df)
        features = self._extract_sentiment_features(features, oos_ledger, master_df)
        features = self._extract_event_features(features, oos_ledger, master_df)
        features = self._extract_setup_internal_features(features, setup_id, oos_ledger, signals_df, signals_metadata)
        features = self._extract_recent_performance_features(features, oos_ledger)
        
        # Fill missing values with median for numerical features
        for col in features.columns:
            if features[col].dtype in ['float64', 'int64']:
                features[col] = features[col].fillna(features[col].median())
            else:
                features[col] = features[col].fillna(0)
        
        # Remove any rows with all NaN values
        features = features.dropna(how='all')
        
        self.logger.info(f"Extracted {len(features.columns)} meta-features for {len(features)} trades")
        return features
    
    def _extract_options_flow_features(self, features: pd.DataFrame, 
                                     oos_ledger: pd.DataFrame, 
                                     master_df: pd.DataFrame) -> pd.DataFrame:
        """Extract options flow features."""
        ticker = oos_ledger.get('specialized_ticker', ['Unknown']).iloc[0] if not oos_ledger.empty else 'Unknown'
        
        for feature_name in self.config.options_flow_features:
            # Look for the feature in master_df
            feature_col = f"{ticker} {feature_name}"
            if feature_col in master_df.columns:
                # Get values at entry dates
                entry_dates = oos_ledger['entry_date']
                feature_values = master_df.loc[entry_dates, feature_col]
                features[f"options_flow_{feature_name}"] = feature_values
            else:
                # Try without ticker prefix
                if feature_name in master_df.columns:
                    entry_dates = oos_ledger['entry_date']
                    feature_values = master_df.loc[entry_dates, feature_name]
                    features[f"options_flow_{feature_name}"] = feature_values
                else:
                    self.logger.warning(f"Options flow feature {feature_name} not found for {ticker}")
        
        return features
    
    def _extract_underlying_context_features(self, features: pd.DataFrame,
                                           oos_ledger: pd.DataFrame,
                                           master_df: pd.DataFrame) -> pd.DataFrame:
        """Extract underlying context features."""
        ticker = oos_ledger.get('specialized_ticker', ['Unknown']).iloc[0] if not oos_ledger.empty else 'Unknown'
        
        for feature_name in self.config.underlying_context_features:
            feature_col = f"{ticker} {feature_name}"
            if feature_col in master_df.columns:
                entry_dates = oos_ledger['entry_date']
                feature_values = master_df.loc[entry_dates, feature_col]
                
                # Calculate additional context features
                if feature_name == "VOLATILITY_90D":
                    # Volatility z-score
                    vol_mean = feature_values.mean()
                    vol_std = feature_values.std()
                    if vol_std > 0:
                        features["vol_90d_z"] = (feature_values - vol_mean) / vol_std
                    else:
                        features["vol_90d_z"] = 0.0
                
                elif feature_name == "PX_LAST":
                    # Price momentum features
                    if len(feature_values) > 21:
                        features["momentum_21d"] = feature_values.pct_change(21)
                    if len(feature_values) > 63:
                        features["momentum_63d"] = feature_values.pct_change(63)
                    
                    # Price trend (linear regression slope over 20 days)
                    if len(feature_values) > 20:
                        features["price_trend_20d"] = self._calculate_trend(feature_values, 20)
                
                features[f"underlying_{feature_name}"] = feature_values
            else:
                self.logger.warning(f"Underlying context feature {feature_name} not found for {ticker}")
        
        return features
    
    def _extract_sentiment_features(self, features: pd.DataFrame,
                                  oos_ledger: pd.DataFrame,
                                  master_df: pd.DataFrame) -> pd.DataFrame:
        """Extract sentiment features."""
        ticker = oos_ledger.get('specialized_ticker', ['Unknown']).iloc[0] if not oos_ledger.empty else 'Unknown'
        
        for feature_name in self.config.sentiment_features:
            feature_col = f"{ticker} {feature_name}"
            if feature_col in master_df.columns:
                entry_dates = oos_ledger['entry_date']
                feature_values = master_df.loc[entry_dates, feature_col]
                
                # Calculate sentiment z-score
                if feature_name in ["TWITTER_COUNT", "NEWS_COUNT", "NET_SENTIMENT"]:
                    mean_val = feature_values.mean()
                    std_val = feature_values.std()
                    if std_val > 0:
                        features[f"sentiment_{feature_name}_z"] = (feature_values - mean_val) / std_val
                    else:
                        features[f"sentiment_{feature_name}_z"] = 0.0
                
                features[f"sentiment_{feature_name}"] = feature_values
            else:
                self.logger.warning(f"Sentiment feature {feature_name} not found for {ticker}")
        
        return features
    
    def _extract_event_features(self, features: pd.DataFrame,
                              oos_ledger: pd.DataFrame,
                              master_df: pd.DataFrame) -> pd.DataFrame:
        """Extract event features."""
        # Event features are global (no ticker prefix)
        for feature_name in self.config.event_features:
            if feature_name in master_df.columns:
                entry_dates = oos_ledger['entry_date']
                feature_values = master_df.loc[entry_dates, feature_name]
                features[f"event_{feature_name}"] = feature_values
            else:
                self.logger.warning(f"Event feature {feature_name} not found")
        
        return features
    
    def _extract_setup_internal_features(self, features: pd.DataFrame,
                                       setup_id: str, oos_ledger: pd.DataFrame,
                                       signals_df: pd.DataFrame,
                                       signals_metadata: List[Dict]) -> pd.DataFrame:
        """Extract setup internal features."""
        # Get setup signals from setup_id
        setup_signals = self._parse_setup_signals(setup_id)
        
        if not setup_signals:
            self.logger.warning(f"Could not parse setup signals from {setup_id}")
            return features
        
        # Calculate trigger z-scores and distances to thresholds
        for signal in setup_signals:
            if signal in signals_df.columns:
                entry_dates = oos_ledger['entry_date']
                signal_values = signals_df.loc[entry_dates, signal]
                
                # Trigger z-score (how strong the signal was)
                features[f"trigger_z_{signal}"] = signal_values
                
                # Distance to threshold (how close to the trigger threshold)
                # Assuming binary signals, distance is 1 - signal_value for positive signals
                features[f"distance_to_threshold_{signal}"] = 1.0 - signal_values
        
        return features
    
    def _extract_recent_performance_features(self, features: pd.DataFrame,
                                           oos_ledger: pd.DataFrame) -> pd.DataFrame:
        """Extract recent performance features."""
        # Calculate recent hit rate
        window = self.config.recent_hit_rate_window
        
        if 'pnl_dollars' in oos_ledger.columns:
            pnl_col = 'pnl_dollars'
        elif 'realized_pnl' in oos_ledger.columns:
            pnl_col = 'realized_pnl'
        elif 'pnl_pct' in oos_ledger.columns:
            pnl_col = 'pnl_pct'
        else:
            self.logger.warning("No P&L column found for recent performance features")
            return features
        
        # Calculate rolling hit rate
        pnl_values = oos_ledger[pnl_col]
        recent_hit_rate = pnl_values.rolling(window=window, min_periods=1).apply(
            lambda x: (x > 0).mean(), raw=False
        )
        
        features["recent_hit_rate"] = recent_hit_rate
        
        # Calculate recent average P&L
        recent_avg_pnl = pnl_values.rolling(window=window, min_periods=1).mean()
        features["recent_avg_pnl"] = recent_avg_pnl
        
        # Calculate recent volatility of P&L
        recent_pnl_vol = pnl_values.rolling(window=window, min_periods=1).std()
        features["recent_pnl_vol"] = recent_pnl_vol
        
        return features
    
    def _parse_setup_signals(self, setup_id: str) -> List[str]:
        """Parse setup signals from setup_id."""
        # Setup ID format: "ticker_direction_signal1_signal2_..."
        parts = setup_id.split('_')
        if len(parts) < 3:
            return []
        
        # Skip ticker and direction, rest are signals
        signals = parts[2:]
        return signals
    
    def _calculate_trend(self, values: pd.Series, window: int) -> pd.Series:
        """Calculate linear trend (slope) over a rolling window."""
        def linear_trend(x):
            if len(x) < 2:
                return 0.0
            y = np.arange(len(x))
            slope, _ = np.polyfit(y, x, 1)
            return slope
        
        return values.rolling(window=window, min_periods=2).apply(linear_trend, raw=False)
