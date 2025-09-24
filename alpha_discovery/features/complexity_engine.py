# complexity_engine.py - Lightweight high-performance complexity features
from __future__ import annotations
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Callable
from functools import lru_cache
import warnings

from . import core as f


class FastComplexityEngine:
    """Ultra-lightweight complexity engine focused on speed."""
    
    def __init__(self):
        # Use simple in-memory LRU cache
        self._cache_size = 1000
        
    @lru_cache(maxsize=1000)
    def _cached_perm_entropy(self, data_tuple: tuple, window: int, order: int) -> tuple:
        """Cached permutation entropy computation."""
        data_array = np.array(data_tuple)
        data_series = pd.Series(data_array)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = f.permutation_entropy(data_series, window, order)
            return tuple(result.values if hasattr(result, 'values') else result)
    
    @lru_cache(maxsize=1000)  
    def _cached_lz_complexity(self, data_tuple: tuple, window: int) -> tuple:
        """Cached LZ complexity computation."""
        data_array = np.array(data_tuple)
        data_series = pd.Series(data_array)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = f.lempel_ziv_complexity(data_series, window)
            return tuple(result.values if hasattr(result, 'values') else result)
    
    @lru_cache(maxsize=500)  # Smaller cache for expensive operations
    def _cached_hurst(self, data_tuple: tuple, window: int) -> tuple:
        """Cached Hurst exponent computation."""
        data_array = np.array(data_tuple)
        data_series = pd.Series(data_array)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = f.hurst_exponent(data_series, window)
            return tuple(result.values if hasattr(result, 'values') else result)
    
    @lru_cache(maxsize=500)  # Smaller cache for expensive operations
    def _cached_dfa_alpha(self, data_tuple: tuple, window: int) -> tuple:
        """Cached DFA alpha computation.""" 
        data_array = np.array(data_tuple)
        data_series = pd.Series(data_array)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = f.dfa_alpha(data_series, window)
            return tuple(result.values if hasattr(result, 'values') else result)
    
    @lru_cache(maxsize=1000)
    def _cached_state_persistence(self, data_tuple: tuple, window: int) -> tuple:
        """Cached state persistence computation."""
        data_array = np.array(data_tuple)
        data_series = pd.Series(data_array)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_dict = f.state_persistence(data_series, window)
            
            run_length = result_dict.get('state_mean_run_length', pd.Series([np.nan] * len(data_series)))
            run_entropy = result_dict.get('state_run_entropy', pd.Series([np.nan] * len(data_series)))
            
            return (
                tuple(run_length.values if hasattr(run_length, 'values') else run_length),
                tuple(run_entropy.values if hasattr(run_entropy, 'values') else run_entropy)
            )
    
    def _prepare_data_for_caching(self, returns_data: pd.Series) -> tuple:
        """Convert pandas Series to tuple for caching (only use recent data to avoid memory issues)."""
        # Only use the most recent 500 data points for caching to balance performance vs memory
        recent_data = returns_data.dropna().tail(500)
        return tuple(recent_data.values)
    
    def compute_complexity_features(self, ticker: str, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute complexity features for a ticker with aggressive caching."""
        # Get price data
        px_col = f"{ticker}_PX_LAST"
        if px_col not in data.columns:
            # Return empty series for missing tickers
            empty_series = pd.Series(np.nan, index=data.index)
            return {
                "cmplx.perm_entropy_63": empty_series,
                "cmplx.perm_entropy_126": empty_series,
                "cmplx.lz_complexity_63": empty_series,
                "cmplx.lz_complexity_126": empty_series,
                "cmplx.hurst_252": empty_series,
                "cmplx.dfa_alpha_252": empty_series,
                "cmplx.run_length_mean": empty_series,
                "cmplx.run_entropy": empty_series,
            }
        
        # Compute returns
        returns_data = f.pct_change_n(pd.to_numeric(data[px_col], errors="coerce"), 1)
        
        # Skip computation if insufficient data
        if len(returns_data.dropna()) < 252:
            empty_series = pd.Series(np.nan, index=data.index) 
            return {
                "cmplx.perm_entropy_63": empty_series,
                "cmplx.perm_entropy_126": empty_series,
                "cmplx.lz_complexity_63": empty_series,
                "cmplx.lz_complexity_126": empty_series,
                "cmplx.hurst_252": empty_series,
                "cmplx.dfa_alpha_252": empty_series,
                "cmplx.run_length_mean": empty_series,
                "cmplx.run_entropy": empty_series,
            }
        
        # Prepare data for caching
        data_tuple = self._prepare_data_for_caching(returns_data)
        
        results = {}
        
        try:
            # Permutation entropy features
            pe_63_result = self._cached_perm_entropy(data_tuple, 63, 3)
            pe_63_series = pd.Series(pe_63_result, index=data.index[:len(pe_63_result)])
            results["cmplx.perm_entropy_63"] = pe_63_series.reindex(data.index)
            
            pe_126_result = self._cached_perm_entropy(data_tuple, 126, 3) 
            pe_126_series = pd.Series(pe_126_result, index=data.index[:len(pe_126_result)])
            results["cmplx.perm_entropy_126"] = pe_126_series.reindex(data.index)
            
            # LZ complexity features
            lz_63_result = self._cached_lz_complexity(data_tuple, 63)
            lz_63_series = pd.Series(lz_63_result, index=data.index[:len(lz_63_result)])
            results["cmplx.lz_complexity_63"] = lz_63_series.reindex(data.index)
            
            lz_126_result = self._cached_lz_complexity(data_tuple, 126)
            lz_126_series = pd.Series(lz_126_result, index=data.index[:len(lz_126_result)])
            results["cmplx.lz_complexity_126"] = lz_126_series.reindex(data.index)
            
            # Hurst and DFA (more expensive, smaller cache)
            hurst_result = self._cached_hurst(data_tuple, 252)
            hurst_series = pd.Series(hurst_result, index=data.index[:len(hurst_result)])
            results["cmplx.hurst_252"] = hurst_series.reindex(data.index)
            
            dfa_result = self._cached_dfa_alpha(data_tuple, 252)
            dfa_series = pd.Series(dfa_result, index=data.index[:len(dfa_result)])
            results["cmplx.dfa_alpha_252"] = dfa_series.reindex(data.index)
            
            # State persistence features
            persistence_results = self._cached_state_persistence(data_tuple, 63)
            run_length_series = pd.Series(persistence_results[0], index=data.index[:len(persistence_results[0])])
            run_entropy_series = pd.Series(persistence_results[1], index=data.index[:len(persistence_results[1])])
            results["cmplx.run_length_mean"] = run_length_series.reindex(data.index)
            results["cmplx.run_entropy"] = run_entropy_series.reindex(data.index)
            
        except Exception as e:
            print(f"[FastComplexityEngine] Error computing features for {ticker}: {e}")
            # Return NaN series on any error
            empty_series = pd.Series(np.nan, index=data.index)
            results = {
                "cmplx.perm_entropy_63": empty_series,
                "cmplx.perm_entropy_126": empty_series,
                "cmplx.lz_complexity_63": empty_series,
                "cmplx.lz_complexity_126": empty_series,
                "cmplx.hurst_252": empty_series,
                "cmplx.dfa_alpha_252": empty_series,
                "cmplx.run_length_mean": empty_series,
                "cmplx.run_entropy": empty_series,
            }
        
        return results


# Global instance
_fast_engine = None

def get_fast_complexity_engine() -> FastComplexityEngine:
    """Get the global fast complexity engine."""
    global _fast_engine
    if _fast_engine is None:
        _fast_engine = FastComplexityEngine()
    return _fast_engine


def compute_complexity_feature(data: pd.DataFrame, ticker: str, feature_name: str) -> pd.Series:
    """Fast computation of a single complexity feature."""
    engine = get_fast_complexity_engine()
    ticker_results = engine.compute_complexity_features(ticker, data)
    return ticker_results.get(feature_name, pd.Series(np.nan, index=data.index))