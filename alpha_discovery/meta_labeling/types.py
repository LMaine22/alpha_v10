# alpha_discovery/meta_labeling/types.py
"""
Data types for Meta-Labeling System
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Any


@dataclass
class MetaLabelingResults:
    """Results from meta-labeling evaluation."""
    setup_id: str
    ticker: str
    direction: str
    
    # Base vs Meta performance
    base_expected_value: float
    meta_expected_value: float
    base_sharpe: float
    meta_sharpe: float
    base_max_drawdown: float
    meta_max_drawdown: float
    
    # Trade statistics
    base_trade_count: int
    meta_trade_count: int
    trade_retention_rate: float
    
    # Model performance
    model_accuracy: float
    model_precision: float
    model_recall: float
    model_f1: float
    model_calibration: float
    
    # Feature importance
    feature_importance: Dict[str, float]
    
    # Meta decisions
    meta_decisions: List[Dict[str, Any]]
    
    # Status
    status: str  # "trained" | "insufficient_data" | "dormant" | "error"
    error_message: Optional[str] = None
