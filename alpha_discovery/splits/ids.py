"""SplitSpec dataclass and deterministic ID generation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd


@dataclass
class SplitSpec:
    """Specification for a single train/test split with metadata."""
    
    # Identifiers (required)
    outer_id: str
    
    # Time boundaries (required)
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    
    # Leakage prevention (required)
    purge_days: int
    embargo_days: int
    
    # Context (required)
    label_horizon: int
    feature_lookback_tail: int
    
    # Optional with defaults
    split_version: str = "PAWF_v1"
    regime_version: str = "R1"
    event_class: str = "normal"
    is_tail: bool = False
    notes: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate timestamps."""
        assert self.train_start < self.train_end, "Invalid train window"
        assert self.test_start < self.test_end, "Invalid test window"
        assert self.train_end <= self.test_start, "Train must end before test"

    @property
    def train_index(self) -> pd.DatetimeIndex:
        """Training date range."""
        return pd.date_range(self.train_start, self.train_end, freq='D')

    @property
    def test_index(self) -> pd.DatetimeIndex:
        """Test date range."""
        return pd.date_range(self.test_start, self.test_end, freq='D')

    @property
    def train_span_days(self) -> int:
        """Training window span in days."""
        return (self.train_end - self.train_start).days

    @property
    def test_span_days(self) -> int:
        """Test window span in days."""
        return (self.test_end - self.test_start).days

    @property
    def split_id(self) -> str:
        """Convenience accessor mirroring generate_split_id."""
        return generate_split_id(self)


def generate_split_id(spec: SplitSpec) -> str:
    """
    Generate deterministic SplitID from spec.
    
    Format: "PAWF_v1|OUTER:{YYYYMM}|H:{HORIZON}|E:{EVENT}|P:{purge}|EMB:{embargo}|REG:{Rver}"
    
    Note: E is always 'normal' in this build (no event calendars).
    
    Args:
        spec: SplitSpec with all metadata
        
    Returns:
        Deterministic split identifier string
        
    Example:
        "PAWF_v1|OUTER:202401|H:5|E:normal|P:5|EMB:10|REG:R1"
    """
    outer_month = spec.test_start.strftime("%Y%m")
    
    parts = [
        spec.split_version,
        f"OUTER:{outer_month}",
        f"H:{spec.label_horizon}",
        f"E:{spec.event_class}",
        f"P:{spec.purge_days}",
        f"EMB:{spec.embargo_days}",
        f"REG:{spec.regime_version}"
    ]
    
    return "|".join(parts)


def parse_split_id(split_id: str) -> dict:
    """
    Parse SplitID back into components.
    
    Args:
        split_id: String like "PAWF_v1|OUTER:202401|H:5|E:normal|P:5|EMB:10|REG:R1"
        
    Returns:
        Dict with parsed components
    """
    parts = split_id.split("|")
    result = {"version": parts[0]}
    
    for part in parts[1:]:
        if ":" in part:
            key, val = part.split(":", 1)
            result[key.lower()] = val
    
    return result
