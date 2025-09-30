"""Calendar orthogonality tests (day-of-week, month-end only - no OPEX/options)."""

from __future__ import annotations
from typing import List, Dict
import pandas as pd
import numpy as np


def make_horizon_holdouts(
    df: pd.DataFrame,
    horizons: List[int] = [1, 5, 21]
) -> Dict[int, pd.DatetimeIndex]:
    """
    Return full index per horizon (evaluation enforces horizon).
    
    Args:
        df: DataFrame with DatetimeIndex
        horizons: List of forecast horizons in days
        
    Returns:
        Dict mapping horizon → full DatetimeIndex
    """
    return {h: df.index.copy() for h in horizons}


def make_calendar_holdouts(
    df: pd.DataFrame,
    include_day_of_week: bool = True,
    include_month_end: bool = True,
) -> Dict[str, pd.DatetimeIndex]:
    """
    Calendar orthogonals without OPEX/options.
    
    Creates holdout sets for:
    - Monday/Friday/Midweek (if include_day_of_week)
    - Month-end vs non-month-end (if include_month_end)
    
    Args:
        df: DataFrame with DatetimeIndex
        include_day_of_week: Include weekday-based holdouts
        include_month_end: Include month-end holdouts
        
    Returns:
        Dict mapping category name → DatetimeIndex for that category
    """
    holdouts: Dict[str, pd.DatetimeIndex] = {}
    
    if include_day_of_week:
        mondays = df.index[df.index.weekday == 0]
        if len(mondays) > 20:
            holdouts["monday"] = mondays
        
        fridays = df.index[df.index.weekday == 4]
        if len(fridays) > 20:
            holdouts["friday"] = fridays
        
        midweek = df.index[df.index.weekday.isin([1, 2, 3])]
        if len(midweek) > 50:
            holdouts["midweek"] = midweek
    
    if include_month_end:
        month_end_dates = [d for d in df.index if is_month_end(d, df.index)]
        if len(month_end_dates) > 20:
            holdouts["month_end"] = pd.DatetimeIndex(month_end_dates)
        
        non_month_end = df.index.difference(pd.DatetimeIndex(month_end_dates))
        if len(non_month_end) > 50:
            holdouts["non_month_end"] = non_month_end
    
    return holdouts


def is_month_end(date: pd.Timestamp, trading_calendar: pd.DatetimeIndex) -> bool:
    """
    True if date is within last 3 trading days of the month.
    
    Args:
        date: Date to check
        trading_calendar: Full trading calendar (DatetimeIndex)
        
    Returns:
        True if date is in last 3 trading days of its month
    """
    month_dates = trading_calendar[
        (trading_calendar.year == date.year) & 
        (trading_calendar.month == date.month)
    ]
    if len(month_dates) == 0:
        return False
    
    return date in month_dates[-3:]


def calculate_orthogonality_score(
    skill_by_category: Dict[str, float],
    min_skill_threshold: float = 0.05
) -> float:
    """
    Consistency of skill across categories (0..1).
    
    Higher score = more consistent skill across different calendar patterns.
    
    Args:
        skill_by_category: Dict mapping category name → skill metric
        min_skill_threshold: Minimum threshold for "pass"
        
    Returns:
        Orthogonality score in [0, 1] (higher is better)
    """
    if not skill_by_category:
        return 0.0
    
    skills = np.array([v for v in skill_by_category.values() if np.isfinite(v)])
    if len(skills) == 0:
        return 0.0
    
    # Pass rate: fraction of categories above threshold
    pass_rate = np.mean(skills >= min_skill_threshold)
    
    # Consistency: inverse of coefficient of variation
    if len(skills) > 1:
        cv = np.std(skills) / (abs(np.mean(skills)) + 1e-9)
        consistency = 1.0 / (1.0 + cv)
    else:
        consistency = 1.0
    
    # Weighted combination
    return float(np.clip(0.6 * pass_rate + 0.4 * consistency, 0.0, 1.0))
