# alpha_discovery/eval/selection.py
"""
Facade that re-exports the selection API from selection_core.
This keeps imports like `from ..eval import selection` working unchanged.
"""

from __future__ import annotations
from .selection_core import *  # noqa: F401,F403

# Keep an explicit __all__ for clarity (optional)
from .selection_core import (
    TickerBest,
    score_ticker_horizon,
    select_best_horizon_per_ticker,
    stepwise_select_portfolio,
    filter_ledger_to_selection,
    portfolio_daily_returns,
    portfolio_metrics,
    assemble_portfolio_stepwise,
    selection_summary,
)

__all__ = [
    "TickerBest",
    "score_ticker_horizon",
    "select_best_horizon_per_ticker",
    "stepwise_select_portfolio",
    "filter_ledger_to_selection",
    "portfolio_daily_returns",
    "portfolio_metrics",
    "assemble_portfolio_stepwise",
    "selection_summary",
]
