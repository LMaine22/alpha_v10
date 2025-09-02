# alpha_discovery/engine/backtester.py
"""
Facade for the backtester. Public API unchanged:
    - ExitPolicy
    - run_setup_backtest_options(...)
"""

from __future__ import annotations

from .bt_common import ExitPolicy, TRADE_HORIZONS_DAYS  # re-export
from .bt_core import run_setup_backtest_options        # re-export

__all__ = ["ExitPolicy", "run_setup_backtest_options", "TRADE_HORIZONS_DAYS"]
