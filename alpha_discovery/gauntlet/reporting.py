# alpha_discovery/gauntlet/reporting.py
from __future__ import annotations

import os
import pandas as pd
from typing import Optional, List

# Core schema for gauntlet ledger (maintains consistency with your enhanced backtester)
GAUNTLET_LEDGER_SCHEMA = [
    # Identifiers & timing
    "setup_id", "ticker", "trigger_date", "exit_date",

    # Trade specification
    "direction", "horizon_days", "option_type", "strike",

    # Underlying & IV data
    "entry_underlying", "exit_underlying", "entry_iv", "exit_iv",

    # Option pricing
    "entry_option_price", "exit_option_price",

    # Position sizing
    "contracts", "capital_allocated", "capital_allocated_used",

    # P&L
    "pnl_dollars", "pnl_pct", "unrealized_pnl", "realized_pnl",

    # Exit attribution
    "exit_reason", "exit_policy_id", "holding_days_actual",

    # Enhanced IV tracking (new fields from your backtester)
    "iv_anchor", "delta_bucket", "iv_ref_days",
    "sigma_anchor", "sigma_entry", "sigma_exit",
    "delta_target", "delta_achieved", "K_over_S", "fallback_to_3M",
]


def write_csv(path: str, df: pd.DataFrame, float_format: str = "%.4f",
              apply_schema: Optional[List[str]] = None) -> str:
    """
    Enhanced CSV writer with optional schema management.

    For the gauntlet ledger, this ensures consistent column ordering.
    For other files, it's just a simple CSV writer.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Apply schema if requested (mainly for gauntlet_ledger.csv)
        if apply_schema and not df.empty:
            df = _apply_schema(df, apply_schema)

        df.to_csv(path, index=False, float_format=float_format)
        print(f"Written: {os.path.basename(path)} ({len(df)} rows)")
        return path

    except Exception as e:
        print(f"Error writing {path}: {e}")
        # Write empty file as fallback
        pd.DataFrame().to_csv(path, index=False)
        return path


def _apply_schema(df: pd.DataFrame, schema: List[str]) -> pd.DataFrame:
    """
    Apply consistent column ordering while preserving all data.

    This ensures the gauntlet ledger has predictable structure for
    downstream analysis tools and manual inspection.
    """
    if df.empty:
        return pd.DataFrame(columns=schema)

    result = df.copy()

    # Add missing schema columns as NaN
    for col in schema:
        if col not in result.columns:
            result[col] = pd.NA

    # Reorder: schema columns first, then any extras
    schema_cols = [c for c in schema if c in result.columns]
    extra_cols = [c for c in result.columns if c not in schema]

    return result[schema_cols + extra_cols]


def ensure_dir(path: str) -> str:
    """Ensure directory exists and return the path."""
    os.makedirs(path, exist_ok=True)
    return path