# alpha_discovery/config.py

from pydantic import BaseModel
from typing import List, Literal, Optional, Dict
from datetime import date


# -----------------------------
# Genetic Algorithm (core)
# -----------------------------
class GaConfig(BaseModel):
    """Genetic Algorithm Search Parameters"""
    population_size: int = 50
    generations: int = 5
    elitism_rate: float = 0.1
    mutation_rate: float = 0.2
    seed: int = 13
    setup_lengths_to_explore: List[int] = [2]

    # Verbosity & debugging used by NSGA layer (added)
    verbose: int = 1              # 0..3 (2 = extra progress summaries)
    debug_sequential: bool = False   # True = evaluate in-process (no joblib)


# -----------------------------
# Data & universe
# -----------------------------
class DataConfig(BaseModel):
    """Data Source and Ticker Configuration"""
    excel_file_path: str = 'data_store/raw/bb_data.xlsx'
    parquet_file_path: str = 'data_store/processed/bb_data.parquet'
    start_date: date = date(2010, 1, 1)
    end_date: date = date(2025, 8, 27)
    holdout_start_date: date = date(2023, 8, 27)

    # Finalized ticker lists
    tradable_tickers: List[str] = [
        'CRWV US Equity', 'TSLA US Equity', 'AMZN US Equity', 'QQQ US Equity',
        'GOOGL US Equity', 'MSFT US Equity', 'AAPL US Equity', 'LLY US Equity',
        'AMD US Equity', 'MSTR US Equity', 'COIN US Equity', 'ARM US Equity'
        #'XLE US Equity', 'XLK US Equity', 'XLRE US Equity', 'XLC US Equity',
        #'XLV US Equity', 'XLP US Equity', 'SPY US Equity', 'QQQ US Equity',
        #'JPM US Equity', 'C US Equity', 'PLTR US Equity',
        #'BMY US Equity', 'PEPS US Equity', 'NKE US Equity',
    ]
    macro_tickers: List[str] = [
        'RTY Index', 'MXWO Index', 'USGG10YR Index', 'USGG2YR Index',
        'DXY Curncy', 'JPY Curncy', 'EUR Curncy', 'EEM US Equity',
        'CL1 Comdty', 'HG1 Comdty', 'XAU Curncy'
    ]


# -----------------------------
# Event calendar
# -----------------------------
class EventsConfig(BaseModel):
    """Event Calendar Integration Settings"""
    file_path: str = "data_store/processed/economic_releases.parquet"

    # Filters
    countries: List[str] = ["US"]
    include_event_types: Optional[List[str]] = None  # e.g., ["CPI", "FOMC", "NFP"]; None = all
    include_tickers: List[str] = []              # reserved (not used yet)
    include_types: List[str] = []                # reserved (not used yet)

    # Relevance & windows
    high_relevance_threshold: float = 70.0
    pre_window_days: int = 2
    post_window_days: int = 2              # reserved (not used yet)
    post_release_lag_days: int = 1   # shift post-release features to T+1 business day


# -----------------------------
# GA diversity (placeholder)
# -----------------------------
class GaDiversityConfig(BaseModel):
    """Controls for diversity and de-duplication (placeholder)."""
    min_unique_setups: int = 10


# -----------------------------
# Validation / splits
# -----------------------------
class ValidationConfig(BaseModel):
    """Validation and support thresholds"""
    min_initial_support: int = 10
    min_portfolio_support: int = 30
    embargo_days: int = 15 # days of post-train embargo before test


# -----------------------------
# Options pricing / backtester knobs
# -----------------------------
class OptionsConfig(BaseModel):
    """
    Options simulation settings.
    Updated for tenor selection, slippage, and guardrails.
    """
    capital_per_trade: float = 10000.0
    contract_multiplier: int = 100
    risk_free_rate_mode: Literal["constant", "macro"] = "constant"
    constant_r: float = 0.0
    allow_nonoptionable: bool = False

    # Tenor selection (business days to expiry)
    tenor_grid_bd: List[int] = [7, 14, 21, 30, 45, 63]
    tenor_buffer_k: float = 1.25

    # Dynamic exit policy defaults (global; GA can override per-setup)
    exit_policies_enabled: bool = True
    exit_pt_multiple: float | None = None
    exit_trail_frac: float | None = 0.7
    exit_sl_multiple: float | None = 0.6
    # If None, default to the current horizon (per-trade) inside the backtester
    exit_time_cap_days: int | None = None

    # IV term structure mapping (fallbacks)
    iv_map_alpha: float = 0.7
    power_law_beta: float = -0.15

    # Slippage and guardrails
    slippage_tiers: Dict[str, float] = {
        "days_15": 0.0075,
        "days_30": 0.0050,
        "days_any": 0.0025,
    }
    min_premium: float = 0.30
    max_contracts: int = 100


# -----------------------------
# Selection / portfolio assembly
# -----------------------------
class SelectionConfig(BaseModel):
    """
    Selection policy for choosing tickers/horizons used in GA scoring.
    Includes knobs expected by selection_core (added) + your existing fields.
    """
    # Ranking -- UPDATED to use new metrics
    metric_primary: Literal[
        "sortino_lb", "expectancy", "sharpe_lb", "omega_ratio", "support"
    ] = "sortino_lb"
    metric_tiebreakers: List[str] = ["expectancy", "support"]  # used as tie-breakers

    # Per-ticker gates (optional)
    per_ticker_min_sharpe_lb: Optional[float] = None
    per_ticker_min_omega: Optional[float] = None
    per_ticker_min_sortino_lb: Optional[float] = None # Added for Sortino
    per_ticker_min_expectancy: Optional[float] = None # Added for Expectancy

    # Support requirements
    min_support_per_ticker: int = 10   # used by selection_core

    # Stepwise assembly thresholds (minimum improvements)
    stepwise_min_delta_sharpe_lb: float = 0.0
    stepwise_min_delta_omega: float = 0.0

    # Portfolio size cap
    max_tickers_in_portfolio: int = 12

    # --- Your existing extra knobs (kept) ---
    stepwise_chunk: int = 3
    top_k_per_ticker: int = 2
    robust_sharpe_alpha: float = 0.2
    winsor_alpha: float = 0.02


# -----------------------------
# Reporting / metrics behavior
# -----------------------------
class ReportingConfig(BaseModel):
    """Reporting and analysis settings."""
    base_capital_for_portfolio: float = 100000.0
    robust_agg_metric: Literal["median", "mean"] = "median"
    trimmed_alpha: float = 0.05
    outlier_factor_flag: float = 10.0

# -----------------------------
# Stage-1 Recency / Liveness
# -----------------------------
class Stage1Config(BaseModel):
    """Stage-1 recency and liveness gates (OOS only)."""
    # Fail if last trigger older than this
    recency_max_days: int = 7
    # Short-window size for liveness/trade checks
    short_window_days: int = 20
    # Require at least this many trades in the short window
    min_trades_short: int = 2
    # Cap on max drawdown in the short window (fractional, e.g. 0.15 = 15%)
    max_drawdown_short: float = 0.15

# -----------------------------
# Stage-2 (MBB) configuration
# -----------------------------
class Stage2Config(BaseModel):
    mbb_B: int = 1000              # bootstrap resamples
    block_len_method: str = "auto"   # "auto" -> sqrt(T), clamped
    block_len_min: int = 5
    block_len_max: int = 50
    seed: int = 42

# -----------------------------
# Stage-3 (FDR/DSR) configuration
# -----------------------------
class Stage3Config(BaseModel):
    fdr_q: float = 0.10              # BH–FDR level




# -----------------------------
# Settings container
# -----------------------------
class Settings(BaseModel):
    """Main container for all project settings"""
    ga: GaConfig = GaConfig()
    ga_diversity: GaDiversityConfig = GaDiversityConfig()
    data: DataConfig = DataConfig()
    events: EventsConfig = EventsConfig()
    validation: ValidationConfig = ValidationConfig()
    options: OptionsConfig = OptionsConfig()
    selection: SelectionConfig = SelectionConfig()
    reporting: ReportingConfig = ReportingConfig()
    stage1: Stage1Config = Stage1Config()
    stage2: Stage2Config = Stage2Config()
    stage3: Stage3Config = Stage3Config()



# Instantiate a global settings object for easy import
settings = Settings()

def gauntlet_cfg(settings: Settings) -> dict:
    """Flattened config keys for gauntlet stages (Stage1, Stage2, Stage3)."""
    return {
        # Stage-1
        "s1_recency_max_days": settings.stage1.recency_max_days,
        "s1_short_window_days": settings.stage1.short_window_days,
        "s1_min_trades_short": settings.stage1.min_trades_short,
        "s1_max_drawdown_short": settings.stage1.max_drawdown_short,

        # Stage-2
        "mbb_B": settings.stage2.mbb_B,
        "block_len_method": settings.stage2.block_len_method,
        "block_len_min": settings.stage2.block_len_min,
        "block_len_max": settings.stage2.block_len_max,
        "seed": settings.stage2.seed,

        # Stage-3
        "fdr_q": settings.stage3.fdr_q,
    }

__all__ = ["Settings", "settings", "gauntlet_cfg"]

