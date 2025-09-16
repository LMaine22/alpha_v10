# alpha_discovery/config.py
from pydantic import BaseModel
from typing import List, Literal, Optional, Dict, Any
from datetime import date


# -----------------------------
# Genetic Algorithm (core)
# -----------------------------
class GaConfig(BaseModel):
    """Genetic Algorithm Search Parameters"""
    population_size: int = 500
    generations: int = 12
    elitism_rate: float = 0.1
    mutation_rate: float = 0.6
    seed: int = 136
    setup_lengths_to_explore: List[int] = [2]

    # Verbosity & debugging used by NSGA layer (added)
    verbose: int = 2  # 0..3 (2 = extra progress summaries)
    debug_sequential: bool = False  # True = evaluate in-process (no joblib)

    # Island Model Configuration
    islands: Optional['IslandConfig'] = None

    # Gauntlet Control
    run_gauntlet: bool = True  # Set to False to skip traditional gauntlet phase entirely
    run_strict_oos_gauntlet: bool = False  # Set to False to skip Strict-OOS gauntlet phase
    run_diagnostic_replay: bool = False  # Set to False to skip diagnostic replay with portfolio analysis

    # Fitness system (legacy)
    fitness_profile: str = "legacy"  # Use legacy metrics system
    objectives: Optional[List[str]] = None


# -----------------------------
# Data & universe
# -----------------------------
class DataConfig(BaseModel):
    """Data Source and Ticker Configuration"""
    excel_file_path: str = 'data_store/raw/bb_data.xlsx'
    parquet_file_path: str = 'data_store/processed/bb_data.parquet'
    start_date: date = date(2018, 1, 1)
    end_date: date = date(2025, 9, 16)
    holdout_start_date: date = date(2023, 8, 27)

    # Finalized ticker lists
    tradable_tickers: List[str] = [
        'MSTR US Equity',

        #'MSTR US Equity', 'SNOW US Equity', 'LLY US Equity', 'COIN US Equity',
        #'QCOM US Equity', 'ULTA US Equity', 'CRM US Equity', 'AAPL US Equity',
        #'AMZN US Equity', 'MSFT US Equity', 'QQQ US Equity', 'SPY US Equity',
        #'TSM US Equity', 'META US Equity', 'TSLA US Equity', 'CRWV US Equity',
        #'VIX Index', 'GOOGL US Equity', 'AMD US Equity',  'ARM US Equity',
        #'PLTR US Equity', 'VIX Index', 'JPM US Equity', 'C US Equity',
        #'BMY US Equity','NKE US Equity',
    ]
    macro_tickers: List[str] = [
        'RTY Index', 'MXWO Index', 'USGG10YR Index', 'USGG2YR Index',
        'DXY Curncy', 'JPY Curncy', 'EUR Curncy', 'EEM US Equity',
        'CL1 Comdty', 'HG1 Comdty', 'XAU Curncy', 'XLE US Equity', 'XLK US Equity',
        #'XLRE US Equity', 'XLC US Equity', 'XLV US Equity', 'XLP US Equity',


        'SNOW US Equity', 'LLY US Equity', 'COIN US Equity',
        'QCOM US Equity', 'ULTA US Equity', 'CRM US Equity', 'AAPL US Equity',
        'AMZN US Equity', 'MSFT US Equity', 'QQQ US Equity', 'SPY US Equity',
        'TSM US Equity', 'META US Equity', 'TSLA US Equity', 'CRWV US Equity',
        'VIX Index', 'GOOGL US Equity', 'AMD US Equity', 'ARM US Equity',
        'PLTR US Equity', 'VIX Index', 'JPM US Equity', 'C US Equity',
        'BMY US Equity', 'NKE US Equity',

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
    include_tickers: List[str] = []  # reserved (not used yet)
    include_types: List[str] = []  # reserved (not used yet)

    # Relevance & windows
    high_relevance_threshold: float = 70.0
    pre_window_days: int = 2
    post_window_days: int = 2  # reserved (not used yet)
    post_release_lag_days: int = 1  # shift post-release features to T+1 business day


# -----------------------------
# GA diversity (placeholder)
# -----------------------------
class GaDiversityConfig(BaseModel):
    """Controls for diversity and de-duplication (placeholder)."""
    min_unique_setups: int = 10


# -----------------------------
# Island Model Configuration
# -----------------------------
class IslandConfig(BaseModel):
    """Island Model Parameters for Genetic Algorithm"""
    enabled: bool = True
    n_islands: int = 4
    migration_interval: int = 5  # generations between migrations
    migration_size: float = 0.2  # % of island population migrated
    replace_strategy: str = "worst"  # "worst" | "random"
    sync_final: bool = True  # merge all islands at end

    # Island-specific population sizes (if different from main)
    island_population_size: Optional[int] = None  # None = use main population_size / n_islands

    # Migration topology
    migration_topology: str = "ring"  # "ring" | "random" | "all_to_all"

    # Logging per island
    log_island_metrics: bool = True


# -----------------------------
# Validation / splits
# -----------------------------
class ValidationConfig(BaseModel):
    """Validation and support thresholds"""
    min_initial_support: int = 10
    min_portfolio_support: int = 30
    embargo_days: int = 10  # days of post-train embargo before test


# -----------------------------
# Options pricing / backtester knobs
# -----------------------------

class OptionsConfig(BaseModel):
    """
    Options simulation settings.
    """
    capital_per_trade: float = 10000.0
    contract_multiplier: int = 100
    risk_free_rate_mode: Literal["constant", "macro"] = "constant"
    constant_r: float = 0.0  # <-- was 0.0; set a realistic annual RF so Sortino's MAR > 0
    allow_nonoptionable: bool = False

    # Tenor selection (business days to expiry)
    tenor_grid_bd: List[int] = [3, 5, 7]
    tenor_buffer_k: float = 1.25

    # Intraday trading patterns
    enable_intraday_patterns: bool = False
    intraday_patterns: List[str] = ['overnight', 'intraday']  # EOD→Open, Open→Open
    intraday_use_regular_horizons: bool = False  # If True, intraday patterns test all horizons; if False, only 1-day

    # Enable exit policies
    exit_policies_enabled: bool = True

    # ===== Policy selection =====
    # 'timebox_be_trail' (new) OR 'arm_trail' / 'exit' / 'scale_out' (legacy family)
    pt_behavior: Literal['exit', 'arm_trail', 'scale_out', 'timebox_be_trail', 'regime_aware'] = 'regime_aware'

    # ===== DISABLED - ONLY REGIME-AWARE EXITS FOR EXTREME RUNNERS =====
    be_trigger_multiple: float = 999.0  # DISABLED - effectively never triggers
    trail_arm_multiple: float = 999.0  # DISABLED - effectively never triggers
    exit_trail_frac: float | None = None  # DISABLED - no traditional trailing
    exit_sl_multiple: float | None = None  # DISABLED - no traditional stops
    exit_time_cap_days: int | None = None  # DISABLED - let regime-aware handle timing

    # ===== DISABLED - ONLY REGIME-AWARE EXITS FOR EXTREME RUNNERS =====
    exit_pt_multiple: float | None = None  # DISABLED - no traditional profit targets
    armed_trail_frac: float | None = None  # DISABLED - no traditional trailing
    scale_out_frac: float = 0.0

    # ===== NEW: Advanced IV Pricing Configuration =====

    # IV anchor: which maturity to use as base volatility
    iv_anchor: Literal['1M', '30D', '3M'] = '1M'  # Default: 1M for short-tenor trades

    # Delta bucket: how to select strike and volatility
    delta_bucket: Literal[
        'ATM', 'AUTO_BY_DIRECTION', 'CALL_40D', 'CALL_25D', 'CALL_10D', 'PUT_40D', 'PUT_25D', 'PUT_10D'] = 'AUTO_BY_DIRECTION'

    # Strict mode: if True and new IV columns missing, raise error; if False, fallback to 3M
    strict_new_iv: bool = True

    # Pricing regime for A/B testing
    pricing_regime: Literal['LEGACY_3M', 'ATM_30D', 'SMILE_1M'] = 'SMILE_1M'

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
    per_ticker_min_sortino_lb: Optional[float] = None  # Added for Sortino
    per_ticker_min_expectancy: Optional[float] = None  # Added for Expectancy

    # Support requirements
    min_support_per_ticker: int = 10  # used by selection_core

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
    recency_max_days: int = 5
    # Short-window size for liveness/trade checks
    short_window_days: int = 5
    # Require at least this many trades in the short window
    min_trades_short: int = 1
    # Cap on max drawdown in the short window (fractional, e.g. 0.15 = 15%)
    max_drawdown_short: float = 0.55


# -----------------------------
# Stage-2 (MBB) configuration
# -----------------------------
class Stage2Config(BaseModel):
    mbb_B: int = 1000  # bootstrap resamples
    block_len_method: str = "auto"  # "auto" -> sqrt(T), clamped
    block_len_min: int = 5
    block_len_max: int = 50
    seed: int = 42


# -----------------------------
# Stage-3 (FDR/DSR) configuration
# -----------------------------
class Stage3Config(BaseModel):
    fdr_q: float = 0.10  # BH–FDR level


# -----------------------------
# Regime-Aware Exit Configuration
# -----------------------------
class RegimeAwareConfig(BaseModel):
    """Configuration for regime-aware exit strategies"""

    # Regime detection thresholds
    risk_on_threshold: float = 0.5
    risk_off_threshold: float = -0.5

    # Default exit profiles
    risk_on_profile: Dict[str, float] = {
        "atr_len": 14, "k_atr_base": 2.5, "k_atr_slope": 0.8, "theta_frac": 0.55,
        "epsilon": 0.03, "m_em": 1.4, "alpha_vol": 0.25, "z_panic": 2.3, "g_atr": 1.0,
        "d_pre_hi": 2, "d_pre_lo": 1, "z_pc_tighten": 1.5, "iv_z_cut": 1.0
    }

    neutral_profile: Dict[str, float] = {
        "atr_len": 14, "k_atr_base": 2.0, "k_atr_slope": 0.8, "theta_frac": 0.50,
        "epsilon": 0.03, "m_em": 1.2, "alpha_vol": 0.15, "z_panic": 2.3, "g_atr": 1.0,
        "d_pre_hi": 2, "d_pre_lo": 1, "z_pc_tighten": 1.5, "iv_z_cut": 1.0
    }

    risk_off_profile: Dict[str, float] = {
        "atr_len": 14, "k_atr_base": 1.6, "k_atr_slope": 1.0, "theta_frac": 0.45,
        "epsilon": 0.03, "m_em": 1.0, "alpha_vol": 0.10, "z_panic": 2.3, "g_atr": 1.2,
        "d_pre_hi": 3, "d_pre_lo": 2, "z_pc_tighten": 1.2, "iv_z_cut": 1.0
    }

    # Position management
    cooldown_days: int = 3
    max_positions_per_setup: int = 1
    signal_reset_days: int = 5  # Force signal reset after being true for this many days

    # Enhanced ledger tracking
    enable_enhanced_ledger: bool = True
    track_mfe_mae: bool = True
    track_regime_data: bool = True
    track_event_data: bool = True


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
    regime_aware: RegimeAwareConfig = RegimeAwareConfig()


# Instantiate a global settings object for easy import
settings = Settings()

# Set default island config
if settings.ga.islands is None:
    settings.ga.islands = IslandConfig()


def gauntlet_cfg(settings: Settings) -> dict:
    """Flattened config keys for gauntlet stages (Stage1, Stage2, Stage3)."""
    return {
        # Stage-1 Health Check (updated keys to match stage functions)
        "s1_rolling_window_days": settings.stage1.short_window_days,
        "s1_min_recent_trades": settings.stage1.min_trades_short,
        "s1_min_total_trades": 5,  # reasonable default
        "s1_momentum_window_days": 30,  # reasonable default
        "s1_min_momentum_trades": 3,   # reasonable default

        # Stage-2 Profitability (updated keys to match stage functions)
        "s2_min_nav_return_pct": 0.0,
        "s2_min_total_pnl": 0.0,
        "s2_min_win_rate": 0.0,
        "s2_min_payoff_ratio": 0.0,
        "s2_max_drawdown_pct": 0.50,
        "s2_recent_days": 30,
        "s2_min_recent_nav_return": 0.0,
        "s2_min_recent_trades": 1,

        # Stage-3 Robustness (updated keys to match stage functions)
        "s3_min_dsr": 0.1,
        "s3_min_ci_lower": 0.0,
        "s3_min_stability_ratio": 0.3,
        "s3_max_stability_ratio": 2.0,
        "s3_min_sharpe_trend": -0.1,
        "s3_n_trials": 1,
        "s3_n_bootstrap": 1000,
        "s3_confidence": 0.95,

        # Legacy MBB settings (for compatibility)
        "mbb_B": settings.stage2.mbb_B,
        "block_len_method": settings.stage2.block_len_method,
        "block_len_min": settings.stage2.block_len_min,
        "block_len_max": settings.stage2.block_len_max,
        "seed": settings.stage2.seed,
        "fdr_q": settings.stage3.fdr_q,
    }


__all__ = ["Settings", "settings", "gauntlet_cfg"]
