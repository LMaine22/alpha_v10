from __future__ import annotations
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Literal, Dict, Any
from datetime import date


class GaConfig(BaseModel):
    """Genetic Algorithm parameters (model-agnostic, NSGA-compatible)."""
    population_size: int = 50  # Reduced for faster testing
    generations: int = 5   # Reduced for faster testing
    elitism_rate: float = 0.1
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    seed: int = 194 # keep visible in run_dir names

    # Setup grammar — PAIRS ONLY
    setup_lengths_to_explore: List[int] = [2]

    # Logging/debug used by nsga/population
    verbose: int = 2
    debug_sequential: bool = False

    # Objectives to MAXIMIZE (NEW: Conservative risk-adjusted metrics)
    # All use bootstrap lower bounds for robustness against overfitting
    objectives: List[str] = [
        "dsr",                         # DSR (Deflated Sharpe Ratio): Corrects for multiple testing/selection bias
        "bootstrap_calmar_lb",         # Conservative CAGR/MaxDD with bootstrap LB
        "bootstrap_profit_factor_lb",  # Trade quality (wins/losses) with bootstrap LB
    ]
    
    # Support & Safety Gates (enforced as constraints, not objectives)
    min_support_bars: int = 252                    # Minimum 1 year of daily returns
    min_trades: int = 50                            # Minimum 50 trades for trade-based metrics
    max_drawdown_threshold: float = -0.6            # Reject if MaxDD worse than -60%
    min_hit_rate: float = 0.30                      # Minimum 30% win rate
    max_hit_rate: float = 0.70                      # Maximum 70% win rate (avoid degenerate filters)
    min_psr: float = 0.60                           # Minimum Probabilistic Sharpe Ratio (60% prob Sharpe > 0)
    cvar_5_threshold: Optional[float] = None        # Optional: CVaR 5% threshold (e.g., -0.10)

    # Penalty objective for GA (complexity vs redundancy)
    # Use 'redundancy_neg' or 'complexity_index_neg'
    complexity_objective: str = "redundancy_neg"

    # Horizon aggregation method for GA objectives ('best', 'mean', 'p75')
    horizon_agg_method: str = "p75"
    # Quantiles for horizon aggregation if method is 'p75'
    h_quantile_low: int = 25
    h_quantile_high: int = Field(75, description="Quantile for aggregating across horizons for metrics where higher is better.")


    # Island model configuration
    class IslandConfig(BaseModel):
        n_islands: int = 4
        migration_interval: int = 2
        migration_size: float = 0.1  # Proportion of population to migrate
        replace_strategy: str = "worst"  # "worst" or "random"
        island_population_size: Optional[int] = None  # If None, uses population_size // n_islands
        migration_topology: str = "ring"  # "ring", "random", or "all_to_all"
        sync_final: bool = True  # Synchronize all islands at the end
        log_island_metrics: bool = True  # Log detailed island metrics

    n_islands: int = 4
    migration_interval: int = 2
    migration_size: float = 0.1  # Proportion of population to migrate
    replace_strategy: str = "worst"  # "worst" or "random"
    island_population_size: Optional[int] = None  # If None, uses population_size // n_islands
    migration_topology: str = "ring"  # "ring", "random", or "all_to_all"
    sync_final: bool = True  # Synchronize all islands at the end
    log_island_metrics: bool = True  # Log detailed island metrics

    # Disable legacy gauntlets in this pivot (compat only)
    run_gauntlet: bool = False
    run_strict_oos_gauntlet: bool = False
    run_diagnostic_replay: bool = False


class PostSimulationConfig(BaseModel):
    """Configuration for the post-discovery options simulation."""
    enabled: bool = True
    hart_index_threshold: int = 55
    top_n_candidates: int = 25


class CombinatorialCVConfig(BaseModel):
    n_splits: int = Field(4, description="[CPCV] Number of groups to divide the time series into.")
    n_test_splits: int = Field(2, description="[CPCV] Number of contiguous groups to use for each test set.")
    embargo_pct: float = Field(0.02, description="[CPCV] Percentage of test set size to embargo after each split.")
    min_test_triggers: int = Field(3)
    max_expand_days: int = Field(120)
    step_days: int = Field(2)
    min_test_days: int = Field(126, description="Minimum test window size in days (~6 months)")
    min_test_obs: int = Field(60, description="Minimum test observations")
    min_train_days: int = Field(378, description="Minimum train window size in days (~1.5 years)")
    min_train_obs: int = Field(180, description="Minimum train observations")

class GauntletConfig(BaseModel):
    """Configuration for the 5-stage Gauntlet evaluation pipeline."""
    run_gauntlet: bool = Field(False, description="Enable the 5-stage gauntlet pipeline")
    run_strict_oos_gauntlet: bool = Field(False, description="Enable strict OOS gauntlet evaluation")
    smoke_mode: bool = Field(False, description="Run in smoke mode (faster, less rigorous)")
    diagnostics: bool = Field(True, description="Generate detailed diagnostic outputs")
    
    # Stage 1: Health & Sanity
    s1_recent_window_days: int = Field(7, description="Stage 1: Recent activity window (days)")
    s1_min_recent_trades: int = Field(1, description="Stage 1: Minimum recent trades")
    s1_min_total_trades: int = Field(5, description="Stage 1: Minimum total trades")
    s1_momentum_window_days: int = Field(30, description="Stage 1: Momentum analysis window")
    s1_min_momentum_trades: int = Field(3, description="Stage 1: Minimum trades for momentum")
    s1_iv_availability_min: float = Field(0.98, description="Stage 1: IV availability threshold")
    s1_strike_success_min: float = Field(0.97, description="Stage 1: Strike selection success rate")
    s1_missing_data_tolerance: float = Field(0.01, description="Stage 1: Missing data tolerance")
    s1_mean_holding_ratio_min: float = Field(0.40, description="Stage 1: Min holding/tenor ratio")
    s1_daily_trade_cap: int = Field(12, description="Stage 1: Max trades per day")
    
    # Stage 2: Profitability
    s2_dsr_wf_min: float = Field(0.25, description="Stage 2: Minimum walk-forward DSR")
    s2_iqr_sharpe_wf_max: float = Field(0.80, description="Stage 2: Maximum Sharpe IQR")
    s2_cvar5_wf_min: float = Field(-0.15, description="Stage 2: Minimum CVaR5")
    s2_support_wf_min: int = Field(50, description="Stage 2: Minimum walk-forward support")
    s2_fold_min_return: float = Field(-0.20, description="Stage 2: Minimum fold return")
    s2_base_capital: float = Field(100_000.0, description="Stage 2: Base capital for calculations")
    
    # Stage 3: CPCV Robustness
    s3_lite_blocks: str = Field("monthly", description="Stage 3: CPCV lite block frequency")
    s3_lite_k: int = Field(2, description="Stage 3: Test blocks per path")
    s3_lite_m_min: int = Field(12, description="Stage 3: Min train blocks")
    s3_lite_m_max: int = Field(24, description="Stage 3: Max train blocks")
    s3_lite_repeats: int = Field(8, description="Stage 3: Repeats per test window")
    s3_lite_paths: int = Field(100, description="Stage 3: Lite paths target")
    s3_full_paths: int = Field(500, description="Stage 3: Full paths target")
    s3_H_days: int = Field(60, description="Stage 3: Purge horizon days")
    s3_embargo_days: int = Field(10, description="Stage 3: Embargo days")
    s3_seed: int = Field(7, description="Stage 3: Random seed")
    
    # Stage 3: Gates
    s3_lite_median_dsr_min: float = Field(0.35, description="Stage 3: Lite median DSR minimum")
    s3_lite_iqr_sharpe_max: float = Field(0.70, description="Stage 3: Lite Sharpe IQR maximum")
    s3_lite_cvar5_min: float = Field(-0.12, description="Stage 3: Lite CVaR5 minimum")
    s3_lite_support_rate_min: float = Field(0.60, description="Stage 3: Lite support rate minimum")
    s3_support_per_path_min: int = Field(30, description="Stage 3: Support per path minimum")
    s3_lite_pbo_binary_max: float = Field(0.30, description="Stage 3: Lite PBO binary maximum")
    s3_lite_spearman_min: float = Field(0.35, description="Stage 3: Lite Spearman minimum")
    
    s3_full_median_dsr_min: float = Field(0.45, description="Stage 3: Full median DSR minimum")
    s3_full_iqr_sharpe_max: float = Field(0.60, description="Stage 3: Full Sharpe IQR maximum")
    s3_full_cvar5_min: float = Field(-0.10, description="Stage 3: Full CVaR5 minimum")
    s3_full_support_rate_min: float = Field(0.70, description="Stage 3: Full support rate minimum")
    s3_full_pbo_binary_max: float = Field(0.20, description="Stage 3: Full PBO binary maximum")
    s3_full_spearman_min: float = Field(0.45, description="Stage 3: Full Spearman minimum")
    
    # Stage 3: Regime gates
    s3_regime_coverage_min: float = Field(0.15, description="Stage 3: Regime coverage minimum")
    s3_regime_support_min: int = Field(20, description="Stage 3: Regime support minimum")
    s3_regime_fragility_max: float = Field(0.35, description="Stage 3: Regime fragility maximum")
    s3_mono_regime_block: bool = Field(True, description="Stage 3: Block mono-regime setups")
    s3_top_quantile: float = Field(0.20, description="Stage 3: Top quantile for PBO")
    allow_lite_only: bool = Field(False, description="Stage 3: Allow lite-only evaluation")
    hart_target_cvar5: float = Field(-0.10, description="Stage 3: Hart target CVaR5")
    
    # Stage 4: Portfolio Fit
    s4_corr_abs_max: float = Field(0.35, description="Stage 4: Max absolute correlation")
    s4_capacity_dsr_min: float = Field(0.20, description="Stage 4: Min DSR after capacity haircut")
    s4_overlap_max: float = Field(0.35, description="Stage 4: Max activation overlap")
    s4_min_observations: int = Field(20, description="Stage 4: Min observations for correlation")
    s4_min_tot_opt_volume: int = Field(1000, description="Stage 4: Min total option volume")
    s4_min_open_interest_sum: int = Field(5000, description="Stage 4: Min open interest sum")
    s4_min_px_volume: float = Field(1_000_000.0, description="Stage 4: Min price volume")
    s4_w_dsr: float = Field(0.6, description="Stage 4: DSR weight in promotion score")
    s4_w_uncorr: float = Field(0.4, description="Stage 4: Uncorrelation weight in promotion score")
    
    # Stage 5: Final Decision
    min_promotion_score: float = Field(0.0, description="Stage 5: Minimum promotion score")
    stage1_weight: float = Field(0.10, description="Stage 5: Stage 1 weight")
    stage2_weight: float = Field(0.25, description="Stage 5: Stage 2 weight")
    stage3_weight: float = Field(0.45, description="Stage 5: Stage 3 weight")
    stage4_weight: float = Field(0.20, description="Stage 5: Stage 4 weight")


class ValidationConfig(BaseModel):
    min_support: int = Field(5, description="Minimum number of trigger days for a setup to be considered valid in a fold.")
    min_initial_support: int = Field(5, description="Minimum number of trigger days for initial setup validation.")
    oos_start_dt: Optional[str] = Field(None, description="Start date for the out-of-sample period. Overrides dynamic splitting if set.")
    gauntlet_start_dt: Optional[str] = Field(None, description="Start date for the gauntlet (final hold-out) period. Set to ~6 months before data end for fresh evaluation.")
    n_discovery_folds: int = Field(5, description="Number of folds for discovery cross-validation (walk-forward only).")
    n_folds: int = Field(4, description="Number of walk-forward folds for GA metrics (legacy alias).")
    purge_pct: float = Field(0.05, description="Percentage of data to purge between train and test sets to prevent leakage.")
    purge_days: int = Field(0, description="Legacy fallback: absolute purge window in days for walk-forward splits (keep 0 to use precise purge)")
    cv_type: str = Field("combinatorial", description="Type of cross-validation: 'walk_forward' or 'combinatorial'.")
    embargo_days: int = Field(7, description="Legacy fallback: absolute embargo window in days for walk-forward splits.")
    cv: CombinatorialCVConfig = Field(default_factory=CombinatorialCVConfig, description="Settings for Combinatorial Purged Cross-Validation.")
    emit_sparsity_report: bool = Field(False, description="If True, write per-candidate CPCV fold sparsity diagnostics to run_dir/diagnostics.")


class ComplexityConfig(BaseModel):
    """Configuration for complexity metrics."""
    # 'permutation' or 'complexity_index'
    metric: str = "permutation"
    # Permutation entropy params
    pe_embedding: int = 3
    pe_tau: int = 1


class HybridSplitConfig(BaseModel):
    """Configuration for the full Discovery->OOS->Gauntlet split."""
    # Discovery CV settings
    n_discovery_folds: int = 4
    discovery_train_years: float = 3.0
    discovery_test_years: float = 1.0
    discovery_step_months: int = 12

    # True OOS settings
    n_oos_folds: int = 1
    oos_fold_months: int = 12

    # Forward Gauntlet settings
    gauntlet_start_offset_months: int = 0  # months after last discovery train end
    gauntlet_end_date: Optional[date] = None # None means use data end_date


class ForecastConfig(BaseModel):
    """Forecast + scoring knobs for return distributions."""
    default_horizon: int = Field(8, description="Fallback horizon when explicit horizons are unavailable.")
    horizons: List[int] = Field(default_factory=lambda: [2, 4, 8, 21], description="Forecast horizons in days.")
    price_field: str = Field("PX_LAST", description="Bloomberg last price field for return calculations.")
    band_edges: List[float] = Field(
        default_factory=lambda: [-0.10, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.10],
        description="Edges for discretizing forecast returns into bands."
    )

    # DFA/Hurst target (shape stability)
    dfa_alpha_target: float = 0.65

    # Bootstrap for CRPS robustness (0 = off)
    crps_bootstrap_samples: int = 20 # Enable for robustness checks

    @field_validator('default_horizon')
    def _validate_default_horizon_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("default_horizon must be a positive integer")
        return value

    @model_validator(mode='after')
    def _ensure_default_horizon_in_horizons(self):
        if not self.horizons:
            self.horizons = [self.default_horizon]
        elif self.default_horizon not in self.horizons:
            self.horizons = list(self.horizons) + [self.default_horizon]
        return self


class RegimeConfig(BaseModel):
    """HMM/GMM configs for regime detection."""
    enabled: bool = True
    features: List[str] = ["volatility", "trend"]
    n_regimes_range: List[int] = [3, 6] # Range of K to test for BIC selection
    vol_window: int = 20
    trend_window: int = 20


class RegimeAwareConfig(BaseModel):
    """Settings for regime-aware runtime behavior."""
    signal_reset_days: int = 5
    anchor_fits_required: int = 3


class OptionsConfig(BaseModel):
    """Configuration for options backtesting."""
    min_premium: float = Field(0.01, description="Minimum premium threshold for options trades.")
    max_premium: float = Field(10.0, description="Maximum premium threshold for options trades.")
    min_dte: int = Field(7, description="Minimum days to expiration for options.")
    max_dte: int = Field(45, description="Maximum days to expiration for options.")


class RobustnessConfig(BaseModel):
    """Perturbation settings for robustness objectives."""
    jitter_flip_prob: float = 0.02
    n_perturbations: int = 5
    mbb_block_length: int = 22 # Approx 1 month of trading days
    page_hinkley_threshold: float = 0.5


class ElvConfig(BaseModel):
    """Configuration for ELV (Expected Live Value) scoring components."""
    # Shrinkage weights for trigger rate
    live_trigger_rate_cv_weight: float = 0.7
    live_trigger_rate_fg_weight: float = 0.3
    trigger_rate_saturation: float = 0.12 # tau_sat

    # Recency-aware trigger prior
    recency_override_days: int = 3           # If triggered within K days, boost to 1.0
    recency_tau_days_default: int = 21       # Default decay constant for recent boost
    trigger_rate_blend_base_weight: float = 0.6  # Weight on base (CV) vs short-term rate

    # Dormancy
    dormancy_eligibility_threshold: float = 0.02 # 2% of days

    # Specialist Rules
    specialist_edge_threshold: float = 0.85
    specialist_trigger_rate_max: float = 0.04
    specialist_coverage_floor: float = 0.40

    # Gates
    gate_min_oos_triggers: int = 2  # Further lowered from 3 to allow more setups to qualify
    gate_crps_percentile: float = 0.45  # Lowered from 0.50 to be more permissive
    gate_ig_percentile: float = 0.55  # Lowered from 0.60 to be more permissive
    gate_mi_max: float = 0.85  # Increased from 0.8 to allow more signal diversity
    gate_sensitivity_max_drop: float = 0.30  # Increased from 0.25 to allow more setups

    # Disqualification
    disqualify_calib_mae_percentile: float = 0.90
    disqualify_crps_percentile: float = 0.10
    disqualify_mbb_p_value: float = 0.01
    disqualify_sensitivity_drop: float = 0.30
    disqualify_mi_max: float = 0.90

    # Edge_OOS component weights
    edge_crps_weight: float = 0.30
    edge_pinball_weight: float = 0.20
    edge_info_gain_weight: float = 0.25
    edge_w1_weight: float = 0.15
    edge_calibration_weight: float = 0.10

    # CoverageFactor component weights
    # Reinterpreted as: regime coverage, support coverage, band certainty
    coverage_regime_breadth_weight: float = 0.5
    coverage_fold_coverage_weight: float = 0.3
    coverage_stability_weight: float = 0.2

    # Penalty adjustment parameters
    penalty_sensitivity_k: float = 4.0
    penalty_mbb_p_value_tiers: List[float] = [0.10, 0.05]
    penalty_mbb_p_value_adj: List[float] = [0.8, 0.6]
    penalty_page_hinkley_adj: float = 0.75
    penalty_redundancy_factor: float = 0.5
    penalty_complexity_factor: float = 0.4
    maturity_n_triggers: int = 40
    maturity_dormant_floor: float = 0.6


class DataConfig(BaseModel):
    excel_file_path: str = 'data_store/raw/bb_data.xlsx'
    parquet_file_path: str = 'data_store/processed/bb_data.parquet'
    start_date: date = date(2018, 1, 1)
    end_date: date = date(2025, 9, 30)

    tradable_tickers: List[str] = [
        'MSTR US Equity','SNOW US Equity','LLY US Equity','COIN US Equity', 
        'NVDA US Equity', 'AVGO US Equity','SMCI US Equity',
        'QCOM US Equity', 'CRM US Equity','AAPL US Equity',
        'AMZN US Equity','MSFT US Equity','QQQ US Equity','SPY US Equity',
        'TSM US Equity','META US Equity','TSLA US Equity','CRWV US Equity',
        'GOOGL US Equity','AMD US Equity','ARM US Equity','PLTR US Equity',
        'IBM US Equity', 'MU US Equity', 'ORCL US Equity', 'ACN US Equity', 
        'NVO US Equity', 'TXN US Equity', 'JPM US Equity','C US Equity',
        'BMY US Equity','NKE US Equity', 'TLT US Equity'

        
    ]
    
    # Single ticker mode: if set, only this ticker will be used for trading setups
    # All other tradable_tickers will be treated like macro_tickers (signals only, no setups)
    single_ticker_mode: Optional[str] = None  # Example: 'AAPL US Equity' to focus on AAPL only

    # Sector mode: pick one or more named groups; union forms tradables for this run
    sector_modes: Optional[List[str]] = ["Semiconductors"]
    include_macro_etfs_in_tradables: bool = True  # If False, SPY/QQQ/TLT remain macro-only
    sector_groups: dict = {
        "Tech & AI Platforms": [
            'AAPL US Equity','MSFT US Equity','AMZN US Equity','GOOGL US Equity','META US Equity',
            'CRM US Equity','ORCL US Equity','SNOW US Equity','PLTR US Equity','IBM US Equity',
            'ACN US Equity','TSLA US Equity'
        ],
        "Semiconductors": [
            'NVDA US Equity','AMD US Equity','AVGO US Equity','QCOM US Equity','TXN US Equity',
            'TSM US Equity','ARM US Equity','MU US Equity','SMCI US Equity','CRWV US Equity' 
       
        ],
        "Crypto Proxies": [
            'COIN US Equity','MSTR US Equity', 
        ],
        "Healthcare": [
            'LLY US Equity','NVO US Equity','BMY US Equity'
        ],
        "Financials": [
            'JPM US Equity','C US Equity', 'NKE US Equity'
        ],
        "MacroETFs": [
            'SPY US Equity','QQQ US Equity','TLT US Equity'
        ],
    }
    macro_etfs: List[str] = ['SPY US Equity','QQQ US Equity','TLT US Equity']
    macro_tickers: List[str] = ['RTY Index','MXWO Index','USGG10YR Index','USGG2YR Index',
                                'DXY Curncy','JPY Curncy','EUR Curncy','CL1 Comdty',
                                'HG1 Comdty','XAU Curncy', 'VIX Index'

    ]
    benchmark_ticker: str = 'SPY US Equity'

    @property
    def effective_tradable_tickers(self) -> List[str]:
        """Returns the list of tickers that will be used for creating trading setups."""
        if self.single_ticker_mode:
            if self.single_ticker_mode not in self.tradable_tickers:
                raise ValueError(f"Single ticker mode ticker '{self.single_ticker_mode}' not found in tradable_tickers")
            return [self.single_ticker_mode]
        # Sector mode override: union of selected sector groups
        if self.sector_modes:
            selected: List[str] = []
            for grp in self.sector_modes:
                members = self.sector_groups.get(grp, [])
                selected.extend(members)
            # Deduplicate and optionally exclude macro ETFs
            selected_unique = list(dict.fromkeys(selected))
            if not self.include_macro_etfs_in_tradables:
                selected_unique = [t for t in selected_unique if t not in self.macro_etfs]
            return selected_unique
        # Default: use configured tradables, optionally exclude macro ETFs
        base = list(self.tradable_tickers)
        if not self.include_macro_etfs_in_tradables:
            base = [t for t in base if t not in self.macro_etfs]
        return base

    @property
    def effective_macro_tickers(self) -> List[str]:
        """Macro tickers for feature-only roles; ensures MacroETFs always included unless promoted."""
        tradables = set(self.effective_tradable_tickers)
        macros = list(dict.fromkeys(self.macro_tickers + self.macro_etfs))
        # Keep only those not in tradables
        macros = [t for t in macros if t not in tradables]
        return macros


class ReportingConfig(BaseModel):
    runs_dir: str = "runs"
    slate_top_n: int = 120  # Increased from 80
    slate_max_per_ticker: int = 7  # Increased from 5


class Settings(BaseModel):
    # Top-level run mode: discover, validate, gauntlet, or full
    run_mode: Literal['discover', 'validate', 'gauntlet', 'full'] = Field(
        default='full',
        description="Controls which parts of the pipeline to run."
    )
    ga: GaConfig = Field(default_factory=GaConfig)
    gauntlet: GauntletConfig = Field(default_factory=GauntletConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    splits: HybridSplitConfig = HybridSplitConfig()
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    regimes: RegimeConfig = RegimeConfig()
    regime_aware: RegimeAwareConfig = RegimeAwareConfig()
    options: OptionsConfig = Field(default_factory=OptionsConfig)
    robustness: RobustnessConfig = RobustnessConfig()
    complexity: ComplexityConfig = ComplexityConfig()
    elv: ElvConfig = ElvConfig()
    data: DataConfig = Field(default_factory=DataConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    simulation: PostSimulationConfig = Field(default_factory=PostSimulationConfig)

    # Back-compat alias so existing code can use settings.validate
    @property
    def validate(self) -> ValidationConfig:
        return self.validation


def gauntlet_cfg(settings: Settings) -> Dict[str, Any]:
    """
    Convert Settings object to gauntlet configuration dictionary.
    
    This function bridges the gap between the structured Settings object
    and the dictionary format expected by the gauntlet system.
    """
    gauntlet_config = settings.gauntlet
    
    return {
        # Main gauntlet settings
        "smoke_mode": gauntlet_config.smoke_mode,
        "diagnostics": gauntlet_config.diagnostics,
        
        # Stage 1: Health & Sanity
        "stage1": {
            "s1_recent_window_days": gauntlet_config.s1_recent_window_days,
            "s1_min_recent_trades": gauntlet_config.s1_min_recent_trades,
            "s1_min_total_trades": gauntlet_config.s1_min_total_trades,
            "s1_momentum_window_days": gauntlet_config.s1_momentum_window_days,
            "s1_min_momentum_trades": gauntlet_config.s1_min_momentum_trades,
            "s1_iv_availability_min": gauntlet_config.s1_iv_availability_min,
            "s1_strike_success_min": gauntlet_config.s1_strike_success_min,
            "s1_missing_data_tolerance": gauntlet_config.s1_missing_data_tolerance,
            "s1_mean_holding_ratio_min": gauntlet_config.s1_mean_holding_ratio_min,
            "s1_daily_trade_cap": gauntlet_config.s1_daily_trade_cap,
        },
        
        # Stage 2: Profitability  
        "stage2": {
            "s2_dsr_wf_min": gauntlet_config.s2_dsr_wf_min,
            "s2_iqr_sharpe_wf_max": gauntlet_config.s2_iqr_sharpe_wf_max,
            "s2_cvar5_wf_min": gauntlet_config.s2_cvar5_wf_min,
            "s2_support_wf_min": gauntlet_config.s2_support_wf_min,
            "s2_fold_min_return": gauntlet_config.s2_fold_min_return,
            "s2_base_capital": gauntlet_config.s2_base_capital,
        },
        
        # Stage 3: CPCV Robustness
        "stage3": {
            # Basic CPCV settings
            "s3_lite_blocks": gauntlet_config.s3_lite_blocks,
            "s3_lite_k": gauntlet_config.s3_lite_k,
            "s3_lite_m_min": gauntlet_config.s3_lite_m_min,
            "s3_lite_m_max": gauntlet_config.s3_lite_m_max,
            "s3_lite_repeats": gauntlet_config.s3_lite_repeats,
            "s3_lite_paths": gauntlet_config.s3_lite_paths,
            "s3_full_paths": gauntlet_config.s3_full_paths,
            "s3_H_days": gauntlet_config.s3_H_days,
            "s3_embargo_days": gauntlet_config.s3_embargo_days,
            "s3_seed": gauntlet_config.s3_seed,
            
            # Lite gates
            "s3_lite_median_dsr_min": gauntlet_config.s3_lite_median_dsr_min,
            "s3_lite_iqr_sharpe_max": gauntlet_config.s3_lite_iqr_sharpe_max,
            "s3_lite_cvar5_min": gauntlet_config.s3_lite_cvar5_min,
            "s3_lite_support_rate_min": gauntlet_config.s3_lite_support_rate_min,
            "s3_support_per_path_min": gauntlet_config.s3_support_per_path_min,
            "s3_lite_pbo_binary_max": gauntlet_config.s3_lite_pbo_binary_max,
            "s3_lite_spearman_min": gauntlet_config.s3_lite_spearman_min,
            
            # Full gates
            "s3_full_median_dsr_min": gauntlet_config.s3_full_median_dsr_min,
            "s3_full_iqr_sharpe_max": gauntlet_config.s3_full_iqr_sharpe_max,
            "s3_full_cvar5_min": gauntlet_config.s3_full_cvar5_min,
            "s3_full_support_rate_min": gauntlet_config.s3_full_support_rate_min,
            "s3_full_pbo_binary_max": gauntlet_config.s3_full_pbo_binary_max,
            "s3_full_spearman_min": gauntlet_config.s3_full_spearman_min,
            
            # Regime gates
            "s3_regime_coverage_min": gauntlet_config.s3_regime_coverage_min,
            "s3_regime_support_min": gauntlet_config.s3_regime_support_min,
            "s3_regime_fragility_max": gauntlet_config.s3_regime_fragility_max,
            "s3_mono_regime_block": gauntlet_config.s3_mono_regime_block,
            "s3_top_quantile": gauntlet_config.s3_top_quantile,
            "allow_lite_only": gauntlet_config.allow_lite_only,
            "hart_target_cvar5": gauntlet_config.hart_target_cvar5,
        },
        
        # Stage 4: Portfolio Fit
        "stage4": {
            "s4_corr_abs_max": gauntlet_config.s4_corr_abs_max,
            "s4_capacity_dsr_min": gauntlet_config.s4_capacity_dsr_min,
            "s4_overlap_max": gauntlet_config.s4_overlap_max,
            "s4_min_observations": gauntlet_config.s4_min_observations,
            "s4_min_tot_opt_volume": gauntlet_config.s4_min_tot_opt_volume,
            "s4_min_open_interest_sum": gauntlet_config.s4_min_open_interest_sum,
            "s4_min_px_volume": gauntlet_config.s4_min_px_volume,
            "s4_w_dsr": gauntlet_config.s4_w_dsr,
            "s4_w_uncorr": gauntlet_config.s4_w_uncorr,
        },
        
        # Stage 5: Final Decision
        "stage5": {
            "min_promotion_score": gauntlet_config.min_promotion_score,
            "stage1_weight": gauntlet_config.stage1_weight,
            "stage2_weight": gauntlet_config.stage2_weight,
            "stage3_weight": gauntlet_config.stage3_weight,
            "stage4_weight": gauntlet_config.stage4_weight,
        }
    }


# Global settings singleton
settings = Settings()

__all__ = ["Settings", "settings", "gauntlet_cfg"]
