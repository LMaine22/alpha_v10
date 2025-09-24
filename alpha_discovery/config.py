from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import date


class GaConfig(BaseModel):
    """Genetic Algorithm parameters (model-agnostic, NSGA-compatible)."""
    population_size: int = 300
    generations: int = 15
    elitism_rate: float = 0.1
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    seed: int = 190 # keep visible in run_dir names

    # Setup grammar — PAIRS ONLY
    setup_lengths_to_explore: List[int] = [2]

    # Logging/debug used by nsga/population
    verbose: int = 2
    debug_sequential: bool = False

    # Objectives to MAXIMIZE (must match keys in ga_core after transformation)
    objectives: List[str] = [
        "crps_neg",
        "pinball_loss_neg_q10",
        "pinball_loss_neg_q90",
        "info_gain",
        "w1_effect",
        "dfa_alpha_closeness_neg",
        "sensitivity_scan_neg",
        "redundancy_neg",  # Default, can be swapped with complexity
        "transfer_entropy_neg",  # Added new Transfer Entropy objective
    ]

    # Penalty objective for GA (complexity vs redundancy)
    # Use 'redundancy_neg' or 'complexity_index_neg'
    complexity_objective: str = "redundancy_neg"

    # Horizon aggregation method for GA objectives ('best', 'mean', 'p75')
    horizon_agg_method: str = "p75"
    # Quantiles for horizon aggregation if method is 'p75'
    h_quantile_low: int = 25
    h_quantile_high: int = 75


    # Island model knobs
    islands: Optional[dict] = {}
    islands_count: int = 4
    migration_interval: int = 2
    migration_size: int = 8

    # Disable legacy gauntlets in this pivot (compat only)
    run_gauntlet: bool = False
    run_strict_oos_gauntlet: bool = False
    run_diagnostic_replay: bool = False


class PostSimulationConfig(BaseModel):
    """Configuration for the post-discovery options simulation."""
    enabled: bool = True
    hart_index_threshold: int = 55
    top_n_candidates: int = 25


class ValidationConfig(BaseModel):
    """Support / split knobs."""
    min_support: int = 5             # friendlier than 30 for cross-ticker coverage
    embargo_days: int = 5
    purge_days: int = 3 # Truncate end of train set to prevent label leakage
    n_folds: int = 5
    # Prefilter: drop ultra-rare primitive signals before GA
    min_signal_fires: int = 6


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
    horizons: List[int] = [2,4]
    default_horizon: int = 5
    price_field: str = "PX_LAST"

    # Probability bands (also used for entropy bins)
    band_edges: List[float] = [-999.0, -0.10, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.10, 999.0]

    # DFA/Hurst target (shape stability)
    dfa_alpha_target: float = 0.65

    # Bootstrap for CRPS robustness (0 = off)
    crps_bootstrap_samples: int = 20 # Enable for robustness checks


class RegimeConfig(BaseModel):
    """HMM/GMM configs for regime detection."""
    enabled: bool = True
    features: List[str] = ["volatility", "trend"]
    n_regimes_range: List[int] = [3, 6] # Range of K to test for BIC selection
    vol_window: int = 20
    trend_window: int = 20


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
    end_date: date = date(2025, 9, 23)

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
    include_macro_etfs_in_tradables: bool = False  # If False, SPY/QQQ/TLT remain macro-only
    sector_groups: dict = {
        "Tech & AI Platforms": [
            'AAPL US Equity','MSFT US Equity','AMZN US Equity','GOOGL US Equity','META US Equity',
            'CRM US Equity','ORCL US Equity','SNOW US Equity','PLTR US Equity','IBM US Equity',
            'ACN US Equity','TSLA US Equity','NKE US Equity'
        ],
        "Semiconductors": [
            'NVDA US Equity','AMD US Equity','AVGO US Equity','QCOM US Equity','TXN US Equity',
            'TSM US Equity','ARM US Equity','MU US Equity','SMCI US Equity','CRWV US Equity'
        ],
        "Crypto Proxies": [
            'COIN US Equity','MSTR US Equity'
        ],
        "Healthcare": [
            'LLY US Equity','NVO US Equity','BMY US Equity'
        ],
        "Financials": [
            'JPM US Equity','C US Equity'
        ],
        "MacroETFs": [
            'SPY US Equity','QQQ US Equity','TLT US Equity'
        ],
    }
    macro_etfs: List[str] = ['SPY US Equity','QQQ US Equity','TLT US Equity']
    macro_tickers: List[str] = ['RTY Index','MXWO Index','USGG10YR Index','USGG2YR Index',
                                'DXY Curncy','JPY Curncy','EUR Curncy','CL1 Comdty',
                                'HG1 Comdty','XAU Curncy', 'VIX Index', 'JPM US Equity',
                                'C US Equity','BMY US Equity','NKE US Equity', 'TLT US Equity'

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
    ga: GaConfig = GaConfig()
    validation: ValidationConfig = ValidationConfig()
    splits: HybridSplitConfig = HybridSplitConfig()
    forecast: ForecastConfig = ForecastConfig()
    regimes: RegimeConfig = RegimeConfig()
    robustness: RobustnessConfig = RobustnessConfig()
    complexity: ComplexityConfig = ComplexityConfig()
    elv: ElvConfig = ElvConfig()
    data: DataConfig = DataConfig()
    reporting: ReportingConfig = ReportingConfig()
    simulation: PostSimulationConfig = PostSimulationConfig()

    # Back-compat alias so existing code can use settings.validate
    @property
    def validate(self) -> ValidationConfig:
        return self.validation


# Global settings singleton
settings = Settings()

__all__ = ["Settings", "settings"]
