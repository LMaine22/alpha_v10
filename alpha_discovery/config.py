from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import date


class GaConfig(BaseModel):
    """Genetic Algorithm parameters (model-agnostic, NSGA-compatible)."""
    population_size: int = 50
    generations: int = 4
    elitism_rate: float = 0.1
    mutation_rate: float = 0.35
    crossover_rate: float = 0.75
    seed: int = 170 # keep visible in run_dir names

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
    islands: Optional[dict] = None
    islands_count: int = 4
    migration_interval: int = 2
    migration_size: int = 8

    # Disable legacy gauntlets in this pivot (compat only)
    run_gauntlet: bool = False
    run_strict_oos_gauntlet: bool = False
    run_diagnostic_replay: bool = False


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
    horizons: List[int] = [3, 5, 8, 10]
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

    # Dormancy
    dormancy_eligibility_threshold: float = 0.02 # 2% of days

    # Specialist Rules
    specialist_edge_threshold: float = 0.85
    specialist_trigger_rate_max: float = 0.04
    specialist_coverage_floor: float = 0.40

    # Gates
    gate_min_oos_triggers: int = 15
    gate_crps_percentile: float = 0.60
    gate_ig_percentile: float = 0.70
    gate_mi_max: float = 0.7
    gate_sensitivity_max_drop: float = 0.20

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
    end_date: date = date(2025, 9, 18)

    tradable_tickers: List[str] = [
        'MSTR US Equity','SNOW US Equity','LLY US Equity','COIN US Equity',
        'QCOM US Equity','ULTA US Equity','CRM US Equity','AAPL US Equity',
        'AMZN US Equity','MSFT US Equity','QQQ US Equity','SPY US Equity',
       # 'TSM US Equity','META US Equity','TSLA US Equity','CRWV US Equity',
       # 'GOOGL US Equity','AMD US Equity','ARM US Equity','PLTR US Equity',
       # 'JPM US Equity','C US Equity','BMY US Equity','NKE US Equity','TLT US Equity'
    ]
    macro_tickers: List[str] = ['RTY Index','MXWO Index','USGG10YR Index','USGG2YR Index',
                                'DXY Curncy','JPY Curncy','EUR Curncy','CL1 Comdty',
                                'HG1 Comdty','XAU Curncy'
    ]
    benchmark_ticker: str = 'SPY US Equity'


class ReportingConfig(BaseModel):
    runs_dir: str = "runs"
    slate_top_n: int = 40
    slate_max_per_ticker: int = 3


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

    # Back-compat alias so existing code can use settings.validate
    @property
    def validate(self) -> ValidationConfig:
        return self.validation


# Global settings singleton
settings = Settings()

__all__ = ["Settings", "settings"]
