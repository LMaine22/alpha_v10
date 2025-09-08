# alpha_discovery/gauntlet/medium.py
"""
Medium Gauntlet: Statistical Rigor with Practical Focus

A 3-stage gauntlet designed for production deployment decisions:
- Stage 1: Recency & Support (basic health checks)
- Stage 2: Profitability & Stability (recent performance + robustness)  
- Stage 3: Confidence (PSR + Purged K-Fold CV)

This replaces the previous academic-focused gauntlet with a more practical,
statistically rigorous approach suitable for live trading decisions.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

try:
    from ..config import Settings
except Exception:
    class Settings:  # type: ignore
        pass

from ..eval.nav import sharpe, nav_daily_returns_from_ledger


# =============================================================================
# CONFIGURATION
# =============================================================================

def get_medium_gauntlet_config() -> Dict[str, Any]:
    """Get configuration for Medium Gauntlet."""
    return {
        # Stage 1: Recency & Support
        "s1_min_trades": 30,                    # Minimum OOS trades required
        "s1_alive_within_days": 30,             # Must have traded within N days
        "s1_max_dd_mult_vs_median": 2.0,        # Max DD vs historical median
        
        # Stage 2: Profitability & Stability
        "s2_recent_k_trades": 30,               # Recent K trades for S2-A
        "s2_hit_rate_min": 0.45,                # Minimum hit rate on recent K
        "s2_ev_per_trade_min": 50.0,            # Minimum EV per trade ($)
        "s2_median_trade_pnl_min": 0.0,         # Minimum median trade PnL
        "s2_recent_dd_cap_mult": 1.5,           # Recent DD cap vs historical
        "s2_rolling_window_size": None,         # Auto: 30 if N>=60, else 20
        "s2_rolling_pass_ratio_min": 0.6,       # Min fraction of good windows
        "s2_rolling_max_consec_bad": 1,         # Max consecutive bad windows
        "s2_bootstrap_n_resamples": 1000,       # Bootstrap resamples
        "s2_bootstrap_lower_pct": 5,            # Lower percentile threshold
        "s2_breadth_lookback_months": 12,       # Lookback for breadth check
        "s2_min_positive_tickers": 2,           # Min tickers with positive EV
        
        # Stage 3: Confidence
        "s3_psr_hurdle": 0.0,                   # PSR hurdle rate
        "s3_psr_min_probability": 0.75,         # Min PSR probability
        "s3_cv_k_folds": 5,                     # K-fold CV folds
        "s3_cv_embargo_days": 5,                # Embargo around test folds
        "s3_cv_q1_threshold": 0.0,              # Q1 threshold for IQR check
        "s3_cv_q1_close_threshold": -0.1,       # Q1 "close to 0" threshold
        
        # Logging & Output
        "logging_out_dir": "gauntlet_medium",
        "generate_regime_tiles": True,          # Generate regime analysis
    }


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SetupResults:
    """Container for setup backtest results."""
    setup_id: str
    ticker: str
    direction: str
    oos_trades: pd.DataFrame
    oos_daily_returns: pd.Series
    oos_equity_curve: pd.Series
    historical_median_drawdown: Optional[float] = None


@dataclass
class Stage1Results:
    """Stage 1 results."""
    passed: bool
    reasons: List[str]
    total_oos_trades: int
    days_since_last_trade: int
    oos_max_drawdown: float
    historical_median_drawdown: Optional[float]
    dd_reference_source: str  # "historical" or "oos_median"


@dataclass
class Stage2Results:
    """Stage 2 results."""
    passed: bool
    reasons: List[str]
    
    # S2-A: Recent K gate
    s2a_passed: bool
    recent_k_trades: int
    hit_rate_k: float
    ev_per_trade: float
    median_trade_pnl: float
    recent_k_max_dd: float
    
    # S2-B: Rolling stability
    s2b_passed: bool
    rolling_pass_ratio: float
    rolling_consec_bad_windows: int
    
    # S2-C: Outlier dependency
    s2c_passed: bool
    trimmed_ev: float
    trimmed_sharpe: float
    
    # S2-D: Bootstrap CI
    s2d_passed: bool
    bootstrap_metric: str
    bootstrap_lower_pctile: float
    
    # S2-E: Breadth (if applicable)
    s2e_passed: Optional[bool]
    breadth_count: Optional[int]


@dataclass
class Stage3Results:
    """Stage 3 results."""
    passed: bool
    reasons: List[str]
    
    # S3-A: PSR
    s3a_passed: bool
    psr_value: float
    psr_hurdle: float
    
    # S3-B: Purged K-Fold CV
    s3b_passed: bool
    cv_median_sharpe: float
    cv_q1: float
    cv_q3: float
    cv_fold_results: List[float]


@dataclass
class GauntletResults:
    """Complete gauntlet results for a setup."""
    setup_id: str
    ticker: str
    direction: str
    final_decision: str  # "Deploy", "Monitor", "Retire"
    
    stage1: Stage1Results
    stage2: Stage2Results
    stage3: Stage3Results
    
    # Summary metrics
    total_oos_trades: int
    hit_rate: float
    ev_per_trade: float
    median_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    
    # Breadth (if multi-ticker)
    breadth_count: Optional[int] = None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    """Compute running drawdown from equity curve."""
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return drawdown


def _compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Compute maximum drawdown from equity curve."""
    dd = _compute_drawdown(equity_curve)
    return float(dd.min())


def _stationary_bootstrap(returns: pd.Series, n_resamples: int, 
                         expected_block_length: float) -> List[float]:
    """
    Stationary bootstrap implementation (Politis-Romano).
    
    Args:
        returns: Daily return series
        n_resamples: Number of bootstrap samples
        expected_block_length: Expected block length (ℓ)
    
    Returns:
        List of bootstrap sample statistics (Sharpe ratios)
    """
    n = len(returns)
    p = 1.0 / expected_block_length  # Probability of starting new block
    
    bootstrap_sharpes = []
    
    for _ in range(n_resamples):
        # Generate bootstrap sample
        bootstrap_returns = []
        i = 0
        start_idx = 0  # Initialize start_idx
        
        while i < n:
            # Start new block with probability p
            if np.random.random() < p:
                # Start new block at random position
                start_idx = np.random.randint(0, n)
                block_length = np.random.geometric(p)
            else:
                # Continue current block
                block_length = 1
            
            # Add block to bootstrap sample
            for j in range(min(block_length, n - i)):
                idx = (start_idx + j) % n
                bootstrap_returns.append(returns.iloc[idx])
                i += 1
                
                if i >= n:
                    break
        
        # Compute Sharpe ratio for this bootstrap sample
        if len(bootstrap_returns) > 1:
            sharpe_val = sharpe(pd.Series(bootstrap_returns))
            bootstrap_sharpes.append(sharpe_val)
    
    return bootstrap_sharpes


def _purged_k_fold_cv(returns: pd.Series, k_folds: int, embargo_days: int) -> List[float]:
    """
    Purged K-Fold Cross-Validation on time series.
    
    Args:
        returns: Daily return series
        k_folds: Number of folds
        embargo_days: Days to embargo around each test fold
    
    Returns:
        List of Sharpe ratios for each fold
    """
    n = len(returns)
    fold_size = n // k_folds
    fold_results = []
    
    for i in range(k_folds):
        # Define test fold boundaries
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, n)
        
        # Apply embargo
        embargo_start = max(0, test_start - embargo_days)
        embargo_end = min(n, test_end + embargo_days)
        
        # Get test fold (excluding embargo)
        test_returns = returns.iloc[test_start:test_end]
        
        if len(test_returns) > 1:
            sharpe_val = sharpe(test_returns)
            fold_results.append(sharpe_val)
    
    return fold_results


def _probabilistic_sharpe_ratio(returns: pd.Series, hurdle: float = 0.0) -> float:
    """
    Probabilistic Sharpe Ratio (PSR) with non-normality adjustment.
    
    Based on Bailey & Lopez de Prado (2012).
    
    Args:
        returns: Daily return series
        hurdle: Hurdle Sharpe ratio
    
    Returns:
        PSR probability
    """
    if len(returns) < 10:
        return 0.0
    
    n = len(returns)
    sr = sharpe(returns)
    skew = returns.skew()
    kurt = returns.kurtosis()
    
    # Non-normality adjustment
    gamma = skew / (n ** 0.5)
    kappa = (kurt - 1) / n
    
    # PSR formula
    if sr <= hurdle:
        return 0.0
    
    # Adjusted Sharpe ratio
    sr_adj = sr * np.sqrt((n - 1) / (n - 3))
    
    # PSR calculation
    numerator = sr_adj - hurdle
    discriminant = 1 - gamma * sr_adj + (gamma - 1) / 4 * sr_adj**2
    
    if discriminant <= 0:
        return 0.0
    
    denominator = np.sqrt(discriminant)
    psr = norm.cdf(numerator / denominator)
    return float(psr)


# =============================================================================
# STAGE IMPLEMENTATIONS
# =============================================================================

def run_stage1_recency_support(setup: SetupResults, config: Dict[str, Any]) -> Stage1Results:
    """Stage 1: Recency & Support checks."""
    reasons = []
    
    # S1-1: Minimum trades
    total_trades = len(setup.oos_trades)
    if total_trades < config["s1_min_trades"]:
        reasons.append(f"insufficient_trades: {total_trades} < {config['s1_min_trades']}")
    
    # S1-2: Days since last trade
    if not setup.oos_trades.empty:
        last_trade_date = pd.to_datetime(setup.oos_trades['exit_date']).max()
        days_since_last = (datetime.now() - last_trade_date).days
    else:
        days_since_last = float('inf')
    
    if days_since_last > config["s1_alive_within_days"]:
        reasons.append(f"stale_trades: {days_since_last} > {config['s1_alive_within_days']}")
    
    # S1-3: Drawdown check
    oos_max_dd = _compute_max_drawdown(setup.oos_equity_curve)
    
    # Determine DD reference
    if setup.historical_median_drawdown is not None:
        dd_reference = setup.historical_median_drawdown
        dd_source = "historical"
    else:
        # Use OOS median DD as fallback
        dd_reference = setup.oos_trades['pnl_dollars'].median()
        dd_source = "oos_median"
        logging.warning(f"Setup {setup.setup_id}: Using OOS median DD as reference")
    
    max_dd_threshold = config["s1_max_dd_mult_vs_median"] * abs(dd_reference)
    if oos_max_dd < -max_dd_threshold:
        reasons.append(f"excessive_drawdown: {oos_max_dd:.3f} < -{max_dd_threshold:.3f}")
    
    passed = len(reasons) == 0
    
    return Stage1Results(
        passed=passed,
        reasons=reasons,
        total_oos_trades=total_trades,
        days_since_last_trade=days_since_last,
        oos_max_drawdown=oos_max_dd,
        historical_median_drawdown=setup.historical_median_drawdown,
        dd_reference_source=dd_source
    )


def run_stage2_profitability_stability(setup: SetupResults, config: Dict[str, Any]) -> Stage2Results:
    """Stage 2: Profitability & Stability checks."""
    reasons = []
    
    N = len(setup.oos_trades)
    K = min(config["s2_recent_k_trades"], N)
    W = 30 if N >= 60 else 20
    if config["s2_rolling_window_size"] is not None:
        W = config["s2_rolling_window_size"]
    
    # S2-A: Recent K gate
    recent_trades = setup.oos_trades.tail(K)
    hit_rate_k = (recent_trades['pnl_dollars'] > 0).mean()
    ev_per_trade = recent_trades['pnl_dollars'].mean()
    median_trade_pnl = recent_trades['pnl_dollars'].median()
    
    # Recent K drawdown
    recent_equity = recent_trades['pnl_dollars'].cumsum()
    recent_k_max_dd = _compute_max_drawdown(recent_equity)
    
    # DD reference (same as Stage 1)
    if setup.historical_median_drawdown is not None:
        dd_reference = setup.historical_median_drawdown
    else:
        dd_reference = setup.oos_trades['pnl_dollars'].median()
    
    s2a_reasons = []
    if hit_rate_k < config["s2_hit_rate_min"]:
        s2a_reasons.append(f"hit_rate: {hit_rate_k:.3f} < {config['s2_hit_rate_min']}")
    if ev_per_trade < config["s2_ev_per_trade_min"]:
        s2a_reasons.append(f"ev_per_trade: {ev_per_trade:.1f} < {config['s2_ev_per_trade_min']}")
    if median_trade_pnl < config["s2_median_trade_pnl_min"]:
        s2a_reasons.append(f"median_pnl: {median_trade_pnl:.1f} < {config['s2_median_trade_pnl_min']}")
    if recent_k_max_dd < -config["s2_recent_dd_cap_mult"] * abs(dd_reference):
        s2a_reasons.append(f"recent_dd: {recent_k_max_dd:.3f} exceeds cap")
    
    s2a_passed = len(s2a_reasons) == 0
    if not s2a_passed:
        reasons.extend([f"s2a_{r}" for r in s2a_reasons])
    
    # S2-B: Rolling stability
    rolling_good_windows = 0
    consec_bad_windows = 0
    max_consec_bad = 0
    
    for i in range(W, N + 1):
        window_trades = setup.oos_trades.iloc[i-W:i]
        window_returns = window_trades['pnl_dollars']
        
        if len(window_returns) > 1:
            window_sharpe = sharpe(window_returns)
            window_ev = window_returns.mean()
            
            is_good = window_sharpe > 0 and window_ev > 0
            if is_good:
                rolling_good_windows += 1
                consec_bad_windows = 0
            else:
                consec_bad_windows += 1
                max_consec_bad = max(max_consec_bad, consec_bad_windows)
    
    rolling_pass_ratio = rolling_good_windows / max(1, N - W + 1)
    s2b_passed = (rolling_pass_ratio >= config["s2_rolling_pass_ratio_min"] and 
                  max_consec_bad <= config["s2_rolling_max_consec_bad"])
    
    if not s2b_passed:
        reasons.append(f"s2b_rolling: ratio={rolling_pass_ratio:.3f}, max_consec={max_consec_bad}")
    
    # S2-C: Outlier dependency
    pnl_values = setup.oos_trades['pnl_dollars'].values
    if len(pnl_values) > 2:
        # Remove best and worst trades
        trimmed_pnl = np.delete(pnl_values, [np.argmax(pnl_values), np.argmin(pnl_values)])
        trimmed_ev = np.mean(trimmed_pnl)
        trimmed_sharpe = sharpe(pd.Series(trimmed_pnl))
        
        s2c_passed = trimmed_ev > 0 and trimmed_sharpe > 0
        if not s2c_passed:
            reasons.append(f"s2c_trimmed: ev={trimmed_ev:.1f}, sharpe={trimmed_sharpe:.3f}")
    else:
        trimmed_ev = 0.0
        trimmed_sharpe = 0.0
        s2c_passed = False
        reasons.append("s2c_trimmed: insufficient_trades_for_trimming")
    
    # S2-D: Bootstrap CI
    if len(setup.oos_daily_returns) > 10:
        T = len(setup.oos_daily_returns)
        expected_block_length = 1.5 * (T ** (1/3))
        
        bootstrap_sharpes = _stationary_bootstrap(
            setup.oos_daily_returns, 
            config["s2_bootstrap_n_resamples"],
            expected_block_length
        )
        
        if bootstrap_sharpes:
            lower_pctile = np.percentile(bootstrap_sharpes, config["s2_bootstrap_lower_pct"])
            s2d_passed = lower_pctile > 0
            if not s2d_passed:
                reasons.append(f"s2d_bootstrap: lower_pctile={lower_pctile:.3f}")
        else:
            s2d_passed = False
            reasons.append("s2d_bootstrap: failed_to_generate_samples")
    else:
        s2d_passed = False
        reasons.append("s2d_bootstrap: insufficient_data")
        lower_pctile = 0.0
    
    # S2-E: Breadth (placeholder - would need multi-ticker data)
    s2e_passed = None
    breadth_count = None
    
    # Overall S2 pass
    s2_passed = (s2a_passed and s2d_passed and 
                 (s2b_passed or s2c_passed) and 
                 (s2e_passed is None or s2e_passed))
    
    return Stage2Results(
        passed=s2_passed,
        reasons=reasons,
        s2a_passed=s2a_passed,
        recent_k_trades=K,
        hit_rate_k=hit_rate_k,
        ev_per_trade=ev_per_trade,
        median_trade_pnl=median_trade_pnl,
        recent_k_max_dd=recent_k_max_dd,
        s2b_passed=s2b_passed,
        rolling_pass_ratio=rolling_pass_ratio,
        rolling_consec_bad_windows=max_consec_bad,
        s2c_passed=s2c_passed,
        trimmed_ev=trimmed_ev,
        trimmed_sharpe=trimmed_sharpe,
        s2d_passed=s2d_passed,
        bootstrap_metric="sharpe",
        bootstrap_lower_pctile=lower_pctile,
        s2e_passed=s2e_passed,
        breadth_count=breadth_count
    )


def run_stage3_confidence(setup: SetupResults, config: Dict[str, Any]) -> Stage3Results:
    """Stage 3: Confidence checks (PSR + Purged K-Fold CV)."""
    reasons = []
    
    # S3-A: PSR
    psr_value = _probabilistic_sharpe_ratio(setup.oos_daily_returns, config["s3_psr_hurdle"])
    s3a_passed = psr_value >= config["s3_psr_min_probability"]
    if not s3a_passed:
        reasons.append(f"s3a_psr: {psr_value:.3f} < {config['s3_psr_min_probability']}")
    
    # S3-B: Purged K-Fold CV
    cv_fold_results = _purged_k_fold_cv(
        setup.oos_daily_returns,
        config["s3_cv_k_folds"],
        config["s3_cv_embargo_days"]
    )
    
    if cv_fold_results:
        cv_median_sharpe = np.median(cv_fold_results)
        cv_q1 = np.percentile(cv_fold_results, 25)
        cv_q3 = np.percentile(cv_fold_results, 75)
        
        # IQR check: Q1 > 0 or Q1 close to 0 with Q2 > 0
        iqr_ok = (cv_q1 > config["s3_cv_q1_threshold"] or 
                 (cv_q1 > config["s3_cv_q1_close_threshold"] and cv_median_sharpe > 0))
        
        s3b_passed = cv_median_sharpe > 0 and iqr_ok
        if not s3b_passed:
            reasons.append(f"s3b_cv: median={cv_median_sharpe:.3f}, q1={cv_q1:.3f}, q3={cv_q3:.3f}")
    else:
        cv_median_sharpe = 0.0
        cv_q1 = 0.0
        cv_q3 = 0.0
        s3b_passed = False
        reasons.append("s3b_cv: failed_to_generate_folds")
    
    s3_passed = s3a_passed and s3b_passed
    
    return Stage3Results(
        passed=s3_passed,
        reasons=reasons,
        s3a_passed=s3a_passed,
        psr_value=psr_value,
        psr_hurdle=config["s3_psr_hurdle"],
        s3b_passed=s3b_passed,
        cv_median_sharpe=cv_median_sharpe,
        cv_q1=cv_q1,
        cv_q3=cv_q3,
        cv_fold_results=cv_fold_results
    )


# =============================================================================
# MAIN GAUNTLET RUNNER
# =============================================================================

def run_medium_gauntlet(setup_results: List[SetupResults], 
                       settings: Settings,
                       config: Optional[Dict[str, Any]] = None) -> List[GauntletResults]:
    """
    Run Medium Gauntlet on a list of setup results.
    
    Args:
        setup_results: List of SetupResults objects
        settings: Settings object
        config: Optional configuration override
    
    Returns:
        List of GauntletResults for each setup
    """
    # Merge config
    cfg = get_medium_gauntlet_config()
    if config:
        cfg.update(config)
    
    results = []
    
    for setup in setup_results:
        logging.info(f"Running Medium Gauntlet for {setup.setup_id}")
        
        # Run all stages
        stage1 = run_stage1_recency_support(setup, cfg)
        stage2 = run_stage2_profitability_stability(setup, cfg)
        stage3 = run_stage3_confidence(setup, cfg)
        
        # Determine final decision
        if stage1.passed and stage2.passed and stage3.passed:
            final_decision = "Deploy"
        elif stage1.passed and stage2.passed:
            final_decision = "Monitor"
        else:
            final_decision = "Retire"
        
        # Compute summary metrics
        total_trades = len(setup.oos_trades)
        hit_rate = (setup.oos_trades['pnl_dollars'] > 0).mean() if total_trades > 0 else 0.0
        ev_per_trade = setup.oos_trades['pnl_dollars'].mean() if total_trades > 0 else 0.0
        median_pnl = setup.oos_trades['pnl_dollars'].median() if total_trades > 0 else 0.0
        max_dd = _compute_max_drawdown(setup.oos_equity_curve)
        sharpe_ratio = sharpe(setup.oos_daily_returns)
        
        # Create result
        result = GauntletResults(
            setup_id=setup.setup_id,
            ticker=setup.ticker,
            direction=setup.direction,
            final_decision=final_decision,
            stage1=stage1,
            stage2=stage2,
            stage3=stage3,
            total_oos_trades=total_trades,
            hit_rate=hit_rate,
            ev_per_trade=ev_per_trade,
            median_pnl=median_pnl,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe_ratio
        )
        
        results.append(result)
        logging.info(f"Setup {setup.setup_id}: {final_decision}")
    
    return results


# =============================================================================
# ARTIFACT GENERATION
# =============================================================================

def write_gauntlet_artifacts(results: List[GauntletResults], 
                           output_dir: str,
                           config: Dict[str, Any]) -> None:
    """Write gauntlet artifacts to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrames
    per_setup_data = []
    survivors_data = []
    non_survivors_data = []
    
    for result in results:
        # Per-setup summary
        setup_data = {
            'setup_id': result.setup_id,
            'ticker': result.ticker,
            'direction': result.direction,
            'final_decision': result.final_decision,
            'total_oos_trades': result.total_oos_trades,
            'hit_rate': result.hit_rate,
            'ev_per_trade': result.ev_per_trade,
            'median_pnl': result.median_pnl,
            'max_drawdown': result.max_drawdown,
            'sharpe_ratio': result.sharpe_ratio,
            'stage1_passed': result.stage1.passed,
            'stage2_passed': result.stage2.passed,
            'stage3_passed': result.stage3.passed,
            'reasons': '; '.join(result.stage1.reasons + result.stage2.reasons + result.stage3.reasons)
        }
        per_setup_data.append(setup_data)
        
        # Categorize results
        if result.final_decision == "Deploy":
            survivors_data.append(setup_data)
        else:
            non_survivors_data.append(setup_data)
    
    # Write CSV files
    pd.DataFrame(per_setup_data).to_csv(
        os.path.join(output_dir, "per_setup_summary.csv"), index=False
    )
    
    if survivors_data:
        pd.DataFrame(survivors_data).to_csv(
            os.path.join(output_dir, "survivors.csv"), index=False
        )
    
    if non_survivors_data:
        pd.DataFrame(non_survivors_data).to_csv(
            os.path.join(output_dir, "non_survivors.csv"), index=False
        )
    
    # Write JSON summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'total_setups': len(results),
        'deploy_count': len(survivors_data),
        'monitor_count': len([r for r in results if r.final_decision == "Monitor"]),
        'retire_count': len([r for r in results if r.final_decision == "Retire"]),
        'results': per_setup_data
    }
    
    with open(os.path.join(output_dir, "gauntlet_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logging.info(f"Gauntlet artifacts written to {output_dir}")
