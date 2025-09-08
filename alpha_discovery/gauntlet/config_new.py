# alpha_discovery/gauntlet/config_new.py
"""
Configuration for the new redesigned gauntlet stages.
This provides sensible defaults that can be overridden in the main config.
"""

from typing import Dict, Any


def get_new_gauntlet_config() -> Dict[str, Any]:
    """
    Get configuration for the new gauntlet stages.
    
    Stage 1 (Health Check):
    - Rolling window recency: Has setup traded recently?
    - Minimum activity: Enough trades in recent period
    - Positive momentum: Not all recent trades are losers
    
    Stage 2 (Profitability Check):
    - Total OOS NAV growth: Positive overall NAV
    - Cumulative $ PnL: Positive total PnL
    - Hit rate / payoff balance: Win% or payoff ratio thresholds
    - Rolling PnL check: Recent trades also profitable
    - Drawdown sanity: Reasonable max drawdown
    
    Stage 3 (Robustness Check):
    - Deflated Sharpe Ratio: Properly calculated DSR
    - Bootstrap confidence intervals: Sharpe CI lower bound > 0
    - Stability ratio: OOS/insample Sharpe within bounds
    - EWMA Sharpe trend: Not collapsing over time
    """
    return {
        # ===== Stage 1: Health Check =====
        "s1_rolling_window_days": 7,           # Must have traded in last 7 days
        "s1_min_recent_trades": 1,             # At least 1 trade in rolling window
        "s1_min_total_trades": 5,              # At least 5 trades total
        "s1_momentum_window_days": 30,         # Check momentum in last 30 days
        "s1_min_momentum_trades": 3,           # Need 3 trades for momentum check
        
        # ===== Stage 2: Profitability Check =====
        "s2_min_nav_return_pct": 0.0,          # Must be profitable OOS
        "s2_min_total_pnl": 0.0,               # Positive total PnL
        "s2_min_win_rate": 0.0,                # Win rate threshold (0 = no minimum)
        "s2_min_payoff_ratio": 0.0,            # Payoff ratio threshold (0 = no minimum)
        "s2_max_drawdown_pct": 0.50,           # Max 50% drawdown
        "s2_recent_days": 30,                  # Recent window for rolling check
        "s2_min_recent_nav_return": 0.0,       # Recent trades must be profitable
        "s2_min_recent_trades": 1,             # Min recent trades for rolling check
        
        # ===== Stage 3: Robustness Check =====
        "s3_min_dsr": 0.1,                     # Min Deflated Sharpe Ratio
        "s3_min_ci_lower": 0.0,                # Min CI lower bound for Sharpe
        "s3_min_stability_ratio": 0.3,         # Min OOS/insample Sharpe ratio
        "s3_max_stability_ratio": 2.0,         # Max OOS/insample Sharpe ratio
        "s3_min_sharpe_trend": -0.1,           # Min Sharpe trend (not collapsing)
        "s3_n_trials": 1,                      # Number of trials for DSR correction
        "s3_n_bootstrap": 1000,                # Bootstrap samples for CI
        "s3_confidence": 0.95,                 # CI confidence level
    }


def get_strict_gauntlet_config() -> Dict[str, Any]:
    """
    Get stricter configuration for more selective gauntlet.
    Use this for higher quality but fewer survivors.
    """
    base_config = get_new_gauntlet_config()
    
    # Stricter Stage 1
    base_config.update({
        "s1_rolling_window_days": 3,           # Must have traded in last 3 days
        "s1_min_recent_trades": 2,             # At least 2 trades in rolling window
        "s1_min_total_trades": 10,             # At least 10 trades total
        "s1_momentum_window_days": 14,         # Check momentum in last 14 days
        "s1_min_momentum_trades": 5,           # Need 5 trades for momentum check
    })
    
    # Stricter Stage 2
    base_config.update({
        "s2_min_nav_return_pct": 5.0,          # Must be 5% profitable OOS
        "s2_min_total_pnl": 1000.0,            # Positive $1000 PnL
        "s2_min_win_rate": 0.4,                # 40% win rate
        "s2_min_payoff_ratio": 1.2,            # 1.2 payoff ratio
        "s2_max_drawdown_pct": 0.30,           # Max 30% drawdown
        "s2_recent_days": 14,                  # Recent window for rolling check
        "s2_min_recent_nav_return": 2.0,       # Recent trades must be 2% profitable
        "s2_min_recent_trades": 2,             # Min 2 recent trades
    })
    
    # Stricter Stage 3
    base_config.update({
        "s3_min_dsr": 0.2,                     # Min 0.2 DSR
        "s3_min_ci_lower": 0.1,                # Min 0.1 CI lower bound
        "s3_min_stability_ratio": 0.5,         # Min 0.5 OOS/insample ratio
        "s3_max_stability_ratio": 1.5,         # Max 1.5 OOS/insample ratio
        "s3_min_sharpe_trend": 0.0,            # Sharpe trend must be non-negative
        "s3_n_trials": 10,                     # 10 trials for DSR correction
        "s3_n_bootstrap": 2000,                # More bootstrap samples
        "s3_confidence": 0.99,                 # 99% confidence level
    })
    
    return base_config


def get_permissive_gauntlet_config() -> Dict[str, Any]:
    """
    Get more permissive configuration for more survivors.
    Use this for broader discovery but potentially lower quality.
    """
    base_config = get_new_gauntlet_config()
    
    # More permissive Stage 1
    base_config.update({
        "s1_rolling_window_days": 14,          # Must have traded in last 14 days
        "s1_min_recent_trades": 1,             # At least 1 trade in rolling window
        "s1_min_total_trades": 3,              # At least 3 trades total
        "s1_momentum_window_days": 60,         # Check momentum in last 60 days
        "s1_min_momentum_trades": 2,           # Need 2 trades for momentum check
    })
    
    # More permissive Stage 2
    base_config.update({
        "s2_min_nav_return_pct": -5.0,         # Allow up to 5% loss
        "s2_min_total_pnl": -500.0,            # Allow up to $500 loss
        "s2_min_win_rate": 0.0,                # No minimum win rate
        "s2_min_payoff_ratio": 0.0,            # No minimum payoff ratio
        "s2_max_drawdown_pct": 0.70,           # Max 70% drawdown
        "s2_recent_days": 60,                  # Recent window for rolling check
        "s2_min_recent_nav_return": -10.0,     # Allow recent losses
        "s2_min_recent_trades": 1,             # Min 1 recent trade
    })
    
    # More permissive Stage 3
    base_config.update({
        "s3_min_dsr": 0.0,                     # Min 0.0 DSR (no minimum)
        "s3_min_ci_lower": -2.0,               # Allow very negative CI lower bound
        "s3_min_stability_ratio": 0.0,         # Min 0.0 OOS/insample ratio (no minimum)
        "s3_max_stability_ratio": 10.0,        # Max 10.0 OOS/insample ratio (very permissive)
        "s3_min_sharpe_trend": -1.0,           # Allow very declining Sharpe
        "s3_n_trials": 1,                      # 1 trial for DSR correction
        "s3_n_bootstrap": 100,                 # Fewer bootstrap samples for speed
        "s3_confidence": 0.80,                 # 80% confidence level (more permissive)
    })
    
    return base_config
