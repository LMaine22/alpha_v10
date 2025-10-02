# alpha_discovery/gauntlet/__init__.py
"""
Gauntlet 2.0 - Multi-Stage Candidate Evaluation System

Extended with CPCV (Combinatorial Purged Cross-Validation) stage
for robust distributional validation with regime analysis.
"""

# Core gauntlet modules
from . import (
    backtester,
    io,
    reporting,
    run,
    stage1_health,
    stage2_profitability, 
    stage3_robustness,
    stage4_portfolio,
    stage5_decision,
    summary
)

# Main run orchestration
from .run import run_gauntlet

__all__ = [
    # Core modules
    "backtester",
    "io", 
    "reporting",
    "run",
    "stage1_health",
    "stage2_profitability",
    "stage3_robustness",
    "stage4_portfolio", 
    "stage5_decision",
    "summary",
    
    # Main orchestration
    "run_gauntlet"
]