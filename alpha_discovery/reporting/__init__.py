"""
Reporting package for alpha discovery and validation.

Includes:
- Eligibility reports from forecast-first validation
- Forecast slates for production trading
- Pareto-optimal setup selection
- Diagnostic visualizations
"""

from . import display_utils
from .eligibility_report import (
    generate_eligibility_report,
    print_eligibility_summary,
    export_reliability_data
)

__all__ = [
    'display_utils',
    'generate_eligibility_report',
    'print_eligibility_summary',
    'export_reliability_data',
]