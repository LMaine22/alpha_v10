# alpha_discovery/eval/__init__.py
from . import metrics, selection
# Legacy modules removed in Phase 10 (CPCV/ELV/Hart):
# - validation.py (use orchestrator.py instead)
# - elv.py (use EligibilityMatrix instead)
# - hart_index.py (use forecast-first metrics instead)

__all__ = [
    "metrics", 
    "selection",
]