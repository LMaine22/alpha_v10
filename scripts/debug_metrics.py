import sys
sys.path.append('.')

from alpha_discovery.config import settings
from alpha_discovery.search.ga_core import _calculate_objectives_with_robust_replacements
import pandas as pd
import numpy as np

# Test with minimal data
print("Testing metrics calculation...")

# Create a minimal test scenario
test_dates = pd.date_range('2025-01-01', periods=50, freq='B')
trigger_dates = pd.DatetimeIndex(test_dates[:10])

# Create some fake returns
returns_8d = pd.Series(np.random.normal(0, 0.02, len(test_dates)), index=test_dates)
unconditional_returns = {8: returns_8d}
horizons = [8]

print(f"Trigger dates: {len(trigger_dates)}")
print(f"Returns shape: {returns_8d.shape}")

try:
    metrics = _calculate_objectives_with_robust_replacements(
        trigger_dates, unconditional_returns, horizons, is_oos_fold=False
    )
    print("\nMetrics calculated successfully:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {type(value)}")
            
except Exception as e:
    print(f"Error calculating metrics: {e}")
    import traceback
    traceback.print_exc()
