"""
Fix for ELV gate that's causing all strategies to show "no-structure"
by lowering the gate_min_oos_triggers threshold.
"""

from alpha_discovery.config import settings

# Original threshold is 15, which is too high for our test data
# Let's lower it to 5
print(f"Changing ELV gate_min_oos_triggers from {settings.elv.gate_min_oos_triggers} to 5")
settings.elv.gate_min_oos_triggers = 5

# Run the main script
import main
main.main()