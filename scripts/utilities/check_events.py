#!/usr/bin/env python3
"""
Smoke test script to verify event features are working properly.
Run this to check if event features are being loaded and built correctly.
"""

import sys
import os
import pandas as pd

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

try:
    from alpha_discovery.data.events import build_event_features
    print("✓ Successfully imported build_event_features")
except ImportError as e:
    print(f"✗ Failed to import build_event_features: {e}")
    sys.exit(1)

def main():
    print("=" * 60)
    print("EVENT FEATURES SMOKE TEST")
    print("=" * 60)
    
    try:
        print("\n1. Building event features...")
        EV = build_event_features()
        print(f"✓ Event features built successfully")
        
        print(f"\n2. Event features shape: {EV.shape}")
        print(f"   Index range: {EV.index.min()} → {EV.index.max()}")
        
        print(f"\n3. First 20 columns:")
        print(f"   {EV.columns[:20].tolist()}")
        
        print(f"\n4. Event feature prefixes found:")
        prefixes = set()
        for col in EV.columns:
            for prefix in ["EV", "days_to_", "COND.", "EXP.", "META.", "day_"]:
                if col.startswith(prefix):
                    prefixes.add(prefix)
                    break
        print(f"   {sorted(prefixes)}")
        
        print(f"\n5. Sample EV_forward_calendar_heat values (last 5 rows):")
        heat_cols = [c for c in EV.columns if "EV_forward_calendar_heat" in c]
        if heat_cols:
            print(f"   Columns: {heat_cols}")
            for col in heat_cols[:3]:  # Show first 3 heat columns
                sample = EV[col].dropna().tail(5)
                if not sample.empty:
                    print(f"   {col}: {sample.values}")
                else:
                    print(f"   {col}: (all NaN)")
        else:
            print("   No EV_forward_calendar_heat columns found!")
        
        print(f"\n6. Non-empty columns count:")
        non_empty = [c for c in EV.columns if not EV[c].dropna().empty]
        print(f"   {len(non_empty)} out of {len(EV.columns)} columns have data")
        
        print(f"\n7. Sample data check:")
        if not EV.empty:
            sample_row = EV.iloc[-1]
            non_nan_cols = sample_row.dropna()
            print(f"   Last row has {len(non_nan_cols)} non-NaN values out of {len(sample_row)}")
            if len(non_nan_cols) > 0:
                print(f"   Sample values: {dict(list(non_nan_cols.head(5).items()))}")
        
        print("\n" + "=" * 60)
        print("✓ EVENT FEATURES TEST COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
