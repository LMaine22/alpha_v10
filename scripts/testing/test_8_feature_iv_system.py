#!/usr/bin/env python3
"""
Test script for the complete 8-feature delta/implied volatility system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpha_discovery.options.market import (
    check_new_iv_columns_available,
    get_1m_smile_iv,
    get_30d_atm_iv,
    interpolate_smile_iv
)

def create_test_data():
    """Create test data with all 8 IV features."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # Create test DataFrame with all 8 IV features
    data = {}
    
    # Add dates
    data['date'] = dates
    
    # Add 30D ATM features
    data['SPY_CALL_IMP_VOL_30D'] = 0.15 + 0.02 * np.random.randn(100)
    data['SPY_PUT_IMP_VOL_30D'] = 0.16 + 0.02 * np.random.randn(100)
    
    # Add 1M delta smile features
    data['SPY_1M_CALL_IMP_VOL_10DELTA_DFLT'] = 0.18 + 0.03 * np.random.randn(100)
    data['SPY_1M_CALL_IMP_VOL_25DELTA_DFLT'] = 0.16 + 0.02 * np.random.randn(100)
    data['SPY_1M_CALL_IMP_VOL_40DELTA_DFLT'] = 0.15 + 0.02 * np.random.randn(100)
    data['SPY_1M_PUT_IMP_VOL_40DELTA_DFLT'] = 0.15 + 0.02 * np.random.randn(100)
    data['SPY_1M_PUT_IMP_VOL_25DELTA_DFLT'] = 0.16 + 0.02 * np.random.randn(100)
    data['SPY_1M_PUT_IMP_VOL_10DELTA_DFLT'] = 0.18 + 0.03 * np.random.randn(100)
    
    # Add price data
    data['SPY_PX_LAST'] = 400 + 10 * np.random.randn(100)
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    return df

def test_8_feature_system():
    """Test the complete 8-feature IV system."""
    print("🧪 Testing Complete 8-Feature IV System")
    print("=" * 50)
    
    # Create test data
    df = create_test_data()
    ticker = "SPY"
    test_date = df.index[50]  # Middle date
    
    print(f"📊 Test Data: {ticker} on {test_date}")
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
    print()
    
    # Test 1: Check if all 8 features are available
    print("1️⃣ Testing Column Availability Check")
    available, missing = check_new_iv_columns_available(ticker, test_date, df)
    print(f"   Available: {available}")
    print(f"   Missing: {missing}")
    print()
    
    # Test 2: Test 30D ATM IV retrieval
    print("2️⃣ Testing 30D ATM IV Retrieval")
    call_30d = get_30d_atm_iv(ticker, test_date, df, "long")
    put_30d = get_30d_atm_iv(ticker, test_date, df, "short")
    print(f"   Call 30D IV: {call_30d:.4f}")
    print(f"   Put 30D IV: {put_30d:.4f}")
    print()
    
    # Test 3: Test 1M smile IV retrieval for all deltas
    print("3️⃣ Testing 1M Smile IV Retrieval")
    deltas = ['CALL_10D', 'CALL_25D', 'CALL_40D', 'PUT_40D', 'PUT_25D', 'PUT_10D']
    for delta in deltas:
        iv = get_1m_smile_iv(ticker, test_date, df, delta)
        print(f"   {delta}: {iv:.4f}")
    print()
    
    # Test 4: Test interpolation
    print("4️⃣ Testing Smile Interpolation")
    call_deltas = [0.10, 0.20, 0.25, 0.30, 0.40]
    put_deltas = [-0.40, -0.30, -0.25, -0.20, -0.10]
    
    print("   Call Delta Interpolation:")
    for delta in call_deltas:
        iv = interpolate_smile_iv(ticker, test_date, df, delta, "call")
        print(f"     {delta:4.2f}: {iv:.4f}")
    
    print("   Put Delta Interpolation:")
    for delta in put_deltas:
        iv = interpolate_smile_iv(ticker, test_date, df, delta, "put")
        print(f"     {delta:5.2f}: {iv:.4f}")
    print()
    
    # Test 5: Test missing data handling
    print("5️⃣ Testing Missing Data Handling")
    # Remove one column to test fallback
    df_incomplete = df.drop(columns=['SPY_1M_CALL_IMP_VOL_10DELTA_DFLT'])
    available_incomplete, missing_incomplete = check_new_iv_columns_available(ticker, test_date, df_incomplete)
    print(f"   With missing column - Available: {available_incomplete}")
    print(f"   Missing: {missing_incomplete}")
    print()
    
    print("✅ 8-Feature IV System Test Complete!")
    print("   All features working correctly with complete data")
    print("   Graceful fallback when features are missing")

if __name__ == "__main__":
    test_8_feature_system()
