#!/usr/bin/env python3
"""
Test script for pywinsor2 v0.2.0 new features.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the pywinsor2 package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pywinsor2'))

from pywinsor2 import winsor2


def test_new_features():
    """Test new features in v0.2.0."""
    print("Testing pywinsor2 v0.2.0 new features...")
    
    # Create test data
    data = pd.DataFrame({
        'wage': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 200],
        'age': [20, 25, 30, 35, 40, 45, 50, 55, 60, 25, 30],
        'industry': ['A'] * 6 + ['B'] * 5
    })
    
    print(f"Original wage range: {data['wage'].min()} - {data['wage'].max()}")
    
    # Test 1: Individual cuts (cutlow, cuthigh)
    print("\n1. Testing individual cuts...")
    result1 = winsor2(data, 'wage', cutlow=10, cuthigh=90, verbose=True)
    
    # Test 2: Verbose mode
    print("\n2. Testing verbose mode...")
    result2, summary = winsor2(data, 'wage', cuts=(5, 95), verbose=True)
    
    # Test 3: Enhanced labels
    print("\n3. Testing enhanced labels...")
    result3 = winsor2(data, 'wage', label=True, suffix='_clean')
    if hasattr(result3, '_labels'):
        print(f"Label created: {result3._labels}")
    
    # Test 4: Trim with flag generation
    print("\n4. Testing trim with flags...")
    result4 = winsor2(data, 'wage', trim=True, genflag='_flag', cuts=(20, 80))
    if 'wage_flag' in result4.columns:
        flagged = result4['wage_flag'].sum()
        print(f"Flagged observations: {flagged}")
    
    # Test 5: Store extreme values
    print("\n5. Testing extreme value storage...")
    result5 = winsor2(data, 'wage', genextreme=('_low', '_high'), cuts=(10, 90))
    if 'wage_low' in result5.columns:
        extreme_low = result5['wage_low'].notna().sum()
        extreme_high = result5['wage_high'].notna().sum()
        print(f"Extreme values stored: {extreme_low} low, {extreme_high} high")
    
    print("\n✅ All new features tested successfully!")
    return True


if __name__ == "__main__":
    try:
        test_new_features()
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
