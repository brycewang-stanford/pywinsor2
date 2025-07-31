#!/usr/bin/env python3
"""
Simple test for basic pywinsor2 functionality.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the pywinsor2 package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pywinsor2'))

from pywinsor2 import winsor2


def test_basic():
    """Test basic functionality."""
    print("Testing basic pywinsor2 functionality...")
    
    # Create simple test data
    data = pd.DataFrame({
        'wage': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],  # outlier: 100
        'industry': ['A'] * 5 + ['B'] * 5
    })
    
    print("Original data:")
    print(data['wage'].describe())
    
    # Test basic winsorizing
    result1 = winsor2(data, 'wage')
    print(f"\nAfter winsorizing: max value = {result1['wage_w'].max()}")
    
    # Test trimming
    result2 = winsor2(data, 'wage', trim=True)
    print(f"After trimming: {result2['wage_tr'].isna().sum()} missing values")
    
    # Test group processing
    result3 = winsor2(data, 'wage', by='industry')
    print("Group processing completed successfully")
    
    print("\nBasic tests PASSED! ✅")
    return True


if __name__ == "__main__":
    try:
        test_basic()
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
