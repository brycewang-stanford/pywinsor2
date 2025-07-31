#!/usr/bin/env python3
"""
Test script for pywinsor2 v0.2.0 new features.

This script tests all the new functionality added in version 0.2.0.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the pywinsor2 package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pywinsor2'))

from pywinsor2 import winsor2


def create_test_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n = 100
    
    data = pd.DataFrame({
        'wage': np.concatenate([
            np.random.normal(50000, 10000, n-10),  # Normal wages
            [20000, 25000, 30000, 150000, 200000, 300000, 35000, 40000, 45000, 180000]  # Some outliers
        ]),
        'age': np.concatenate([
            np.random.normal(35, 8, n-5),  # Normal ages
            [18, 19, 65, 70, 75]  # Some extreme ages
        ]),
        'industry': np.random.choice(['Tech', 'Finance', 'Healthcare'], n),
        'experience': np.concatenate([
            np.random.normal(10, 5, n-8),  # Normal experience
            [0, 1, 25, 30, 35, 40, 2, 3]  # Some outliers
        ])
    })
    
    # Ensure no negative values for age and experience
    data['age'] = np.maximum(data['age'], 18)
    data['experience'] = np.maximum(data['experience'], 0)
    
    print(f"Created test dataset with {len(data)} observations")
    print(f"Wage range: ${data['wage'].min():.0f} - ${data['wage'].max():.0f}")
    print(f"Age range: {data['age'].min():.1f} - {data['age'].max():.1f}")
    print(f"Experience range: {data['experience'].min():.1f} - {data['experience'].max():.1f}")
    
    return data


def test_individual_cuts():
    """Test cutlow and cuthigh parameters."""
    print("\n" + "="*60)
    print("Testing Individual Cut Parameters")
    print("="*60)
    
    data = create_test_data()
    
    # Test cutlow only
    print("\n1. Testing cutlow only (left-side winsorizing):")
    result1 = winsor2(data, ['wage'], cutlow=5, verbose=True)
    
    # Test cuthigh only  
    print("\n2. Testing cuthigh only (right-side winsorizing):")
    result2 = winsor2(data, ['wage'], cuthigh=95, verbose=True)
    
    # Test both cutlow and cuthigh
    print("\n3. Testing both cutlow and cuthigh:")
    result3 = winsor2(data, ['wage'], cutlow=10, cuthigh=90, verbose=True)
    
    return result3


def test_enhanced_labels():
    """Test enhanced variable labeling."""
    print("\n" + "="*60)
    print("Testing Enhanced Variable Labels")
    print("="*60)
    
    data = create_test_data()
    
    result = winsor2(data, ['wage', 'age'], cuts=(5, 95), label=True, suffix='_clean')
    
    print("\nEnhanced labels created:")
    if hasattr(result, '_labels'):
        for var, label in result._labels.items():
            print(f"  {var}: {label}")
    else:
        print("  Labels stored in DataFrame metadata")
    
    return result


def test_trim_flags():
    """Test trimming with flag generation."""
    print("\n" + "="*60)
    print("Testing Trim with Flag Variables")
    print("="*60)
    
    data = create_test_data()
    
    result, summary = winsor2(data, ['wage'], trim=True, cuts=(10, 90), 
                             genflag='_flag', verbose=True)
    
    print(f"\nFlag variable 'wage_flag' created:")
    print(f"  Observations flagged: {result['wage_flag'].sum()}")
    print(f"  Percentage flagged: {result['wage_flag'].mean()*100:.1f}%")
    
    # Show some flagged observations
    flagged = result[result['wage_flag'] == 1]
    if len(flagged) > 0:
        print(f"\nFirst few flagged observations:")
        print(flagged[['wage', 'wage_tr', 'wage_flag']].head())
    
    return result


def test_extreme_values():
    """Test storing extreme values."""
    print("\n" + "="*60)
    print("Testing Extreme Value Storage")
    print("="*60)
    
    data = create_test_data()
    
    result, summary = winsor2(data, ['wage'], cuts=(5, 95), 
                             genextreme=('_low', '_high'), verbose=True)
    
    print(f"\nExtreme value variables created:")
    print(f"  wage_low: {result['wage_low'].notna().sum()} low extreme values stored")
    print(f"  wage_high: {result['wage_high'].notna().sum()} high extreme values stored")
    
    # Show extreme values
    low_extremes = result[result['wage_low'].notna()]
    high_extremes = result[result['wage_high'].notna()]
    
    if len(low_extremes) > 0:
        print(f"\nLow extreme values:")
        print(low_extremes[['wage', 'wage_w', 'wage_low']].head())
    
    if len(high_extremes) > 0:
        print(f"\nHigh extreme values:")
        print(high_extremes[['wage', 'wage_w', 'wage_high']].head())
    
    return result


def test_variable_specific_cuts():
    """Test variable-specific cuts."""
    print("\n" + "="*60)
    print("Testing Variable-Specific Cuts")
    print("="*60)
    
    data = create_test_data()
    
    # Different cuts for different variables
    var_cuts = {
        'wage': (5, 95),  # More aggressive for wage
        'age': (2, 98),   # Less aggressive for age
        'experience': (10, 90)  # Moderate for experience
    }
    
    result, summary = winsor2(data, ['wage', 'age', 'experience'], 
                             var_cuts=var_cuts, verbose=True)
    
    print(f"\nVariable-specific cuts applied:")
    for var, cuts in var_cuts.items():
        changed = summary['observations_changed'][var]
        print(f"  {var} (cuts {cuts}): {changed} observations changed")
    
    return result


def test_group_processing():
    """Test enhanced group processing."""
    print("\n" + "="*60)
    print("Testing Enhanced Group Processing")
    print("="*60)
    
    data = create_test_data()
    
    result, summary = winsor2(data, ['wage'], by='industry', cuts=(10, 90), 
                             verbose=True, label=True)
    
    print(f"\nGroup processing results:")
    group_details = summary['processing_details']['wage']['group_details']
    for group, details in group_details.items():
        print(f"  {group}: {details['changed']} observations changed")
    
    # Compare group ranges
    print(f"\nWinsorized wage ranges by group:")
    for industry in data['industry'].unique():
        mask = data['industry'] == industry
        wages = result.loc[mask, 'wage_w']
        print(f"  {industry}: ${wages.min():.0f} - ${wages.max():.0f}")
    
    return result


def test_comprehensive_example():
    """Test multiple features together."""
    print("\n" + "="*60)
    print("Testing Comprehensive Example")
    print("="*60)
    
    data = create_test_data()
    
    # Use multiple features together
    result, summary = winsor2(
        data, 
        ['wage'], 
        cutlow=5, 
        cuthigh=95,
        by='industry',
        label=True,
        genextreme=('_original_low', '_original_high'),
        suffix='_processed',
        verbose=True
    )
    
    print(f"\nComprehensive processing completed:")
    print(f"  New variables created: {[col for col in result.columns if col not in data.columns]}")
    
    return result, summary


def main():
    """Run all tests."""
    print("PyWinsor2 v0.2.0 - New Features Testing")
    print("="*60)
    
    try:
        # Run all test functions
        test_individual_cuts()
        test_enhanced_labels()
        test_trim_flags()
        test_extreme_values()
        test_variable_specific_cuts()
        test_group_processing()
        result, summary = test_comprehensive_example()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\nFinal dataset shape: {result.shape}")
        print(f"Final columns: {list(result.columns)}")
        
        return True
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
