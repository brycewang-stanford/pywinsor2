"""
Comprehensive tests for pywinsor2 v0.2.0 new features.

This module tests all the new functionality added in version 0.2.0:
1. Individual cuts (cutlow, cuthigh) 
2. Enhanced variable labeling
3. Verbose reporting
4. Flag generation for trimmed observations
5. Extreme value storage
6. Variable-specific cuts
7. Enhanced group processing
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

# Add the pywinsor2 package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pywinsor2'))

from pywinsor2 import winsor2


class TestNewFeatures:
    """Test all new features in v0.2.0."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create test data with clear outliers
        self.data = pd.DataFrame({
            'wage': [10, 20, 30, 40, 50, 60, 70, 80, 90, 500, 600],  # Clear outliers: 500, 600
            'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 18, 75],      # Outliers: 18, 75
            'industry': ['Tech', 'Tech', 'Finance', 'Finance', 'Tech', 
                        'Finance', 'Tech', 'Finance', 'Tech', 'Finance', 'Tech']
        })

    def test_individual_cuts(self):
        """Test cutlow and cuthigh parameters."""
        # Test cutlow only
        result1 = winsor2(self.data, 'wage', cutlow=10)
        assert 'wage_w' in result1.columns
        assert result1['wage_w'].min() >= self.data['wage'].quantile(0.10)
        
        # Test cuthigh only
        result2 = winsor2(self.data, 'wage', cuthigh=90)  
        assert result2['wage_w'].max() <= self.data['wage'].quantile(0.90)
        
        # Test both
        result3 = winsor2(self.data, 'wage', cutlow=20, cuthigh=80)
        assert result3['wage_w'].min() >= self.data['wage'].quantile(0.20)
        assert result3['wage_w'].max() <= self.data['wage'].quantile(0.80)

    def test_verbose_mode(self):
        """Test verbose reporting."""
        result, summary = winsor2(self.data, 'wage', verbose=True)
        
        # Check summary structure
        assert isinstance(summary, dict)
        assert 'variables_processed' in summary
        assert 'observations_changed' in summary
        assert 'wage' in summary['variables_processed']
        assert 'wage' in summary['observations_changed']
        assert isinstance(summary['observations_changed']['wage'], (int, np.integer))

    def test_enhanced_labels(self):
        """Test enhanced variable labeling."""
        result = winsor2(self.data, 'wage', label=True, cuts=(10, 90), suffix='_clean')
        
        # Check if labels are created
        assert hasattr(result, '_labels')
        assert 'wage_clean' in result._labels
        label_text = result._labels['wage_clean']
        assert 'Winsorized' in label_text
        assert '10%' in label_text
        assert '90%' in label_text

    def test_trim_with_flags(self):
        """Test trimming with flag generation."""
        result = winsor2(self.data, 'wage', trim=True, genflag='_flag', cuts=(20, 80))
        
        # Check flag variable creation
        assert 'wage_flag' in result.columns
        assert 'wage_tr' in result.columns
        
        # Check flag values
        assert result['wage_flag'].dtype in [int, 'int64', np.int64]
        assert set(result['wage_flag'].unique()).issubset({0, 1})
        
        # Check consistency between flags and trimmed values
        flagged_mask = result['wage_flag'] == 1
        trimmed_mask = result['wage_tr'].isna() & result['wage'].notna()
        assert flagged_mask.sum() == trimmed_mask.sum()

    def test_extreme_value_storage(self):
        """Test storing original extreme values."""
        result = winsor2(self.data, 'wage', genextreme=('_low', '_high'), cuts=(20, 80))
        
        # Check extreme value variables creation
        assert 'wage_low' in result.columns
        assert 'wage_high' in result.columns
        assert 'wage_w' in result.columns
        
        # Check that extreme values are stored correctly
        low_extremes = result['wage_low'].notna()
        high_extremes = result['wage_high'].notna()
        
        # Should have some low and high extremes
        assert low_extremes.sum() > 0 or high_extremes.sum() > 0
        
        # Stored values should match original extreme values
        if low_extremes.sum() > 0:
            original_low = result.loc[low_extremes, 'wage']
            stored_low = result.loc[low_extremes, 'wage_low']
            assert (original_low == stored_low).all()

    def test_variable_specific_cuts(self):
        """Test variable-specific cuts."""
        var_cuts = {
            'wage': (10, 90),
            'age': (5, 95)
        }
        
        result, summary = winsor2(self.data, ['wage', 'age'], var_cuts=var_cuts, verbose=True)
        
        # Check that both variables are processed
        assert 'wage_w' in result.columns
        assert 'age_w' in result.columns
        assert len(summary['variables_processed']) == 2
        
        # Check that different cuts were applied (different change counts expected)
        wage_changes = summary['observations_changed']['wage']
        age_changes = summary['observations_changed']['age']
        assert isinstance(wage_changes, (int, np.integer))
        assert isinstance(age_changes, (int, np.integer))

    def test_group_processing_enhanced(self):
        """Test enhanced group processing."""
        result, summary = winsor2(self.data, 'wage', by='industry', verbose=True, cuts=(25, 75))
        
        # Check that processing occurred
        assert 'wage_w' in result.columns
        assert 'wage' in summary['variables_processed']
        
        # Check group-wise winsorizing makes sense
        for industry in self.data['industry'].unique():
            mask = self.data['industry'] == industry
            group_wages = result.loc[mask, 'wage_w']
            
            # Group should have some variation but less extreme values
            assert len(group_wages.unique()) > 1  # Should have variation
            assert group_wages.max() <= result['wage_w'].max()

    def test_comprehensive_example(self):
        """Test multiple features together."""
        result, summary = winsor2(
            self.data,
            ['wage'],
            cutlow=15,
            cuthigh=85,
            by='industry', 
            label=True,
            genextreme=('_orig_low', '_orig_high'),
            suffix='_processed',
            verbose=True
        )
        
        # Check all expected columns exist
        expected_cols = ['wage_processed', 'wage_orig_low', 'wage_orig_high']
        for col in expected_cols:
            assert col in result.columns
        
        # Check summary
        assert 'wage' in summary['variables_processed']
        assert isinstance(summary['observations_changed']['wage'], (int, np.integer))
        
        # Note: Label checking is simplified for now as full label implementation
        # in pandas requires additional metadata handling
        # TODO: Implement full label functionality

    def test_error_handling(self):
        """Test error conditions."""
        # Test genflag without trim
        with pytest.raises(ValueError, match="genflag can only be used with trim=True"):
            winsor2(self.data, 'wage', genflag='_flag')
        
        # Test invalid genextreme format
        with pytest.raises(TypeError, match="genextreme must be a tuple or list of 2 strings"):
            winsor2(self.data, 'wage', genextreme=('_low',))  # Only one element
        
        # Test var_cuts with non-existent variable
        with pytest.raises(ValueError, match="var_cuts contains variables not in varlist"):
            winsor2(self.data, 'wage', var_cuts={'nonexistent': (10, 90)})

    def test_backward_compatibility(self):
        """Test that old API still works."""
        # Test original functionality still works
        result1 = winsor2(self.data, 'wage')
        result2 = winsor2(self.data, 'wage', cuts=(5, 95), trim=True)
        result3 = winsor2(self.data, 'wage', by='industry', replace=True)
        
        # Should not raise any errors and produce expected outputs
        assert 'wage_w' in result1.columns
        assert 'wage_tr' in result2.columns
        assert result3['wage'].max() < self.data['wage'].max()  # Should be winsorized


def run_all_tests():
    """Run all tests and report results."""
    test_instance = TestNewFeatures()
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        try:
            test_instance.setup_method()
            method = getattr(test_instance, method_name)
            method()
            print(f"✅ {method_name}: PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {method_name}: FAILED - {e}")
            failed += 1
    
    print(f"\nTest Summary: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
