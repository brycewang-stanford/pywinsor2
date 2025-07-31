"""
Enhanced core functionality for pywinsor2 package v0.2.0.

This module implements additional helper functions with clean syntax.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import numpy as np
import pandas as pd
from .utils import compute_percentiles


def _process_variable_ungrouped_v2(
    df: pd.DataFrame,
    var: str,
    cuts: Tuple[float, float],
    suffix: str,
    replace: bool,
    trim: bool,
    label: bool,
    genflag: Optional[str] = None,
    genextreme: Optional[Tuple[str, str]] = None,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Process a single variable without grouping - enhanced version."""

    # Get non-missing data for percentile calculation
    non_missing_mask = df[var].notna()
    
    # Initialize summary
    summary = {
        'changed_count': 0,
        'trimmed_count': 0,
        'winsorized_lower': 0,
        'winsorized_upper': 0,
        'extreme_count': 0,
        'percentiles': None
    }

    if not non_missing_mask.any():
        warnings.warn(f"Variable '{var}' has no non-missing values.")
        # Create new variable if not replacing
        if not replace:
            new_var = f"{var}{suffix}"
            df[new_var] = np.nan
        return df, summary

    # Compute percentiles
    lower_pct, upper_pct = compute_percentiles(df[var], cuts, non_missing_mask)
    summary['percentiles'] = (lower_pct, upper_pct)

    # Identify extreme values
    lower_extreme_mask = non_missing_mask & (df[var] < lower_pct)
    upper_extreme_mask = non_missing_mask & (df[var] > upper_pct)
    
    # Store extreme values if requested
    if genextreme:
        low_var = f"{var}{genextreme[0]}"
        high_var = f"{var}{genextreme[1]}"
        
        # Initialize extreme value variables
        df[low_var] = np.nan
        df[high_var] = np.nan
        
        # Store original extreme values
        df.loc[lower_extreme_mask, low_var] = df.loc[lower_extreme_mask, var]
        df.loc[upper_extreme_mask, high_var] = df.loc[upper_extreme_mask, var]
        
        summary['extreme_count'] = lower_extreme_mask.sum() + upper_extreme_mask.sum()

    # Create new variable name
    if replace:
        new_var = var
    else:
        new_var = f"{var}{suffix}"

    # Apply winsorizing or trimming
    if trim:
        # Trimming: set extreme values to NaN
        if replace:
            # Convert to float to allow NaN values
            if df[var].dtype != "float64":
                df[var] = df[var].astype(float)
            df.loc[lower_extreme_mask, var] = np.nan
            df.loc[upper_extreme_mask, var] = np.nan
        else:
            df[new_var] = df[var].copy().astype(float)
            df.loc[lower_extreme_mask, new_var] = np.nan
            df.loc[upper_extreme_mask, new_var] = np.nan
            
        # Generate flag variables if requested
        if genflag:
            flag_var = f"{var}{genflag}"
            df[flag_var] = 0  # Initialize with 0
            df.loc[lower_extreme_mask | upper_extreme_mask, flag_var] = 1
            
        summary['trimmed_count'] = lower_extreme_mask.sum() + upper_extreme_mask.sum()
        summary['changed_count'] = summary['trimmed_count']
    else:
        # Winsorizing: replace extreme values with percentiles
        if replace:
            # Convert to float to avoid dtype incompatibility
            if df[var].dtype != "float64":
                df[var] = df[var].astype(float)
            df.loc[lower_extreme_mask, var] = lower_pct
            df.loc[upper_extreme_mask, var] = upper_pct
        else:
            df[new_var] = df[var].copy().astype(float)
            df.loc[lower_extreme_mask, new_var] = lower_pct
            df.loc[upper_extreme_mask, new_var] = upper_pct
            
        summary['winsorized_lower'] = lower_extreme_mask.sum()
        summary['winsorized_upper'] = upper_extreme_mask.sum()
        summary['changed_count'] = summary['winsorized_lower'] + summary['winsorized_upper']

    # Add enhanced label if requested
    if label and not replace:
        operation = "Trimmed" if trim else "Winsorized"
        low_str = f"{cuts[0]:g}" if cuts[0] >= 1 else f"0{cuts[0]:g}"
        high_str = f"{cuts[1]:g}"

        # Create enhanced label with operation details
        original_label = getattr(df[var], "name", var)
        new_label = f"{original_label} - {operation} at {low_str}%-{high_str}%"

        # Use setattr to avoid pandas warning about attribute assignment
        if not hasattr(df, "_labels"):
            object.__setattr__(df, "_labels", {})
        df._labels[new_var] = new_label

    if verbose:
        print(f"  - {summary['changed_count']} observations modified")
        if trim:
            print(f"  - {summary['trimmed_count']} observations trimmed")
        else:
            print(f"  - {summary['winsorized_lower']} lower, {summary['winsorized_upper']} upper winsorized")

    return df, summary


def _process_variable_grouped_v2(
    df: pd.DataFrame,
    var: str,
    cuts: Tuple[float, float],
    suffix: str,
    replace: bool,
    trim: bool,
    label: bool,
    by: Union[str, List[str]],
    genflag: Optional[str] = None,
    genextreme: Optional[Tuple[str, str]] = None,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Process a single variable with grouping - enhanced version."""

    # Initialize summary
    summary = {
        'changed_count': 0,
        'trimmed_count': 0,
        'winsorized_lower': 0,
        'winsorized_upper': 0,
        'extreme_count': 0,
        'group_details': {}
    }

    # Create new variable name
    if replace:
        new_var = var
        # We need a temporary variable for group processing
        temp_var = f"_temp_{var}"
        df[temp_var] = df[var].copy()
        source_var = temp_var
    else:
        new_var = f"{var}{suffix}"
        source_var = var

    # Initialize extreme value variables if requested
    if genextreme:
        low_var = f"{var}{genextreme[0]}"
        high_var = f"{var}{genextreme[1]}"
        df[low_var] = np.nan
        df[high_var] = np.nan

    # Initialize flag variable if requested
    if genflag and trim:
        flag_var = f"{var}{genflag}"
        df[flag_var] = 0

    # Get group-wise percentiles and process
    def process_group(group):
        group_summary = {'changed': 0, 'lower': 0, 'upper': 0}
        
        non_missing_mask = group[source_var].notna()
        if not non_missing_mask.any():
            return group, group_summary

        lower_pct, upper_pct = compute_percentiles(
            group[source_var], cuts, non_missing_mask
        )
        
        # Identify extreme values within group
        lower_extreme_mask = non_missing_mask & (group[source_var] < lower_pct)
        upper_extreme_mask = non_missing_mask & (group[source_var] > upper_pct)
        
        # Store extreme values if requested
        if genextreme:
            group.loc[lower_extreme_mask, low_var] = group.loc[lower_extreme_mask, source_var]
            group.loc[upper_extreme_mask, high_var] = group.loc[upper_extreme_mask, source_var]
        
        # Apply transformation
        if trim:
            group.loc[lower_extreme_mask, new_var] = np.nan
            group.loc[upper_extreme_mask, new_var] = np.nan
            
            if genflag:
                group.loc[lower_extreme_mask | upper_extreme_mask, flag_var] = 1
                
            group_summary['changed'] = lower_extreme_mask.sum() + upper_extreme_mask.sum()
        else:
            group.loc[lower_extreme_mask, new_var] = lower_pct
            group.loc[upper_extreme_mask, new_var] = upper_pct
            
            group_summary['lower'] = lower_extreme_mask.sum()
            group_summary['upper'] = upper_extreme_mask.sum()
            group_summary['changed'] = group_summary['lower'] + group_summary['upper']
        
        return group, group_summary

    # Apply group processing
    if isinstance(by, str):
        by_list = [by]
    else:
        by_list = by

    # Initialize new variable
    if not replace:
        df[new_var] = df[var].astype(float)
    elif df[var].dtype != "float64":
        df[var] = df[var].astype(float)

    # Process each group
    for group_keys, group_data in df.groupby(by_list):
        group_indices = group_data.index
        processed_group, group_summary = process_group(group_data)
        
        # Update main dataframe
        df.loc[group_indices, new_var] = processed_group[new_var]
        
        if genextreme:
            df.loc[group_indices, low_var] = processed_group[low_var]
            df.loc[group_indices, high_var] = processed_group[high_var]
            
        if genflag and trim:
            df.loc[group_indices, flag_var] = processed_group[flag_var]
        
        # Update summary
        summary['changed_count'] += group_summary['changed']
        if trim:
            summary['trimmed_count'] += group_summary['changed']
        else:
            summary['winsorized_lower'] += group_summary['lower']
            summary['winsorized_upper'] += group_summary['upper']
            
        summary['group_details'][str(group_keys)] = group_summary

    if genextreme:
        summary['extreme_count'] = summary['changed_count']

    # Clean up temporary variable if used
    if replace and temp_var in df.columns:
        df.drop(columns=[temp_var], inplace=True)

    # Add enhanced label if requested
    if label and not replace:
        operation = "Trimmed" if trim else "Winsorized"
        low_str = f"{cuts[0]:g}" if cuts[0] >= 1 else f"0{cuts[0]:g}"
        high_str = f"{cuts[1]:g}"

        original_label = getattr(df[var], "name", var)
        by_str = "+".join(by_list) if isinstance(by, list) else by
        new_label = f"{original_label} - {operation} at {low_str}%-{high_str}% by {by_str}"

        # Use setattr to avoid pandas warning about attribute assignment
        if not hasattr(df, "_labels"):
            object.__setattr__(df, "_labels", {})
        df._labels[new_var] = new_label

    if verbose:
        print(f"  - {summary['changed_count']} observations modified across groups")
        if trim:
            print(f"  - {summary['trimmed_count']} observations trimmed")
        else:
            print(f"  - {summary['winsorized_lower']} lower, {summary['winsorized_upper']} upper winsorized")

    return df, summary


def _print_summary(summary):
    """Print detailed processing summary."""
    print("\n" + "="*50)
    print("PYWINSOR2 PROCESSING SUMMARY")
    print("="*50)
    
    print(f"Variables processed: {len(summary['variables_processed'])}")
    print(f"Variable names: {', '.join(summary['variables_processed'])}")
    
    print("\nObservations changed per variable:")
    for var, count in summary['observations_changed'].items():
        print(f"  {var}: {count} observations")
    
    if summary['extreme_values_stored']:
        print("\nExtreme values stored:")
        for var, count in summary['extreme_values_stored'].items():
            print(f"  {var}: {count} extreme values saved")
    
    total_changed = sum(summary['observations_changed'].values())
    print(f"\nTotal observations modified: {total_changed}")
    print("="*50)
