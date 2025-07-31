"""
Core functionality for pywinsor2 package.

This module implements the main winsor2 function that provides
winsorizing and trimming capabilities for pandas DataFrames.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .utils import compute_percentiles, validate_inputs


def winsor2(
    data: pd.DataFrame,
    varlist: Union[str, List[str]],
    cuts: Optional[Tuple[float, float]] = None,
    cutlow: Optional[float] = None,
    cuthigh: Optional[float] = None,
    suffix: Optional[str] = None,
    replace: bool = False,
    trim: bool = False,
    by: Optional[Union[str, List[str]]] = None,
    label: bool = False,
    copy: bool = True,
    verbose: bool = False,
    genflag: Optional[str] = None,
    genextreme: Optional[Tuple[str, str]] = None,
    var_cuts: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Winsorize or trim variables in a pandas DataFrame with enhanced features.

    This function replicates and extends the functionality of Stata's winsor2 command,
    allowing you to winsorize (replace extreme values with percentile values)
    or trim (remove extreme values) variables in a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing the variables to process.
    varlist : str or list of str
        Variable name(s) to winsorize or trim.
    cuts : tuple of float, default (1, 99)
        Percentiles at which to winsorize/trim (lower, upper).
        Values should be between 0 and 100. Ignored if cutlow/cuthigh specified.
    cutlow : float, optional
        Lower percentile cut. If specified, overrides cuts lower bound.
    cuthigh : float, optional
        Upper percentile cut. If specified, overrides cuts upper bound.
    suffix : str, optional
        Suffix for new variable names. If None, defaults to '_w' for
        winsorizing or '_tr' for trimming.
    replace : bool, default False
        If True, replace original variables. Cannot be used with suffix.
    trim : bool, default False
        If True, trim (set to NaN) instead of winsorize.
    by : str or list of str, optional
        Variable name(s) for group-wise processing.
    label : bool, default False
        If True, add enhanced descriptive labels to new variables.
    copy : bool, default True
        If True, return a copy of the DataFrame. If False, modify in place.
    verbose : bool, default False
        If True, print detailed processing information.
    genflag : str, optional
        Suffix for generating flag variables indicating trimmed observations.
        Only works with trim=True.
    genextreme : tuple of str, optional
        Tuple of (low_suffix, high_suffix) for generating variables that store
        original extreme values before winsorizing/trimming.
    var_cuts : dict, optional
        Dictionary mapping variable names to their specific cuts.
        Format: {'var1': (low, high), 'var2': (low, high)}

    Returns
    -------
    pd.DataFrame or tuple
        If verbose=False: DataFrame with winsorized/trimmed variables.
        If verbose=True: Tuple of (DataFrame, summary_dict) with processing statistics.

    Examples
    --------
    >>> import pandas as pd
    >>> import pywinsor2 as pw2
    >>>
    >>> # Create sample data
    >>> data = pd.DataFrame({
    ...     'wage': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
    ...     'industry': ['A'] * 5 + ['B'] * 5
    ... })
    >>>
    >>> # Basic winsorizing
    >>> result = pw2.winsor2(data, 'wage')
    >>>
    >>> # Independent cuts
    >>> result = pw2.winsor2(data, 'wage', cutlow=5, cuthigh=95)
    >>>
    >>> # Trim with flag variables
    >>> result = pw2.winsor2(data, 'wage', trim=True, genflag='_flag')
    >>>
    >>> # Store extreme values
    >>> result = pw2.winsor2(data, 'wage', genextreme=('_low', '_high'))
    >>>
    >>> # Variable-specific cuts
    >>> result = pw2.winsor2(data, ['wage'], var_cuts={'wage': (5, 95)})
    >>>
    >>> # Verbose output
    >>> result, summary = pw2.winsor2(data, 'wage', verbose=True)
    """

    # Set default cuts if not specified
    if cuts is None and cutlow is None and cuthigh is None:
        cuts = (1, 99)
    
    # Handle individual cut specifications
    if cutlow is not None or cuthigh is not None:
        if cuts is not None:
            warnings.warn("cuts parameter ignored when cutlow/cuthigh specified")
        cuts = (cutlow if cutlow is not None else 0, 
                cuthigh if cuthigh is not None else 100)

    # Input validation with new parameters
    if var_cuts is None and genflag is None and genextreme is None:
        # Use original validation for backward compatibility
        data, varlist, cuts, suffix, by = validate_inputs(
            data, varlist, cuts, suffix, replace, trim, by
        )
    else:
        # Use enhanced validation
        data, varlist, cuts, suffix, by = validate_inputs(
            data, varlist, cuts, suffix, replace, trim, by,
            genflag=genflag, genextreme=genextreme, var_cuts=var_cuts
        )

    # Create working copy if needed
    if copy or not replace:
        df = data.copy()
    else:
        df = data

    # Set default suffix if not provided
    if suffix is None:
        suffix = "_tr" if trim else "_w"

    # Initialize summary statistics for verbose mode
    summary = {
        'variables_processed': [],
        'observations_changed': {},
        'extreme_values_stored': {},
        'processing_details': {}
    } if verbose else None

    # Validate that new variables don't already exist (if not replacing)
    if not replace:
        for var in varlist:
            new_var = f"{var}{suffix}"
            if new_var in df.columns:
                raise ValueError(
                    f"Variable '{new_var}' already exists. "
                    f"Use a different suffix or set replace=True."
                )
            
            # Check for flag variables
            if genflag:
                flag_var = f"{var}{genflag}"
                if flag_var in df.columns:
                    raise ValueError(
                        f"Flag variable '{flag_var}' already exists."
                    )
            
            # Check for extreme value variables
            if genextreme:
                low_var = f"{var}{genextreme[0]}"
                high_var = f"{var}{genextreme[1]}"
                if low_var in df.columns or high_var in df.columns:
                    raise ValueError(
                        f"Extreme value variables already exist."
                    )

    # Process each variable
    for var in varlist:
        # Get cuts for this variable
        if var_cuts and var in var_cuts:
            var_cuts_tuple = var_cuts[var]
        else:
            var_cuts_tuple = cuts
            
        if verbose:
            print(f"Processing variable '{var}' with cuts {var_cuts_tuple}")

        if by is None:
            # Process without grouping
            df = _process_variable_ungrouped_enhanced(
                df, var, var_cuts_tuple, suffix, replace, trim, label,
                genflag, genextreme, verbose, summary
            )
        else:
            # Process with grouping
            df = _process_variable_grouped_enhanced(
                df, var, var_cuts_tuple, suffix, replace, trim, label, by,
                genflag, genextreme, verbose, summary
            )

    if verbose:
        _print_summary(summary)
        return df, summary
    else:
        return df
    """
    Winsorize or trim variables in a pandas DataFrame.

    This function replicates the functionality of Stata's winsor2 command,
    allowing you to winsorize (replace extreme values with percentile values)
    or trim (remove extreme values) variables in a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing the variables to process.
    varlist : str or list of str
        Variable name(s) to winsorize or trim.
    cuts : tuple of float, default (1, 99)
        Percentiles at which to winsorize/trim (lower, upper).
        Values should be between 0 and 100.
    suffix : str, optional
        Suffix for new variable names. If None, defaults to '_w' for
        winsorizing or '_tr' for trimming.
    replace : bool, default False
        If True, replace original variables. Cannot be used with suffix.
    trim : bool, default False
        If True, trim (set to NaN) instead of winsorize.
    by : str or list of str, optional
        Variable name(s) for group-wise processing.
    label : bool, default False
        If True, add descriptive labels to new variables.
    copy : bool, default True
        If True, return a copy of the DataFrame. If False, modify in place.

    Returns
    -------
    pd.DataFrame
        DataFrame with winsorized/trimmed variables.

    Examples
    --------
    >>> import pandas as pd
    >>> import pywinsor2 as pw2
    >>>
    >>> # Create sample data
    >>> data = pd.DataFrame({
    ...     'wage': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
    ...     'industry': ['A'] * 5 + ['B'] * 5
    ... })
    >>>
    >>> # Basic winsorizing
    >>> result = pw2.winsor2(data, 'wage')
    >>>
    >>> # Winsorize with custom cuts
    >>> result = pw2.winsor2(data, 'wage', cuts=(5, 95))
    >>>
    >>> # Trim instead of winsorize
    >>> result = pw2.winsor2(data, 'wage', trim=True)
    >>>
    >>> # Group-wise processing
    >>> result = pw2.winsor2(data, 'wage', by='industry')
    """

    # Input validation
    data, varlist, cuts, suffix, by = validate_inputs(
        data, varlist, cuts, suffix, replace, trim, by
    )

    # Create working copy if needed
    if copy or not replace:
        df = data.copy()
    else:
        df = data

    # Set default suffix if not provided
    if suffix is None:
        suffix = "_tr" if trim else "_w"

    # Validate that new variables don't already exist (if not replacing)
    if not replace:
        for var in varlist:
            new_var = f"{var}{suffix}"
            if new_var in df.columns:
                raise ValueError(
                    f"Variable '{new_var}' already exists. "
                    f"Use a different suffix or set replace=True."
                )

    # Process each variable
    for var in varlist:
        if by is None:
            # Process without grouping
            df = _process_variable_ungrouped(
                df, var, cuts, suffix, replace, trim, label
            )
        else:
            # Process with grouping
            df = _process_variable_grouped(
                df, var, cuts, suffix, replace, trim, label, by
            )

    return df


def _process_variable_ungrouped(
    df: pd.DataFrame,
    var: str,
    cuts: Tuple[float, float],
    suffix: str,
    replace: bool,
    trim: bool,
    label: bool,
) -> pd.DataFrame:
    """Process a single variable without grouping."""

    # Get non-missing data for percentile calculation
    non_missing_mask = df[var].notna()

    if not non_missing_mask.any():
        warnings.warn(f"Variable '{var}' has no non-missing values.")
        # Create new variable if not replacing
        if not replace:
            new_var = f"{var}{suffix}"
            df[new_var] = np.nan
        return df

    # Compute percentiles
    lower_pct, upper_pct = compute_percentiles(df[var], cuts, non_missing_mask)

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
            df.loc[non_missing_mask & (df[var] < lower_pct), var] = np.nan
            df.loc[non_missing_mask & (df[var] > upper_pct), var] = np.nan
        else:
            df[new_var] = df[var].copy().astype(float)
            df.loc[non_missing_mask & (df[var] < lower_pct), new_var] = np.nan
            df.loc[non_missing_mask & (df[var] > upper_pct), new_var] = np.nan
    else:
        # Winsorizing: replace extreme values with percentiles
        if replace:
            # Convert to float to avoid dtype incompatibility
            if df[var].dtype != "float64":
                df[var] = df[var].astype(float)
            df.loc[non_missing_mask & (df[var] < lower_pct), var] = lower_pct
            df.loc[non_missing_mask & (df[var] > upper_pct), var] = upper_pct
        else:
            df[new_var] = df[var].copy().astype(float)
            df.loc[non_missing_mask & (df[var] < lower_pct), new_var] = lower_pct
            df.loc[non_missing_mask & (df[var] > upper_pct), new_var] = upper_pct

    # Add label if requested
    if label and not replace:
        operation = "Trim" if trim else "Winsor"
        low_str = f"{cuts[0]:g}" if cuts[0] >= 1 else f"0{cuts[0]:g}"
        high_str = f"{cuts[1]:g}"

        # Try to preserve original label
        original_label = getattr(df[var], "name", var)
        new_label = f"{original_label}-{operation}(p{low_str},p{high_str})"

        # Use setattr to avoid pandas warning about attribute assignment
        if not hasattr(df, "_labels"):
            object.__setattr__(df, "_labels", {})
        df._labels[new_var] = new_label

    return df


def _process_variable_grouped(
    df: pd.DataFrame,
    var: str,
    cuts: Tuple[float, float],
    suffix: str,
    replace: bool,
    trim: bool,
    label: bool,
    by: Union[str, List[str]],
) -> pd.DataFrame:
    """Process a single variable with grouping."""

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

    # Get group-wise percentiles
    def compute_group_percentiles(group):
        non_missing_mask = group[source_var].notna()
        if not non_missing_mask.any():
            return pd.Series([np.nan, np.nan], index=["lower_pct", "upper_pct"])

        lower_pct, upper_pct = compute_percentiles(
            group[source_var], cuts, non_missing_mask
        )
        return pd.Series([lower_pct, upper_pct], index=["lower_pct", "upper_pct"])

    # Compute percentiles for each group
    group_percentiles = df.groupby(by).apply(
        compute_group_percentiles, include_groups=False
    )

    # Merge percentiles back to main dataframe
    if isinstance(by, str):
        by_list = [by]
    else:
        by_list = by

    # Reset index to get group variables as columns
    group_percentiles = group_percentiles.reset_index()

    # Merge with original data
    df_with_pcts = df.merge(group_percentiles, on=by_list, how="left")

    # Apply winsorizing or trimming
    non_missing_mask = df[var].notna()

    # Initialize new variable
    if replace:
        new_var = var
        # Convert to float for proper assignment
        if df[var].dtype != "float64":
            df[var] = df[var].astype(float)
    else:
        new_var = f"{var}{suffix}"
        df[new_var] = df[var].astype(float)

    if trim:
        # Trimming: set extreme values to NaN
        mask_lower = non_missing_mask & (df[var] < df_with_pcts["lower_pct"])
        mask_upper = non_missing_mask & (df[var] > df_with_pcts["upper_pct"])

        df.loc[mask_lower, new_var] = np.nan
        df.loc[mask_upper, new_var] = np.nan
    else:
        # Winsorizing: replace extreme values with percentiles
        mask_lower = non_missing_mask & (df[var] < df_with_pcts["lower_pct"])
        mask_upper = non_missing_mask & (df[var] > df_with_pcts["upper_pct"])

        df.loc[mask_lower, new_var] = df_with_pcts.loc[mask_lower, "lower_pct"]
        df.loc[mask_upper, new_var] = df_with_pcts.loc[mask_upper, "upper_pct"]

    # Clean up temporary variable if used
    if replace and temp_var in df.columns:
        df.drop(columns=[temp_var], inplace=True)

    # Add label if requested
    if label and not replace:
        operation = "Trim" if trim else "Winsor"
        low_str = f"{cuts[0]:g}" if cuts[0] >= 1 else f"0{cuts[0]:g}"
        high_str = f"{cuts[1]:g}"

        original_label = getattr(df[var], "name", var)
        new_label = f"{original_label}-{operation}(p{low_str},p{high_str})"

        # Use setattr to avoid pandas warning about attribute assignment
        if not hasattr(df, "_labels"):
            object.__setattr__(df, "_labels", {})
        df._labels[new_var] = new_label

    return df

def _process_variable_ungrouped_enhanced(
    df, var, cuts, suffix, replace, trim, label, genflag=None, genextreme=None, verbose=False, summary=None
):
    """Enhanced processing for single variable without grouping."""
    # Get non-missing data for percentile calculation
    non_missing_mask = df[var].notna()

    if not non_missing_mask.any():
        warnings.warn(f"Variable '{var}' has no non-missing values.")
        if not replace:
            new_var = f"{var}{suffix}"
            df[new_var] = np.nan
        return df

    # Compute percentiles
    lower_pct, upper_pct = compute_percentiles(df[var], cuts, non_missing_mask)
    
    # Identify extreme values
    lower_extreme_mask = non_missing_mask & (df[var] < lower_pct)
    upper_extreme_mask = non_missing_mask & (df[var] > upper_pct)
    
    # Store extreme values if requested
    if genextreme:
        low_var = f"{var}{genextreme[0]}"
        high_var = f"{var}{genextreme[1]}"
        df[low_var] = np.nan
        df[high_var] = np.nan
        df.loc[lower_extreme_mask, low_var] = df.loc[lower_extreme_mask, var]
        df.loc[upper_extreme_mask, high_var] = df.loc[upper_extreme_mask, var]

    # Create new variable name
    if replace:
        new_var = var
    else:
        new_var = f"{var}{suffix}"

    # Apply winsorizing or trimming
    if trim:
        if replace:
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
            df[flag_var] = 0
            df.loc[lower_extreme_mask | upper_extreme_mask, flag_var] = 1
    else:
        if replace:
            if df[var].dtype != "float64":
                df[var] = df[var].astype(float)
            df.loc[lower_extreme_mask, var] = lower_pct
            df.loc[upper_extreme_mask, var] = upper_pct
        else:
            df[new_var] = df[var].copy().astype(float)
            df.loc[lower_extreme_mask, new_var] = lower_pct
            df.loc[upper_extreme_mask, new_var] = upper_pct

    # Enhanced labeling
    if label and not replace:
        operation = "Trimmed" if trim else "Winsorized"
        low_str = f"{cuts[0]:g}" if cuts[0] >= 1 else f"0{cuts[0]:g}"
        high_str = f"{cuts[1]:g}"
        
        original_label = getattr(df[var], "name", var)
        new_label = f"{original_label} - {operation} at {low_str}%-{high_str}%"
        
        if not hasattr(df, "_labels"):
            object.__setattr__(df, "_labels", {})
        df._labels[new_var] = new_label

    # Update summary if verbose
    if summary is not None:
        changed_count = lower_extreme_mask.sum() + upper_extreme_mask.sum()
        summary['variables_processed'].append(var)
        summary['observations_changed'][var] = changed_count
        
        if verbose:
            print(f"  - {changed_count} observations modified")

    return df


def _process_variable_grouped_enhanced(
    df, var, cuts, suffix, replace, trim, label, by, genflag=None, genextreme=None, verbose=False, summary=None
):
    """Enhanced processing for single variable with grouping."""
    import numpy as np
    
    if suffix is None:
        suffix = '_w' if not trim else '_tr'
    
    # Determine output variable name
    output_var = f"{var}{suffix}" if not replace else var
    
    # Initialize change counter
    total_changed = 0
    
    # Process each group
    grouped_data = []
    for name, group in df.groupby(by):
        group_copy = group.copy()
        
        # Compute percentiles for this group
        values = group_copy[var].dropna()
        if len(values) > 0:
            lower_val = np.percentile(values, cuts[0])
            upper_val = np.percentile(values, cuts[1])
            
            # Apply winsorization/trimming
            if trim:
                # Trimming: set extreme values to NaN
                mask = (group_copy[var] < lower_val) | (group_copy[var] > upper_val)
                group_copy[output_var] = group_copy[var].copy()
                group_copy.loc[mask & group_copy[var].notna(), output_var] = np.nan
            else:
                # Winsorization: replace extreme values with percentiles
                group_copy[output_var] = group_copy[var].astype(float).copy()
                lower_mask = group_copy[var] < lower_val
                upper_mask = group_copy[var] > upper_val
                group_copy.loc[lower_mask & group_copy[var].notna(), output_var] = float(lower_val)
                group_copy.loc[upper_mask & group_copy[var].notna(), output_var] = float(upper_val)
            
            # Count changes for this group
            if trim:
                changed_mask = (group_copy[var] < lower_val) | (group_copy[var] > upper_val)
                group_changed = changed_mask.sum()
            else:
                original_vals = group_copy[var].fillna(-999999)
                new_vals = group_copy[output_var].fillna(-999999)
                group_changed = (original_vals != new_vals).sum()
            
            total_changed += group_changed
            
            # Handle additional features (flags, extremes) - simplified for groups
            if genflag and not replace:
                flag_var = f"{var}{genflag}"
                if trim:
                    group_copy[flag_var] = ((group_copy[var] < lower_val) | 
                                          (group_copy[var] > upper_val)).astype(int)
                else:
                    group_copy[flag_var] = ((group_copy[var] < lower_val) | 
                                          (group_copy[var] > upper_val)).astype(int)
            
            # Handle extreme value storage
            if genextreme and not replace:
                low_var = f"{var}{genextreme[0]}"
                high_var = f"{var}{genextreme[1]}"
                
                # Initialize extreme value columns
                group_copy[low_var] = np.nan
                group_copy[high_var] = np.nan
                
                # Store original extreme values
                low_mask = group_copy[var] < lower_val
                high_mask = group_copy[var] > upper_val
                
                group_copy.loc[low_mask & group_copy[var].notna(), low_var] = group_copy.loc[low_mask & group_copy[var].notna(), var]
                group_copy.loc[high_mask & group_copy[var].notna(), high_var] = group_copy.loc[high_mask & group_copy[var].notna(), var]
        
        grouped_data.append(group_copy)
    
    # Combine all groups
    result_df = pd.concat(grouped_data, ignore_index=False)
    result_df = result_df.sort_index()
    
    # Update summary
    if summary is not None:
        if var not in summary['variables_processed']:
            summary['variables_processed'].append(var)
        summary['observations_changed'][var] = total_changed
    
    # Add variable label if requested
    if label and not replace:
        # This would typically be handled by pandas metadata
        # For now, we skip detailed labeling for grouped processing
        pass
    
    return result_df


def _print_summary(summary):
    """Print detailed processing summary."""
    if summary is None:
        return
        
    print("")
    print("=" * 50)
    print("PYWINSOR2 PROCESSING SUMMARY")
    print("=" * 50)
    
    num_vars = len(summary['variables_processed'])
    var_names = ", ".join(summary['variables_processed'])
    print(f"Variables processed: {num_vars}")
    print(f"Variable names: {var_names}")
    
    print("")
    print("Observations changed per variable:")
    for var, count in summary['observations_changed'].items():
        print(f"  {var}: {count} observations")
    
    total_changed = sum(summary['observations_changed'].values())
    print(f"")
    print(f"Total observations modified: {total_changed}")
    print("=" * 50)
