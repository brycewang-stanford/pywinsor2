"""
Utility functions for pywinsor2 package.

This module contains helper functions for input validation,
percentile computation, and other supporting functionality.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def validate_inputs(
    data: pd.DataFrame,
    varlist: Union[str, List[str]],
    cuts: Tuple[float, float],
    suffix: Optional[str],
    replace: bool,
    trim: bool,
    by: Optional[Union[str, List[str]]],
    genflag: Optional[str] = None,
    genextreme: Optional[Tuple[str, str]] = None,
    var_cuts: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[
    pd.DataFrame, List[str], Tuple[float, float], Optional[str], Optional[List[str]]
]:
    """
    Validate and standardize input parameters with enhanced features.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    varlist : str or list of str
        Variable names to process
    cuts : tuple of float
        Percentile cuts
    suffix : str or None
        Suffix for new variables
    replace : bool
        Whether to replace original variables
    trim : bool
        Whether to trim instead of winsorize
    by : str, list of str, or None
        Grouping variables
    genflag : str or None
        Suffix for flag variables
    genextreme : tuple of str or None
        Suffixes for extreme value variables
    var_cuts : dict or None
        Variable-specific cuts

    Returns
    -------
    tuple
        Validated and standardized inputs

    Raises
    ------
    ValueError
        If inputs are invalid
    TypeError
        If inputs have wrong types
    """

    # Validate data
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    if data.empty:
        raise ValueError("data cannot be empty")

    # Validate and standardize varlist
    if isinstance(varlist, str):
        varlist = [varlist]
    elif not isinstance(varlist, list):
        raise TypeError("varlist must be a string or list of strings")

    if not varlist:
        raise ValueError("varlist cannot be empty")

    # Check that all variables exist in the data
    missing_vars = [var for var in varlist if var not in data.columns]
    if missing_vars:
        raise ValueError(f"Variables not found in data: {missing_vars}")

    # Validate cuts
    if not isinstance(cuts, (tuple, list)) or len(cuts) != 2:
        raise ValueError("cuts must be a tuple or list of exactly 2 numbers")

    low, high = cuts
    if not all(isinstance(x, (int, float)) for x in cuts):
        raise TypeError("cuts must contain numeric values")

    if not (0 <= low <= 100) or not (0 <= high <= 100):
        raise ValueError("cuts must be between 0 and 100")

    # Ensure low <= high (swap if necessary, like Stata)
    if low > high:
        low, high = high, low
        cuts = (low, high)

    # Check for no-op case
    if low == 0 and high == 100:
        raise ValueError("cuts(0 100) would have no effect")

    # Validate suffix and replace options
    if replace and suffix is not None:
        raise ValueError("suffix cannot be specified with replace=True")

    if suffix is not None and not isinstance(suffix, str):
        raise TypeError("suffix must be a string")

    # Validate by parameter
    if by is not None:
        if isinstance(by, str):
            by = [by]
        elif not isinstance(by, list):
            raise TypeError("by must be a string or list of strings")

        # Check that all by variables exist
        missing_by_vars = [var for var in by if var not in data.columns]
        if missing_by_vars:
            raise ValueError(f"Grouping variables not found in data: {missing_by_vars}")

    # Validate new parameters
    if genflag is not None:
        if not isinstance(genflag, str):
            raise TypeError("genflag must be a string")
        if not trim:
            raise ValueError("genflag can only be used with trim=True")

    if genextreme is not None:
        if not isinstance(genextreme, (tuple, list)) or len(genextreme) != 2:
            raise TypeError("genextreme must be a tuple or list of 2 strings")
        if not all(isinstance(s, str) for s in genextreme):
            raise TypeError("genextreme elements must be strings")

    if var_cuts is not None:
        if not isinstance(var_cuts, dict):
            raise TypeError("var_cuts must be a dictionary")
        
        # Validate that all keys are in varlist
        invalid_vars = [var for var in var_cuts.keys() if var not in varlist]
        if invalid_vars:
            raise ValueError(f"var_cuts contains variables not in varlist: {invalid_vars}")
        
        # Validate cuts format for each variable
        for var, var_cut in var_cuts.items():
            if not isinstance(var_cut, (tuple, list)) or len(var_cut) != 2:
                raise ValueError(f"var_cuts['{var}'] must be a tuple or list of 2 numbers")
            
            low, high = var_cut
            if not all(isinstance(x, (int, float)) for x in var_cut):
                raise TypeError(f"var_cuts['{var}'] must contain numeric values")
            
            if not (0 <= low <= 100) or not (0 <= high <= 100):
                raise ValueError(f"var_cuts['{var}'] must be between 0 and 100")

    return data, varlist, cuts, suffix, by


def compute_percentiles(
    series: pd.Series, cuts: Tuple[float, float], mask: Optional[pd.Series] = None
) -> Tuple[float, float]:
    """
    Compute percentiles for a pandas Series.

    Parameters
    ----------
    series : pd.Series
        Input series
    cuts : tuple of float
        Percentile cuts (lower, upper)
    mask : pd.Series of bool, optional
        Boolean mask for valid observations

    Returns
    -------
    tuple of float
        Lower and upper percentile values
    """

    low, high = cuts

    if mask is not None:
        valid_data = series[mask]
    else:
        valid_data = series.dropna()

    if len(valid_data) == 0:
        return np.nan, np.nan

    # Handle edge cases
    if low == 0:
        lower_pct = valid_data.min()
    else:
        lower_pct = valid_data.quantile(low / 100)

    if high == 100:
        upper_pct = valid_data.max()
    else:
        upper_pct = valid_data.quantile(high / 100)

    return lower_pct, upper_pct


def format_label_percentile(percentile: float) -> str:
    """
    Format percentile for labels, matching Stata's style.

    Parameters
    ----------
    percentile : float
        Percentile value (0-100)

    Returns
    -------
    str
        Formatted percentile string
    """
    if percentile < 1:
        return f"0{percentile:g}"
    else:
        return f"{percentile:g}"


def check_variable_types(df: pd.DataFrame, varlist: List[str]) -> None:
    """
    Check that variables are numeric.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    varlist : list of str
        Variable names to check

    Raises
    ------
    TypeError
        If any variables are not numeric
    """
    non_numeric = []
    for var in varlist:
        if not pd.api.types.is_numeric_dtype(df[var]):
            non_numeric.append(var)

    if non_numeric:
        raise TypeError(f"Variables must be numeric: {non_numeric}")


def get_variable_info(df: pd.DataFrame, var: str) -> dict:
    """
    Get information about a variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    var : str
        Variable name

    Returns
    -------
    dict
        Variable information including dtype, missing count, etc.
    """
    series = df[var]

    info = {
        "name": var,
        "dtype": str(series.dtype),
        "count": len(series),
        "non_missing": series.notna().sum(),
        "missing": series.isna().sum(),
        "min": series.min() if pd.api.types.is_numeric_dtype(series) else None,
        "max": series.max() if pd.api.types.is_numeric_dtype(series) else None,
        "mean": series.mean() if pd.api.types.is_numeric_dtype(series) else None,
        "std": series.std() if pd.api.types.is_numeric_dtype(series) else None,
    }

    return info
