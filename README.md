# pywinsor2

[![PyPI version](https://badge.fury.io/py/pywinsor2.svg)](https://badge.fury.io/py/pywinsor2)
[![Downloads](https://static.pepy.tech/badge/pywinsor2)](https://pepy.tech/project/pywinsor2)
[![Downloads](https://static.pepy.tech/badge/pywinsor2/month)](https://pepy.tech/project/pywinsor2)
[![Downloads](https://static.pepy.tech/badge/pywinsor2/week)](https://pepy.tech/project/pywinsor2)
[![Python Versions](https://img.shields.io/pypi/pyversions/pywinsor2.svg)](https://pypi.org/project/pywinsor2/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/brycewang-stanford/pywinsor2.svg?style=social&label=Star)](https://github.com/brycewang-stanford/pywinsor2)

Python implementation of Stata's `winsor2` command for winsorizing and trimming data.

## Installation

```bash
pip install pywinsor2
```

## Quick Start

```python
import pandas as pd
import pywinsor2 as pw2

# Load sample data
data = pd.DataFrame({
    'wage': [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0, 100.0],
    'industry': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B', 'A', 'B']
})

# Winsorize at 1st and 99th percentiles (default)
result = pw2.winsor2(data, ['wage'])

# Winsorize with custom cuts
result = pw2.winsor2(data, ['wage'], cuts=(5, 95))

# Trim instead of winsorize
result = pw2.winsor2(data, ['wage'], trim=True)

# Winsorize by group
result = pw2.winsor2(data, ['wage'], by='industry')

# Replace original variables
pw2.winsor2(data, ['wage'], replace=True)
```

## Features

- **Winsorizing**: Replace extreme values with percentile values
- **Trimming**: Remove extreme values (set to NaN)
- **Group-wise processing**: Process data within groups
- **Flexible percentiles**: Specify custom cut-off percentiles
- **Multiple variables**: Process multiple columns simultaneously
- **Stata compatibility**: API designed to match Stata's `winsor2` command

## Main Function

### `winsor2(data, varlist, cuts=(1, 99), suffix=None, replace=False, trim=False, by=None, label=False)`

**Parameters:**
- `data` (DataFrame): Input pandas DataFrame
- `varlist` (list): List of column names to process
- `cuts` (tuple): Percentiles for winsorizing/trimming (default: (1, 99))
- `suffix` (str): Suffix for new variables (default: '_w' for winsor, '_tr' for trim)
- `replace` (bool): Replace original variables (default: False)
- `trim` (bool): Trim instead of winsorize (default: False)
- `by` (str or list): Group variables for group-wise processing
- `label` (bool): Add descriptive labels to new columns (default: False)

**Returns:**
- `DataFrame`: Processed DataFrame with winsorized/trimmed variables

## Examples

### Basic Usage

```python
import pandas as pd
import pywinsor2 as pw2

# Create sample data
data = pd.DataFrame({
    'wage': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],  # outlier: 100
    'age': [20, 25, 30, 35, 40, 45, 50, 55, 60, 25]
})

# Winsorize at default percentiles (1, 99)
result = pw2.winsor2(data, ['wage'])
print(result['wage_w'])  # New winsorized variable

# Winsorize multiple variables
result = pw2.winsor2(data, ['wage', 'age'], cuts=(5, 95))

# Trim outliers
result = pw2.winsor2(data, ['wage'], trim=True, cuts=(10, 90))
print(result['wage_tr'])  # Trimmed variable
```

### Group-wise Processing

```python
# Winsorize within groups
data = pd.DataFrame({
    'wage': [1, 2, 3, 10, 1, 2, 3, 15],
    'industry': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
})

result = pw2.winsor2(data, ['wage'], by='industry', cuts=(25, 75))
```

### Advanced Options

```python
# Replace original variables
pw2.winsor2(data, ['wage'], replace=True, cuts=(2, 98))

# Custom suffix and labels
result = pw2.winsor2(data, ['wage'], suffix='_clean', label=True)
```

## Comparison with Stata

| Stata Command | Python Equivalent |
|---------------|-------------------|
| `winsor2 wage` | `pw2.winsor2(df, ['wage'])` |
| `winsor2 wage, cuts(5 95)` | `pw2.winsor2(df, ['wage'], cuts=(5, 95))` |
| `winsor2 wage, trim` | `pw2.winsor2(df, ['wage'], trim=True)` |
| `winsor2 wage, by(industry)` | `pw2.winsor2(df, ['wage'], by='industry')` |
| `winsor2 wage, replace` | `pw2.winsor2(df, ['wage'], replace=True)` |

## License

MIT License

## Author

Bryce Wang - brycew6m@stanford.edu

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.
