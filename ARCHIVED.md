# ⚠️ This Repository Has Been Archived

This repository has been **archived** and is no longer maintained. 

All functionality from `pywinsor2` has been integrated into the unified **Py-Stata-Commands** package.

## 🔗 New Repository

Please use the new unified repository:
**https://github.com/brycewang-stanford/Py-Stata-Commands**

## 🚀 New Installation

```bash
pip install py-stata-commands
```

## 📖 New Usage

```python
from py_stata_commands import winsor2

# Same functionality, new location
result = winsor2.winsor2(df, ['income'], cuts=(1, 99))
result = winsor2.winsor2(df, ['income'], by='industry')
```

## 📚 Benefits of the New Package

- **Unified installation**: All Stata-equivalent commands in one package
- **Consistent API**: Familiar syntax across all modules
- **Better maintenance**: Single repository for all related functionality
- **Comprehensive documentation**: Complete examples and guides

---

**Migration**: Replace `import pywinsor2` with `from py_stata_commands import winsor2`