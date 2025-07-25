"""
Example usage of pywinsor2 package.

This script demonstrates the key features of pywinsor2,
replicating common Stata winsor2 usage patterns.
"""

import numpy as np
import pandas as pd

from pywinsor2 import winsor2


def main():
    """Run example demonstrations."""
    print("PyWinsor2 Example Usage")
    print("=" * 50)

    # Create sample data similar to what you might see in economic research
    np.random.seed(42)
    n = 1000

    # Generate data with outliers
    wages = np.random.lognormal(mean=2.5, sigma=0.8, size=n)
    wages = np.concatenate([wages, [wages.max() * 5, wages.max() * 3]])  # Add outliers

    hours = np.random.normal(40, 10, size=len(wages))
    hours = np.maximum(hours, 1)  # Ensure positive hours

    education = np.random.choice(
        ["High School", "College", "Graduate"], size=len(wages), p=[0.4, 0.4, 0.2]
    )

    data = pd.DataFrame({"wage": wages, "hours": hours, "education": education})

    print(f"Created dataset with {len(data)} observations")
    print("Wage statistics before winsorizing:")
    print(data["wage"].describe())
    print()

    # Example 1: Basic winsorizing (default 1% and 99% percentiles)
    print("Example 1: Basic winsorizing")
    print("-" * 30)
    result1 = winsor2(data, "wage")
    print(
        "Variables created:",
        [col for col in result1.columns if col not in data.columns],
    )
    print(f"Original max wage: ${data['wage'].max():.2f}")
    print(f"Winsorized max wage: ${result1['wage_w'].max():.2f}")
    print()

    # Example 2: Custom percentiles
    print("Example 2: More aggressive winsorizing (5th and 95th percentiles)")
    print("-" * 30)
    result2 = winsor2(data, "wage", cuts=(5, 95))
    print(f"5th-95th percentile max wage: ${result2['wage_w'].max():.2f}")
    print()

    # Example 3: Trimming instead of winsorizing
    print("Example 3: Trimming (setting extreme values to missing)")
    print("-" * 30)
    result3 = winsor2(data, "wage", trim=True)
    print(f"Observations trimmed: {result3['wage_tr'].isna().sum()}")
    print(f"Percentage trimmed: {result3['wage_tr'].isna().mean() * 100:.1f}%")
    print()

    # Example 4: Multiple variables
    print("Example 4: Winsorizing multiple variables")
    print("-" * 30)
    result4 = winsor2(data, ["wage", "hours"])
    new_vars = [col for col in result4.columns if col.endswith("_w")]
    print("New variables created:", new_vars)
    print()

    # Example 5: Group-wise winsorizing
    print("Example 5: Group-wise winsorizing by education")
    print("-" * 30)
    result5 = winsor2(data, "wage", by="education")

    # Compare group-wise vs overall winsorizing
    overall_result = winsor2(data, "wage")

    print("Group-wise winsorized wage ranges:")
    for edu in data["education"].unique():
        mask = data["education"] == edu
        group_wages = result5.loc[mask, "wage_w"]
        print(f"  {edu}: ${group_wages.min():.2f} - ${group_wages.max():.2f}")

    overall_min = overall_result["wage_w"].min()
    overall_max = overall_result["wage_w"].max()
    print(f"\nOverall winsorized range: ${overall_min:.2f} - ${overall_max:.2f}")
    print()

    # Example 6: Replace original variable
    print("Example 6: Replace original variable")
    print("-" * 30)
    data_copy = data.copy()
    result6 = winsor2(data_copy, "wage", replace=True)
    print("Original wage column modified in-place")
    print(f"New max wage: ${result6['wage'].max():.2f}")
    print()

    # Example 7: Custom suffix
    print("Example 7: Custom suffix")
    print("-" * 30)
    result7 = winsor2(data, "wage", suffix="_clean")
    print(
        "Variable created:", [col for col in result7.columns if col.endswith("_clean")]
    )
    print()

    # Example 8: Comparison with Stata-like workflow
    print("Example 8: Stata-like workflow")
    print("-" * 30)
    print("Original Stata commands and Python equivalents:")
    print()

    stata_commands = [
        "winsor2 wage, cuts(1 99)",
        "winsor2 wage, cuts(5 95) trim",
        "winsor2 wage hours, replace",
        "winsor2 wage, by(education)",
        "winsor2 wage, suffix(_adj) cuts(2.5 97.5)",
    ]

    python_commands = [
        "winsor2(data, 'wage')",
        "winsor2(data, 'wage', cuts=(5, 95), trim=True)",
        "winsor2(data, ['wage', 'hours'], replace=True)",
        "winsor2(data, 'wage', by='education')",
        "winsor2(data, 'wage', suffix='_adj', cuts=(2.5, 97.5))",
    ]

    for stata, python in zip(stata_commands, python_commands):
        print(f"Stata:  {stata}")
        print(f"Python: {python}")
        print()

    # Example 9: Statistical comparison
    print("Example 9: Statistical impact of winsorizing")
    print("-" * 30)

    # Compare statistics before and after
    original_stats = data["wage"].describe()
    winsorized = winsor2(data, "wage")["wage_w"]
    winsor_stats = winsorized.describe()

    print("Statistic comparison:")
    print(f"{'Statistic':<10} {'Original':<12} {'Winsorized':<12} {'Change':<12}")
    print("-" * 50)

    for stat in ["mean", "std", "min", "max"]:
        orig = original_stats[stat]
        wins = winsor_stats[stat]
        change = ((wins - orig) / orig) * 100
        print(f"{stat:<10} {orig:<12.2f} {wins:<12.2f} {change:<12.1f}%")

    print()
    print("Example completed! Check the results above.")


if __name__ == "__main__":
    main()
