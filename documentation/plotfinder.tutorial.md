# PlotFinder Tutorial: Advanced Regression Analysis Made Simple

## Overview

PlotFinder is a comprehensive Python library that automatically determines the best mathematical function to fit your data. Instead of manually testing different regression types, PlotFinder tests multiple approaches and returns the one with the lowest error.

## Quick Start

### Basic Usage

```python
from Plot_Finder import find_best_fit

# Your data
x = [1, 2, 3, 4, 5]
y = [2, 5, 10, 17, 26]

# Find the best fit
method, error, formula = find_best_fit(x, y, plot=True)

print(f"Best method: {method}")
print(f"Error: {error}")
```

### Installation Requirements

```bash
pip install numpy pandas matplotlib sympy
```

## Core Functions

### 1. `find_best_fit(x, y, plot=False, maxPolynomial=7, methods="all")`

**The main function** - automatically tests multiple regression types and returns the best fit.

**Parameters:**
- `x, y`: Your data arrays
- `plot`: Whether to display a plot (default: False)
- `maxPolynomial`: Maximum polynomial degree to test (default: 7)
- `methods`: Which methods to test (default: "all")

**Returns:**
- `method`: String name of the best method
- `error`: Mean absolute error of the fit
- `formula`: Standardized regression output dictionary

### 2. `find_fourier(x, y, Iterations=2, plot=False, maxPolynomial=3, methods="all")`

**Advanced function** - performs iterative regression to capture multi-component functions.

**Use Case:** When your data contains multiple mathematical components (e.g., `y = 2x + 3sin(x)`)

## Supported Regression Types

| Type | Example Function | When to Use |
|------|------------------|-------------|
| **Linear** | `y = 2x + 3` | Straight-line relationships |
| **Quadratic** | `y = x² + 2x + 1` | Parabolic curves |
| **Cubic** | `y = x³ - 2x² + x + 1` | S-curves, inflection points |
| **Polynomial (4-7)** | `y = x⁵ + x³ + x` | Complex curves |
| **Exponential** | `y = 3e^(0.5x)` | Growth/decay processes |
| **Logarithmic** | `y = 2ln(x) + 1` | Diminishing returns |
| **Sinusoidal** | `y = 4sin(x)` | Periodic/oscillating data |

## Method Selection

### Test All Methods (Default)
```python
method, error, formula = find_best_fit(x, y, methods="all")
```

### Test Specific Methods Only
```python
# Only linear and exponential
method, error, formula = find_best_fit(x, y, methods="linear, exponential")

# Only polynomials up to degree 3
method, error, formula = find_best_fit(x, y, methods="all", maxPolynomial=3)
```

### Available Method Names
- `"linear"`, `"quadratic"`, `"cubic"`
- `"exponential"`, `"logarithmic"`, `"sine"`
- `"all"` (tests everything)

## Working with Results

### Understanding the Output

```python
method, error, formula = find_best_fit(x, y)

# Method name (string)
print(f"Best fit: {method}")  # e.g., "Quadratic"

# Numerical error (float)
print(f"Error: {error:.6f}")  # e.g., 0.000123

# Standardized formula (dictionary)
print(formula)
# {
#     "sin_terms": [(amplitude, frequency, phase)],
#     "exponential_terms": [(coefficient, exponent)],
#     "logarithmic_terms": [(coefficient, shift)],
#     "polynomial_terms": {power: coefficient}
# }
```

### Interpreting Formula Components

```python
from Regression_Standards import print_non_zero_terms

# Print only the active terms
print_non_zero_terms(formula)
```

## Real-World Examples

### Example 1: Simple Linear Data
```python
import numpy as np

# Create linear data: y = 2x + 3
x = np.linspace(1, 10, 50)
y = 2 * x + 3

method, error, formula = find_best_fit(x, y, plot=True)
# Expected: method="Linear", error≈0
```

### Example 2: Growth Data (Exponential)
```python
# Population growth: y = 100 * e^(0.1x)
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([100, 110.5, 122.1, 135.0, 149.2, 164.9])

method, error, formula = find_best_fit(x, y)
# Expected: method="Exponential"
```

### Example 3: CSV Data Analysis
```python
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")
x = df["independent_variable"].values
y = df["dependent_variable"].values

# Remove any missing values
mask = ~(np.isnan(x) | np.isnan(y))
x_clean = x[mask]
y_clean = y[mask]

# Find best fit
method, error, formula = find_best_fit(x_clean, y_clean, plot=True)
```

### Example 4: Complex Multi-Component Data
```python
# Data with multiple components: y = x + sin(2x)
x = np.linspace(0, 10, 100)
y = 2*x + 3*np.sin(2*x)

# Standard regression might not capture both components
method1, error1, formula1 = find_best_fit(x, y)

# Fourier regression can separate components
funclist, fourier_formula = find_fourier(x, y, Iterations=3, plot=True)
```

## Advanced Features

### Custom Error Handling

PlotFinder automatically handles common edge cases:
- **Logarithmic regression**: Shifts negative x-values to positive
- **Exponential regression**: Handles negative y-values with shifting
- **Division by zero**: Removes problematic data points

### Plotting and Visualization

```python
# Basic plot
find_best_fit(x, y, plot=True)

# Advanced plotting with custom ranges
from Regression_Standards import plot_function_data
plot_function_data(formula, x, y, x_range=(-5, 15), title="My Analysis")
```

### Performance Tips

1. **Large datasets**: Consider sampling for initial analysis
2. **High polynomial degrees**: May overfit; use `maxPolynomial=3-5` for noisy data
3. **Method selection**: Restrict methods if you know the expected relationship type

## Common Use Cases

### Scientific Data Analysis
- **Physics**: Position vs. time, force vs. displacement
- **Biology**: Population growth, enzyme kinetics
- **Chemistry**: Reaction rates, concentration changes

### Business Analytics
- **Sales forecasting**: Revenue vs. time
- **Market analysis**: Price vs. demand
- **Growth modeling**: User adoption curves

### Engineering Applications
- **Signal processing**: Frequency analysis
- **Control systems**: Response characteristics
- **Materials testing**: Stress-strain relationships

## Troubleshooting

### Common Issues and Solutions

**Issue**: "divide by zero encountered in log"
```python
# Solution: Remove zero or negative values for log regression
x = x[x > 0]
y = y[x > 0]
```

**Issue**: Very high error for all methods
```python
# Try Fourier regression for multi-component data
funclist, formula = find_fourier(x, y, Iterations=5)
```

**Issue**: Overfitting with high-degree polynomials
```python
# Reduce maximum polynomial degree
method, error, formula = find_best_fit(x, y, maxPolynomial=3)
```
## Next Steps

1. **Explore the source code** in `Regression_Finder.py` to understand implementation details
2. **Contribute to the project** by testing edge cases or adding new regression types

## Support and Documentation

- **GitHub Repository**: Check the README.md for latest updates
- **Test Files**: Review `Plot_Finder_Test.py` for comprehensive usage examples
- **Schema Documentation**: See `schema.md` for output format specifications

---