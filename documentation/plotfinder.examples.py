# NOTE: PlotFinder Tutorial Examples: Guide to using the PlotFinder regression library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Plot_Finder import find_best_fit, find_fourier
from Regression_Standards import print_non_zero_terms, plot_function_data

# =============================================================================
# EXAMPLE 1: Basic Linear Data
# =============================================================================
print("=== EXAMPLE 1: Linear Data ===")

# Create simple linear data: y = 2x + 3
x1 = np.linspace(0, 10, 50)
x1 = x1[x1 != 0]  # Remove zero to avoid log issues
y1 = 2 * x1 + 3

# OPTIONAL: add some noise
np.random.seed(0)  # Set seed for reproducibility
y1 += np.random.normal(0, 1, len(y1))

# Use PlotFinder to find the best fit; this automatically tests multiple regression methods and return the one with the lowest error;
method1, error1, formula1 = find_best_fit(x1, y1, plot=True)

print(f"Best method: {method1}")
print(f"Error: {error1:.6f}")
print("Non-zero terms:")
print_non_zero_terms(formula1)
print("\n" + "="*50 + "\n")

# =============================================================================
# EXAMPLE 2: Quadratic Data
# =============================================================================
print("=== EXAMPLE 2: Quadratic Data ===")

# Create quadratic data: y = x² - 2x + 1
x2 = np.linspace(-3, 5, 50)
y2 = x2**2 - 2*x2 + 1

method2, error2, formula2 = find_best_fit(x2, y2, plot=True)

print(f"Best method: {method2}")
print(f"Error: {error2:.6f}")
print("Non-zero terms:")
print_non_zero_terms(formula2)
print("\n" + "="*50 + "\n")

# =============================================================================
# EXAMPLE 3: Exponential Data
# =============================================================================
print("=== EXAMPLE 3: Exponential Data ===")

# Create exponential data: y = 3 * e^(0.5x)
x3 = np.linspace(0, 5, 50)
y3 = 3 * np.exp(0.5 * x3)

method3, error3, formula3 = find_best_fit(x3, y3, plot=True)

print(f"Best method: {method3}")
print(f"Error: {error3:.6f}")
print("Non-zero terms:")
print_non_zero_terms(formula3)
print("\n" + "="*50 + "\n")

# =============================================================================
# EXAMPLE 4: Logarithmic Data
# =============================================================================
print("=== EXAMPLE 4: Logarithmic Data ===")

# Create logarithmic data: y = 2 * ln(x) + 1
x4 = np.linspace(1, 10, 50)  # Start from 1 to avoid ln(0)
y4 = 2 * np.log(x4) + 1

method4, error4, formula4 = find_best_fit(x4, y4, plot=True)

print(f"Best method: {method4}")
print(f"Error: {error4:.6f}")
print("Non-zero terms:")
print_non_zero_terms(formula4)
print("\n" + "="*50 + "\n")

# =============================================================================
# EXAMPLE 5: Sinusoidal Data
# =============================================================================
print("=== EXAMPLE 5: Sinusoidal Data ===")

# Create sine data: y = 4 * sin(x)
x5 = np.linspace(0, 2*np.pi, 50) # Generates 50 points over one full sine wave cycle  (from 0 to 2𝜋)
y5 = 4 * np.sin(x5) # Computes the corresponding y-values for a sine wave with amplitude 4.

method5, error5, formula5 = find_best_fit(x5, y5, plot=True)

print(f"Best method: {method5}")
print(f"Error: {error5:.6f}")
print("Non-zero terms:")
print_non_zero_terms(formula5)
print("\n" + "="*50 + "\n")

# =============================================================================
# EXAMPLE 6: Custom Method Selection
# =============================================================================
print("=== EXAMPLE 6: Custom Method Selection ===")

# Use the same linear data but only test specific methods
method6, error6, formula6 = find_best_fit(
    x1, y1, 
    plot=False,
    methods="linear, exponential",  # Only test these specified methods
    maxPolynomial=2
)

print(f"Best method (limited selection): {method6}")
print(f"Error: {error6:.6f}")
print("\n" + "="*50 + "\n")

# =============================================================================
# EXAMPLE 7: Working with Real CSV Data
# =============================================================================
print("=== EXAMPLE 7: Real CSV Data ===")

# Load the possum dataset (if available)
try:
    df = pd.read_csv("data/possum.csv")
    df_filtered = df[["hdlngth", "age"]].dropna()
    x_real = df_filtered["age"].values
    y_real = df_filtered["hdlngth"].values
    
    method_real, error_real, formula_real = find_best_fit(x_real, y_real, plot=True)
    
    print(f"Best method for possum data: {method_real}")
    print(f"Error: {error_real:.6f}")
    print("Non-zero terms:")
    print_non_zero_terms(formula_real)
    
except FileNotFoundError:
    print("Possum dataset not found. Creating synthetic data instead...")
    
    # Create synthetic realistic data with noise
    x_real = np.random.uniform(1, 9, 50)
    y_real = 90 + 0.5 * x_real + np.random.normal(0, 2, 50)
    
    method_real, error_real, formula_real = find_best_fit(x_real, y_real, plot=True)
    
    print(f"Best method for synthetic data: {method_real}")
    print(f"Error: {error_real:.6f}")
    print("Non-zero terms:")
    print_non_zero_terms(formula_real)

print("\n" + "="*50 + "\n")

# =============================================================================
# EXAMPLE 8: Advanced Fourier Series Regression
# =============================================================================
print("=== EXAMPLE 8: Fourier Series (Multi-function) Regression ===")

# Create complex data: y = x + sin(2x)
x8 = np.linspace(0, 10, 100) # 100 evenly spaced values from 0 to 10
x8 = x8[x8 != 0]
y8 = 2*x8 + 3*np.sin(2*x8) # Computes a function that combines a linear component with a sine wave oscillation

# Use Fourier series regression to capture multiple components
try:
    funclist, fourier_formula = find_fourier(
        x8, y8, 
        Iterations=3,  # Try multiple iterations
        plot=True,
        maxPolynomial=3,
        methods="all"
    )
    
    print(f"Fourier regression found {len(funclist)} components")
    print("Final combined formula:")
    print_non_zero_terms(fourier_formula)
    
except Exception as e:
    print(f"Fourier regression encountered an issue: {e}")
    print("Falling back to standard regression...")
    method8, error8, formula8 = find_best_fit(x8, y8, plot=True)
    print(f"Standard best method: {method8}")

print("\n" + "="*50 + "\n")

# =============================================================================
# EXAMPLE 9: Noisy Data Analysis
# =============================================================================
print("=== EXAMPLE 9: Handling Noisy Data ===")

# Create data with significant noise
x9 = np.linspace(1, 10, 50)
y9_clean = 2 * x9**2 + 3  # Clean quadratic
y9_noisy = y9_clean + np.random.normal(0, 5, 50)  # Add noise

print("Analyzing clean data:")
method9a, error9a, formula9a = find_best_fit(x9, y9_clean, plot=False)
print(f"Clean data - Method: {method9a}, Error: {error9a:.6f}")

print("\nAnalyzing noisy data:")
method9b, error9b, formula9b = find_best_fit(x9, y9_noisy, plot=True)
print(f"Noisy data - Method: {method9b}, Error: {error9b:.6f}")

print("Non-zero terms for noisy data:")
print_non_zero_terms(formula9b)
print("\n" + "="*50 + "\n")

# =============================================================================
# EXAMPLE 10: High-Degree Polynomial Data
# =============================================================================
print("=== EXAMPLE 10: High-Degree Polynomial ===")


x10 = np.linspace(-2, 2, 50)
y10 = x10**5 - 2*x10**3 + x10 + 1 # Create 5th degree polynomial data: y = x^5 - 2x^3 + x + 1

method10, error10, formula10 = find_best_fit(
    x10, y10, 
    plot=True,
    maxPolynomial=7  # Allow up to 7th degree polynomials
)

print(f"Best method: {method10}")
print(f"Error: {error10:.6f}")
print("Non-zero terms:")
print_non_zero_terms(formula10)

# =============================================================================
print("\n" + "="*80)
print("TUTORIAL COMPLETE!")
print("="*80)
print("\nKey Takeaways:")
print("1. PlotFinder automatically tests multiple regression types")
print("2. It returns the method with the lowest error")
print("3. You can restrict which methods to test")
print("4. Fourier regression can capture multi-component functions")
print("5. The library handles edge cases like log(0) and negative values")
print("6. Results are returned in a standardized format for further analysis")