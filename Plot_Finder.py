import pandas as pd
import numpy as np
import seaborn as sns  
# import statsmodels.api as sm
# import statsmodels.formula.api as smf  # type: ignore
import matplotlib.pyplot as plt
from sympy import symbols, sympify, solve, diff, lambdify, exp, log, sin

from Regression_Finder import *
from Regression_Standards import *

df = pd.read_csv("data/possum.csv"); df_filtered = df[["hdlngth", "age"]].dropna()
y = df_filtered["hdlngth"].values; x = df_filtered["age"].values; age = symbols('x')

def plot_best_fit(x, y, best_fit_method_name="Best Fit Method", best_fit_formula=None): # plot our function:
    
    print(f"\nplotting: {best_fit_formula}")

    plt.scatter(x, y, label="Data points", color="blue")        # data points as scatter plot
    x_range = np.linspace(min(x), max(x), 100)                  # domain for the plot
    sym_func = generate_sympy_function(best_fit_formula)
    y_range = lambdify(age, sym_func, 'numpy')(x_range) # best-fit formula over x range
    y_range = np.nan_to_num(y_range, nan=0.0, posinf=1e6, neginf=-1e6)

    print("Generated expression:", sym_func) # print("x_range shape:", x_range.shape); print("y_range shape:", y_range.shape)

    # plot our regression curve:
    plt.plot(x_range, y_range, color='red', label=f'Best fit: {best_fit_method_name} Regression')  # Best-fit curve
    plt.xlabel('X'); plt.ylabel('Y'); plt.title(f'{best_fit_method_name} Regression'); plt.legend(); plt.show()

# main function
def find_best_fit(x, y, plot=False, maxPolynomial=7, methods="all"):

    all_methods = {
        "linear": ("Linear", linear_regression),
        "quadratic": ("Quadratic", quadratic_regression),
        "cubic": ("Cubic", cubic_regression),
        # polynomial is included
        
        "exponential": ("Exponential", exp_regression),
        "logarithmic": ("Logarithmic", logarithmic_regression),
        "sine": ("Sine", sin_regression),
    }

    # methods = [ ("Linear", linear_regression), ("Quadratic", quadratic_regression), ("Cubic", cubic_regression), ("Exponential", exp_regression), ("Logarithmic", logarithmic_regression), ("Sine", sin_regression), ]
    
    if isinstance(methods, str): methods = [m.strip().lower() for m in methods.split(",")] #cleanup input
    
    selected_methods = []
    if "all" in methods:
        if maxPolynomial >= 1: selected_methods.append(("Linear", linear_regression))
        if maxPolynomial >= 2: selected_methods.append(("Quadratic", quadratic_regression))
        if maxPolynomial >= 3: selected_methods.append(("Cubic", cubic_regression))
        for key in ["exponential", "logarithmic", "sine"]: selected_methods.append(all_methods[key])
    else:
        for key in methods:
            if key in all_methods:
                if key == "linear": selected_methods.append(all_methods[key])
                elif key == "quadratic": selected_methods.append(all_methods[key])
                elif key == "cubic": selected_methods.append(all_methods[key])
                elif key not in ["linear", "quadratic", "cubic"]: selected_methods.append(all_methods[key]) #adds exp, log, or sin
    
    error_list = []; best_method = None; best_fit_formula = None # saves candidate for best
    
    # non polynomial regression
    for name, method in selected_methods:
        error, formula = method(x, y)
        error_list.append((name, error, formula))
        if best_method == None or error < min_error: # if a new best is found
            best_method = name; best_fit_formula = formula; min_error = error
    
    # polynomial regression degrees 4 to 7
    if(maxPolynomial>3): # if 4 or more then it prints the necessary terms
        for degree in range(4, (maxPolynomial+1)):
            error, formula = poly_regression(x, y, degree)
            error_list.append((f"Polynomial (x^{degree})", error, formula))
            if best_method is None or error < min_error: # if a new best is found
                best_method = f"Polynomial (x^{degree})"; best_fit_formula = formula; min_error = error
    
    # out stuff
    print("\n--- Regression Errors ---")
    for name, error, _ in error_list:
        print(f"{name} Error: {error}")
    
    print(f"\nThe method with the smallest error is: {best_method} Regression with an error of {min_error}")
    print(f"\nApproximate function: {get_non_zero_terms(best_fit_formula)}")
    
    if(plot):
        plot_best_fit(x, y, best_method, best_fit_formula)
    
    return best_method, min_error, best_fit_formula


def find_fourier(x, y, Iterations=2, plot=False, maxPolynomial=3, methods="all", verbose=False):
    """
    Perform iterative regression to decompose data into multiple components (Fourier-like series).
    
    This function repeatedly fits regression models to residuals, building up a 
    multi-component approximation. For example, it might find:
    - Iteration 1: Quadratic trend (y = ax² + bx + c)
    - Iteration 2: Linear correction on residuals
    - Iteration 3: Sinusoidal component on remaining residuals
    
    Parameters:
    -----------
    x : array-like
        Independent variable data
    y : array-like
        Dependent variable data
    Iterations : int, default=2
        Number of regression iterations to perform
    plot : bool, default=False
        Whether to plot the progressive approximation
    maxPolynomial : int, default=3
        Maximum polynomial degree to consider in each iteration
    methods : str, default="all"
        Which regression methods to test in each iteration
    
    Returns:
    --------
    tuple : (funclist, full_formula)
        - funclist: List of individual regression formulas from each iteration
        - full_formula: Combined formula adding all components together
    """
    
    # Adjust iterations (since first iteration is counted separately)
    Iterations = Iterations - 1
    
    if verbose:
        print(f"\n______ Iteration {1} ______")
    
    # Initialize variables
    residuals = y.copy()  # Start with original y values
    full_formula = None
    cumulative_formulas = []  # Track cumulative formulas for plotting
    
    # First iteration: fit the original data
    best_method, min_error, best_fit_formula = find_best_fit(
        x, y, False, maxPolynomial=maxPolynomial, methods=methods
    )
    
    # Initialize tracking lists
    funclist = []
    funclist.append(best_fit_formula)
    full_formula = best_fit_formula
    cumulative_formulas.append(full_formula)  # Store first approximation
    
    if verbose:
        print(f"current best method : {best_method}")
    
    # Subsequent iterations: fit the residuals
    for i in range(Iterations):
        if verbose:
            print(f"\n______ Iteration {i+2} ______")
        
        # Calculate predictions from current best fit formula
        pred_y = lambdify(age, generate_sympy_function(best_fit_formula), 'numpy')(x)
        
        # Update residuals by subtracting predictions
        residuals = residuals - pred_y
        
        # Find best fit for the residuals
        best_method, min_error, best_fit_formula = find_best_fit(
            x, residuals, False, maxPolynomial=maxPolynomial, methods=methods
        )
        
        if verbose:
            print(f"current best method : {best_method}")
        
        # Add new component to our lists
        funclist.append(best_fit_formula)
        
        # Combine all formulas found so far
        full_formula = add_regression_outputs(full_formula, best_fit_formula)
        cumulative_formulas.append(full_formula)
    
    # Plot the progressive approximation if requested
    if plot:
        plot_fourier(x, y, cumulative_formulas)
    
    return funclist, full_formula


def plot_fourier(x, y, cumulative_formulas):
    """
    Plot progressive Fourier series approximation showing F(1), F(1)+F(2), F(1)+F(2)+F(3), etc.
    
    Parameters:
    -----------
    x : array-like
        Independent variable data points
    y : array-like  
        Dependent variable data points
    cumulative_formulas : list
        List of cumulative regression formulas from each iteration
    """
    plt.figure(figsize=(12, 8))
    
    # Plot original data points
    plt.scatter(x, y, label="Original Data", color="black", alpha=0.7, s=30, zorder=5)
    
    # Colors for different iterations
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    # Create smooth x values for plotting continuous curves
    x_smooth = np.linspace(min(x), max(x), 300)
    
    # Plot each cumulative approximation
    for i, formula in enumerate(cumulative_formulas):
        sym_func = generate_sympy_function(formula)
        numpy_func = lambdify(age, sym_func, 'numpy')
        
        y_smooth = numpy_func(x_smooth)
        y_smooth = np.nan_to_num(y_smooth, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Create descriptive label
        if i == 0:
            label = f"F(1): {get_dominant_term(formula)}"
        else:
            components = "+".join([f"F({j+1})" for j in range(i+1)])
            label = f"{components}: {get_dominant_term(formula)}"
        
        # Style: final approximation gets thicker line, others are dashed
        if i == len(cumulative_formulas) - 1:
            plt.plot(x_smooth, y_smooth, color=colors[i % len(colors)], 
                    linewidth=2.5, label=label, alpha=1.0)
        else:
            plt.plot(x_smooth, y_smooth, color=colors[i % len(colors)], 
                    linewidth=1.5, label=label, alpha=0.7, linestyle='--')
    
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('Fourier Series Progressive Approximation', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def get_dominant_term(formula):
    """Extract human-readable description of dominant terms in regression formula."""
    terms = []
    
    # Check polynomial terms
    poly_terms = formula.get("polynomial_terms", {})
    for power, coef in poly_terms.items():
        if abs(coef) > 1e-6:
            if power == 0:
                terms.append("const")
            elif power == 1:
                terms.append("linear")
            elif power == 2:
                terms.append("quad")
            elif power == 3:
                terms.append("cubic")
            else:
                terms.append(f"x^{power}")
    
    # Check sine terms
    sin_terms = formula.get("sin_terms", [])
    for amp, freq, phase in sin_terms:
        if abs(amp) > 1e-6:
            terms.append("sin")
            break
    
    # Check exponential terms
    exp_terms = formula.get("exponential_terms", [])
    for coef, base in exp_terms:
        if abs(coef) > 1e-6:
            terms.append("exp")
            break
    
    # Check logarithmic terms
    log_terms = formula.get("logarithmic_terms", [])
    for coef, shift in log_terms:
        if abs(coef) > 1e-6:
            terms.append("log")
            break
    
    return "+".join(terms[:3]) if terms else "none"

# a, b, c = find_best_fit(x, y, True); print(a); print(b); print_non_zero_terms(c) #uncomment for debug

# a = fourier(x, y, 8, True); print(a)#; print(b) #uncomment for debug
