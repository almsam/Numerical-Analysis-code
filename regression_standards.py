import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sin, exp, log, sympify, Piecewise

def print_all_terms(regression_output):
    """Print all terms in the regression output."""
    print("All terms in regression:")
    for key, value in regression_output.items():
        print(f"{key}:")
        if key == "polynomial_terms":
            for power, coef in value.items():
                print(f"  {coef}x^{power}")
        else:
            for term in value:
                print(f"  {term}")

def print_non_zero_terms(regression_output):
    """Print only non-zero terms in the regression output."""
    print("Non-zero terms in regression:")
    for key, value in regression_output.items():
        if key == "polynomial_terms":
            non_zero = {p: c for p, c in value.items() if abs(c) > 1e-10}
            if non_zero:
                print(f"{key}:")
                for power, coef in non_zero.items():
                    print(f"  {coef}x^{power}")
        else:
            non_zero = [term for term in value if any(abs(x) > 1e-10 for x in term)]
            if non_zero:
                print(f"{key}:")
                for term in non_zero:
                    print(f"  {term}")

def add_regression_outputs(reg1, reg2):
    """Add two regression outputs together."""
    result = {
        "sin_terms": reg1["sin_terms"] + reg2["sin_terms"],
        "exponential_terms": reg1["exponential_terms"] + reg2["exponential_terms"],
        "logarithmic_terms": reg1["logarithmic_terms"] + reg2["logarithmic_terms"],
        "polynomial_terms": {}
    }
    
    # Add polynomial terms
    all_powers = set(reg1["polynomial_terms"].keys()) | set(reg2["polynomial_terms"].keys())
    for power in all_powers:
        coef1 = reg1["polynomial_terms"].get(power, 0)
        coef2 = reg2["polynomial_terms"].get(power, 0)
        result["polynomial_terms"][power] = coef1 + coef2
    
    return result

def generate_sympy_function(regression_output):
    """Generate a SymPy function from the regression output."""
    x = symbols('x')
    expr = 0
    
    # Add sine terms
    for A, f, p in regression_output["sin_terms"]:
        expr += A * sin(f * x + p)
    
    # Add exponential terms
    for c, b in regression_output["exponential_terms"]:
        expr += c * exp(b * x)
    
    # Add logarithmic terms
    for c, b in regression_output["logarithmic_terms"]:
        # expr += c * log(b * x)
        expr += Piecewise((c * log(b * x), b * x > 0), (0, True))
    
    # Add polynomial terms
    for power, coef in regression_output["polynomial_terms"].items():
        expr += coef * x**power
    
    return expr

def plot_function(regression_output, x_range=(-10, 10), num_points=1000, title="Regression Function"):
    """Plot the regression function."""
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.zeros_like(x)
    
    # Add sine terms
    for A, f, p in regression_output["sin_terms"]:
        y += A * np.sin(f * x + p)
    
    # Add exponential terms
    for c, b in regression_output["exponential_terms"]:
        y += c * np.exp(b * x)
    
    # Add logarithmic terms
    for c, b in regression_output["logarithmic_terms"]:
        # Avoid log of negative numbers
        # mask = x > 0
        z = b * x; mask = z > 0
        y[mask] += c * np.log(z[mask])
    
    # Add polynomial terms
    for power, coef in regression_output["polynomial_terms"].items():
        y += coef * x**power
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.grid(True)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_function_data(regression_output, x_data, y_data, x_range=(-10, 10), num_points=1000, title="Regression Function with Data Points"):
    """Plot both the regression function and the original data points."""
    # Plot the continuous function
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.zeros_like(x)
    
    # Add sine terms
    for A, f, p in regression_output["sin_terms"]:
        y += A * np.sin(f * x + p)
    
    # Add exponential terms
    for c, b in regression_output["exponential_terms"]:
        y += c * np.exp(b * x)
    
    # Add logarithmic terms
    for c, b in regression_output["logarithmic_terms"]:
        # mask = x > 0
        z = b * x; mask = z > 0
        y[mask] += c * np.log(z[mask])
    
    # Add polynomial terms
    for power, coef in regression_output["polynomial_terms"].items():
        y += coef * x**power
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='Regression Function')
    plt.scatter(x_data, y_data, color='red', alpha=0.5, label='Data Points')
    plt.grid(True)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show() 