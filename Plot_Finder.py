import pandas as pd
import numpy as np
import seaborn as sns  
# import statsmodels.api as sm
# import statsmodels.formula.api as smf  # type: ignore
import matplotlib.pyplot as plt
from sympy import symbols, sympify, solve, diff, lambdify, exp, log, sin

from Regression_Finder import *
from Regression_Standards import *



df = pd.read_csv("c:/Users/samia/OneDrive/Desktop/Numerical-Analysis-code/data/possum.csv"); df_filtered = df[["hdlngth", "age"]].dropna()
y = df_filtered["hdlngth"].values; x = df_filtered["age"].values; age = symbols('x')







def plot_best_fit(x, y, best_fit_method, best_fit_formula): # plot our function:
    
    print(f"\nplotting: {best_fit_formula}")

    plt.scatter(x, y, label="Data points", color="blue")        # data points as scatter plot
    x_range = np.linspace(min(x), max(x), 100)                  # domain for the plot
    sym_func = generate_sympy_function(best_fit_formula)
    y_range = lambdify(age, sym_func, 'numpy')(x_range) # best-fit formula over x range
    y_range = np.nan_to_num(y_range, nan=0.0, posinf=1e6, neginf=-1e6)

    print("Generated expression:", sym_func) # print("x_range shape:", x_range.shape); print("y_range shape:", y_range.shape)

    # plot our regression curve:
    plt.plot(x_range, y_range, color='red', label=f'Best fit: {best_fit_method} Regression')  # Best-fit curve
    plt.xlabel('X'); plt.ylabel('Y'); plt.title(f'{best_fit_method} Regression'); plt.legend(); plt.show()

# main function
def find_best_fit(x, y, plot=False, maxPolynomial=7):

    methods = [
        ("Linear", linear_regression),
        ("Quadratic", quadratic_regression),
        ("Cubic", cubic_regression),
        # polynomial is included
        
        ("Exponential", exp_regression),
        ("Logarithmic", logarithmic_regression),
        ("Sine", sin_regression),
    ]
    
    error_list = []; best_method = None; best_fit_formula = None # saves candidate for best
    
    # non polynomial regression
    for name, method in methods:
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



def fourier(x, y, n, plot=False):
    # first we need to regress on the data & get the approximation function
    n = n - 1
    
    print(f"\n______ Iteration {1} ______")
    residuals = y.copy()
    full_formula = None
    
    best_method, min_error, best_fit_formula = find_best_fit(x, y, False)
    funclist = []; funclist.append(best_fit_formula)
    full_formula = best_fit_formula
    
    print(f"current best method : {best_method}")
    
    for i in range(n):
        
        print(f"\n______ Iteration {i+2} ______")
        pred_y = lambdify(age, best_fit_formula, 'numpy')(x);   residuals = residuals - pred_y
        best_method, min_error, best_fit_formula = find_best_fit(x, residuals, False) # regress on error
        
        print(f"current best method : {best_method}")
        funclist.append(best_fit_formula); full_formula += best_fit_formula # type: ignore
    
    # repeat the above until n = 0
    
    if(plot): plot_best_fit(x, y, "Fourier Series", full_formula)
    
    return full_formula# the output should be a list of fuctions








# a, b, c = find_best_fit(x, y, True); print(a); print(b); print_non_zero_terms(c) #uncomment for debug

# a = fourier(x, y, 8, True); print(a)#; print(b) #uncomment for debug