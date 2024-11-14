import pandas as pd
import numpy as np
import seaborn as sns  
import statsmodels.api as sm
import statsmodels.formula.api as smf  # type: ignore
import matplotlib.pyplot as plt
from sympy import symbols, sympify, solve, diff, lambdify, exp, log, sin


df = pd.read_csv("c:/Users/samia/OneDrive/Desktop/Numerical-Analysis-code/data/possum.csv")
df_filtered = df[["hdlngth", "age"]].dropna()
y = df_filtered["hdlngth"].values
x = df_filtered["age"].values

age = symbols('x')

# Original regression methods

def linear_regression(x, y):
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    intercept, slope = model.params
    error = np.mean(np.abs(y - model.predict(x)))
    return error, intercept + slope * age

def quadratic_regression(x, y):
    X = sm.add_constant(np.column_stack((x, np.power(x, 2))))
    model = sm.OLS(y, X).fit()
    intercept, linear, quadratic = model.params
    error = np.mean(np.abs(y - model.predict(X)))
    return error, intercept + linear * age + quadratic * age**2

def cubic_regression(x, y):
    X = sm.add_constant(np.column_stack((x, np.power(x, 2), np.power(x, 3))))
    model = sm.OLS(y, X).fit()
    intercept, linear, quadratic, cubic = model.params
    error = np.mean(np.abs(y - model.predict(X)))
    return error, intercept + linear * age + quadratic * age**2 + cubic * age**3

def poly_regression(x, y, degree):
    poly_terms = [np.power(x, i) for i in range(1, degree + 1)]
    X = sm.add_constant(np.column_stack(poly_terms))
    model = sm.OLS(y, X).fit()
    params = model.params
    error = np.mean(np.abs(y - model.predict(X)))
    polynomial_formula = sum(params[i] * age**i for i in range(degree + 1))
    return error, polynomial_formula

def exp_regression(x, y):
    log_y = np.log(y)
    X = sm.add_constant(x)
    model = sm.OLS(log_y, X).fit()
    intercept, slope = model.params
    error = np.mean(np.abs(y - np.exp(model.predict(X))))
    # return error, np.exp(intercept) * np.exp(slope * age)
    formula = exp(intercept) * exp(slope * age) # type: ignore
    return error, formula

def logarithmic_regression(x, y):
    X = sm.add_constant(np.log(x))
    model = sm.OLS(y, X).fit()
    intercept, log_coeff = model.params
    error = np.mean(np.abs(y - model.predict(X)))
    formula = intercept + log_coeff * log(age)
    return error, formula

def sin_regression(x, y):
    sin_x = np.sin(x)
    X = sm.add_constant(sin_x)
    model = sm.OLS(y, X).fit()
    intercept, sin_coeff = model.params
    error = np.mean(np.abs(y - model.predict(X)))
    formula = intercept + sin_coeff * sin(age)
    return error, formula

def logistic_regression(x, y):
    y_transformed = (y - np.min(y)) / (np.max(y) - np.min(y))
    X = sm.add_constant(x)
    model = sm.Logit(y_transformed, X).fit()
    intercept, slope = model.params
    error = np.mean(np.abs(y_transformed - model.predict(X)))
    # formula = 1 / (1 + np.exp(-(intercept + slope * age)))
    age = symbols('age'); formula = 1 / (1 + exp(-(intercept + slope * age))) # type: ignore
    return error, formula

def loess_regression(x, y, frac=0.3):
    loess_model = sm.nonparametric.lowess(y, x, frac=frac)
    pred_loess = loess_model[:, 1]
    residuals_loess = y - pred_loess
    error = np.mean(np.abs(residuals_loess))
    return error, "non-parametric"


def plot_best_fit(x, y, best_fit_method, best_fit_formula):
    
    print(f"\nplotting: {best_fit_formula}")

    # data points as scatter plot
    plt.scatter(x, y, label="Data points", color="blue")
    
    # domain for the regression function
    x_range = np.linspace(min(x), max(x), 100)
    
    # best-fit formula over x
    y_range = lambdify(age, best_fit_formula, 'numpy')(x_range)
    
    # plot our regression curve
    plt.plot(x_range, y_range, color='red', label=f'Best fit: {best_fit_method} Regression')  # Best-fit curve
    plt.xlabel('X'); plt.ylabel('Y')
    plt.title(f'{best_fit_method} Regression'); plt.legend(); plt.show()

# Modified main function
def find_best_fit(x, y):
    # error_list = []

    # error_list.append(("Linear", linear_regression(x, y)))
    # error_list.append(("Quadratic", quadratic_regression(x, y)))
    # error_list.append(("Cubic", cubic_regression(x, y)))
    
    # polynomial_errors = poly_regression(x, y)
    # for i, poly_error in enumerate(polynomial_errors, start=4):
    #     error_list.append((f"Polynomial (x^{i})", poly_error))
    
    # error_list.append(("Exponential", exp_regression(x, y)))
    # error_list.append(("Logarithmic", logarithmic_regression(x, y)))
    # error_list.append(("Sine", sin_regression(x, y)))
    # error_list.append(("LOESS", loess_regression(x, y)))



    methods = [
        ("Linear", linear_regression),
        ("Quadratic", quadratic_regression),
        ("Cubic", cubic_regression),
        ("Exponential", exp_regression),
        ("Logarithmic", logarithmic_regression),
        ("Sine", sin_regression),
        # ("Logistic", logistic_regression),
        ("LOESS", loess_regression)
    ]
    
    error_list = []; best_method = None; best_fit_formula = None
    
    for name, method in methods:
        error, formula = method(x, y)
        error_list.append((name, error, formula))
        if best_method is None or error < min_error:
            best_method = name; best_fit_formula = formula; min_error = error
    
    # Polynomial regression for degrees 4 to 7
    for degree in range(4, 8):
        error, formula = poly_regression(x, y, degree)
        error_list.append((f"Polynomial (x^{degree})", error, formula))
        if best_method is None or error < min_error:
            best_method = f"Polynomial (x^{degree})"; best_fit_formula = formula; min_error = error
    
    # Find the method with the least error
    min_error_method = min(error_list, key=lambda x: x[1])
    method_name, min_error, min_formula = min_error_method
    
    # print("\n--- Regression Errors ---")
    # for method_name, error in error_list:
    #     print(f"{method_name} Error: {error}")

    # Designate the most accurate method
    # min_error_method = min(error_list, key=lambda x: x[1])
    # print(f"\nThe method with the smallest error is: {min_error_method[0]} Regression with an error of {min_error_method[1]}")
    # print(f"\nTherefore, the function is likely a {min_error_method[0]} function")
    
    print("\n--- Regression Errors ---")
    for name, error, _ in error_list:
        print(f"{name} Error: {error}")

    print(f"\nThe method with the smallest error is: {method_name} Regression with an error of {min_error}")
    print(f"\nApproximate function: {min_formula}")
    
    plot_best_fit(x, y, best_method, best_fit_formula)
    
    return method_name, min_error, min_formula



# find_best_fit(x, y) #uncomment for debug
