import pandas as pd
import numpy as np
import seaborn as sns  
import statsmodels.api as sm
import statsmodels.formula.api as smf  # type: ignore
from sympy import symbols, sympify, solve, diff, lambdify


df = pd.read_csv("c:/Users/samia/OneDrive/Desktop/Numerical-Analysis-code/data/possum.csv")
df_filtered = df[["hdlngth", "age"]].dropna()
y = df_filtered["hdlngth"].values
x = df_filtered["age"].values


# Original regression methods
def linear_regression(x, y):
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    pred = model.predict(x)
    residuals = y - pred
    LinearError = np.mean(np.abs(residuals))
    print("Linear regression error: ", LinearError)
    return LinearError

def quadratic_regression(x, y):
    x_quad = np.power(x, 2)
    X = pd.DataFrame({"age": x, "age_squared": x_quad})
    X = sm.add_constant(X)
    model_quad = sm.OLS(y, X).fit()
    pred_quad = model_quad.predict(X)
    residuals_quad = y - pred_quad
    QuadraticError = np.mean(np.abs(residuals_quad))
    print("Quadratic regression error: ", QuadraticError)
    return QuadraticError

def cubic_regression(x, y):
    x_cube = np.power(x, 3)
    X = pd.DataFrame({"age": x, "age_cubed": x_cube})
    X = sm.add_constant(X)
    model_cube = sm.OLS(y, X).fit()
    pred_cube = model_cube.predict(X)
    residuals_cube = y - pred_cube
    CubeError = np.mean(np.abs(residuals_cube))
    print("Cubic regression error: ", CubeError)
    return CubeError

def poly_regression(x, y):
    errors = []
    for power in range(4, 8):
        X_poly = pd.DataFrame({"age": x, f"age_{power}": np.power(x, power)})
        X_poly = sm.add_constant(X_poly)
        model_poly = sm.OLS(y, X_poly).fit()
        pred_poly = model_poly.predict(X_poly)
        residuals_poly = y - pred_poly
        error = np.mean(np.abs(residuals_poly))
        errors.append(error)
        print(f"Polynomial regression (x^{power}) error: ", error)
    return errors

def exp_regression(x, y):
    log_y = np.log(y)
    X = sm.add_constant(x)
    model_exp = sm.OLS(log_y, X).fit()
    log_pred = model_exp.predict(X)
    pred_exp = np.exp(log_pred)
    residuals_exp = y - pred_exp
    ExponentialError = np.mean(np.abs(residuals_exp))
    print("Exponential regression error: ", ExponentialError)
    return ExponentialError

def logarithmic_regression(x, y):
    log_x = np.log(x)
    X = sm.add_constant(log_x)
    model_log = sm.OLS(y, X).fit()
    pred_log = model_log.predict(X)
    residuals_log = y - pred_log
    LogarithmicError = np.mean(np.abs(residuals_log))
    print("Logarithmic regression error: ", LogarithmicError)
    return LogarithmicError

def sin_regression(x, y):
    sin_x = np.sin(x)
    X = sm.add_constant(sin_x)
    model_sin = sm.OLS(y, X).fit()
    pred_sin = model_sin.predict(X)
    residuals_sine = y - pred_sin
    SinError = np.mean(np.abs(residuals_sine))
    print("Sine regression error: ", SinError)
    return SinError

def logistic_regression(x, y):
    y_transformed = (y - np.min(y)) / (np.max(y) - np.min(y))
    X = sm.add_constant(x)
    model_logistic = sm.Logit(y_transformed, X).fit()
    pred_logistic = model_logistic.predict(X)
    residuals_logistic = y_transformed - pred_logistic
    LogisticError = np.mean(np.abs(residuals_logistic))
    print("Logistic regression error: ", LogisticError)
    return LogisticError

def loess_regression(x, y, frac=0.3):
    loess_model = sm.nonparametric.lowess(y, x, frac=frac)
    pred_loess = loess_model[:, 1]
    residuals_loess = y - pred_loess
    LoessError = np.mean(np.abs(residuals_loess))
    print("LOESS regression error: ", LoessError)
    return LoessError


# Modified main function
def main(x, y):
    error_list = []

    error_list.append(("Linear", linear_regression(x, y)))
    error_list.append(("Quadratic", quadratic_regression(x, y)))
    error_list.append(("Cubic", cubic_regression(x, y)))
    
    polynomial_errors = poly_regression(x, y)
    for i, poly_error in enumerate(polynomial_errors, start=4):
        error_list.append((f"Polynomial (x^{i})", poly_error))
    
    error_list.append(("Exponential", exp_regression(x, y)))
    error_list.append(("Logarithmic", logarithmic_regression(x, y)))
    error_list.append(("Sine", sin_regression(x, y)))
    error_list.append(("LOESS", loess_regression(x, y)))



    
    print("\n--- Regression Errors ---")
    for method_name, error in error_list:
        print(f"{method_name} Error: {error}")

    # Designate the most accurate method
    min_error_method = min(error_list, key=lambda x: x[1])
    print(f"\nThe method with the smallest error is: {min_error_method[0]} Regression with an error of {min_error_method[1]}")
    print(f"\nTherefore, the function is likely a {min_error_method[0]} function")


main(x, y)
