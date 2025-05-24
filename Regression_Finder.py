import pandas as pd
import numpy as np
from sympy import symbols

df = pd.read_csv("c:/Users/samia/OneDrive/Desktop/Numerical-Analysis-code/data/possum.csv"); df_filtered = df[["hdlngth", "age"]].dropna()
y = df_filtered["hdlngth"].values; x = df_filtered["age"].values; age = symbols('x')

def linear_regression(x, y):
        # so first we need to fit the regression & get the m(x) & b
    x_mean = np.mean(x); y_mean = np.mean(y)
    Sxx = np.sum((x - x_mean)**2); Sxy = np.sum((x - x_mean)*(y - y_mean))
    slope = Sxy/Sxx; intercept = y_mean - (slope*x_mean)
        # then somehow find the error
    def predict(x): return (intercept + (slope*x))
    error = np.mean(np.abs(y - predict(x)))
        # and return the result (this time in the new format)
    regression = {
        "sin_terms": [(0, 0, 0)], "exponential_terms": [(0, 0)], "logarithmic_terms": [(0, 0)],
        "polynomial_terms": {0: intercept, 1: slope}
    }
    return error, regression # intercept + slope * age


def quadratic_regression(x, y):
        # define quadratic term & fit the regression & to get the ax**2 & bx & c
    X = np.column_stack((np.ones(len(x)), x, x**2))
    intercept, linear, quadratic = np.linalg.inv(X.T @ X) @ X.T @ y
        # then somehow find the error
    def predict(x): return (intercept + (linear*x) + (quadratic*(x**2)))
    error = np.mean(np.abs(y - predict(x)))
        # and return the result (this time in the new format)
    regression = {
        "sin_terms": [(0, 0, 0)], "exponential_terms": [(0, 0)], "logarithmic_terms": [(0, 0)],
        "polynomial_terms": {0: intercept, 1: linear, 2: quadratic}
    }
    return error, regression # intercept + linear * age + quadratic * age**2


def cubic_regression(x, y):
        # define cubic term & fit the regression to get ax^3 + bx^2 + cx + d
    X = np.column_stack((np.ones(len(x)), x, x**2, x**3))
    intercept, linear, quadratic, cubic = np.linalg.inv(X.T @ X) @ X.T @ y
    def predict(x): return intercept + linear * x + quadratic * x**2 + cubic * x**3
        # then somehow find the error
    error = np.mean(np.abs(y - predict(x)))
        # and return the result (this time in the new format)
    regression = {
        "sin_terms": [(0, 0, 0)], "exponential_terms": [(0, 0)], "logarithmic_terms": [(0, 0)],
        "polynomial_terms": {0: intercept, 1: linear, 2: quadratic, 3: cubic}
    }
    return error, regression # intercept + linear * age + quadratic * age**2 + cubic * age**3


def poly_regression(x, y, degree):
        # generate polynomial features up to the given degree
    X = np.column_stack([x**i for i in range(degree + 1)])
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    def predict(x_input): return sum(beta[i] * x_input**i for i in range(degree + 1))
        # then somehow find the error
    error = np.mean(np.abs(y - predict(x)))
        # and return the result (this time in the new format)
    poly_terms = {i: beta[i] for i in range(degree + 1)}
    regression = {
        "sin_terms": [(0, 0, 0)], "exponential_terms": [(0, 0)], "logarithmic_terms": [(0, 0)],
        "polynomial_terms": poly_terms
    }
    return error, regression # predict(age)


def exp_regression(x, y):
        # generate polynomial features up to the given degree
    
    shift = 0
    if np.any(y <= 0):
        shift = abs(np.min(y)) + 1e-6
        y = y + shift
    
    log_y = np.log(y)
    X = np.column_stack((np.ones(len(x)), x))
    intercept, slope = np.linalg.inv(X.T @ X) @ X.T @ log_y
    
    #     # then somehow find the error
    # def predict(x): return np.exp(intercept + slope * x) - shift
    # error = np.mean(np.abs(y - (np.exp(intercept + slope * x))))
    #     # and return the result (this time in the new format)
    # coefficient = np.exp(intercept)#; base = np.exp(slope) #removed base to eliminate massive bug
    # regression = {
    #     "sin_terms": [(0, 0, 0)],
    #     "exponential_terms": [(coefficient, slope)], # base/(2*np.sqrt(np.e)))], # (coefficient/(2*np.sqrt(np.e)), base/(2*np.sqrt(np.e)))], # coefficient, base)], #removed base to eliminate massive bug
    #     "logarithmic_terms": [(0, 0)],
    #     "polynomial_terms": {0: -shift, 1: 0, 2: 0} # {0: intercept, 1: 0, 2: 0}
    # }
    # return error, regression # intercept + (slope * exp(age))
    
    
    def build_result(sign=1.0):
        predicted = sign * np.exp(intercept + slope * x) - shift
        error = np.mean(np.abs(y - predicted))
        coefficient = sign * np.exp(intercept)
        regression = {
            "sin_terms": [(0, 0, 0)],
            "exponential_terms": [(coefficient, slope)],
            "logarithmic_terms": [(0, 0)],
            "polynomial_terms": {0: -shift, 1: 0, 2: 0}
        }
        return error, regression

    # Try both +exp and -exp fits, return the one with lower error
    pos_error, pos_reg = build_result(sign=1.0)
    neg_error, neg_reg = build_result(sign=-1.0)

    return (pos_error, pos_reg) if pos_error < neg_error else (neg_error, neg_reg)



def logarithmic_regression(x, y):
        # generate polynomial features up to the given degree
    # Determine shift if any x values are non-positive
    shift = 0
    if np.any(x <= 0):
        shift = abs(np.min(x)) + 1e-6; x_shifted = x + shift  # ensure all values are > 0
    else: x_shifted = x.copy()  # no shift case
    log_x = np.log(x_shifted)
    X = np.column_stack((np.ones(len(log_x)), log_x))
    intercept, log_coef = np.linalg.inv(X.T @ X) @ X.T @ y
        # then somehow find the error
    def predict(x): return intercept + log_coef * np.log(x + shift)
    error = np.mean(np.abs(y - predict(x)))
        # and return the result (this time in the new format)
    regression = {
        "sin_terms": [(0, 0, 0)],
        "exponential_terms": [(0, 0)],
        "logarithmic_terms": [(log_coef, shift)], #np.e)],
        "polynomial_terms": {0: intercept, 1: 0, 2: 0}
    }
    return error, regression # intercept + (log_coef * log(age))


def sin_regression(x, y):
        # generate polynomial features up to the given degree
    sin_x = np.sin(x)
    X = np.column_stack((np.ones(len(sin_x)), sin_x))
    intercept, sin_coef = np.linalg.inv(X.T @ X) @ X.T @ y
        # then somehow find the error
    def predict(x): return intercept + sin_coef * np.sin(x)
    error = np.mean(np.abs(y - predict(x)))
        # and return the result (this time in the new format)
    regression = {
        "sin_terms": [(sin_coef, 1, 0)],
        "exponential_terms": [(0, 0)],
        "logarithmic_terms": [(0, 0)],
        "polynomial_terms": {0: intercept, 1: 0, 2: 0}
    }
    return error, regression # intercept + (sin_coef * sin(age))