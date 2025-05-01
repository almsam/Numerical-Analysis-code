import pandas as pd
import numpy as np
import seaborn as sns  
import matplotlib.pyplot as plt
from sympy import symbols, sympify, solve, diff, lambdify, exp, log, sin

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
    return error, intercept + slope * age


def quadratic_regression(x, y):
        # define quadratic term & fit the regression & to get the ax**2 & bx & c
    X = np.column_stack((np.ones(len(x)), x, x**2))
    intercept, linear, quadratic = np.linalg.inv(X.T @ X) @ X.T @ y
        # then somehow find the error
    def predict(x): return (intercept + (linear*x) + (quadratic*(x**2)))
    error = np.mean(np.abs(y - predict(x)))
        # and return the result (this time in the new format)
    return error, intercept + linear * age + quadratic * age**2
