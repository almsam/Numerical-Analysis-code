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


# Function input and manipulation
def input_function():
    func_str = input("Enter a function (e.g., '2*x**2 + 4*x + 1'): ")
    x = symbols('x')
    return sympify(func_str)

def evaluate_function(func, x_val):
    x = symbols('x')
    return float(func.subs(x, x_val))

def find_min_max(func, range_start, range_end):
    x = symbols('x')
    critical_points = solve(diff(func, x), x)
    valid_points = [p for p in critical_points if range_start <= p <= range_end]
    valid_points.extend([range_start, range_end])
    values = [evaluate_function(func, p) for p in valid_points]
    return min(values), max(values)

def calculate_derivative(func, x_val):
    x = symbols('x')
    deriv = diff(func, x)
    return float(deriv.subs(x, x_val))