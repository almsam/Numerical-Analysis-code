import pandas as pd
import numpy as np
import seaborn as sns  
import statsmodels.api as sm
import statsmodels.formula.api as smf  # type: ignore
from sympy import symbols, sympify, solve, diff, lambdify


# df = pd.read_csv("data/possum.csv")
# df_filtered = df[["hdlngth", "age"]].dropna()
# y = df_filtered["hdlngth"].values
# x = df_filtered["age"].values


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

def calculate_derivative(x_val, func):
    x = symbols('x')
    deriv = diff(func, x)
    return float(deriv.subs(x, x_val)) # type: ignore



def main():
    user_func = input_function()
    min, max = find_min_max(user_func, -100, 100)

    # min max
    print(f"\nMin value of user-defined function: {min}")
    print(f"Max value of user-defined function: {max}")

    # derivative at x = 0
    deriv_at_mean = calculate_derivative(0, user_func)
    print(f"Derivative of user-defined function at mx = ({0}): {deriv_at_mean}")


main()







### code dump:
### this space was originally for regressing on a custom function, but is no longer in use

# # user-defined function regression
# def user_defined_regression(x, y, func):
#     x_sym = symbols('x')
#     func_lambda = lambdify(x_sym, func, "numpy")
#     pred = func_lambda(x)
#     residuals = y - pred
#     UserDefinedError = np.mean(np.abs(residuals))
#     print("User-defined function regression error: ", UserDefinedError)
#     return UserDefinedError

# def main(x, y):

#     # User-defined function regression
#     user_func = input_function()
#     user_func_error = user_defined_regression(x, y, user_func)
#     error_list.append(("User-defined", user_func_error))

#     # Find min and max of user-defined function
#     min_val, max_val = find_min_max(user_func, min(x), max(x))
#     print(f"\nMin value of user-defined function: {min_val}")
#     print(f"Max value of user-defined function: {max_val}")

#     # Calculate derivative at mean x
#     mean_x = np.mean(x)
#     deriv_at_mean = calculate_derivative(user_func, mean_x)
#     print(f"Derivative of user-defined function at mean x ({mean_x}): {deriv_at_mean}")