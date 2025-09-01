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

def shift_away_from_inf(f, x, step=1e-6, direction=1):
    while np.isinf(f(x)) or np.isnan(f(x)):
        x += step * direction
    return x

def bisection_search(func, range_start, range_end, tolerance=1e-6, max_iter=1000):
    x = symbols('x')
    f = lambdify(x, func)
    a = shift_away_from_inf(f, range_start, direction=1)
    b = shift_away_from_inf(f, range_end, direction=-1)
    if f(a) * f(b) >= 0:
        return None # both range values have same sign, so no root. Better way to signify this?
    for _ in range(max_iter):
        m = (a+b)/2 # center of current interval
        m = shift_away_from_inf(f, m)

        if abs(f(m)) < tolerance or (b-a) / 2 < tolerance: # y-val close enough to 0, or interval small enough
            return m
        if f(a) * f(m) < 0:
            b = m
        else:
            a = m
    return (a+b) / 2

def bisection_all(func, range_start, range_end, step, tolerance=1e-6, max_iter=1000):
    roots = []
    x = symbols('x')
    f = lambdify(x, func)

    for x in np.arange(range_start, range_end, step):
        # a = x
        # b = x + step
        a = shift_away_from_inf(f, x, direction=1)
        b = shift_away_from_inf(f, x+step, direction=-1)
        if(f(a) == 0):
            if a not in roots:
                roots.append(a)
        elif(f(b) == 0):
            if b not in roots:
                roots.append(b)
        elif f(a) * f(b) < 0:
            root = bisection_search(func, a, b, tolerance, max_iter)
            if root not in roots:
                roots.append(root)
    
    # print(roots)
    return roots

def trapezoid_basic(func, range_start, range_end, num_traps=1500):
    x = symbols('x')
    f = lambdify(x, func)
    range_start = np.float64(range_start)
    range_end = np.float64(range_end)

    range_start = shift_away_from_inf(f, range_start, step=1e-10, direction=1)
    range_end = shift_away_from_inf(f, range_end, step=1e-10, direction=-1)

    step_size = (range_end - range_start) / num_traps

    # calc: step_size / 2 ( f(range_start) + 2(Σf(x_i)) + f(range_end) )
    # -> ((f(range_start) + f(range_end)) / 2 + Σf(x_i)) * step size
    area_sum = f(range_start) + f(range_end) * 0.5

    for i in range(1, num_traps):
        x_i = range_start + i * step_size
        area_sum += f(x_i)

    return area_sum * step_size
    
def trapezoid(func, range_start, range_end, init_num_traps=1000, tolerance=1e-5, max_iter=1000):
    n = init_num_traps
    x = symbols('x')
    f = lambdify(x, func)
    range_start = np.float64(range_start)
    range_end = np.float64(range_end)

    range_start = shift_away_from_inf(f, range_start, step=1e-10, direction=1)
    range_end = shift_away_from_inf(f, range_end, step=1e-10, direction=-1)

    prev_approx = trapezoid_basic(func, range_start, range_end, n)
    current_approx = 0

    if(max_iter <= 0):
        raise ValueError("Please ensure max_iter is greater than 0.")

    for iter in range(max_iter):
        n *= 2 # double trapezoid number
        current_approx = trapezoid_basic(func, range_start, range_end, n)

        # result stabilized?
        if abs(current_approx - prev_approx) < tolerance:
            return current_approx

        prev_approx = current_approx

    return current_approx


def newton_method(func, guess, multiplicity=1, tolerance=1e-5, max_iter=1000):
    x = symbols('x')
    f = lambdify(x, func)
    guess = np.float64(guess)
    if np.isinf(f(x)):
        if not np.isinf(f(x+1e-6)):
            guess = shift_away_from_inf(f, guess, direction=1)
        else:
            guess = shift_away_from_inf(f, guess, direction=-1)

    func_prime = diff(func, x)
    f_prime = lambdify(x, func_prime)

    prev_x = guess
    current_x = guess
    for _ in range(max_iter):
        fx = f(prev_x)
        fpx = f_prime(prev_x)

        if fpx == 0:
           raise ValueError("Derivative is zero - no convergence") 

        current_x = prev_x - multiplicity * fx / fpx

        if abs(current_x - prev_x) < tolerance:
            return current_x

    return current_x





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
