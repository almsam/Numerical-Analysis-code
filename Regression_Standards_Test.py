import numpy as np
from Regression_Standards import (
    print_all_terms, get_all_terms,
    print_non_zero_terms, get_non_zero_terms,
    add_regression_outputs,
    generate_sympy_function,
    plot_function,
    plot_function_data
)

def test_regression_standards():
    # Create a sample regression output
    regression1 = {
        "sin_terms": [(1.0, 2.0, 0.0)],  # sin(2x)
        "exponential_terms": [(0.5, 1.0)],  # 0.5*exp(x)
        "logarithmic_terms": [(1.0, 1.0)],  # ln(x)
        "polynomial_terms": {0: 1.0, 1: 2.0, 2: 0.0}  # 1 + 2x
    }

    regression2 = {
        "sin_terms": [(0.5, 1.0, np.pi/2)],  # 0.5*sin(x + pi/2)
        "exponential_terms": [(0.0, 2.0)],  # zero term
        "logarithmic_terms": [(0.5, 2.0)],  # 0.5*ln(2x)
        "polynomial_terms": {1: 1.0, 2: 3.0}  # x + 3x^2
    }

    print("\nTesting print_all_terms:")
    print_all_terms(regression1)
    
    print("\nTesting get_all_terms:")
    print(get_all_terms(regression1))

    print("\nTesting print_non_zero_terms:")
    print_non_zero_terms(regression1)

    print("\nTesting get_non_zero_terms:")
    print(get_non_zero_terms(regression1))

    print("\nTesting add_regression_outputs:")
    combined = add_regression_outputs(regression1, regression2)
    print_all_terms(combined)

    print("\nTesting generate_sympy_function:")
    expr = generate_sympy_function(regression1)
    print("SymPy expression:", expr)

    print("\nTesting plotting:")
    # Generate some sample data
    x_data = np.linspace(0.1, 5, 50)
    y_data = np.sin(2*x_data) + 0.5*np.exp(x_data) + np.log(x_data) + 1 + 2*x_data + np.random.normal(0, 0.1, 50)
    
    print("Plotting function...")
    plot_function(regression1)
    
    print("Plotting function with data...")
    plot_function_data(regression1, x_data, y_data)

if __name__ == "__main__":
    test_regression_standards() 