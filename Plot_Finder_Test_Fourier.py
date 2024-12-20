import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from sympy import symbols, lambdify, exp, log, sin
import statsmodels.api as sm
import statsmodels.formula.api as smf  # type: ignore
from Plot_Finder import linear_regression, quadratic_regression, cubic_regression, poly_regression, exp_regression, logarithmic_regression, sin_regression, logistic_regression, loess_regression, find_best_fit, plot_best_fit, fourier
import matplotlib.pyplot as plt

class TestFourierRegression(unittest.TestCase):
    def setUp(self):
        # make data for y = x + sin(x)
        self.x = np.linspace(1, 10, 101)
        self.y = self.x #+ np.sin(self.x)  # target func
        self.n = 5  # num iterations
        self.age = symbols("age")

    # def test_fourier_regression(self):
    #     full_formula = fourier(self.x, self.y, 5)
    #     full_formula_func = lambdify(self.age, sum(full_formula), 'numpy') # type: ignore

    #     # eval
    #     predictions = full_formula_func(self.x)

    #     # check MAE acceptable
    #     mae = np.mean(np.abs(self.y - predictions))
    #     self.assertLess(mae, 0.1, f"MAE is too high: {mae}")


    # def test_linear_regression(self):
    #     full_formula = fourier(self.x, self.y, 1)
    #     full_formula_func = lambdify(self.age, full_formula, 'numpy')

    #     # eval
    #     predictions = full_formula_func(self.x)

    #     # check MAE acceptable
    #     mae = self.y - predictions
    #     # mae = np.mean(np.abs(mae))
    #     # self.assertLess(mae, 1000000, f"MAE is too high: {mae}")
    #     print(type(mae))
    #     # self.assertAlmostEqual(mae, 0, places=5, msg="Expected zero error for perfect linear data")
    #     self.assertGreaterEqual(mae, 0)

    def test_fourier_regression_perfect_linear_data(self):
        print("\n\n\nFourier:\n\n\n")
        x = np.linspace(0, 10, 50); x = x[x != 0] #avoid x 0
        y = 3 * x + 2  # y = 3x + 2
        formula = fourier(x, y, 1)
        expected_formula = 3 * symbols("age") + 2
        # self.assertEqual(method, "Linear")
        
        age = symbols("age")
        full_formula_func = lambdify(age, formula, 'numpy')
        expected_func = lambdify(age, expected_formula, 'numpy')
        points = np.linspace(0, 10, 100)  #  points for comparison
        formula_values = full_formula_func(points)
        expected_values = expected_func(points)

        # calc mae
        error = np.mean(np.abs(formula_values - expected_values))
        
        print(type(error))
        # print("Linear:\nexpected:", "3.0*x + 2.0", "\nrecieved: ", str(formula))
        # self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect linear data")

if __name__ == '__main__':
    unittest.main()
