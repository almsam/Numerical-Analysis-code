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


if __name__ == '__main__':
    unittest.main()
