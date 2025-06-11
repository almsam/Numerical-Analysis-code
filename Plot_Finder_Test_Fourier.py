import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from sympy import symbols, lambdify, exp, log, sin
import statsmodels.api as sm
import statsmodels.formula.api as smf  # type: ignore
from Plot_Finder import find_best_fit, plot_best_fit, find_fourier
# from Numerical_Methods import *
import matplotlib.pyplot as plt

class TestFourierRegression(unittest.TestCase):
    def setUp(self):
        # make data for y = x + sin(x)
        self.x = np.linspace(1, 10, 101)
        self.y = self.x + np.sin(np.linspace(1, 20, 101))  # target func
        self.n = 5  # num iterations
        self.age = symbols("age")

    def test_fourier_regression(self):
        formula = find_fourier(self.x, self.y, 8, True)
        # formula_func = lambdify(self.x, formula, 'numpy')
        y_range = lambdify(self.age, formula, 'numpy')(self.x)
        residuals = self.y.copy(); residuals = residuals - y_range
        print( formula )


if __name__ == '__main__':
    unittest.main()
