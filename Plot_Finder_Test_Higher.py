import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from sympy import symbols, exp, log, sin
import statsmodels.api as sm
import statsmodels.formula.api as smf  # type: ignore
from Plot_Finder import linear_regression, quadratic_regression, cubic_regression, poly_regression, exp_regression, logarithmic_regression, sin_regression, logistic_regression, loess_regression, find_best_fit

class TestRegressionMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # load data
        cls.df = pd.read_csv("c:/Users/samia/OneDrive/Desktop/Numerical-Analysis-code/data/possum.csv")
        df_filtered = cls.df[["hdlngth", "age"]].dropna()
        cls.y = df_filtered["hdlngth"].values
        cls.x = df_filtered["age"].values


    def test_linear_regression_perfect_linear_data(self):
        x = np.linspace(0, 10, 50)
        y = 3 * x + 2  # y = 3x + 2
        error, formula = linear_regression(x, y)
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect linear data")

    def test_quadratic_regression_with_perfect_quadratic_data(self):
        x = np.linspace(-5, 5, 50)
        y = 2 * x**2 + 3 * x + 1  # y = 2x^2 + 3x + 1
        error, formula = quadratic_regression(x, y)
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect quadratic data")

    def test_cubic_regression_perfect_cubic_data(self):
        x = np.linspace(-3, 3, 50)
        y = x**3 - 2 * x**2 + 3 * x + 1  # y = x^3 - 2x^2 + 3x + 1
        error, formula = cubic_regression(x, y)
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect cubic data")



    def test_polynomial_regression_perfect_quartic_data(self):
        x = np.linspace(-3, 3, 50)
        y = x**4 - x**3 + 2 * x**2 + x + 1  # y = x^4 - x^3 + 2x^2 + x + 1
        error, formula = poly_regression(x, y, degree=4)
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect p4 data")

    def test_polynomial_regression_perfect_quintic_data(self):
        x = np.linspace(-3, 3, 50)
        y = x**5 + x**4 - x**3 + 2 * x**2 + x + 1  # y = x^5 + x^4 - x^3 + 2x^2 + x + 1
        error, formula = poly_regression(x, y, degree=4)
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect p5 data")
        
    def test_polynomial_regression_perfect_hexic_data(self):
        x = np.linspace(-3, 3, 50)
        y = x**6 + x**5 + x**4 - x**3 + 2 * x**2 + x + 1  # y = x^6 + x^5 + x^4 - x^3 + 2x^2 + x + 1
        error, formula = poly_regression(x, y, degree=4)
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect p6 data")
        
    def test_polynomial_regression_perfect_heptic_data(self):
        x = np.linspace(-3, 3, 50)
        y = x**7 + x**6 + x**5 + x**4 - x**3 + 2 * x**2 + x + 1  # y = x^7 + x^6 + x^5 + x^4 - x^3 + 2x^2 + x + 1
        error, formula = poly_regression(x, y, degree=4)
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect p7 data")



    def test_exp_regression_perfect_exp_data(self):
        x = np.linspace(0, 5, 50)
        y = 2 * np.exp(0.5 * x)  # y = 2 * e^(0.5 * x)
        error, formula = exp_regression(x, y)
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect exponential data")

    def test_logarithmic_regression_perfect_log_data(self):
        x = np.linspace(1, 10, 50)  # Avoid x = 0 to prevent log(0)
        y = 3 * np.log(x) + 1  # y = 3 * log(x) + 1
        error, formula = logarithmic_regression(x, y)
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect logarithmic data")

    def test_sin_regression_perfect_sin_data(self):
        x = np.linspace(0, 2 * np.pi, 50)
        y = 5 * np.sin(x)  # y = 5 * sin(x)
        error, formula = sin_regression(x, y)
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect sine data")

    def test_logistic_regression_perfect_logistic_data(self):
        x = np.linspace(-6, 6, 50)
        y = 1 / (1 + np.exp(-x))  # Standard logistic function
        error, formula = logistic_regression(x, y)
        self.assertAlmostEqual(error, 0, places=0, msg="Expected zero error for perfect logistic data")

    def test_loess_regression_perfect_quadratic_data(self):
        x = np.linspace(-5, 5, 50)
        y = 2 * x**2 + 3 * x + 1  # y = 2x^2 + 3x + 1
        error, formula = loess_regression(x, y)
        self.assertLessEqual(error, 1, msg="Expected zero error for LOESS on perfect quadratic data") # expect error under 1




if __name__ == '__main__':
    unittest.main()

# Explanation
#     Linear Data:      y = 3x + 2 tests linear_regression
#     Quadratic Data:   y = 2x^2 + 3x + 1 tests quadratic_regression
#     Cubic Data:       y = x^3 - 2x^2 + 3x + 1 tests cubic_regression

#     Quartic Data:     y = x^4 - x^3 + 2x^2 + x + 1    tests poly_regression with degree 4
#     Quinic Data:      y = x^5 + [quartic]             tests poly_regression with degree 5
#     Hexic Data:       y = x^6 + [Quinic]              tests poly_regression with degree 6
#     Heptic Data:      y = x^7 + [Hexic]               tests poly_regression with degree 7

#     Exponential Data: y = 2 * e^(0.5 * x) tests exp_regression
#     Logarithmic Data: y = 3 * log(x) + 1 tests logarithmic_regression
#     Sine Data:        y = 5 * sin(x) tests sin_regression
#     Logistic Data:    y = 1 / (1 + exp(-x)) tests logistic_regression
#     LOESS on Quad Data: also uses y = 2x^2 + 3x + 1 to test loess_regression