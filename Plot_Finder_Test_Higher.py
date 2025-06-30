import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from sympy import symbols, exp, log, sin
import statsmodels.api as sm
import statsmodels.formula.api as smf  # type: ignore
# from Plot_Finder import linear_regression, quadratic_regression, cubic_regression, poly_regression, exp_regression, logarithmic_regression, sin_regression, find_best_fit, plot_best_fit
from Plot_Finder import find_best_fit, plot_best_fit
from Regression_Finder import *
import matplotlib.pyplot as plt

class TestRegressionMethods(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        # load data
        cls.df = pd.read_csv("data/possum.csv")
        df_filtered = cls.df[["hdlngth", "age"]].dropna()
        cls.y = df_filtered["hdlngth"].values
        cls.x = df_filtered["age"].values
        cls.zero = 0.000000000000001

    # def plot_regression(self, x, y, predicted_y, title):
    #     plt.scatter(x, y, label="Data Points", color="blue"); plt.plot(x, predicted_y, label="Fitted Curve", color="red")
    #     plt.title(title); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

# simple

    def test_nonperfect_fit_returns_nonzero_error(self):
        print("\n\n\nNonzero:\n\n\n")
        x = np.linspace(0, 10, 50)
        y = x**2 + np.random.normal(0, 5, 50)
        method, error, formula = find_best_fit(x, y, True)
        # self.assertEqual(method, "Linear")
        self.assertGreater(error, 0.1)


    def test_linear_regression_perfect_linear_data(self):
        print("\n\n\nLinear:\n\n\n")
        x = np.linspace(0, 10, 50); x = x[x != 0] #avoid x 0
        y = 3 * x + 2  # y = 3x + 2
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Linear")
        print("Linear:\nexpected:", "3.0*x + 2.0", "\nrecieved: ", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect linear data")
        # self.plot_regression(x, y, formula(x), "Linear Regression")
        # plot_best_fit(x, y, best_method, best_fit_formula)

    def test_quadratic_regression_with_perfect_quadratic_data(self):
        print("\n\n\nQuadratic:\n\n\n")
        x = np.linspace(self.zero, 5, 50); x = x[x != 0]
        y = x**2 + 3 * x + 1  # y = 2x^2 + 3x + 1
        method, error, formula = find_best_fit(x, y, True)
        self.assertTrue((method == "Quadratic") or (method == "Cubic"))
        print("Quadtatic:\nexpected:", "x**2 + 3 * x + 1", "\n recieved: ", str(formula),)
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect quadratic data")

    def test_cubic_regression_perfect_cubic_data(self):
        print("\n\n\nCubic:\n\n\n")
        x = np.linspace(self.zero, 3, 50)
        y = x**3 - 2 * x**2 + 3 * x + 1  # y = x^3 - 2x^2 + 3x + 1
        method, error, formula = find_best_fit(x, y, True)
        self.assertTrue((method == "Cubic"))
        print("Cubic:\nexpected:", "x**3 - 2 * x**2 + 3 * x + 1", "\nrecieved: ", str(formula),)
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect cubic data")

# polynomial

    def test_polynomial_regression_perfect_quartic_data(self):
        print("\n\n\nP4:\n\n\n")
        x = np.linspace(self.zero, 3, 50)
        y = x**4 - x**3 + 2 * x**2 + x + 1  # y = x^4 - x^3 + 2x^2 + x + 1
        method, error, formula = find_best_fit(x, y, True)
        self.assertTrue((method == "Polynomial (x^4)"))
        print("P4:\nexpected:", "x**4 - x**3 + 2 * x**2 + x + 1", "\nrecieved: ", str(formula),)
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect p4 data")

    def test_polynomial_regression_perfect_quintic_data(self):
        print("\n\n\nP5:\n\n\n")
        x = np.linspace(self.zero, 3, 50)
        y = x**5 + x**4 - x**3 + 2 * x**2 + x + 1  # y = x^5 + x^4 - x^3 + 2x^2 + x + 1
        method, error, formula = find_best_fit(x, y, True)
        self.assertTrue((method == "Polynomial (x^5)"))
        print("P5:\nexpected:", "x**5 + x**4 - x**3 + 2 * x**2 + x + 1", "\nrecieved: ", str(formula),)
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect p5 data")
        
    def test_polynomial_regression_perfect_hexic_data(self):
        print("\n\n\nP6:\n\n\n")
        x = np.linspace(self.zero, 3, 50)
        y = x**6 + x**5 + x**4 - x**3 + 2 * x**2 + x + 1  # y = x^6 + x^5 + x^4 - x^3 + 2x^2 + x + 1
        method, error, formula = find_best_fit(x, y, True)
        self.assertTrue((method == "Polynomial (x^6)"))
        print("P6:\nexpected:", "x**6 + x**5 + x**4 - x**3 + 2 * x**2 + x + 1", "\nrecieved: ", str(formula),)
        self.assertAlmostEqual(error, 0, places=4, msg="Expected zero error for perfect p6 data")
        
    def test_polynomial_regression_perfect_heptic_data(self):
        print("\n\n\nP7:\n\n\n")
        x = np.linspace(self.zero, 3, 50)
        y = x**7 + x**6 + x**5 + x**4 - x**3 + 2 * x**2 + x + 1  # y = x^7 + x^6 + x^5 + x^4 - x^3 + 2x^2 + x + 1
        method, error, formula = find_best_fit(x, y, True)
        self.assertTrue((method == "Polynomial (x^7)"))
        print("P7:\nexpected:", "x**7 + x**6 + x**5 + x**4 - x**3 + 2 * x**2 + x + 1", "\nrecieved: ", str(formula),)
        self.assertAlmostEqual(error, 0, places=3, msg="Expected zero error for perfect p7 data")

# special

# exp

    def test_exp_regression_perfect_exp_data(self):
        print("\n\n\nexp:\n\n\n")
        x = np.linspace(self.zero, 5, 50)
        y = 2 * np.exp(0.5 * x)  # y = 2 * e^(0.5 * x)
        method, error, formula = find_best_fit(x, y, True)
        self.assertTrue((method == "Exponential"))
        print("exp:\nexpected:", "2 * np.exp(0.5 * x)", "\nrecieved: ", str(formula),)
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect exponential data")

    def test_exp_regression_shifted_left(self):
        x = np.linspace(-5, 0, 50)
        y = 4 * np.exp(0.3 * x)
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Exponential")
        print("exp:\nexpected:", "4 * np.exp(0.3 * x)", "\nrecieved: ", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect left shift exponential data")

    def test_exp_regression_centered_around_zero(self):
        x = np.linspace(-2, 2, 50)
        y = 1.5 * np.exp(0.7 * x)
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Exponential")
        print("exp:\nexpected:", "1.5 * np.exp(0.7 * x)", "\nrecieved: ", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect center zero exponential data")

    def test_exp_regression_positive_quadrant(self):
        x = np.linspace(1, 10, 50)
        y = 2.2 * np.exp(0.2 * x)
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Exponential")
        print("exp:\nexpected:", "2.2 * np.exp(0.2 * x)", "\nrecieved: ", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect positive exponential data")


# log

    def test_logarithmic_regression_perfect_log_data(self):
        print("\n\n\nlog:\n\n\n")
        x = np.linspace(1, 10, 50)  # Avoid x = 0 to prevent log(0)
        y = 3 * np.log(x) + 1  # y = 3 * log(x) + 1
        method, error, formula = find_best_fit(x, y, True)
        self.assertTrue((method == "Logarithmic"))
        print("log:\nexpected:", "3 * np.log(x) + 1", "\nrecieved: ", str(formula),)
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect logarithmic data")

    def test_log_regression_positive_x_large_values(self):
        x = np.linspace(10, 100, 50)
        y = 2.5 * np.log(x) - 4
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Logarithmic")
        print("log:\nexpected:", "2.5 * np.log(x) - 4", "\nrecieved: ", str(formula),)
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect x large logarithmic data")

    def test_log_regression_near_one(self):
        x = np.linspace(1.1, 5, 50)
        y = 0.8 * np.log(x) + 2
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Logarithmic")
        print("log:\nexpected:", "0.8 * np.log(x) + 2", "\nrecieved: ", str(formula),)
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect near 1 logarithmic data")

    def test_log_regression_shifted_upward(self):
        x = np.linspace(1, 15, 50)
        y = 3.3 * np.log(x) + 7
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Logarithmic")
        print("log:\nexpected:", "3.3 * np.log(x) + 7", "\nrecieved: ", str(formula),)
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect shifted up logarithmic data")

# sin

    def test_sin_regression_perfect_sin_data(self):
        print("\n\n\nsin:\n\n\n")
        x = np.linspace(self.zero, 2 * np.pi, 50)
        y = 5 * np.sin(x)  # y = 5 * sin(x)
        method, error, formula = find_best_fit(x, y, True)
        self.assertTrue((method == "Sine"))
        print("sin:\nexpected:", "3 * np.log(x) + 1", "\nrecieved: ", str(formula),)
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect sine data")

    # def test_logistic_regression_perfect_logistic_data(self):
    #     print("\n\n\nlogistic:\n\n\n")
    #     x = np.linspace(-6, 6, 50)
    #     y = 1 / (1 + np.exp(-x))  # Standard logistic function
    #     method, error, formula = find_best_fit(x, y) # , True)
    #     self.assertTrue((method == "Logistic"))
    #     print("P7:\nexpected:", "3 * np.log(x) + 1", "\nrecieved: ", str(formula),)
    #     self.assertAlmostEqual(error, 0, places=0, msg="Expected zero error for perfect logistic data")

    # def test_loess_regression_perfect_quadratic_data(self): # removed method for now but test remains
    #     print("\n\n\nLOESS:\n\n\n")
    #     x = np.linspace(self.zero, 5, 50)
    #     y = 2 * x**2 + 3 * x + 1  # y = 2x^2 + 3x + 1 # y = 2 * x**2 + 3 * x + 1 + np.random.normal(0, 1, 100)
    #     method, error, formula = find_best_fit(x, y, True)
    #     # self.assertTrue((method == "LOESS") or (method == "Cubic"))
    #     print("loess:\nexpected:", "2 * x**2 + 3 * x + 1", "\nrecieved: ", str(formula),)
    #     self.assertLessEqual(error, 1, msg="Expected zero error for LOESS on perfect quadratic data") # expect error under 1




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