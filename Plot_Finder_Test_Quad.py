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
        cls.df = pd.read_csv("c:/Users/samia/OneDrive/Desktop/Numerical-Analysis-code/data/possum.csv")
        df_filtered = cls.df[["hdlngth", "age"]].dropna()
        cls.y = df_filtered["hdlngth"].values
        cls.x = df_filtered["age"].values
        cls.zero = 0.000000000000001

######################### linear #################################

    def test_linear_q1(self):
        print("\n\n\nLinear:\n\n\n")
        x = np.linspace(0, 10, 50); x = x[x != 0] #avoid x 0
        y = 3 * x + 2  # y = 3x + 2
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Linear")
        print("Linear:\nexpected:", "3.0*x + 2.0", "\nrecieved: ", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for q1 linear data")
    def test_linear_q2(self):
        x = np.linspace(-10, 0, 50); x = x[x != 0]  # avoid x = 0
        y = 3 * x + 2
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Linear")
        print("Linear Q2:\nexpected:", "3.0*x + 2.0", "\nreceived: ", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for q2 linear data")
    def test_linear_q3(self):
        x = np.linspace(-10, -1, 50)
        y = -2 * x - 3  # Another line with both x and y negative
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Linear")
        print("Linear Q3:\nexpected:", "-2.0*x - 3.0", "\nreceived: ", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for q3 linear data")
    def test_linear_q4(self):
        x = np.linspace(1, 10, 50)
        y = -2 * x + 5  # y is negative, x is positive
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Linear")
        print("Linear Q4:\nexpected:", "-2.0*x + 5.0", "\nreceived: ", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for q4 linear data")

######################### quadratic #################################

    def test_quadratic_q1(self):
        print("\n\n\nQuadratic Q1:\n\n\n")
        x = np.linspace(0.1, 5, 50)
        y = x**2 + 2 * x + 1  # Opens upward, all y > 0
        method, error, formula = find_best_fit(x, y, True)
        self.assertIn(method, ["Quadratic", "Cubic"])
        print("Quadratic Q1:\nexpected:", "x**2 + 2*x + 1", "\nreceived: ", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for q1 quadratic data")
    def test_quadratic_q2(self):
        x = np.linspace(-5, -0.1, 50)
        y = x**2 - 3 * x + 2  # Opens upward, y > 0
        method, error, formula = find_best_fit(x, y, True)
        self.assertIn(method, ["Quadratic", "Cubic"])
        print("Quadratic Q2:\nexpected:", "x**2 - 3*x + 2", "\nreceived: ", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for q2 quadratic data")
    def test_quadratic_q3(self):
        x = np.linspace(-5, -0.1, 50)
        y = -x**2 - 2 * x - 3  # Opens downward, all y < 0
        method, error, formula = find_best_fit(x, y, True)
        self.assertIn(method, ["Quadratic", "Cubic"])
        print("Quadratic Q3:\nexpected:", "-x**2 - 2*x - 3", "\nreceived: ", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for q3 quadratic data")
    def test_quadratic_q4(self):
        x = np.linspace(0.1, 5, 50)
        y = -x**2 + 3 * x - 2  # Opens downward, mostly y < 0
        method, error, formula = find_best_fit(x, y, True)
        self.assertIn(method, ["Quadratic", "Cubic"])
        print("Quadratic Q4:\nexpected:", "-x**2 + 3*x - 2", "\nreceived: ", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for q4 quadratic data")
    
    


if __name__ == '__main__':
    unittest.main()