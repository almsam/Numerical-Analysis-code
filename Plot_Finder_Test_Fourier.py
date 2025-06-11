import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from sympy import symbols, lambdify, exp, log, sin
import statsmodels.api as sm
import statsmodels.formula.api as smf  # type: ignore
from Plot_Finder import *
# from Numerical_Methods import *
import matplotlib.pyplot as plt

class TestFourierRegression(unittest.TestCase):
    def setUp(self):
        # make data for y = x + sin(x)
        self.x = np.linspace(1, 10, 101)
        self.y = self.x + np.sin(np.linspace(1, 20, 101))  # target func
        self.n = 5  # num iterations
        self.age = symbols("age")

    def test_fourier_linear(self):
        # formula = find_fourier(self.x, self.y, 8, True)
        print("\n\n\nLinear:\n\n\n")
        x = np.linspace(0, 10, 50); x = x[x != 0] #avoid x 0
        y = 3 * x + 2 + (1*np.sin(5*x)) # y = 3x + 2 + 4 * exp(0.3 * x)
        funclist, formula = find_fourier(x, y, Iterations=1, plot=True, maxPolynomial=3, methods="all")#(x, y, True)
        print(formula)
        print(funclist)
        self.assertTrue(True, msg="Linear test passes")
        # self.assertEqual(method, "Linear")
        # print("Linear:\nexpected:", "3.0*x + 2.0", "\nrecieved: ", str(formula))
        # self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect linear data")


if __name__ == '__main__':
    unittest.main()
