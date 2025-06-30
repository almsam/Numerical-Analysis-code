import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from sympy import symbols
from Regression_Finder import *

class TestLinearRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # load and clean data
        df = pd.read_csv("data/possum.csv")
        df_filtered = df[["hdlngth", "age"]].dropna()
        cls.x = df_filtered["age"].values
        cls.y = df_filtered["hdlngth"].values

    def test_linear_regression(self):
        error, formula = linear_regression(self.x, self.y)

        self.assertIsInstance(error, float)
        self.assertGreater(error, 0)

        self.assertIn('x', str(formula))
        
    def test_quadratic_regression(self):
        error, formula = quadratic_regression(self.x, self.y)

        self.assertIsInstance(error, float)
        self.assertGreater(error, 0)

        self.assertIn('x', str(formula))

    def test_cubic_regression(self):
        error, formula = cubic_regression(self.x, self.y)

        self.assertIsInstance(error, float)
        self.assertGreater(error, 0)

        self.assertIn('x', str(formula))

    def test_poly_regression(self):
        error, formula = poly_regression(self.x, self.y, 5)

        self.assertIsInstance(error, float)
        self.assertGreater(error, 0)

        self.assertIn('x', str(formula))

    def test_exp_regression(self):
        error, formula = exp_regression(self.x, self.y)

        self.assertIsInstance(error, float)
        self.assertGreater(error, 0)

        self.assertIn('x', str(formula))

    def test_log_regression(self):
        error, formula = logarithmic_regression(self.x, self.y)

        self.assertIsInstance(error, float)
        self.assertGreater(error, 0)

        self.assertIn('x', str(formula))

    def test_sin_regression(self):
        error, formula = sin_regression(self.x, self.y)

        self.assertIsInstance(error, float)
        self.assertGreater(error, 0)

        self.assertIn('x', str(formula))


if __name__ == '__main__':
    unittest.main()
