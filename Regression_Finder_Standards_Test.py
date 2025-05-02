import unittest
import numpy as np
from Regression_Finder import *
from Regression_Standards import *

class TestRegressionStandards(unittest.TestCase):

    @classmethod
    def setUp(cls):
        df = pd.read_csv("c:/Users/samia/OneDrive/Desktop/Numerical-Analysis-code/data/possum.csv")
        df_filtered = df[["hdlngth", "age"]].dropna()
        cls.x = df_filtered["age"].values
        cls.y = df_filtered["hdlngth"].values

    def test_linear_standards(self):
        error, regression = linear_regression(self.x, self.y)

    def test_quadratic_standards(self):
        error, regression = quadratic_regression(self.x, self.y)

    def test_cubic_standards(self):
        error, regression = cubic_regression(self.x, self.y)

    def test_poly_standards(self):
        error, regression = poly_regression(self.x, self.y, 4)

    def test_sin_standards(self):
        error, regression = exp_regression(self.x, self.y)

    def test_log_standards(self):
        error, regression = logarithmic_regression(self.x, self.y)

    def test_exponential_standards(self):
        error, regression = exp_regression(self.x, self.y)


if __name__ == '__main__':
    unittest.main()
