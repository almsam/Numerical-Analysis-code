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

    def check_standards_methods(self, model):
        print_all_terms(model)
        
        print_non_zero_terms(model)
        
        # model.add_regression_outputs()
        
        func = generate_sympy_function(model)
        
        # self.assertIsNotNone(func)
        # model.plot_function()
        
        # model.plot_function_data()
        # results = model.evaluate()
        # self.assertIn('r_squared', results)
        # self.assertIn('mae', results)
        # self.assertIn('rmse', results)

    def test_linear_standards(self):
        error, regression = linear_regression(self.x, self.y)
        self.check_standards_methods(regression)

    def test_quadratic_standards(self):
        error, regression = quadratic_regression(self.x, self.y)
        self.check_standards_methods(regression)

    def test_cubic_standards(self):
        error, regression = cubic_regression(self.x, self.y)
        self.check_standards_methods(regression)

    def test_poly_standards(self):
        error, regression = poly_regression(self.x, self.y, 4)
        self.check_standards_methods(regression)

    def test_sin_standards(self):
        error, regression = exp_regression(self.x, self.y)
        self.check_standards_methods(regression)

    def test_log_standards(self):
        error, regression = logarithmic_regression(self.x, self.y)
        self.check_standards_methods(regression)

    def test_exponential_standards(self):
        error, regression = exp_regression(self.x, self.y)
        self.check_standards_methods(regression)


if __name__ == '__main__':
    unittest.main()
