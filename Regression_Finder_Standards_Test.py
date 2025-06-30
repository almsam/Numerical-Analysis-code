import unittest
import numpy as np
from Regression_Finder import *
from Regression_Standards import *

class TestRegressionStandards(unittest.TestCase):

    @classmethod
    def setUp(cls):
        df = pd.read_csv("data/possum.csv")
        df_filtered = df[["hdlngth", "age"]].dropna()
        cls.x = df_filtered["age"].values
        cls.y = df_filtered["hdlngth"].values

    def check_standards_methods(self, model, model_name):
        
        print("\n \n ############################################################")
        print("\n \n \t \t Model: " + model_name)

        print("\n \n \n Printing all terms for model " + model_name)
        print_all_terms(model)
        
        print("\n \n \n Printing all non 0 terms for model " + model_name)
        print_non_zero_terms(model)
        
        print("\n \n \n Testing sympy func for model " + model_name)
        func = generate_sympy_function(model)
        self.assertIsNotNone(func)
        
        # model.add_regression_outputs() #this test case is skipped
        
        print("\n \n \n Testing graphing model" + model_name)
        
        x_range = (0, 10)
        plot_function(model, x_range=x_range, title=model_name)
        
        plot_function_data(model, self.x, self.y, x_range=x_range, title=model_name)


    def test_linear_standards(self):
        error, regression = linear_regression(self.x, self.y)
        self.check_standards_methods(regression, "linear")

    def test_quadratic_standards(self):
        error, regression = quadratic_regression(self.x, self.y)
        self.check_standards_methods(regression, "quadratic")

    def test_cubic_standards(self):
        error, regression = cubic_regression(self.x, self.y)
        self.check_standards_methods(regression, "cubic")

    def test_poly_standards(self):
        error, regression = poly_regression(self.x, self.y, 4)
        self.check_standards_methods(regression, "poly")

    def test_sin_standards(self):
        error, regression = sin_regression(self.x, self.y)
        self.check_standards_methods(regression, "sin")

    def test_log_standards(self):
        error, regression = logarithmic_regression(self.x, self.y)
        self.check_standards_methods(regression, "log")

    def test_exponential_standards(self):
        error, regression = exp_regression(self.x, self.y)
        self.check_standards_methods(regression, "exp")


if __name__ == '__main__':
    unittest.main()
