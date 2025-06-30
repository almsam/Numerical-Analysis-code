import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from sympy import symbols, exp, log, sin
import statsmodels.api as sm
import statsmodels.formula.api as smf  # type: ignore
# from Plot_Finder import linear_regression, quadratic_regression, cubic_regression, poly_regression, exp_regression, logarithmic_regression, sin_regression, find_best_fit
from Regression_Finder import *

class TestRegressionMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # load data
        cls.df = pd.read_csv("data/possum.csv")
        df_filtered = cls.df[["hdlngth", "age"]].dropna()
        cls.y = df_filtered["hdlngth"].values
        cls.x = df_filtered["age"].values

    @patch('builtins.print')
    def test_linear_regression(self, mock_print):
        error, formula = linear_regression(self.x, self.y)
        self.assertIsInstance(error, float)
        self.assertGreater(error, 0)  # expect positive error
        self.assertIn('x', str(formula))
    
    def test_quadratic_regression(self):
        error, formula = quadratic_regression(self.x, self.y)
        self.assertIsInstance(error, float)
        self.assertGreater(error, 0)
        self.assertIn('x', str(formula))  # formula should contain 'x'

    def test_cubic_regression(self):
        error, formula = cubic_regression(self.x, self.y)
        self.assertIsInstance(error, float)
        self.assertGreater(error, 0)
        self.assertIn('x', str(formula))

    def test_poly_regression(self):
        for degree in range(4, 8):
            error, formula = poly_regression(self.x, self.y, degree)
            self.assertIsInstance(error, float)
            self.assertGreater(error, 0)
            self.assertIn('x', str(formula))

    def test_exp_regression(self):
        error, formula = exp_regression(self.x, self.y)
        self.assertIsInstance(error, float)
        self.assertGreater(error, 0)
        self.assertIn('x', str(formula))

    def test_logarithmic_regression(self):
        error, formula = logarithmic_regression(self.x, self.y)
        self.assertIsInstance(error, float)
        self.assertGreater(error, 0)
        self.assertIn('x', str(formula))

    def test_sin_regression(self):
        error, formula = sin_regression(self.x, self.y)
        self.assertIsInstance(error, float)
        self.assertGreater(error, 0)
        self.assertIn('x', str(formula))

    # def test_loess_regression(self):
    #     error, formula = loess_regression(self.x, self.y)
    #     self.assertIsInstance(error, float)
    #     self.assertGreater(error, 0)
    #     self.assertEqual(formula, "non-parametric")

    # def test_main_function(self):
    #     # get the printed output
    #     from unittest.mock import patch
    #     with patch('builtins.print') as mocked_print:
    #         main(self.x, self.y)
    #         mocked_print.assert_any_call("The method with the smallest error is:")

    def test_method_selection(self):
        error_list = []
        methods = [
            ("Linear", linear_regression),
            ("Quadratic", quadratic_regression),
            ("Cubic", cubic_regression),
            ("Exponential", exp_regression),
            ("Logarithmic", logarithmic_regression),
            ("Sine", sin_regression),
            # ("LOESS", loess_regression) #removed but this shall remain as it wont prevent tests
        ]
        for name, method in methods:
            error, formula = method(self.x, self.y)
            error_list.append((name, error, formula))
        
        #polynomial regression for degrees 4 to 7
        for degree in range(4, 8):
            error, formula = poly_regression(self.x, self.y, degree)
            error_list.append((f"Polynomial (x^{degree})", error, formula))
        
        min_error_method = min(error_list, key=lambda x: x[1])
        self.assertIn(min_error_method[0], ['Linear', 'Quadratic', 'Cubic', 'Exponential', 'Logarithmic', 'Sine']) # , 'LOESS'])
        self.assertIsInstance(min_error_method[1], float)
        self.assertGreater(min_error_method[1], 0)



if __name__ == '__main__':
    unittest.main()