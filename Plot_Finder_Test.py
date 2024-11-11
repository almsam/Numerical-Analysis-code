import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from Plot_Finder import linear_regression, quadratic_regression, cubic_regression, poly_regression, exp_regression, logarithmic_regression, sin_regression, logistic_regression, loess_regression

class TestRegressionMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # load data
        cls.df = pd.read_csv("c:/Users/samia/OneDrive/Desktop/Numerical-Analysis-code/data/possum.csv")
        df_filtered = cls.df[["hdlngth", "age"]].dropna()
        cls.y = df_filtered["hdlngth"].values
        cls.x = df_filtered["age"].values

    @patch('builtins.print')
    def test_linear_regression(self, mock_print):
        error = linear_regression(self.x, self.y)
        mock_print.assert_called_with("Linear regression error: ", error)
        self.assertIsInstance(error, float)

    @patch('builtins.print')
    def test_quadratic_regression(self, mock_print):
        error = quadratic_regression(self.x, self.y)
        mock_print.assert_called_with("Quadratic regression error: ", error)
        self.assertIsInstance(error, float)

    @patch('builtins.print')
    def test_cubic_regression(self, mock_print):
        error = cubic_regression(self.x, self.y)
        mock_print.assert_called_with("Cubic regression error: ", error)
        self.assertIsInstance(error, float)

    @patch('builtins.print')
    def test_polynomial_regression(self, mock_print):
        errors = poly_regression(self.x, self.y)
        self.assertEqual(len(errors), 4)  # Ensure four polynomial degrees are tested (x^4 to x^7)
        for i, error in enumerate(errors, start=4):
            mock_print.assert_any_call(f"Polynomial regression (x^{i}) error: ", error)
            self.assertIsInstance(error, float)

    @patch('builtins.print')
    def test_exponential_regression(self, mock_print):
        error = exp_regression(self.x, self.y)
        mock_print.assert_called_with("Exponential regression error: ", error)
        self.assertIsInstance(error, float)

    @patch('builtins.print')
    def test_logarithmic_regression(self, mock_print):
        error = logarithmic_regression(self.x, self.y)
        mock_print.assert_called_with("Logarithmic regression error: ", error)
        self.assertIsInstance(error, float)

    @patch('builtins.print')
    def test_sine_regression(self, mock_print):
        error = sin_regression(self.x, self.y)
        mock_print.assert_called_with("Sine regression error: ", error)
        self.assertIsInstance(error, float)

    @patch('builtins.print')
    def test_logistic_regression(self, mock_print):
        error = logistic_regression(self.x, self.y)
        mock_print.assert_called_with("Logistic regression error: ", error)
        self.assertIsInstance(error, float)

    @patch('builtins.print')
    def test_loess_regression(self, mock_print):
        error = loess_regression(self.x, self.y)
        mock_print.assert_called_with("LOESS regression error: ", error)
        self.assertIsInstance(error, float)


if __name__ == '__main__':
    unittest.main()
