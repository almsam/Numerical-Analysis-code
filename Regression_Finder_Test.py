import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from sympy import symbols
from Regression_Finder import *
from Regression_Standards import *

class TestRegressionGeneral(unittest.TestCase):
    """Test cases from Regression_Finder_General_Test.py"""

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


class TestRegressionStandards(unittest.TestCase):
    """Test cases from Regression_Finder_Standards_Test.py"""

    @classmethod
    def setUpClass(cls):
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
        
        print("\n \n \n Testing graphing model " + model_name)
        
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


class TestExponentialRegression(unittest.TestCase):
    """Test cases for the exponential regression fix mentioned in test report.txt"""

    def test_exp_regression_centered_around_zero(self):
        print("\n\n\n=== EXPONENTIAL REGRESSION FIX TEST ===")
        print("Testing: test_exp_regression_centered_around_zero")
        x = np.linspace(-2, 2, 50)
        y = 1.5 * np.exp(0.7 * x)
        
        print(f"Expected: 1.5 * np.exp(0.7 * x)")
        error, regression = exp_regression(x, y)
        print(f"Received: {regression}")
        
        # Check that we get a reasonable fit
        self.assertIsInstance(error, float)
        self.assertLess(error, 1.0, msg="Error should be small for perfect exponential data")
        
        # Check the exponential terms are populated
        exp_terms = regression["exponential_terms"]
        self.assertEqual(len(exp_terms), 1, msg="Should have exactly one exponential term")
        coeff, slope = exp_terms[0]
        
        # Should be close to the original values (within reasonable tolerance)
        self.assertAlmostEqual(coeff, 1.5, places=1, msg="Coefficient should be close to 1.5")
        self.assertAlmostEqual(slope, 0.7, places=1, msg="Slope should be close to 0.7")

    def test_exp_regression_perfect_exp_data(self):
        print("\n\n\n=== EXPONENTIAL REGRESSION FIX TEST ===")
        print("Testing: test_exp_regression_perfect_exp_data")
        x = np.linspace(0.1, 5, 50)
        y = 2 * np.exp(0.5 * x)
        
        print(f"Expected: 2 * np.exp(0.5 * x)")
        error, regression = exp_regression(x, y)
        print(f"Received: {regression}")
        
        # Check that we get a very small error for perfect data
        self.assertIsInstance(error, float)
        self.assertLess(error, 0.01, msg="Error should be very small for perfect exponential data")
        
        # Check the exponential terms are populated correctly
        exp_terms = regression["exponential_terms"]
        self.assertEqual(len(exp_terms), 1, msg="Should have exactly one exponential term")
        coeff, slope = exp_terms[0]
        
        # Should be very close to the original values
        self.assertAlmostEqual(coeff, 2.0, places=2, msg="Coefficient should be close to 2.0")
        self.assertAlmostEqual(slope, 0.5, places=2, msg="Slope should be close to 0.5")

    def test_exp_regression_shifted_left(self):
        print("\n\n\n=== EXPONENTIAL REGRESSION FIX TEST ===")
        print("Testing: test_exp_regression_shifted_left")
        x = np.linspace(-5, 0, 50)
        y = 4 * np.exp(0.3 * x)
        
        print(f"Expected: 4 * np.exp(0.3 * x)")
        error, regression = exp_regression(x, y)
        print(f"Received: {regression}")
        
        # Check that we get a reasonable fit
        self.assertIsInstance(error, float)
        self.assertLess(error, 1.0, msg="Error should be small for perfect exponential data")
        
        # Check the exponential terms are populated
        exp_terms = regression["exponential_terms"]
        self.assertEqual(len(exp_terms), 1, msg="Should have exactly one exponential term")
        coeff, slope = exp_terms[0]
        
        # Should be close to the original values
        self.assertAlmostEqual(coeff, 4.0, places=1, msg="Coefficient should be close to 4.0")
        self.assertAlmostEqual(slope, 0.3, places=1, msg="Slope should be close to 0.3")

    def test_exp_regression_positive_quadrant(self):
        print("\n\n\n=== EXPONENTIAL REGRESSION FIX TEST ===")
        print("Testing: test_exp_regression_positive_quadrant")
        x = np.linspace(1, 10, 50)
        y = 2.2 * np.exp(0.2 * x)
        
        print(f"Expected: 2.2 * np.exp(0.2 * x)")
        error, regression = exp_regression(x, y)
        print(f"Received: {regression}")
        
        # Check that we get a reasonable fit
        self.assertIsInstance(error, float)
        self.assertLess(error, 1.0, msg="Error should be small for perfect exponential data")
        
        # Check the exponential terms are populated
        exp_terms = regression["exponential_terms"]
        self.assertEqual(len(exp_terms), 1, msg="Should have exactly one exponential term")
        coeff, slope = exp_terms[0]
        
        # Should be close to the original values
        self.assertAlmostEqual(coeff, 2.2, places=1, msg="Coefficient should be close to 2.2")
        self.assertAlmostEqual(slope, 0.2, places=1, msg="Slope should be close to 0.2")


if __name__ == '__main__':
    # Create test suite with all test classes
    suite = unittest.TestSuite()
    
    # Add test classes in logical order
    suite.addTest(unittest.makeSuite(TestRegressionGeneral))
    suite.addTest(unittest.makeSuite(TestRegressionStandards))
    suite.addTest(unittest.makeSuite(TestExponentialRegression))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)