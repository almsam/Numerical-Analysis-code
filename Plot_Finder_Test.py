import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from sympy import symbols, exp, log, sin
import statsmodels.api as sm
import statsmodels.formula.api as smf  # type: ignore
from Plot_Finder import find_best_fit, plot_best_fit, find_fourier
from Regression_Finder import *
import matplotlib.pyplot as plt

class TestRegressionMethods(unittest.TestCase):
    """Test cases from Plot_Finder_Test_Higher.py"""

    @classmethod
    def setUpClass(cls):
        # load data
        cls.df = pd.read_csv("data/possum.csv") # made this path file-dependent; adjust path as needed
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
        print("sin:\nexpected:", "5 * np.sin(x)", "\nrecieved: ", str(formula),)
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for perfect sine data")


class TestQuadrantRegression(unittest.TestCase):
    """Test cases from Plot_Finder_Test_Quad.py"""

    @classmethod
    def setUpClass(cls):
        cls.zero = 0.000000000000001

######################### linear #################################

    def test_linear_q1(self):
        print("\n\n\nLinear Q1:\n\n\n")
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

######################### cubic #################################

    def test_cubic_q1(self):
        print("\n\n\nCubic Q1:\n\n\n")
        x = np.linspace(0.1, 5, 50)
        y = x**3 + 2 * x**2 + x + 1  # All x > 0, y > 0
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Cubic")
        print("Cubic Q1:\nexpected:", "x**3 + 2*x**2 + x + 1", "\nreceived: ", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for q1 cubic data")

    def test_cubic_q2(self):
        print("\n\n\nCubic Q2:\n\n\n")
        x = np.linspace(-5, -0.1, 50)
        y = -x**3 - x**2 + 2 * x - 1  # x < 0, mostly y > 0
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Cubic")
        print("Cubic Q2:\nexpected:", "-x**3 - x**2 + 2*x - 1", "\nreceived: ", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for q2 cubic data")

    def test_cubic_q3(self):
        print("\n\n\nCubic Q3:\n\n\n")
        x = np.linspace(-5, -0.1, 50)
        y = -x**3 - 2 * x**2 - x - 2  # x < 0, y < 0
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Cubic")
        print("Cubic Q3:\nexpected:", "-x**3 - 2*x**2 - x - 2", "\nreceived: ", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for q3 cubic data")

    def test_cubic_q4(self):
        print("\n\n\nCubic Q4:\n\n\n")
        x = np.linspace(0.1, 5, 50)
        y = -x**3 + x**2 - 2 * x + 3  # x > 0, y < 0
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Cubic")
        print("Cubic Q4:\nexpected:", "-x**3 + x**2 - 2*x + 3", "\nreceived: ", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for q4 cubic data")

######################### exp #################################

    def test_exp_q1(self):
        print("\n\n\n Exp Q1:\n\n\n")
        x = np.linspace(0.1, 5, 50)
        y = 2 * np.exp(0.3 * x)
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Exponential")
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for exp q1 data")

    def test_exp_q2(self):
        print("\n\n\n Exp Q2:\n\n\n")
        x = -np.linspace(0.1, 5, 50)
        y = 2 * np.exp(0.3 * x)
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Exponential")
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for exp q2 data")

    def test_exp_q3(self):
        print("\n\n\n Exp Q3:\n\n\n")
        x = -np.linspace(0.1, 5, 50)
        y = 2 * np.exp(-0.3 * x)
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Exponential")
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for exp q3 data")

    def test_exp_q4(self):
        print("\n\n\n Exp Q4:\n\n\n")
        x = np.linspace(0.1, 5, 50)
        y = 2 * np.exp(-0.3 * x)
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Exponential")
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for exp q4 data")

######################### log #################################

    def test_log_q1(self):
        print("\n\n\n Log Q1:\n\n\n")
        x = np.linspace(1, 10, 50)
        y = 2 * np.log(x) + 5
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Logarithmic")
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for log q1 data")

    def test_log_q2(self):
        print("\n\n\n Log Q2:\n\n\n")
        x = np.linspace(1, 10, 50)
        y = -2 * np.log(x) + 5
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Logarithmic")
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for log q2 data")

    def test_log_q3(self):
        print("\n\n\n Log Q3:\n\n\n")
        x = np.linspace(1, 10, 50)
        y = -2 * np.log(x) - 5
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Logarithmic")
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for log q3 data")

    def test_log_q4(self):
        print("\n\n\n Log Q4:\n\n\n")
        x = np.linspace(1, 10, 50)
        y = 2 * np.log(x) - 5
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Logarithmic")
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for log q4 data")

######################### sin #################################

    def test_sin_q1(self):
        print("\n\n\n Sin Q1:\n\n\n")
        x = np.linspace(0, np.pi / 2, 50)
        y = 4 * np.sin(x)
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Sine")
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for sin q1 data")

    def test_sin_q2(self):
        print("\n\n\n Sin Q2:\n\n\n")
        x = np.linspace(np.pi / 2, np.pi, 50)
        y = 4 * np.sin(x)
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Sine")
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for sin q2 data")

    def test_sin_q3(self):
        print("\n\n\n Sin Q3:\n\n\n")
        x = np.linspace(np.pi, 3 * np.pi / 2, 50)
        y = 4 * np.sin(x)
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Sine")
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for sin q3 data")

    def test_sin_q4(self):
        print("\n\n\n Sin Q4:\n\n\n")
        x = np.linspace(3 * np.pi / 2, 2 * np.pi, 50)
        y = 4 * np.sin(x)
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Sine")
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for sin q4 data")


class TestBestFitParameters(unittest.TestCase):
    """Test cases from Plot_Finder_Test_Control.py"""

    def setUp(self):
        np.random.seed(42)
        self.x = np.linspace(1, 10, 100)  # avoid 0 to keep log defined
        self.y = 3 * self.x + 2 + np.random.normal(0, 0.5, 100)  # mostly linear

    def test_max_polynomial_limits_high_degree(self):
        """Test that polynomial degrees up to maxPolynomial are included."""
        method, error, formula = find_best_fit(self.x, self.y, methods="all", maxPolynomial=5)
        found_degrees = [d for d in range(4, 6) if f"x^{d}" in method]
        if "Polynomial" in method: # type: ignore
            self.assertTrue(any(found_degrees), msg=f"Expected polynomial degree 4 or 5 in method, got {method}")
        else:
            self.assertNotIn("Polynomial", method)

    def test_methods_filter_includes_only_exponential(self):
        """Test that only exponential method is considered."""
        method, error, formula = find_best_fit(self.x, self.y, methods="exponential", maxPolynomial=1)
        self.assertEqual(method, "Exponential", msg="Only 'Exponential' should be used")

    def test_methods_filter_combined_methods(self):
        """Test that linear and logarithmic are the only ones used."""
        method, error, formula = find_best_fit(self.x, self.y, methods="linear, logarithmic", maxPolynomial=1)
        self.assertIn(method, ["Linear", "Logarithmic"], msg="Only 'Linear' and 'Logarithmic' should be considered")

    def test_sine_excluded_when_not_requested(self):
        """Test that sine regression is excluded when not in method list."""
        method, error, formula = find_best_fit(self.x, self.y, methods="linear, exponential", maxPolynomial=3)
        self.assertNotEqual(method, "Sine", msg="Sine should not be tested when not requested")

    def test_high_degree_polynomial_win(self):
        """Test that high-degree polynomials can win with appropriate data."""
        x = np.linspace(0, 2, 100)
        y = 5 * x**5 - 3 * x**3 + x + 1  # 5th degree poly
        method, error, formula = find_best_fit(x, y, maxPolynomial=6)
        self.assertIn("x^5", method, msg="Expected 5th-degree polynomial to be selected")

    def test_default_behavior_includes_all(self):
        """Test that default parameters include all models including polynomials up to 7."""
        method, error, formula = find_best_fit(self.x, self.y)
        self.assertIsInstance(method, str)
        self.assertGreaterEqual(error, 0)


class TestFourierRegression(unittest.TestCase):
    """Test cases from Plot_Finder_Test_Fourier.py"""
    
    def setUp(self):
        # make data for y = x + sin(x)
        self.x = np.linspace(1, 10, 101)
        self.y = self.x + np.sin(np.linspace(1, 20, 101))  # target func
        self.n = 5  # num iterations
        self.age = symbols("age")

    def test_fourier_linear(self):
        print("\n\n\nLinear Fourier:\n\n\n")
        x = np.linspace(0, 10, 50); x = x[x != 0] #avoid x 0
        y = 3 * x + 2 + (1*np.sin(5*x)) # y = 3x + 2 + sin(5x)
        funclist, formula = find_fourier(x, y, Iterations=1, plot=True, maxPolynomial=3, methods="all")
        print(formula)
        print(funclist)
        self.assertTrue(True, msg="Linear test passes")


class TestLowerLevel(unittest.TestCase):
    """Test cases from Plot_Finder_Test_Lower.py"""

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

    def test_method_selection(self):
        error_list = []
        methods = [
            ("Linear", linear_regression),
            ("Quadratic", quadratic_regression),
            ("Cubic", cubic_regression),
            ("Exponential", exp_regression),
            ("Logarithmic", logarithmic_regression),
            ("Sine", sin_regression),
        ]
        for name, method in methods:
            error, formula = method(self.x, self.y)
            error_list.append((name, error, formula))
        
        #polynomial regression for degrees 4 to 7
        for degree in range(4, 8):
            error, formula = poly_regression(self.x, self.y, degree)
            error_list.append((f"Polynomial (x^{degree})", error, formula))
        
        min_error_method = min(error_list, key=lambda x: x[1])
        self.assertIn(min_error_method[0], ['Linear', 'Quadratic', 'Cubic', 'Exponential', 'Logarithmic', 'Sine', 'Polynomial (x^4)', 'Polynomial (x^5)', 'Polynomial (x^6)', 'Polynomial (x^7)'])
        self.assertIsInstance(min_error_method[1], float)
        self.assertGreater(min_error_method[1], 0)


if __name__ == '__main__':
    # Create test suite with all test classes
    suite = unittest.TestSuite()
    
    # Add test classes in logical order
    suite.addTest(unittest.makeSuite(TestRegressionMethods))
    suite.addTest(unittest.makeSuite(TestQuadrantRegression))  
    suite.addTest(unittest.makeSuite(TestBestFitParameters))
    suite.addTest(unittest.makeSuite(TestFourierRegression))
    suite.addTest(unittest.makeSuite(TestLowerLevel))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)