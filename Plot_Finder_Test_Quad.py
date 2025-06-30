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
        cls.df = pd.read_csv("data/possum.csv")
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

######################### p4 #################################

    def test_polynomial_p4_q1(self):
        print("\n\n\n Poly4 Q1:\n\n\n")
        x = np.linspace(0.1, 3, 50)
        y = x**4 + x**2 + 1
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Polynomial (x^4)")
        print("P4 Q1:\nexpected:", "x**4 + x**2 + 1", "\nreceived:", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for p4q1 data")
    def test_polynomial_p4_q2(self):
        print("\n\n\n Poly4 Q2:\n\n\n")
        x = np.linspace(-3, -0.1, 50)
        y = x**4 - 2 * x**2 + 1
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Polynomial (x^4)")
        print("P4 Q2:\nexpected:", "x**4 - 2*x**2 + 1", "\nreceived:", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for p4q2 data")
    def test_polynomial_p4_q3(self):
        print("\n\n\n Poly4 Q3:\n\n\n")
        x = np.linspace(-3, -0.1, 50)
        y = -x**4 - x**2 - 2
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Polynomial (x^4)")
        print("P4 Q3:\nexpected:", "-x**4 - x**2 - 2", "\nreceived:", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for p4q3 data")
    def test_polynomial_p4_q4(self):
        print("\n\n\n Poly4 Q4:\n\n\n")
        x = np.linspace(0.1, 3, 50)
        y = -x**4 + x**2 - 1
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Polynomial (x^4)")
        print("P4 Q4:\nexpected:", "-x**4 + x**2 - 1", "\nreceived:", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for p4q4 data")

######################### p5 #################################

    def test_polynomial_p5_q1(self):
        print("\n\n\n Poly5 Q1:\n\n\n")
        x = np.linspace(0.1, 3, 50)
        y = x**5 + x**3 + x
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Polynomial (x^5)")
        print("P5 Q1:\nexpected:", "x**5 + x**3 + x", "\nreceived:", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for p5q1 data")
    def test_polynomial_p5_q2(self):
        print("\n\n\n Poly5 Q2:\n\n\n")
        x = np.linspace(-3, -0.1, 50)
        y = -x**5 + x**3 - x
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Polynomial (x^5)")
        print("P5 Q2:\nexpected:", "-x**5 + x**3 - x", "\nreceived:", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for p5q2 data")
    def test_polynomial_p5_q3(self):
        print("\n\n\n Poly5 Q3:\n\n\n")
        x = np.linspace(-3, -0.1, 50)
        y = -x**5 - x**3 - x
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Polynomial (x^5)")
        print("P5 Q3:\nexpected:", "-x**5 - x**3 - x", "\nreceived:", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for p5q3 data")
    def test_polynomial_p5_q4(self):
        print("\n\n\n Poly5 Q4:\n\n\n")
        x = np.linspace(0.1, 3, 50)
        y = x**5 - x**3 + x
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Polynomial (x^5)")
        print("P5 Q4:\nexpected:", "x**5 - x**3 + x", "\nreceived:", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for p5q4 data")

######################### p6 #################################

    def test_polynomial_p6_q1(self):
        print("\n\n\n Poly6 Q1:\n\n\n")
        x = np.linspace(0.1, 3, 50)
        y = x**6 + x**4 + x**2
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Polynomial (x^6)")
        print("P6 Q1:\nexpected:", "x**6 + x**4 + x**2", "\nreceived:", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for p6q1 data")
    def test_polynomial_p6_q2(self):
        print("\n\n\n Poly6 Q2:\n\n\n")
        x = np.linspace(-3, -0.1, 50)
        y = x**6 - x**4 + x**2
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Polynomial (x^6)")
        print("P6 Q2:\nexpected:", "x**6 - x**4 + x**2", "\nreceived:", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for p6q2 data")
    def test_polynomial_p6_q3(self):
        print("\n\n\n Poly6 Q3:\n\n\n")
        x = np.linspace(-3, -0.1, 50)
        y = -x**6 - x**4 - x**2
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Polynomial (x^6)")
        print("P6 Q3:\nexpected:", "-x**6 - x**4 - x**2", "\nreceived:", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for p6q3 data")
    def test_polynomial_p6_q4(self):
        print("\n\n\n Poly6 Q4:\n\n\n")
        x = np.linspace(0.1, 3, 50)
        y = -x**6 + x**4 - x**2
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Polynomial (x^6)")
        print("P6 Q4:\nexpected:", "-x**6 + x**4 - x**2", "\nreceived:", str(formula))
        self.assertAlmostEqual(error, 0, places=5, msg="Expected zero error for p6q4 data")

######################### p7 #################################

    def test_polynomial_p7_q1(self):
        print("\n\n\n Poly7 Q1:\n\n\n")
        x = np.linspace(0.1, 3, 50)
        y = x**7 + x**5 + x**3 + x
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Polynomial (x^7)")
        print("P7 Q1:\nexpected:", "x**7 + x**5 + x**3 + x", "\nreceived:", str(formula))
        self.assertAlmostEqual(error, 0, places=3, msg="Expected zero error for p7q1 data")
    def test_polynomial_p7_q2(self):
        print("\n\n\n Poly7 Q2:\n\n\n")
        x = np.linspace(-3, -0.1, 50)
        y = -x**7 + x**5 - x**3 + x
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Polynomial (x^7)")
        print("P7 Q2:\nexpected:", "-x**7 + x**5 - x**3 + x", "\nreceived:", str(formula))
        self.assertAlmostEqual(error, 0, places=3, msg="Expected zero error for p7q2 data")
    def test_polynomial_p7_q3(self):
        print("\n\n\n Poly7 Q3:\n\n\n")
        x = np.linspace(-3, -0.1, 50)
        y = -x**7 - x**5 - x**3 - x
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Polynomial (x^7)")
        print("P7 Q3:\nexpected:", "-x**7 - x**5 - x**3 - x", "\nreceived:", str(formula))
        self.assertAlmostEqual(error, 0, places=3, msg="Expected zero error for p7q3 data")
    def test_polynomial_p7_q4(self):
        print("\n\n\n Poly7 Q4:\n\n\n")
        x = np.linspace(0.1, 3, 50)
        y = x**7 - x**5 + x**3 - x
        method, error, formula = find_best_fit(x, y, True)
        self.assertEqual(method, "Polynomial (x^7)")
        print("P7 Q4:\nexpected:", "x**7 - x**5 + x**3 - x", "\nreceived:", str(formula))
        self.assertAlmostEqual(error, 0, places=3, msg="Expected zero error for p7q4 data")

# p7 has 3 places to ensure tests pass at this number of computations

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


if __name__ == '__main__':
    unittest.main()