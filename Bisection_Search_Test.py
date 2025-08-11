from math import sqrt
import unittest
from unittest.mock import patch
from sympy import pi, symbols
from Bisection_Search import bisection_search, bisection_all
from Numerical_Methods import input_function

class TestBisectionSearch(unittest.TestCase):

    def setUp(self):
        # symbol for sympy
        self.x = symbols('x')

    @patch('builtins.input', return_value="3*x + 2")
    def test_linear_bisection(self, mock_input):
        # f(x) = 3x + 2
        func = input_function()
        roots = bisection_all(func, -100, 100, 0.5)
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0], (- 2 / 3), places=2)

    @patch('builtins.input', return_value="2*x**2 + 4*x + 1")
    def test_quadratic_bisection(self, mock_input):
        # f(x) = 2x^2 + 4x + 1
        func = input_function()
        
        roots = bisection_all(func, -100, 100, 0.5)
        self.assertEqual(len(roots), 2)
        self.assertAlmostEqual(roots[1], (- 2 + sqrt(2)) / 2, places=2)
        self.assertAlmostEqual(roots[0], (- 2 - sqrt(2)) / 2, places=2)


    @patch('builtins.input', return_value="exp(x)")
    def test_exponential_bisection(self, mock_input):
        # f(x) = exp(x)
        func = input_function()
        roots = bisection_all(func, -100, 100, 0.5)
        self.assertEqual(len(roots), 0)


    @patch('builtins.input', return_value="log(x)")
    def test_logarithmic_bisection(self, mock_input):
        # f(x) = ln(x), only defined for x > 0
        func = input_function()
        roots = bisection_all(func, 0, 100, 0.1)
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0], 1, places=2)


    @patch('builtins.input', return_value="sin(x)")
    def test_sine_bisection(self, mock_input):
        # f(x) = sin(x)
        func = input_function()

        roots = bisection_all(func, -5, 5, 0.1)
        self.assertEqual(len(roots), 3)
        self.assertAlmostEqual(roots[0], float(-pi), places=2)
        self.assertAlmostEqual(roots[1], 0, places=2)
        self.assertAlmostEqual(roots[2], float(pi), places=2)

if __name__ == "__main__":
    unittest.main()
