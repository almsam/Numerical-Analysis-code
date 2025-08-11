import unittest
from math import sqrt
import numpy as np
from unittest.mock import patch
from sympy import symbols, exp, log, sin, sympify, pi
from Numerical_Methods import input_function, evaluate_function, find_min_max, calculate_derivative, bisection_all # type: ignore

class TestMathFunctions(unittest.TestCase):
    
    def setUp(self):
        # symbol for sympy
        self.x = symbols('x')
    
    @patch('builtins.input', return_value="3*x + 2")
    def test_linear_function(self, mock_input):
        # f(x) = 3x + 2
        func = input_function()
        
        # eval
        self.assertAlmostEqual(evaluate_function(func, 1), 5.0, places=2)
        self.assertAlmostEqual(evaluate_function(func, -1), -1.0, places=2)
        
        # min max
        min_val, max_val = find_min_max(func, -10, 10)
        self.assertAlmostEqual(min_val, -28.0, places=2)
        self.assertAlmostEqual(max_val, 32.0, places=2)
        
        # derivative
        self.assertAlmostEqual(calculate_derivative(0, func), 3.0, places=2)

        # bisection
        roots = bisection_all(func, -100, 100, 0.5)
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0], (- 2 / 3), places=2)

    @patch('builtins.input', return_value="2*x**2 + 4*x + 1")
    def test_quadratic_function(self, mock_input):
        # f(x) = 2x^2 + 4x + 1
        func = input_function()
        
        # evaluate
        self.assertAlmostEqual(evaluate_function(func, -1), -1.0, places=2)
        self.assertAlmostEqual(evaluate_function(func, 2), 17.0, places=2)
        
        # min max
        min_val, max_val = find_min_max(func, -10, 10)
        self.assertAlmostEqual(min_val, -1.0, places=2)
        self.assertAlmostEqual(max_val, 241.0, places=2)
        
        # derivative
        self.assertAlmostEqual(calculate_derivative(0, func), 4.0, places=2)

        # bisection
        roots = bisection_all(func, -100, 100, 0.5)
        self.assertEqual(len(roots), 2)
        self.assertAlmostEqual(roots[1], (- 2 + sqrt(2)) / 2, places=2)
        self.assertAlmostEqual(roots[0], (- 2 - sqrt(2)) / 2, places=2)

    @patch('builtins.input', return_value="exp(x)")
    def test_exponential_function(self, mock_input):
        # f(x) = exp(x)
        func = input_function()
        
        # evaluate
        self.assertAlmostEqual(evaluate_function(func, 0), 1.0, places=2)
        self.assertAlmostEqual(evaluate_function(func, 1), float(exp(1)), places=2)
        
        # min max
        min_val, max_val = find_min_max(func, -2, 2)
        self.assertAlmostEqual(min_val, float(exp(-2)), places=2)
        self.assertAlmostEqual(max_val, float(exp(2)), places=2)
        
        # derivative
        self.assertAlmostEqual(calculate_derivative(1, func), float(exp(1)), places=2)

        # bisection
        roots = bisection_all(func, -100, 100, 0.5)
        self.assertEqual(len(roots), 0)

    @patch('builtins.input', return_value="log(x)")
    def test_logarithmic_function(self, mock_input):
        # f(x) = ln(x), only defined for x > 0
        func = input_function()
        
        # evaluate
        # evaluate
        self.assertAlmostEqual(evaluate_function(func, 1), 0.0, places=2)
        self.assertAlmostEqual(evaluate_function(func, float(exp(1))), 1.0, places=2)
        
        # min max
        min_val, max_val = find_min_max(func, 0.1, 2)
        self.assertAlmostEqual(min_val, float(log(0.1)), places=2)
        self.assertAlmostEqual(max_val, float(log(2)), places=2)
        
        # derivative
        self.assertAlmostEqual(calculate_derivative(1, func), 1.0, places=2)

        # bisection
        roots = bisection_all(func, 0, 100, 0.1)
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0], 1, places=2)

    @patch('builtins.input', return_value="sin(x)")
    def test_sine_function(self, mock_input):
        # f(x) = sin(x)
        func = input_function()
        
        # evaluate
        self.assertAlmostEqual(evaluate_function(func, 0), 0.0, places=2)
        self.assertAlmostEqual(evaluate_function(func, 3.14159 / 2), 1.0, places=2)
        
        # min max
        min_val, max_val = find_min_max(func, 0, 2 * 3.14159)
        self.assertAlmostEqual(min_val, -1.0, places=2)
        self.assertAlmostEqual(max_val, 1.0, places=2)
        
        # derivative
        self.assertAlmostEqual(calculate_derivative(3.14159 / 2, func), 0.0, places=2)

        # bisection
        roots = bisection_all(func, -5, 5, 0.1)
        self.assertEqual(len(roots), 3)
        self.assertAlmostEqual(roots[0], float(-pi), places=2)
        self.assertAlmostEqual(roots[1], 0, places=2)
        self.assertAlmostEqual(roots[2], float(pi), places=2)

if __name__ == "__main__":
    unittest.main()
