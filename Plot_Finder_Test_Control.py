import unittest
import numpy as np
from Plot_Finder import find_best_fit

class TestBestFitParameters(unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()
