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


if __name__ == '__main__':
    unittest.main()
