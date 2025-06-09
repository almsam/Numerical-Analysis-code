import unittest
import numpy as np
from Plot_Finder import find_best_fit

class TestBestFitParameters(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.x = np.linspace(1, 10, 100)  # avoid 0 to keep log defined
        self.y = 3 * self.x + 2 + np.random.normal(0, 0.5, 100)  # mostly linear

    

