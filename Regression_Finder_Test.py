import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from sympy import symbols
from Regression_Finder import *

class TestLinearRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # load and clean data
        df = pd.read_csv("c:/Users/samia/OneDrive/Desktop/Numerical-Analysis-code/data/possum.csv")
        df_filtered = df[["hdlngth", "age"]].dropna()
        cls.x = df_filtered["age"].values
        cls.y = df_filtered["hdlngth"].values


if __name__ == '__main__':
    unittest.main()
