import pandas as pd
import numpy as np
import seaborn as sns  
import matplotlib.pyplot as plt
from sympy import symbols, sympify, solve, diff, lambdify, exp, log, sin

df = pd.read_csv("c:/Users/samia/OneDrive/Desktop/Numerical-Analysis-code/data/possum.csv"); df_filtered = df[["hdlngth", "age"]].dropna()
y = df_filtered["hdlngth"].values; x = df_filtered["age"].values; age = symbols('x')

