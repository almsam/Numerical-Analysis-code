import pandas as pd
import numpy as np
import seaborn as sns  
import matplotlib.pyplot as plt
from sympy import symbols, sympify, solve, diff, lambdify, exp, log, sin

df = pd.read_csv("c:/Users/samia/OneDrive/Desktop/Numerical-Analysis-code/data/possum.csv"); df_filtered = df[["hdlngth", "age"]].dropna()
y = df_filtered["hdlngth"].values; x = df_filtered["age"].values; age = symbols('x')

def linear_regression(x, y):
    
    # x = sm.add_constant(x)
    # model = sm.OLS(y, x).fit()
    # intercept, slope = model.params
    
    # -- # so first we need to fit the regression & get the m(x) & b
    
    # error = np.mean(np.abs(y - model.predict(x)))
    
    # -- # then somehow find the error
    
    # return error, intercept + slope * age
    
    # -- # and return the result (this time in the new format)