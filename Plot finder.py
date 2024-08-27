import pandas as pd
import numpy as np
import seaborn as sns  # type: ignore
import statsmodels.api as sm
import statsmodels.formula.api as smf  # type: ignore

# Lets start out by adding a dataset
df = pd.read_csv("c:/Users/samia/OneDrive/Desktop/Numerical-Analysis-code/data/possum.csv")
# print(df.head(10)); print("ran success")
df_filtered = df[["hdlngth", "age"]].dropna()
y = df_filtered["hdlngth"]; x = df_filtered["age"]

#   Then we can do a linear regression
sm.add_constant(x)
model = sm.OLS(y, x).fit()
#   And then measure how accurate it is
pred = model.predict(x); residuals = y - pred
LinearError = np.mean(np.abs(residuals))
print("lin reg error: ", LinearError)



# Then we can do a quadratic regression
x_quad = np.power(x, 2)
X = pd.DataFrame({"age": x, "age_squared": x_quad})
X = sm.add_constant(X)
model_quad = sm.OLS(y, X).fit()
# And then measure how accurate it is
pred_quad = model_quad.predict(X)
residuals_quad = y - pred_quad
QuadraticError = np.mean(np.abs(residuals_quad))
print("quad reg error: ", QuadraticError)



# Then we can do a cubic regression
x_cube = np.power(x, 3)
X = pd.DataFrame({"age": x, "age_cubed": x_cube})
X = sm.add_constant(X)
model_cube = sm.OLS(y, X).fit()
# And then measure how accurate it is
pred_cube = model_cube.predict(X)
residuals_cube = y - pred_cube
CubeError = np.mean(np.abs(residuals_cube))
print("cube reg error: ", CubeError)



# Then we can do an exp regression
# And then measure how accurate it is

# & then a sign wave regression
# And then measure how accurate it is

# Then compare how accurate all of the above is
# & designate the most accurate
