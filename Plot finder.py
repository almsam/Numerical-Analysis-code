import pandas as pd
import numpy as np
import seaborn as sns  # type: ignore
import statsmodels.api as sm
import statsmodels.formula.api as smf  # type: ignore

# Lets start out by adding a dataset
df = pd.read_csv("c:/Users/samia/OneDrive/Desktop/Numerical-Analysis-code/data/possum.csv")
# print(df.head(10)); print("ran success")

#   Then we can do a linear regression
df_filtered = df[["hdlngth", "age"]].dropna()
y = df_filtered["hdlngth"]; x = df_filtered["age"]

sm.add_constant(x)
model = sm.OLS(y, x).fit()

# And then measure how accurate it is
pred = model.predict(x); residuals = y - pred
LinearError = np.mean(np.abs(residuals))

print("lin reg error: ", LinearError)


# Then we can do a quadratic regression
# And then measure how accurate it is

# Then we can do an exp regression
# And then measure how accurate it is

# & then a sign wave regression
# And then measure how accurate it is

# Then compare how accurate all of the above is
# & designate the most accurate
