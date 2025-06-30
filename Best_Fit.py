from Plot_Finder import find_best_fit
import pandas as pd

# import n clean data
data_path = "data/possum.csv"
df = pd.read_csv(data_path)
df_filtered = df[["hdlngth", "age"]].dropna()
x = df_filtered["age"].values; y = df_filtered["hdlngth"].values

# call main from Plot_Finder.py
find_best_fit(x, y, True)
