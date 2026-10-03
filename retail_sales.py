import numpy as np
import pandas as pd
from scipy.stats import zscore

df = pd.DataFrame({
    "Day": range(1, 11),
    "Revenue": [45, 52, np.nan, 48, 390, 55, np.nan, 50, 47, 53],
})

# (a) missing values -> median fill
print("Missing values in Revenue:", df["Revenue"].isnull().sum())
df["Revenue"] = df["Revenue"].fillna(df["Revenue"].median())
print("After median fill:")
print(df[["Day", "Revenue"]])

# (b) normalization
rev = df["Revenue"]
df["Revenue_MinMax"] = (rev - rev.min()) / (rev.max() - rev.min())
df["Revenue_Zscore"] = zscore(rev)

# (c) IQR outliers
q1, q3 = rev.quantile(0.25), rev.quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
df["Outlier"] = (rev < lower) | (rev > upper)
print("IQR bounds:", lower, "-", upper)
print("IQR outliers detected on day(s):", df.loc[df["Outlier"], "Day"].tolist())

# (d) final table
print(df[["Day", "Revenue", "Revenue_MinMax", "Revenue_Zscore", "Outlier"]])
