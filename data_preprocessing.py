import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Analyse dataset
# Load dataset
df = pd.read_csv('Dataset/FidelFolio_Dataset.csv')

# # Display basic info
# print("Shape of dataset:", df.shape)
# print(df.head())

# # Check datatypes and missing values
# print(df.info())

# # # Summary stats
# print(df.describe())

other_cols_object = [f"Feature{i}" for i in [4, 5, 6, 7, 9]]
other_cols_object.append(" Target 1 ")
other_cols_object.append(" Target 2 ")
other_cols_object.append(" Target 3 ")

for col in other_cols_object:
    # Convert to string first, then remove commas, then convert to float
    df[col] = df[col].astype(str).str.replace(',', '', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')
# print(df.columns.tolist())

# Sort and prepare data by time
df_sorted = df.sort_values(by=["Company", "Year"])

features=[f'Feature{i}' for i in range (1,29)]
num_cols=features

# Clean targets
targets = [' Target 1 ', ' Target 2 ', ' Target 3 ']
df[targets] = df[targets].apply(pd.to_numeric, errors='coerce')






# 2. Fill Missing Values
# Fill NaNs with company-wise mean, then global mean as fallback
for target in targets:
    company_mean = df.groupby('Company')[target].transform(lambda x: x.fillna(x.mean()))
    global_mean = df[target].mean()
    df[target] = company_mean.fillna(global_mean)

# Fill NaNs for each feature by company-wise mean
for feature in features:
    feature_mean = df.groupby('Company')[feature].transform(lambda x: x.fillna(x.mean()))
    global_mean = df[feature].mean()
    df[feature] = feature_mean.fillna(global_mean)
# print(df.info())






# 3. Fixing Outliers
# Winsorization: cap values outside IQR bounds
def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series.clip(lower, upper)

df[num_cols] = df[num_cols].apply(cap_outliers)
# print(df)





# 4. Normalization
from sklearn.preprocessing import StandardScaler

# Exclude target columns from scaling
target_cols = [' Target 1 ', ' Target 2 ', ' Target 3 ']
feature_cols = num_cols

# Scale features
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])




# 5. Saving modified dataset
df.to_csv('Dataset/FidelFolio_Dataset_Cleaned.csv', index=False)