import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("C:/Users/LENOVO/OneDrive/Desktop/semester 4/Python Toolbox/Air-Quality-Analysis-Python/data/AQI Dataset for project.csv")
#pd.set_option('display.max_rows', 5000)
#pd.set_option('display.max_columns', 50)

#print(df.head(100))
print(df.describe())
print(df.info())
print(df.isna().sum()) 


df = df.dropna()
print(df.isna().sum())


df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')
print(df.info())
df['city'] = df['city'].str.lower()
df['state'] = df['state'].str.lower()


df["Pollution Level"] = None
df.loc[df["pollutant_avg"]<50, "Pollution Level"] = "Good"
df.loc[((df["pollutant_avg"]>=50) & (df["pollutant_avg"]<100)), "Pollution Level"] = "Moderate"
df.loc[df["pollutant_avg"]>=100, "Pollution Level"] = "Unhealthy"

print(df.head(10))
print(df.info())