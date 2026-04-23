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


# Detecting outliers and inliers
numeric_cols = ["pollutant_min", "pollutant_max", "pollutant_avg"]
outliers = []
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)      # It marks the point below which 25% of the data falls.
    Q3 = df[col].quantile(0.75)      # It marks the point below which 75% of the data falls.
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers.extend(df[(df[col] < lower_bound) | (df[col] > upper_bound)].index)

print(outliers)
s1 = df.drop(set(outliers))
print(s1.info())


# Normalising the data
df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
print(df)
print(df.describe())


# Visualization
# Plot 1
df.groupby('pollutant_id')['pollutant_avg'].mean().plot(kind='bar')
plt.title("Average Pollution by Pollutant")
plt.xlabel("Pollutant")
plt.ylabel("Average Value")
plt.show()


# Plot 2
top_states = df.groupby('state')['pollutant_avg'].mean().sort_values(ascending=False).head(10)
top_states.plot(kind='bar')
plt.title("Top 10 Polluted States")
plt.show()


# Plot 3
counts = df['Pollution Level'].value_counts()
plt.figure()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title("Pollution Level Distribution")
plt.show()



# Plot 4   // Scatter plot which show pollution changes with geographical position. You may see clusters (north vs south differences)
plt.figure(figsize=(10, 5))
sns.scatterplot(x='latitude', y='pollutant_avg', hue='Pollution Level', data=df, palette='pastel')
plt.xlabel("Latitude")
plt.ylabel("Pollution Level")
plt.title("Pollution vs Latitude")
plt.show()


# Plot 5
plt.figure()
sns.heatmap(df[['pollutant_min','pollutant_max','pollutant_avg']].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Between Pollutants")
plt.show()


# Plot 6
plt.figure()
plt.hist(df['pollutant_avg'], bins=20, color='purple', edgecolor='black', alpha=0.3)
plt.title("Distribution of Pollution Levels")
plt.xlabel("Pollution")
plt.ylabel("Frequency")
plt.show()




# plot 7
pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'OZONE']
df[pollutants].boxplot()
plt.title("Box Plot of Pollutants")
plt.xlabel("Pollutants")
plt.ylabel("Values")
plt.xticks(rotation=45)
plt.show()