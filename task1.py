import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1 Load Dataset
df = pd.read_csv("StudentsPerformance.csv")

# 2 Display first & last few rows
print("FIRST 5 ROWS:\n", df.head())
print("\nLAST 5 ROWS:\n", df.tail())

# 3 Dataset shape
print("\nDataset Shape (rows, columns): ", df.shape)

# 4 Dataset info
print("\nDATA INFO:")
print(df.info())

# 5 Statistical Summary
print("\nSTATISTICAL SUMMARY:")
print(df.describe())

# 6 Check Missing Values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# 7 Unique values in categorical columns
print("\nUNIQUE VALUES (categorical columns):")
for col in df.columns:
    if df[col].dtype == 'object':
        print(col, ":", df[col].unique())

# 8 Identify target + input features
print("\nTARGET VARIABLE: 'math score'")
print("INPUT FEATURES:", df.columns[:-1].tolist())

# 9 Simple Visualizations
plt.figure(figsize=(6,4))
sns.histplot(df['math score'], kde=True)
plt.title("Math Score Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x=df['gender'])
plt.title("Gender Count")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(data=df[['math score', 'reading score', 'writing score']])
plt.title("Score Comparison")
plt.show()

# 10 Correlation Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

#Simple ML Model (Linear Regression)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = df[['reading score', 'writing score']]
y = df['math score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nMODEL ACCURACY (R2 Score): ", r2_score(y_test, y_pred))