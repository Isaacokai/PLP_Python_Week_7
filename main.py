# Assignment: Pandas & Matplotlib Data Analysis


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load & Explore Data


try:
    # Load the Iris dataset from sklearn and convert to DataFrame
    iris = load_iris(as_frame=True)
    df = iris.frame  # DataFrame version
    df['species'] = df['target'].map(dict(enumerate(iris.target_names)))

    print("First 5 rows of dataset:")
    print(df.head(), "\n")

    print("Dataset Info:")
    print(df.info(), "\n")

    # Check for missing values
    print("Missing values per column:")
    print(df.isnull().sum(), "\n")

    # Clean dataset (Iris has no missing values, but demo filling)
    df.fillna(df.mean(numeric_only=True), inplace=True)

except FileNotFoundError:
    print("✗ Dataset file not found. Please check the path.")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")

# Task 2: Basic Data Analysis


# Statistical summary
print("Statistical Summary:")
print(df.describe(), "\n")

# Group by species and compute mean of numerical columns
grouped = df.groupby("species").mean(numeric_only=True)
print("Mean values grouped by species:")
print(grouped, "\n")

# Simple finding
print("Observation: Setosa flowers have smaller petal sizes compared to Versicolor and Virginica.\n")

# Task 3: Data Visualization

# Use Seaborn style for better visuals
sns.set(style="whitegrid")

# 1. Line Chart (just as demo, sepal length across index)
plt.figure(figsize=(8,5))
plt.plot(df.index, df['sepal length (cm)'], label="Sepal Length")
plt.title("Line Chart: Sepal Length Trend Across Samples")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# 2. Bar Chart: Average petal length per species
plt.figure(figsize=(8,5))
sns.barplot(x="species", y="petal length (cm)", data=df, estimator="mean", errorbar=None)
plt.title("Bar Chart: Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram: Distribution of Sepal Length
plt.figure(figsize=(8,5))
plt.hist(df["sepal length (cm)"], bins=15, color="skyblue", edgecolor="black")
plt.title("Histogram: Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot: Sepal Length vs Petal Length
plt.figure(figsize=(8,5))
sns.scatterplot(
    x="sepal length (cm)",
    y="petal length (cm)",
    hue="species",
    data=df,
    palette="deep"
)
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
