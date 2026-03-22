import pandas as pd

# Load the dataset
url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)
# Display the first few rows of the dataset
print(df.head())
# Display summary statistics of the dataset
print(df.describe())
# Check for missing values
print(df.isnull().sum())
# Group the data by species and calculate the mean of each group
grouped = df.groupby('species').mean()
print(grouped)
