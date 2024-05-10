# Task 1: Load a CSV file into a pandas DataFrame
import pandas as pd
file_path = 'breast-cancer.csv'
data = pd.read_csv(file_path)
print(data.head())
