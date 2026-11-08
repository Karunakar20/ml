import pandas as pd

file_path = 'pandas/files/employee_sample_100_records.csv'

df = pd.read_csv(file_path)

# Perform modifications
df["Bonus"] = df["Salary_INR"] * 0.10

# Write updated DataFrame
# df.to_csv("updated_employees.csv", index=False)

# df.to_csv("filename.csv", index=False)  # Save the DataFrame to a CSV file

# df.to_parquet("employees.parquet", index=False)
# df = pd.read_pickle("employees.pkl")
# df.to_pickle("employees.pkl")
# print(df.describe())
# print(df.head())

print(df.isnull().sum())