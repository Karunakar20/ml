import pandas as pd

# To check the version
# print(pd.__version__)

# Series - one-dimensional labeled array 
series = pd.Series(["10", "20", "30", 40, True])

seriess = pd.Series([10, 20, 30, 40, True], index=["a", "b", "c", "d", "e"])
# print(seriess)  # Display the first few rows of the Series

data = pd.Series({
    'a': 10,
    'b': 20,
    'c': 30,
    'd': 40,
    'e': True
})

# print(data.get('a'))  # Display the first few rows of the Series
# print(data.get('f', 'Not Found'))  # Display the first few rows of the Series

# DataFrame  - two-dimensional labeled data structure with columns of potentially different types
data = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Age": [25, 30, 35, 40],
    "City": ["New York", "Los Angeles", "Chicago", "Houston"]
}
data_frame = pd.DataFrame(data)
# print(data_frame)
# print(data_frame.iloc[1])  # Display the first few rows of the DataFrame
# print(data_frame.info())  # Display the first few rows of the DataFrame

# empty DataFrame with specified columns
df = pd.DataFrame(columns=["Name", "Age", "Salary"])
# print(df)

