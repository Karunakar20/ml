# Pandas

## 1. What is Pandas?

Pandas is an open-source Python library used for:

- Data manipulation
- Data analysis
- Data cleaning
- Data transformation
- Reading and writing datasets
- Handling missing values
- Working with large tabular datasets
- Preparing data for Machine Learning

## 2. Series and DataFrames

In Pandas, Series and DataFrame are the two fundamental data structures used to store, manipulate, and analyze data.

The main difference is:
- Series: One-dimensional data (single column).
- DataFrame: Two-dimensional data (multiple rows and columns).

## 3. Why Doesn't Pandas Support Native 3D DataFrames?
Pandas was primarily designed for structured and tabular data analysis.

# Its two main data structures, Series and DataFrame, are optimized for one-dimensional and two-dimensional data.

A traditional DataFrame represents information using rows and columns. Adding a third axis would require another level of indexing and data organization.

Instead of introducing a native 3D DataFrame, Pandas focuses on providing powerful operations for tabular data.

However, Pandas can still store NumPy arrays as individual values inside DataFrame cells. This is different from having a dedicated 3D data structure.

Important: Pandas can work with multidimensional data through integrations with other Python libraries.

