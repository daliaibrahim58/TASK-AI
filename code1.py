import numpy as np
import pandas as pd

lst = [12.23, 13.32, 100, 36.32]
arr = np.array(lst)
print("Original List:", lst)
print("One-dimensional NumPy array:", arr)

matrix = np.arange(2, 11).reshape(3, 3)
print(matrix)

arr_int = np.array([1, 2, 3, 4])
print(arr_int.astype(float))

arr2 = np.array([[1, 4, 2],
                 [3, 0, 5]])
print(np.argmax(arr2, axis=1))
print(np.argmin(arr2, axis=1))

a = np.array([1, 2])
b = np.array([4, 5])
print(a > b)
print(a >= b)
print(a < b)
print(a <= b)

df = pd.read_csv('./data.csv')

print(df.head())
print(df.info())
print(df.describe())

numeric_cols = df.select_dtypes(include='number').columns

df['Total'] = df[numeric_cols[0]] + df[numeric_cols[1]]
print(df[df[numeric_cols[0]] > 120])
print(df.sort_values(by=numeric_cols[0], ascending=True))
