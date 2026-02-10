import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df = pd.read_csv('./data2.csv')

print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())

plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['age', 'bmi']])
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x='smoker', data=df)
plt.show()

le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])

scaler = MinMaxScaler()
df[['age', 'bmi']] = scaler.fit_transform(df[['age', 'bmi']])

print(df.head())