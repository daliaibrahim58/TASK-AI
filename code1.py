import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



df = pd.read_csv("./credit_card_fraud.csv")

print("First rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nInfo:")
print(df.info())


print("\nSummary statistics:")
print(df.describe())

print("\nMode:")
print(df.mode().iloc[0])


outlier_columns = []

for col in df.select_dtypes(include=np.number).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    if ((df[col] < lower) | (df[col] > upper)).any():
        outlier_columns.append(col)

print("\nColumns with outliers:")
print(outlier_columns)


print("\nClass distribution:")
print(df["is_fraud"].value_counts())

print("\nClass percentage:")
print(df["is_fraud"].value_counts(normalize=True))



categorical_cols = df.select_dtypes(include="object").columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


for col in outlier_columns:
    if col in df.columns and col != "is_fraud":
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]


X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)



model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nResults:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
