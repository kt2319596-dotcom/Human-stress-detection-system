
import pandas as pd

df = pd.read_csv("student_stress_dataset.csv")

print("Rows and Columns:", df.shape)
print("Missing Values:")
print(df.isnull().sum())
print("Duplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates()

numeric_cols = df.select_dtypes(include="number").columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df.to_csv("student_stress_cleaned.csv", index=False)
print("Data cleaning completed.")
