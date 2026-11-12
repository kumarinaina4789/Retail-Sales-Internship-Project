"""
Retail Sales Dataset — Task 1: Data Cleaning
task1_data_cleaning .
"""
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "retail_sales_raw.csv"
OUTPUT_FILE = BASE_DIR / "retail_sales_cleaned.csv"

df = pd.read_csv(INPUT_FILE)
print("=== BEFORE CLEANING ===")
print("Shape:", df.shape)
print("\nMissing values by column:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nData types:\n", df.dtypes)

# Standardize text fields and category labels
text_cols = ["Customer_ID", "Customer_Name", "Gender", "Product_ID", "Product_Name", "Category"]
for col in text_cols:
    df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

df["Gender"] = df["Gender"].replace({
    "female": "Female", "F": "Female", "f": "Female",
    "male": "Male", "M": "Male", "m": "Male"
})

# Convert data types
for col in ["Age", "Quantity", "Price_Per_Unit"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"], errors="coerce")

# Flag impossible numeric values as missing
df.loc[~df["Age"].between(18, 100), "Age"] = np.nan
df.loc[df["Quantity"] <= 0, "Quantity"] = np.nan
df.loc[df["Price_Per_Unit"] <= 0, "Price_Per_Unit"] = np.nan

# Replace implausibly high prices with product reference prices.
reference_prices = {
    "P101": 799, "P102": 299, "P103": 149, "P104": 99, "P105": 399,
    "P106": 1299, "P107": 2499, "P108": 699, "P109": 1799, "P110": 1199
}
for idx, row in df.iterrows():
    pid = row["Product_ID"]
    price = row["Price_Per_Unit"]
    if pd.notna(pid) and pd.notna(price) and pid in reference_prices:
        if price > 10 * reference_prices[pid]:
            df.at[idx, "Price_Per_Unit"] = reference_prices[pid]

# Remove exact duplicate rows
duplicates_removed = int(df.duplicated().sum())
df = df.drop_duplicates()

# Remove records without a Customer_ID (needed for customer-level analysis)
missing_customer_id_removed = int(df["Customer_ID"].isna().sum())
df = df.dropna(subset=["Customer_ID"])

# Impute numeric values with median
for col in ["Age", "Quantity", "Price_Per_Unit"]:
    df[col] = df[col].fillna(df[col].median())

df["Age"] = df["Age"].round().astype(int)
df["Quantity"] = df["Quantity"].round().astype(int)

# Fill categorical/date missing values with mode
for col in ["Gender", "Purchase_Date"]:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mode().iloc[0])

# Recalculate Total_Amount from quantity × unit price
df["Total_Amount"] = (df["Quantity"] * df["Price_Per_Unit"]).round(2)

# Remove rows missing essential transaction/product fields
df = df.dropna(subset=["Transaction_ID", "Product_ID", "Product_Name", "Category"])

# Ensure final data types
df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"])
df["Age"] = df["Age"].astype(int)
df["Quantity"] = df["Quantity"].astype(int)
df["Price_Per_Unit"] = df["Price_Per_Unit"].astype(float)
df["Total_Amount"] = df["Total_Amount"].astype(float)

df.to_csv(OUTPUT_FILE, index=False)

print("\n=== CLEANING SUMMARY ===")
print("Exact duplicate rows removed:", duplicates_removed)
print("Rows removed for missing Customer_ID:", missing_customer_id_removed)
print("Final shape:", df.shape)
print("\nMissing values after cleaning:\n", df.isnull().sum())
print("\nFinal data types:\n", df.dtypes)
print("\nSaved cleaned dataset to:", OUTPUT_FILE)
