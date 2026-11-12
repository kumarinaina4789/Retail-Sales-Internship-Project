
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set project folder
folder = Path(__file__).resolve().parent

# =====================================
# TASK 2: DATA TRANSFORMATION
# =====================================

# Import cleaned dataset from Task 1
df = pd.read_csv(folder / "retail_sales_cleaned.csv")

print("ORIGINAL CLEANED DATASET")
print(df.head())

# Convert numerical columns
df["Quantity"] = pd.to_numeric(df["Quantity"])
df["Price_Per_Unit"] = pd.to_numeric(df["Price_Per_Unit"])

# Convert date column into datetime
df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"])

# Create Total Sales column
df["Total_Sales"] = (
    df["Quantity"] * df["Price_Per_Unit"]
)

# Extract date features
df["Day_of_Week"] = df["Purchase_Date"].dt.day_name()

df["Month"] = df["Purchase_Date"].dt.month_name()

df["Year"] = df["Purchase_Date"].dt.year

df["Week_Number"] = (
    df["Purchase_Date"].dt.isocalendar().week
)

# Save transformed dataset
df.to_csv(folder / "retail_sales_transformed.csv", index=False)

print("\nTASK 2 COMPLETED SUCCESSFULLY!")
print(df.head())


# =====================================
# TASK 3: EXPLORATORY DATA ANALYSIS
# =====================================

print("\nTASK 3: EXPLORATORY DATA ANALYSIS")

# 1. DESCRIPTIVE STATISTICS

print("\nDESCRIPTIVE STATISTICS")

print(df[[
    "Total_Sales",
    "Quantity",
    "Price_Per_Unit"
]].describe())

# Save statistics
df[[
    "Total_Sales",
    "Quantity",
    "Price_Per_Unit"
]].describe().to_csv(
    folder / "eda_descriptive_statistics.csv"
)


# 2. CUSTOMER SPENDING HISTOGRAM

customer_spending = df.groupby(
    "Customer_ID"
)["Total_Sales"].sum()

plt.figure(figsize=(8, 5))

plt.hist(customer_spending, bins=12, edgecolor="black")

plt.title("Customer Spending Distribution")
plt.xlabel("Total Spending")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig(folder / "customer_spending.png")
plt.show()
plt.close()


# 3. TOP-SELLING PRODUCTS BY QUANTITY

top_quantity = df.groupby(
    "Product_Name"
)["Quantity"].sum().sort_values(ascending=False)

plt.figure(figsize=(9, 5))

top_quantity.plot(kind="bar")

plt.title("Top Selling Products by Quantity")
plt.xlabel("Product Name")
plt.ylabel("Quantity Sold")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(folder / "top_products_quantity.png")
plt.show()
plt.close()


# 4. TOP PRODUCTS BY REVENUE

top_revenue = df.groupby(
    "Product_Name"
)["Total_Sales"].sum().sort_values(ascending=False)

plt.figure(figsize=(9, 5))

top_revenue.plot(kind="bar")

plt.title("Top Products by Revenue")
plt.xlabel("Product Name")
plt.ylabel("Total Sales")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(folder / "top_products_revenue.png")
plt.show()
plt.close()


# 5. MONTHLY SALES TREND

monthly_sales = df.groupby(
    df["Purchase_Date"].dt.month
)["Total_Sales"].sum()

plt.figure(figsize=(9, 5))

monthly_sales.plot(kind="line", marker="o")

plt.title("Monthly Sales Trend")
plt.xlabel("Month Number")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.savefig(folder / "monthly_sales_trend.png")
plt.show()
plt.close()


# 6. WEEKLY SALES TREND

weekly_sales = df.groupby(
    df["Purchase_Date"].dt.isocalendar().week
)["Total_Sales"].sum()

plt.figure(figsize=(9, 5))

weekly_sales.plot(kind="line", marker="o")

plt.title("Weekly Sales Trend")
plt.xlabel("Week Number")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.savefig(folder / "weekly_sales_trend.png")
plt.show()
plt.close()


# 7. PURCHASING PATTERNS BY DAY

day_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

day_sales = df.groupby(
    "Day_of_Week"
)["Total_Sales"].sum().reindex(day_order, fill_value=0)

plt.figure(figsize=(9, 5))

day_sales.plot(kind="bar")

plt.title("Sales by Day of Week")
plt.xlabel("Day")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.savefig(folder / "sales_by_day.png")
plt.show()
plt.close()


# 8. HEATMAP: MONTH AND DAY

pivot = df.pivot_table(
    index="Month",
    columns="Day_of_Week",
    values="Total_Sales",
    aggfunc="sum",
    fill_value=0
)

month_order = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

pivot = pivot.reindex(
    index=month_order,
    columns=day_order,
    fill_value=0
)

plt.figure(figsize=(10, 6))

plt.imshow(pivot, aspect="auto")
plt.colorbar(label="Total Sales")

plt.xticks(range(7), day_order, rotation=45)
plt.yticks(range(12), month_order)

plt.title("Purchasing Pattern Heatmap")
plt.xlabel("Day of Week")
plt.ylabel("Month")

plt.tight_layout()
plt.savefig(folder / "purchasing_heatmap.png")
plt.show()
plt.close()


# =====================================
# FINAL RESULTS
# =====================================

print("\nTOP SELLING PRODUCT BY QUANTITY:")
print(top_quantity.head())

print("\nTOP SELLING PRODUCT BY REVENUE:")
print(top_revenue.head())

print("\nHIGHEST SALES DAY:")
print(day_sales.idxmax())

print("\nHIGHEST SALES MONTH:")
print(monthly_sales.idxmax())

print("\nALL TASKS COMPLETED SUCCESSFULLY!")

