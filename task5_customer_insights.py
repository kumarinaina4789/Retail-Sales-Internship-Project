"""
PROJECT: Retail Sales Data Analysis
TASK 5: CUSTOMER INSIGHTS

Objectives:
1. Identify top-performing products.
2. Analyze customer purchasing patterns.
3. Identify repeat customers.
4. Generate product recommendations.

Libraries:
Pandas, Matplotlib, Seaborn
"""

# ------------------------------------------
# 1. IMPORT REQUIRED LIBRARIES
# ------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from itertools import combinations


# ------------------------------------------
# 2. LOAD TRANSFORMED DATASET
# ------------------------------------------

folder = Path(__file__).resolve().parent

file_path = folder / "retail_sales_transformed.csv"

if not file_path.exists():
    file_path = folder / "retail_sales_cleaned.csv"

if not file_path.exists():
    raise FileNotFoundError("Retail sales dataset not found.")

df = pd.read_csv(file_path)

print("\n========== DATASET LOADED ==========")

print(df.head())


# ------------------------------------------
# 3. CHECK REQUIRED COLUMNS
# ------------------------------------------

required_columns = [
    "Customer_ID",
    "Transaction_ID",
    "Product_Name",
    "Quantity",
    "Purchase_Date"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# Calculate Total Sales if necessary

if "Total_Sales" not in df.columns:

    if "Price_Per_Unit" in df.columns:

        df["Total_Sales"] = (
            df["Quantity"] * df["Price_Per_Unit"]
        )

    elif "Total_Amount" in df.columns:

        df["Total_Sales"] = df["Total_Amount"]

    else:

        raise ValueError("Sales column not found.")


# Convert purchase date

df["Purchase_Date"] = pd.to_datetime(
    df["Purchase_Date"],
    errors="coerce"
)

df = df.dropna(subset=["Purchase_Date"])


# ------------------------------------------
# 4. TOP-PERFORMING PRODUCTS
# ------------------------------------------

print("\n========== TOP-PERFORMING PRODUCTS ==========")

product_performance = df.groupby(
    "Product_Name"
).agg({

    "Quantity": "sum",

    "Total_Sales": "sum"

}).sort_values(
    by="Total_Sales",
    ascending=False
)

print(product_performance)

product_performance.to_csv(
    folder / "top_performing_products.csv"
)


# Top products by revenue

plt.figure(figsize=(11, 6))

product_performance["Total_Sales"].head(10).sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Products by Revenue")

plt.xlabel("Total Revenue")

plt.ylabel("Product Name")

plt.tight_layout()

plt.savefig(
    folder / "top_products_revenue_task5.png",
    dpi=300
)

plt.show()


# Top products by quantity

plt.figure(figsize=(11, 6))

product_performance["Quantity"].sort_values(
    ascending=False
).head(10).sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Products by Quantity Sold")

plt.xlabel("Quantity Sold")

plt.ylabel("Product Name")

plt.tight_layout()

plt.savefig(
    folder / "top_products_quantity_task5.png",
    dpi=300
)

plt.show()


# ------------------------------------------
# 5. PURCHASING PATTERN ANALYSIS
# ------------------------------------------

print("\n========== PURCHASING PATTERNS ==========")

# Extract day and month

df["Day"] = df["Purchase_Date"].dt.day_name()

df["Month"] = df["Purchase_Date"].dt.month_name()

df["Year"] = df["Purchase_Date"].dt.year


# Sales by day of week

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekly_sales = df.groupby("Day")[
    "Total_Sales"
].sum().reindex(day_order)

print("\nSales by Day:")

print(weekly_sales)

weekly_sales.to_csv(
    folder / "sales_by_weekday_task5.csv"
)


# Visualize weekday sales

plt.figure(figsize=(10, 5))

weekly_sales.plot(kind="bar")

plt.title("Customer Purchasing Patterns by Day")

plt.xlabel("Day of Week")

plt.ylabel("Total Sales")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    folder / "purchasing_by_weekday.png",
    dpi=300
)

plt.show()


# Monthly sales

monthly_sales = df.groupby(
    ["Year", "Purchase_Date"]
)["Total_Sales"].sum()

# Group using year and month

df["Month_Number"] = df["Purchase_Date"].dt.month

monthly_sales = df.groupby(
    ["Year", "Month_Number"]
)["Total_Sales"].sum()

print("\nMonthly Sales:")

print(monthly_sales)

monthly_sales.to_csv(
    folder / "monthly_purchasing_patterns.csv"
)


# Monthly sales visualization

monthly_sales.plot(
    kind="line",
    marker="o",
    figsize=(11, 5)
)

plt.title("Monthly Customer Purchasing Patterns")

plt.xlabel("Year and Month")

plt.ylabel("Total Sales")

plt.tight_layout()

plt.savefig(
    folder / "monthly_purchasing_patterns.png",
    dpi=300
)

plt.show()


# ------------------------------------------
# 6. PRODUCT CATEGORY PURCHASING PATTERNS
# ------------------------------------------

if "Category" in df.columns:

    category_month = pd.pivot_table(

        df,

        values="Total_Sales",

        index="Category",

        columns="Month",

        aggfunc="sum",

        fill_value=0

    )

    print("\n========== CATEGORY PURCHASING PATTERNS ==========")

    print(category_month)

    plt.figure(figsize=(12, 6))

    sns.heatmap(
        category_month,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu"
    )

    plt.title("Product Category Purchasing Patterns")

    plt.xlabel("Month")

    plt.ylabel("Product Category")

    plt.tight_layout()

    plt.savefig(
        folder / "category_purchasing_heatmap.png",
        dpi=300
    )

    plt.show()


# ------------------------------------------
# 7. REPEAT CUSTOMER ANALYSIS
# ------------------------------------------

print("\n========== REPEAT CUSTOMER ANALYSIS ==========")

customer_purchases = df.groupby(
    "Customer_ID"
).agg({

    "Transaction_ID": "nunique",

    "Total_Sales": "sum"

}).rename(columns={

    "Transaction_ID": "Total_Purchases",

    "Total_Sales": "Total_Spending"

})

customer_purchases["Customer_Type"] = (
    customer_purchases["Total_Purchases"]
    .apply(
        lambda x: "Repeat Customer"
        if x > 1
        else "One-Time Customer"
    )
)

print(customer_purchases)

customer_purchases.to_csv(
    folder / "repeat_customer_analysis.csv"
)


# Count customer types

customer_count = customer_purchases[
    "Customer_Type"
].value_counts()

print("\nCustomer Type Summary:")

print(customer_count)


# Repeat customer visualization

plt.figure(figsize=(8, 5))

customer_count.plot(
    kind="bar"
)

plt.title("Repeat Customers vs One-Time Customers")

plt.xlabel("Customer Type")

plt.ylabel("Number of Customers")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    folder / "repeat_customer_analysis.png",
    dpi=300
)

plt.show()


# ------------------------------------------
# 8. PRODUCT PREFERENCES OF REPEAT CUSTOMERS
# ------------------------------------------

repeat_ids = customer_purchases[
    customer_purchases["Total_Purchases"] > 1
].index

repeat_products = df[
    df["Customer_ID"].isin(repeat_ids)
].groupby("Product_Name")[
    "Quantity"
].sum().sort_values(ascending=False)

print("\nProducts Purchased by Repeat Customers:")

print(repeat_products)

repeat_products.to_csv(
    folder / "repeat_customer_products.csv"
)


# ------------------------------------------
# 9. PRODUCT RECOMMENDATIONS
# ------------------------------------------

print("\n========== PRODUCT RECOMMENDATIONS ==========")

# Find products purchased together
# using Transaction_ID

transaction_products = df.groupby(
    "Transaction_ID"
)["Product_Name"].apply(
    lambda x: sorted(set(x.dropna()))
)

pair_counts = {}

for products in transaction_products:

    if len(products) >= 2:

        for pair in combinations(products, 2):

            pair_counts[pair] = (
                pair_counts.get(pair, 0) + 1
            )


if pair_counts:

    recommendations = pd.DataFrame(

        [
            {
                "Product_1": pair[0],
                "Product_2": pair[1],
                "Times_Purchased_Together": count
            }

            for pair, count in pair_counts.items()
        ]

    )

    recommendations = recommendations.sort_values(
        by="Times_Purchased_Together",
        ascending=False
    )

    print(recommendations.head(10))

    recommendations.to_csv(
        folder / "product_recommendations.csv",
        index=False
    )

else:

    print(
        "No products purchased together were found."
    )

    print(
        "Each transaction may contain only one product. "
        "Multi-product transaction data is needed "
        "for reliable cross-selling recommendations."
    )


# ------------------------------------------
# 10. FINAL BUSINESS INSIGHTS
# ------------------------------------------

print("\n========== FINAL BUSINESS INSIGHTS ==========")

print(
    "Highest Revenue Product:",
    product_performance["Total_Sales"].idxmax()
)

print(
    "Most Purchased Product:",
    product_performance["Quantity"].idxmax()
)

print(
    "Best Sales Day:",
    weekly_sales.idxmax()
)

print(
    "Total Customers:",
    len(customer_purchases)
)

print(
    "Repeat Customers:",
    customer_count.get("Repeat Customer", 0)
)

print(
    "One-Time Customers:",
    customer_count.get("One-Time Customer", 0)
)

print("\nTASK 5 COMPLETED SUCCESSFULLY!")