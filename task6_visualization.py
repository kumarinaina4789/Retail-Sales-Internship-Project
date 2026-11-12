# ==========================================
# TASK 6: DATA VISUALIZATION
# RETAIL SALES DATA ANALYSIS
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

print("\n========== TASK 6: DATA VISUALIZATION ==========\n")

df = pd.read_csv("retail_sales_transformed.csv")

# Convert date column into datetime
df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"])

print("Dataset Loaded Successfully!")
print(df.head())


# ==========================================
# STEP 2: CREATE OUTPUT FOLDER
# ==========================================

output_folder = "task6_visualizations"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# ==========================================
# STEP 3: HISTOGRAM
# CUSTOMER SPENDING DISTRIBUTION
# ==========================================

customer_spending = df.groupby("Customer_ID")["Total_Sales"].sum()

plt.figure(figsize=(9, 5))

plt.hist(customer_spending, bins=10, edgecolor="black")

plt.title("Customer Spending Distribution")
plt.xlabel("Total Customer Spending")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(output_folder + "/customer_spending_histogram.png")

plt.show()

print("\nHistogram Created Successfully!")


# ==========================================
# STEP 4: BAR CHART
# TOP SELLING PRODUCTS
# ==========================================

product_sales = df.groupby("Product_Name")["Quantity"].sum()

product_sales = product_sales.sort_values(ascending=False)

plt.figure(figsize=(10, 6))

product_sales.plot(kind="bar")

plt.title("Top Selling Products")
plt.xlabel("Product Name")
plt.ylabel("Total Quantity Sold")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig(output_folder + "/top_selling_products.png")

plt.show()

print("\nTop Selling Products Chart Created!")


# ==========================================
# STEP 5: BAR CHART
# PRODUCT CATEGORY SALES
# ==========================================

category_column = next(
    (col for col in ["Category", "Product_Category"] if col in df.columns),
    None
)

if category_column:

    category_sales = df.groupby(category_column)["Total_Sales"].sum()

    category_sales = category_sales.sort_values(ascending=False)

    plt.figure(figsize=(9, 5))

    category_sales.plot(kind="bar")

    plt.title("Sales by Product Category")
    plt.xlabel("Product Category")
    plt.ylabel("Total Sales")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(output_folder + "/category_sales.png")

    plt.show()

    print("\nCategory Sales Chart Created!")

else:
    print("\nCategory column not found. Category chart skipped.")


# ==========================================
# STEP 6: LINE PLOT
# MONTHLY SALES TREND
# ==========================================

monthly_sales = df.groupby(
    [df["Purchase_Date"].dt.year,
     df["Purchase_Date"].dt.month]
)["Total_Sales"].sum()

monthly_sales.index = [
    f"{year}-{month:02d}"
    for year, month in monthly_sales.index
]

plt.figure(figsize=(10, 5))

plt.plot(monthly_sales.index, monthly_sales.values,
         marker="o", linewidth=2)

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)

plt.grid(True)

plt.tight_layout()

plt.savefig(output_folder + "/monthly_sales_trend.png")

plt.show()

print("\nMonthly Sales Trend Created!")


# ==========================================
# STEP 7: LINE PLOT
# WEEKLY SALES TREND
# ==========================================

weekly_sales = df.groupby("Week_Number")["Total_Sales"].sum()

plt.figure(figsize=(10, 5))

plt.plot(weekly_sales.index, weekly_sales.values,
         marker="o", linewidth=2)

plt.title("Weekly Sales Trend")
plt.xlabel("Week Number")
plt.ylabel("Total Sales")

plt.grid(True)

plt.tight_layout()

plt.savefig(output_folder + "/weekly_sales_trend.png")

plt.show()

print("\nWeekly Sales Trend Created!")


# ==========================================
# STEP 8: PIE CHART
# CUSTOMER SEGMENT DISTRIBUTION
# ==========================================

segments = pd.read_csv("customer_segmentation.csv")

# Match customer segments with transactions
purchase_segments = df.merge(
    segments[["Customer_ID", "Customer_Segment"]],
    on="Customer_ID",
    how="left"
)

# Count purchases made by each segment
segment_purchases = purchase_segments[
    "Customer_Segment"
].value_counts()

plt.figure(figsize=(8, 6))

plt.pie(
    segment_purchases,
    labels=segment_purchases.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Percentage of Purchases by Customer Segment")

plt.tight_layout()

plt.savefig(output_folder + "/customer_segment_pie.png")

plt.show()

print("\nCustomer Segment Pie Chart Created!")


# ==========================================
# STEP 9: HEATMAP
# DAY-WISE AND WEEK-WISE SALES
# ==========================================

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

heatmap_data = df.pivot_table(
    values="Total_Sales",
    index="Week_Number",
    columns="Day_of_Week",
    aggfunc="sum",
    fill_value=0
)

heatmap_data = heatmap_data.reindex(columns=day_order)

plt.figure(figsize=(11, 8))

sns.heatmap(
    heatmap_data,
    cmap="YlGnBu",
    linewidths=0.3
)

plt.title("Weekly and Daily Sales Heatmap")
plt.xlabel("Day of Week")
plt.ylabel("Week Number")

plt.tight_layout()

plt.savefig(output_folder + "/sales_heatmap.png")

plt.show()

print("\nSales Heatmap Created Successfully!")


# ==========================================
# STEP 10: FINAL SUMMARY
# ==========================================

print("\n======================================")
print("       TASK 6 COMPLETED SUCCESSFULLY")
print("======================================")

print("\nGenerated Visualizations:")

print("1. Customer Spending Histogram")
print("2. Top Selling Products Bar Chart")
print("3. Product Category Sales Bar Chart (if available)")
print("4. Monthly Sales Trend")
print("5. Weekly Sales Trend")
print("6. Customer Segment Pie Chart")
print("7. Daily and Weekly Sales Heatmap")

print("\nAll charts are saved inside:", output_folder)

print("\nThank You!")