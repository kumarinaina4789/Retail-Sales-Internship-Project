"""
PROJECT: Retail Sales Data Analysis
TASK 4: CUSTOMER SEGMENTATION

Objective:
To segment customers based on their purchasing behaviour
using Frequency, Monetary analysis and K-Means Clustering.

Libraries:
1. Pandas
2. Matplotlib
3. Scikit-learn
"""

# ------------------------------------------
# 1. IMPORT REQUIRED LIBRARIES
# ------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ------------------------------------------
# 2. LOAD TRANSFORMED DATASET
# ------------------------------------------

folder = Path(__file__).resolve().parent

file_path = folder / "retail_sales_transformed.csv"

if not file_path.exists():
    file_path = folder / "retail_sales_cleaned.csv"

if not file_path.exists():
    raise FileNotFoundError(
        "Dataset not found. Keep the CSV file in the project folder."
    )

df = pd.read_csv(file_path)

print("\n========== DATASET LOADED SUCCESSFULLY ==========")
print(df.head())


# ------------------------------------------
# 3. CHECK REQUIRED COLUMNS
# ------------------------------------------

required_columns = ["Customer_ID", "Transaction_ID"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Required column '{column}' is missing.")

# Calculate Total_Sales if it does not exist

if "Total_Sales" not in df.columns:

    if "Quantity" in df.columns and "Price_Per_Unit" in df.columns:

        df["Total_Sales"] = (
            df["Quantity"] * df["Price_Per_Unit"]
        )

    elif "Total_Amount" in df.columns:

        df["Total_Sales"] = df["Total_Amount"]

    else:
        raise ValueError("Sales amount columns are missing.")


# ------------------------------------------
# 4. CALCULATE CUSTOMER FREQUENCY
# ------------------------------------------

customer_frequency = df.groupby("Customer_ID")[
    "Transaction_ID"
].nunique()

print("\n========== CUSTOMER FREQUENCY ==========")
print(customer_frequency.head())


# ------------------------------------------
# 5. CALCULATE CUSTOMER MONETARY VALUE
# ------------------------------------------

customer_monetary = df.groupby("Customer_ID")[
    "Total_Sales"
].sum()

print("\n========== CUSTOMER MONETARY VALUE ==========")
print(customer_monetary.head())


# ------------------------------------------
# 6. CREATE CUSTOMER SEGMENTATION DATAFRAME
# ------------------------------------------

customer_data = pd.DataFrame({

    "Frequency": customer_frequency,

    "Monetary": customer_monetary

})

customer_data = customer_data.reset_index()

print("\n========== CUSTOMER ANALYSIS ==========")
print(customer_data.head())


# ------------------------------------------
# 7. APPLY STANDARD SCALING
# ------------------------------------------

features = customer_data[["Frequency", "Monetary"]]

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)


# ------------------------------------------
# 8. APPLY K-MEANS CLUSTERING
# ------------------------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

customer_data["Cluster"] = kmeans.fit_predict(
    scaled_features
)


# ------------------------------------------
# 9. IDENTIFY CUSTOMER SEGMENTS
# ------------------------------------------

cluster_summary = customer_data.groupby("Cluster").agg({

    "Frequency": "mean",

    "Monetary": "mean"

})

# Calculate combined standardized score
# to identify the highest-value group.

cluster_summary["Score"] = (
    scaler.transform(cluster_summary[["Frequency", "Monetary"]])
    .mean(axis=1)
)

# Highest combined score = High Value
high_value_cluster = cluster_summary["Score"].idxmax()

remaining_clusters = cluster_summary.drop(
    high_value_cluster
)

# Highest frequency among remaining clusters
# = Frequent Buyers

frequent_cluster = remaining_clusters[
    "Frequency"
].idxmax()

# Remaining cluster = Infrequent / Low Value

low_value_cluster = [
    cluster for cluster in cluster_summary.index
    if cluster not in [high_value_cluster, frequent_cluster]
][0]

segment_names = {

    high_value_cluster: "High Value Customers",

    frequent_cluster: "Frequent Buyers",

    low_value_cluster: "Infrequent / Low Value Customers"

}

customer_data["Customer_Segment"] = (
    customer_data["Cluster"].map(segment_names)
)


# ------------------------------------------
# 10. DISPLAY CUSTOMER SEGMENTS
# ------------------------------------------

print("\n========== CUSTOMER SEGMENTATION RESULTS ==========")

print(
    customer_data[
        [
            "Customer_ID",
            "Frequency",
            "Monetary",
            "Customer_Segment"
        ]
    ].to_string(index=False)
)


# ------------------------------------------
# 11. CREATE SEGMENT SUMMARY
# ------------------------------------------

segment_summary = customer_data.groupby(
    "Customer_Segment"
).agg({

    "Customer_ID": "count",

    "Frequency": "mean",

    "Monetary": "mean"

}).rename(columns={

    "Customer_ID": "Number_of_Customers",

    "Frequency": "Average_Purchases",

    "Monetary": "Average_Spending"

})

print("\n========== CUSTOMER SEGMENT SUMMARY ==========")

print(segment_summary)


# ------------------------------------------
# 12. SAVE CUSTOMER SEGMENTATION RESULTS
# ------------------------------------------

customer_data.to_csv(
    folder / "customer_segmentation.csv",
    index=False
)

segment_summary.to_csv(
    folder / "customer_segment_summary.csv"
)

print("\nCustomer segmentation files saved successfully!")


# ------------------------------------------
# 13. VISUALIZE CUSTOMER SEGMENTS
# ------------------------------------------

plt.figure(figsize=(10, 6))

for segment in customer_data["Customer_Segment"].unique():

    segment_data = customer_data[
        customer_data["Customer_Segment"] == segment
    ]

    plt.scatter(

        segment_data["Frequency"],

        segment_data["Monetary"],

        label=segment,

        s=80

    )

plt.title("Customer Segmentation Using K-Means Clustering")

plt.xlabel("Purchase Frequency")

plt.ylabel("Total Customer Spending")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    folder / "customer_segmentation.png",
    dpi=300
)

plt.show()


# ------------------------------------------
# 14. FINAL PROJECT SUMMARY
# ------------------------------------------

print("\n========== FINAL SUMMARY ==========")

print("Total Customers:", len(customer_data))

print(
    "High Value Customers:",
    len(customer_data[
        customer_data["Customer_Segment"] == "High Value Customers"
    ])
)

print(
    "Frequent Buyers:",
    len(customer_data[
        customer_data["Customer_Segment"] == "Frequent Buyers"
    ])
)

print(
    "Infrequent / Low Value Customers:",
    len(customer_data[
        customer_data["Customer_Segment"] == "Infrequent / Low Value Customers"
    ])
)

print("\nTASK 4 COMPLETED SUCCESSFULLY!")