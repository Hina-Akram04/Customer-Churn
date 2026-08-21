import pandas as pd

IN_PATH = "data/processed/churn_clean.csv"
OUT_PATH = "data/processed/churn_features.csv"

df = pd.read_csv(IN_PATH)
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# how long they've been a customer, in plain business terms
df["Customer_Lifetime_Months"] = df["Tenure_Months"]

# average spend per month, safe against tenure = 0
df["Average_Monthly_Spend"] = df["Total_Charges"] / df["Tenure_Months"].replace(0, 1)

# how many of the 6 add-on services they use
service_cols = ["Online_Security", "Online_Backup", "Device_Protection",
                 "Tech_Support", "Streaming_TV", "Streaming_Movies"]
df["Service_Count"] = df[service_cols].sum(axis=1)

# top 25% spenders flagged as high monthly charge
df["High_Monthly_Charge"] = (df["Monthly_Charges"] > df["Monthly_Charges"].quantile(0.75)).astype(int)

df["Tenure_Group"] = pd.cut(df["Tenure_Months"], bins=[-1, 12, 48, 1000],
                             labels=["New", "Established", "Loyal"])
df = pd.get_dummies(df, columns=["Tenure_Group"], drop_first=True)
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

print(f"After feature engineering: {df.shape[1]} columns")

df.to_csv(OUT_PATH, index=False)
print(f"Saved to {OUT_PATH}")
