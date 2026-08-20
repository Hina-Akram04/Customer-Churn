import pandas as pd
from sklearn.preprocessing import LabelEncoder

RAW_PATH = "data/raw/customer_churn.xlsx"
OUT_PATH = "data/processed/churn_clean.csv"

df = pd.read_excel(RAW_PATH)
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Total Charges is stored as text and has blanks for tenure=0 customers
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0)

df = df.drop_duplicates()

# keep CustomerID in its own file, row-aligned with the processed data,
# so predictions can be matched back to a customer later
df[["CustomerID"]].to_csv("data/processed/customer_ids.csv", index=False)

# columns with no predictive value, or that leak the target
drop_cols = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude",   
    "Churn Label",                          # duplicate of Churn Value
    "Churn Reason",                         # free text, mostly missing, not used for modeling
    "Churn Score",                          
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

print(f"After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Churn rate: {df['Churn Value'].mean():.2%}")

binary_cols = [
    "Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service",
    "Multiple Lines", "Online Security", "Online Backup", "Device Protection",
    "Tech Support", "Streaming TV", "Streaming Movies", "Paperless Billing",
]
for col in binary_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

df = pd.get_dummies(df, columns=["Internet Service", "Contract", "Payment Method"], drop_first=True)
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

df.columns = [c.replace(" ", "_") for c in df.columns]

df.to_csv(OUT_PATH, index=False)
print(f"Saved to {OUT_PATH}")
