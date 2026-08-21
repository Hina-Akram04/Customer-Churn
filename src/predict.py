
import json
import argparse
import joblib
import numpy as np
import pandas as pd

NUMERIC_COLS = ["Tenure_Months", "Monthly_Charges", "Total_Charges", "CLTV",
                 "Customer_Lifetime_Months", "Average_Monthly_Spend"]

READABLE_NAMES = {
    "Tenure_Months": "short customer tenure",
    "Monthly_Charges": "high monthly charges",
    "Contract_One Year": "not on a long-term contract",
    "Contract_Two Year": "not on a long-term contract",
    "Internet_Service_Fiber Optic": "fiber optic service (higher-cost tier)",
    "Payment_Method_Electronic Check": "electronic check payment method",
    "Tech_Support": "no tech support add-on",
    "Online_Security": "no online security add-on",
    "Service_Count": "few add-on services",
    "High_Monthly_Charge": "monthly charge in the top 25%",
    "CLTV": "low customer lifetime value",
}


def risk_level(prob):
    if prob >= 0.6:
        return "HIGH"
    if prob >= 0.3:
        return "MEDIUM"
    return "LOW"


def top_factors_batch(X, model, feature_names, train_means, top_n=4):
    
    importances = pd.Series(model.feature_importances_, index=feature_names)
    deviation = (X[feature_names] - train_means).abs()
    norm_dev = deviation / (train_means.abs() + 1e-6)
    score = norm_dev.mul(importances, axis=1)

    top_idx = np.argsort(-score.values, axis=1)[:, :top_n]
    feature_arr = np.array(feature_names)

    factors_list = []
    for idx_row in top_idx:
        labels = []
        for feat in feature_arr[idx_row]:
            label = READABLE_NAMES.get(feat, feat.replace("_", " "))
            if label not in labels:
                labels.append(label)
        factors_list.append(", ".join(labels))
    return factors_list


def predict_batch(df):
    model = joblib.load("models/churn_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    with open("models/feature_names.json") as f:
        feature_names = json.load(f)
    with open("models/best_model_name.json") as f:
        meta = json.load(f)

    reference = pd.read_csv("data/processed/churn_features.csv")
    train_means = reference[feature_names].mean()

    X = df[feature_names].copy()
    X_scaled = X.copy()
    if meta["uses_scaling"]:
        X_scaled[NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS])
    probs = model.predict_proba(X_scaled)[:, 1]

    risk_levels = [risk_level(p) for p in probs]
    factors = top_factors_batch(X, model, feature_names, train_means)
    customer_ids = df["CustomerID"] if "CustomerID" in df.columns else [f"row_{i}" for i in range(len(df))]

    return pd.DataFrame({
        "CustomerID": list(customer_ids),
        "Churn_Probability": np.round(probs, 4),
        "Risk_Level": risk_levels,
        "Top_Risk_Factors": factors,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    input_df = pd.read_csv(args.input)
    result = predict_batch(input_df)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} predictions -> {args.output}")
