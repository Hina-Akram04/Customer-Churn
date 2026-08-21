import joblib
import json
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

DATA_PATH = "data/processed/churn_features.csv"
NUMERIC_COLS = ["Tenure_Months", "Monthly_Charges", "Total_Charges", "CLTV",
                 "Customer_Lifetime_Months", "Average_Monthly_Spend"]

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Churn_Value"])
y = df["Churn_Value"]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
X_test_scaled[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

churn_rate = y_train.mean()

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced",
                                             random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss",
                              random_state=42, scale_pos_weight=(1 - churn_rate) / churn_rate),
}

results = {}
fitted = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "ROC-AUC": roc_auc_score(y_test, probs),
    }
    fitted[name] = model
    print(f"{name:22s} -> " + ", ".join(f"{k}: {v:.4f}" for k, v in results[name].items()))

comparison = pd.DataFrame(results).T.round(4)
comparison.to_csv("reports/model_comparison.csv")
print(f"\nSaved comparison table to reports/model_comparison.csv")

best_name = comparison["ROC-AUC"].idxmax()
best_model = fitted[best_name]
print(f"\nBest model: {best_name} (ROC-AUC = {comparison.loc[best_name, 'ROC-AUC']:.4f})")

joblib.dump(best_model, "models/churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
with open("models/feature_names.json", "w") as f:
    json.dump(feature_names, f)
with open("models/best_model_name.json", "w") as f:
    json.dump({"best_model": best_name, "uses_scaling": best_name == "Logistic Regression"}, f)

print("Saved model to models/churn_model.pkl")
