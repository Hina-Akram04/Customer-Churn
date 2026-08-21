import json
import joblib
import shap
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/churn_features.csv"

model = joblib.load("models/churn_model.pkl")
with open("models/feature_names.json") as f:
    feature_names = json.load(f)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Churn_Value"])
y = df["Churn_Value"]
_, X_test, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

importances = pd.Series(model.feature_importances_, index=feature_names)
importances = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(8, 6))
importances.sort_values().plot(kind="barh", color="#4C72B0")
plt.title("Top Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("reports/figures/feature_importance.png", dpi=150)
plt.close()
print("Saved -> reports/figures/feature_importance.png")

#SHAP summary plot 
sample = X_test.sample(min(300, len(X_test)), random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)
values = shap_values[1] if isinstance(shap_values, list) else shap_values

plt.figure()
shap.summary_plot(values, sample, show=False)
plt.tight_layout()
plt.savefig("reports/figures/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved to reports/figures/shap_summary.png")
