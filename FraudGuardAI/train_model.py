
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    roc_auc_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "creditcard.csv")
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.pkl")
RESULTS_PATH = os.path.join(BASE_DIR, "pr_results.csv")

# Load dataset
df = pd.read_csv(DATASET_PATH)

if "Class" not in df.columns:
    raise ValueError("Dataset must contain a 'Class' column.")

df = df.dropna(subset=["Class"]).copy()
df["Class"] = pd.to_numeric(df["Class"], errors="raise").astype(int)

X = df.drop(columns=["Class"])
y = df["Class"]

if y.nunique() != 2:
    raise ValueError("Dataset must contain both legitimate (0) and fraud (1) classes.")

# Convert features to numeric and handle missing values
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([float("inf"), float("-inf")], float("nan"))
X = X.fillna(X.median()).fillna(0)

# Keep the test set separate from model training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Scaling and model in one pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

print("Training Random Forest model...")
model.fit(X_train, y_train)

# Fraud probability scores
y_scores = model.predict_proba(X_test)[:, 1]
y_pred = (y_scores >= 0.5).astype(int)

# Actual evaluation results
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
average_precision = average_precision_score(y_test, y_scores)
roc_auc = roc_auc_score(y_test, y_scores)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

# Save model and PR curve data
joblib.dump(model, MODEL_PATH)

pd.DataFrame({
    "recall": recall,
    "precision": precision
}).to_csv(RESULTS_PATH, index=False)

print("\nMODEL EVALUATION")
print("----------------")
print(f"Test transactions: {len(y_test)}")
print(f"Actual fraud cases: {int((y_test == 1).sum())}")
print(f"Average Precision (AP): {average_precision:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("\nConfusion Matrix [ [TN, FP], [FN, TP] ]:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print(f"\nModel saved: {MODEL_PATH}")
print(f"PR results saved: {RESULTS_PATH}")
