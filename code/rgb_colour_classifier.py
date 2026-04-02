import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("rgb_data.csv", header=None)

df.columns = [
    "timestamp",
    "session_id",
    "repeat_id",
    "distance_mm",
    "target_id",
    "current_uA",
    "sample_index",
    "ambient",
    "R",
    "G",
    "B",
    "total"
]

# -----------------------------
# Force numeric types
# -----------------------------
numeric_cols = [
    "timestamp",
    "session_id",
    "repeat_id",
    "distance_mm",
    "current_uA",
    "sample_index",
    "ambient",
    "R",
    "G",
    "B",
    "total"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Keep target_id as string
df["target_id"] = df["target_id"].astype(str)

# -----------------------------
# Drop invalid rows
# -----------------------------
df = df.dropna(subset=[
    "distance_mm", "current_uA", "ambient", "R", "G", "B", "target_id"
]).copy()

# Recompute total
df["total"] = df["R"] + df["G"] + df["B"]

# Keep only physically valid rows
df = df[
    (df["total"] > 0) &
    (df["current_uA"] > 0) &
    (df["distance_mm"] > 0)
].copy()

# -----------------------------
# Aggregate per repeat-condition
# -----------------------------
grouped = df.groupby(
    ["session_id", "repeat_id", "target_id", "distance_mm", "current_uA"],
    as_index=False
).mean(numeric_only=True)

grouped["total"] = grouped["R"] + grouped["G"] + grouped["B"]
grouped = grouped[
    (grouped["total"] > 0) &
    (grouped["current_uA"] > 0) &
    (grouped["distance_mm"] > 0)
].copy()

# -----------------------------
# Feature engineering
# -----------------------------
eps = 1e-9

grouped["log_total"] = np.log(grouped["total"] + eps)
grouped["log_current"] = np.log(grouped["current_uA"] + eps)
grouped["log_ambient"] = np.log(grouped["ambient"] + eps)

grouped["R_norm"] = grouped["R"] / (grouped["total"] + eps)
grouped["G_norm"] = grouped["G"] / (grouped["total"] + eps)
grouped["B_norm"] = grouped["B"] / (grouped["total"] + eps)

grouped["R_over_G"] = grouped["R"] / (grouped["G"] + eps)
grouped["R_over_B"] = grouped["R"] / (grouped["B"] + eps)
grouped["G_over_B"] = grouped["G"] / (grouped["B"] + eps)

grouped["log_R_over_G"] = np.log(grouped["R_over_G"] + eps)
grouped["log_R_over_B"] = np.log(grouped["R_over_B"] + eps)
grouped["log_G_over_B"] = np.log(grouped["G_over_B"] + eps)

# -----------------------------
# Select features
# -----------------------------
features = [
    "log_total",
    "log_current",
    # "log_ambient",   # uncomment if you want to include ambient
    "R_norm",
    "G_norm",
    "B_norm",
    "R_over_G",
    "R_over_B",
    "G_over_B",
    "log_R_over_G",
    "log_R_over_B",
    "log_G_over_B"
]

X = grouped[features]
y = grouped["target_id"]

# -----------------------------
# Encode labels
# -----------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# -----------------------------
# XGBoost classifier
# -----------------------------
model = xgb.XGBClassifier(
    n_estimators=1500,
    max_depth=8,
    learning_rate=0.05,
    min_child_weight=1,
    gamma=0.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",
    random_state=42,
    use_label_encoder=False
)

model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)

# -----------------------------
# Evaluation
# -----------------------------
acc = accuracy_score(y_test, y_pred)

print(f"Number of grouped samples: {len(grouped)}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Accuracy: {acc:.4f}")
print("\nClassification report:\n")
print(classification_report(y_test_labels, y_pred_labels))

# -----------------------------
# Save model + label encoder
# -----------------------------
model.save_model("color_classifier.json")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Saved model to color_classifier.json")
print("Saved label encoder to label_encoder.pkl")

# -----------------------------
# Confusion matrix
# -----------------------------
cm = confusion_matrix(
    y_test_labels,
    y_pred_labels,
    labels=label_encoder.classes_
)

print("\nConfusion Matrix:")
print(pd.DataFrame(
    cm,
    index=[f"true_{c}" for c in label_encoder.classes_],
    columns=[f"pred_{c}" for c in label_encoder.classes_]
))

fig, ax = plt.subplots(figsize=(8, 7))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
plt.title("Confusion Matrix - Color Classifier")
plt.tight_layout()
plt.show()

# -----------------------------
# Feature importance
# -----------------------------
importance = model.feature_importances_
order = np.argsort(importance)[::-1]

plt.figure(figsize=(10, 5))
plt.bar(np.array(features)[order], importance[order])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Importance")
plt.title("Feature Importance - Color Classifier")
plt.tight_layout()
plt.show()

# -----------------------------
# Predicted probabilities
# -----------------------------
probs = model.predict_proba(X_test)

probs_df = pd.DataFrame(
    probs,
    columns=[f"p_{cls}" for cls in label_encoder.classes_]
)

print("\nExample predicted probabilities:")
print(probs_df.head())