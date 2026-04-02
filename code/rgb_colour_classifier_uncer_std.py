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

df["target_id"] = df["target_id"].astype(str)

# -----------------------------
# Clean data
# -----------------------------
df = df.dropna(subset=[
    "distance_mm", "current_uA", "ambient", "R", "G", "B", "target_id"
]).copy()

df["total"] = df["R"] + df["G"] + df["B"]

df = df[
    (df["total"] > 0) &
    (df["current_uA"] > 0) &
    (df["distance_mm"] > 0)
].copy()

# -----------------------------
# Aggregate mean + std per repeat-condition
# -----------------------------
group_cols = ["session_id", "repeat_id", "target_id", "distance_mm", "current_uA"]

agg = df.groupby(group_cols).agg(
    ambient_mean=("ambient", "mean"),
    ambient_std=("ambient", "std"),
    ambient_n=("ambient", "count"),

    R_mean=("R", "mean"),
    R_std=("R", "std"),
    R_n=("R", "count"),

    G_mean=("G", "mean"),
    G_std=("G", "std"),
    G_n=("G", "count"),

    B_mean=("B", "mean"),
    B_std=("B", "std"),
    B_n=("B", "count")
).reset_index()

# Replace NaN std values (can happen if only one sample exists)
for col in ["ambient_std", "R_std", "G_std", "B_std"]:
    agg[col] = agg[col].fillna(0.0)

# -----------------------------
# Standard uncertainty of the mean
# -----------------------------
eps = 1e-9

agg["u_ambient"] = agg["ambient_std"] / np.sqrt(agg["ambient_n"].clip(lower=1))
agg["u_R"] = agg["R_std"] / np.sqrt(agg["R_n"].clip(lower=1))
agg["u_G"] = agg["G_std"] / np.sqrt(agg["G_n"].clip(lower=1))
agg["u_B"] = agg["B_std"] / np.sqrt(agg["B_n"].clip(lower=1))

# -----------------------------
# Derived means
# -----------------------------
agg["ambient"] = agg["ambient_mean"]
agg["R"] = agg["R_mean"]
agg["G"] = agg["G_mean"]
agg["B"] = agg["B_mean"]
agg["total"] = agg["R"] + agg["G"] + agg["B"]

agg = agg[
    (agg["total"] > 0) &
    (agg["current_uA"] > 0) &
    (agg["distance_mm"] > 0)
].copy()

# -----------------------------
# Uncertainty propagation
# -----------------------------
# Total uncertainty (RSS)
agg["u_total"] = np.sqrt(agg["u_R"]**2 + agg["u_G"]**2 + agg["u_B"]**2)

# Logs
agg["log_total"] = np.log(agg["total"] + eps)
agg["log_current"] = np.log(agg["current_uA"] + eps)
agg["log_ambient"] = np.log(agg["ambient"] + eps)

agg["u_log_total"] = agg["u_total"] / (agg["total"] + eps)
agg["u_log_ambient"] = agg["u_ambient"] / (agg["ambient"] + eps)

# Normalized channels
agg["R_norm"] = agg["R"] / (agg["total"] + eps)
agg["G_norm"] = agg["G"] / (agg["total"] + eps)
agg["B_norm"] = agg["B"] / (agg["total"] + eps)

agg["u_R_norm"] = agg["R_norm"] * np.sqrt(
    (agg["u_R"] / (agg["R"] + eps))**2 +
    (agg["u_total"] / (agg["total"] + eps))**2
)

agg["u_G_norm"] = agg["G_norm"] * np.sqrt(
    (agg["u_G"] / (agg["G"] + eps))**2 +
    (agg["u_total"] / (agg["total"] + eps))**2
)

agg["u_B_norm"] = agg["B_norm"] * np.sqrt(
    (agg["u_B"] / (agg["B"] + eps))**2 +
    (agg["u_total"] / (agg["total"] + eps))**2
)

# Ratios
agg["R_over_G"] = agg["R"] / (agg["G"] + eps)
agg["R_over_B"] = agg["R"] / (agg["B"] + eps)
agg["G_over_B"] = agg["G"] / (agg["B"] + eps)

agg["u_R_over_G"] = agg["R_over_G"] * np.sqrt(
    (agg["u_R"] / (agg["R"] + eps))**2 +
    (agg["u_G"] / (agg["G"] + eps))**2
)

agg["u_R_over_B"] = agg["R_over_B"] * np.sqrt(
    (agg["u_R"] / (agg["R"] + eps))**2 +
    (agg["u_B"] / (agg["B"] + eps))**2
)

agg["u_G_over_B"] = agg["G_over_B"] * np.sqrt(
    (agg["u_G"] / (agg["G"] + eps))**2 +
    (agg["u_B"] / (agg["B"] + eps))**2
)

# Log-ratios
agg["log_R_over_G"] = np.log(agg["R_over_G"] + eps)
agg["log_R_over_B"] = np.log(agg["R_over_B"] + eps)
agg["log_G_over_B"] = np.log(agg["G_over_B"] + eps)

agg["u_log_R_over_G"] = np.sqrt(
    (agg["u_R"] / (agg["R"] + eps))**2 +
    (agg["u_G"] / (agg["G"] + eps))**2
)

agg["u_log_R_over_B"] = np.sqrt(
    (agg["u_R"] / (agg["R"] + eps))**2 +
    (agg["u_B"] / (agg["B"] + eps))**2
)

agg["u_log_G_over_B"] = np.sqrt(
    (agg["u_G"] / (agg["G"] + eps))**2 +
    (agg["u_B"] / (agg["B"] + eps))**2
)

# -----------------------------
# Select features
# -----------------------------
features = [
    # Original features
    "log_total",
    "log_current",
    # "log_ambient",
    "R_norm",
    "G_norm",
    "B_norm",
    "R_over_G",
    "R_over_B",
    "G_over_B",
    "log_R_over_G",
    "log_R_over_B",
    "log_G_over_B",

    # Uncertainty-aware features
    "u_total",
    "u_log_total",
    # "u_log_ambient",
    "u_R_norm",
    "u_G_norm",
    "u_B_norm",
    "u_R_over_G",
    "u_R_over_B",
    "u_G_over_B",
    "u_log_R_over_G",
    "u_log_R_over_B",
    "u_log_G_over_B"
]

X = agg[features]
y = agg["target_id"]

# Encode class labels
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
    gamma=0.00,
    reg_alpha=0.0,
    reg_lambda=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",
    random_state=42
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

print(f"Number of grouped samples: {len(agg)}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Accuracy: {acc:.4f}")
print("\nClassification report:\n")
print(classification_report(y_test_labels, y_pred_labels))

# -----------------------------
# Save model + label encoder + feature list
# -----------------------------
model.save_model("color_classifier_uncertainty.json")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# with open("feature_columns.pkl", "wb") as f:
#     pickle.dump(features, f)

print("Saved model to color_classifier_uncertainty.json")
print("Saved label encoder to label_encoder.pkl")
print("Saved feature list to feature_columns.pkl")

# -----------------------------
# Confusion matrix
# -----------------------------
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)

fig, ax = plt.subplots(figsize=(8, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
plt.title("Confusion Matrix - Uncertainty-Aware Color Classifier")
plt.tight_layout()
plt.show()

# -----------------------------
# Feature importance
# -----------------------------
importance = model.feature_importances_
order = np.argsort(importance)[::-1]

plt.figure(figsize=(12, 6))
plt.bar(np.array(features)[order], importance[order])
plt.xticks(rotation=60, ha="right")
plt.ylabel("Importance")
plt.title("Feature Importance - Uncertainty-Aware Color Classifier")
plt.tight_layout()
plt.show()

# -----------------------------
# Optional: predicted probabilities
# -----------------------------
probs = model.predict_proba(X_test)

probs_df = pd.DataFrame(
    probs,
    columns=[f"p_{cls}" for cls in label_encoder.classes_]
)

print("\nExample predicted probabilities:")
print(probs_df.head())