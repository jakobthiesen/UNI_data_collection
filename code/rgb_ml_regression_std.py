import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

# Drop rows with missing critical values
df = df.dropna(subset=[
    "distance_mm", "current_uA", "ambient", "R", "G", "B"
]).copy()

# Recompute total safely
df["total"] = df["R"] + df["G"] + df["B"]

# Keep only physically valid rows
df = df[(df["total"] > 0) & (df["current_uA"] > 0)].copy()

# -----------------------------
# Aggregate per repeat-condition
# KEEP mean and std this time
# -----------------------------
grouped = df.groupby(
    ["session_id", "repeat_id", "target_id", "distance_mm", "current_uA"],
    as_index=False
).agg({
    "ambient": ["mean", "std"],
    "R": ["mean", "std"],
    "G": ["mean", "std"],
    "B": ["mean", "std"],
    "total": ["mean", "std"]
})

# Flatten multi-index column names
grouped.columns = [
    "_".join(col).strip("_") if isinstance(col, tuple) else col
    for col in grouped.columns
]

# Rename for convenience
grouped = grouped.rename(columns={
    "ambient_mean": "ambient",
    "ambient_std": "ambient_std",
    "R_mean": "R",
    "R_std": "R_std",
    "G_mean": "G",
    "G_std": "G_std",
    "B_mean": "B",
    "B_std": "B_std",
    "total_mean": "total",
    "total_std": "total_std"
})

# Replace NaN std values (can happen if only one sample in a group)
for col in ["ambient_std", "R_std", "G_std", "B_std", "total_std"]:
    grouped[col] = grouped[col].fillna(0.0)

# Recompute total from means, just to be safe
grouped["total"] = grouped["R"] + grouped["G"] + grouped["B"]
grouped = grouped[(grouped["total"] > 0) & (grouped["current_uA"] > 0)].copy()

# -----------------------------
# Feature engineering
# -----------------------------
eps = 1e-9

grouped["log_total"] = np.log(grouped["total"] + eps)
grouped["log_current"] = np.log(grouped["current_uA"] + eps)

# Material / spectral information
grouped["R_norm"] = grouped["R"] / (grouped["total"] + eps)
grouped["G_norm"] = grouped["G"] / (grouped["total"] + eps)
grouped["B_norm"] = grouped["B"] / (grouped["total"] + eps)

grouped["R_over_G"] = grouped["R"] / (grouped["G"] + eps)
grouped["R_over_B"] = grouped["R"] / (grouped["B"] + eps)
grouped["G_over_B"] = grouped["G"] / (grouped["B"] + eps)

# Ambient feature
grouped["log_ambient"] = np.log(grouped["ambient"] + eps)

# -----------------------------
# NEW: uncertainty / variability features
# -----------------------------
grouped["log_total_std"] = np.log(grouped["total_std"] + eps)
grouped["log_R_std"] = np.log(grouped["R_std"] + eps)
grouped["log_G_std"] = np.log(grouped["G_std"] + eps)
grouped["log_B_std"] = np.log(grouped["B_std"] + eps)
grouped["log_ambient_std"] = np.log(grouped["ambient_std"] + eps)

# Optional SNR-like feature
grouped["snr_total"] = grouped["total"] / (grouped["total_std"] + eps)
grouped["log_snr_total"] = np.log(grouped["snr_total"] + eps)

# -----------------------------
# Select features
# Start simple: total std only
# -----------------------------
features = [
    "log_total",
    "log_current",
    "R_norm",
    "G_norm",
    "B_norm",
    "R_over_G",
    "R_over_B",
    "G_over_B",
    "log_ambient",

    # uncertainty-aware additions
    "log_total_std"
]

# If you want a slightly richer version later, use this instead:
# features = [
#     "log_total",
#     "log_current",
#     "R_norm",
#     "G_norm",
#     "B_norm",
#     "R_over_G",
#     "R_over_B",
#     "G_over_B",
#     "log_ambient",
#     "log_total_std",
#     "log_R_std",
#     "log_G_std",
#     "log_B_std",
#     "log_ambient_std",
#     "log_snr_total"
# ]

X = grouped[features]
y = grouped["distance_mm"]

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# XGBoost regressor
# -----------------------------
model = xgb.XGBRegressor(
    n_estimators=20000,
    max_depth=10,
    learning_rate=0.003,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Number of grouped samples: {len(grouped)}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"MAE:  {mae:.2f} mm")
print(f"RMSE: {rmse:.2f} mm")

print("Train distances:", sorted(y_train.unique()))
print("Test distances:", sorted(y_test.unique()))

# model.save_model("distance_model_with_std.json")

# -----------------------------
# Plot: True vs Predicted
# -----------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("True Distance (mm)")
plt.ylabel("Predicted Distance (mm)")
plt.title("XGBoost with Std Feature Included")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Plot: Error vs Distance
# -----------------------------
error = np.abs(y_test - y_pred)

plt.figure(figsize=(7, 6))
plt.scatter(y_test, error, alpha=0.7)
plt.xlabel("Distance (mm)")
plt.ylabel("Absolute Error (mm)")
plt.title("Absolute Error vs Distance")
plt.grid(True)
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
plt.title("Feature Importance")
plt.tight_layout()
plt.show()