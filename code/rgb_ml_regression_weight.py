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
df = df[
    (df["total"] > 0) &
    (df["current_uA"] > 0) &
    (df["distance_mm"] > 0)
].copy()

# -----------------------------
# Aggregate per repeat-condition
# KEEP mean and std
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

# Replace NaN std values
for col in ["ambient_std", "R_std", "G_std", "B_std", "total_std"]:
    grouped[col] = grouped[col].fillna(0.0)

# Recompute total from means
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
grouped["log_distance"] = np.log(grouped["distance_mm"] + eps)

# Material / spectral information
grouped["R_norm"] = grouped["R"] / (grouped["total"] + eps)
grouped["G_norm"] = grouped["G"] / (grouped["total"] + eps)
grouped["B_norm"] = grouped["B"] / (grouped["total"] + eps)

grouped["R_over_G"] = grouped["R"] / (grouped["G"] + eps)
grouped["R_over_B"] = grouped["R"] / (grouped["B"] + eps)
grouped["G_over_B"] = grouped["G"] / (grouped["B"] + eps)

# -----------------------------
# Uncertainty / variability features
# -----------------------------
grouped["log_total_std"] = np.log(grouped["total_std"] + eps)
grouped["log_R_std"] = np.log(grouped["R_std"] + eps)
grouped["log_G_std"] = np.log(grouped["G_std"] + eps)
grouped["log_B_std"] = np.log(grouped["B_std"] + eps)
grouped["log_ambient_std"] = np.log(grouped["ambient_std"] + eps)

grouped["snr_total"] = grouped["total"] / (grouped["total_std"] + eps)
grouped["log_snr_total"] = np.log(grouped["snr_total"] + eps)

# -----------------------------
# Select features
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
    "log_total_std"
]

X = grouped[features]
y = grouped["log_distance"]          # train in log-space
y_mm = grouped["distance_mm"]        # keep original for evaluation

# -----------------------------
# Train/test split
# Stratify by original distance, not log-distance
# -----------------------------
X_train, X_test, y_train, y_test, y_train_mm, y_test_mm = train_test_split(
    X, y, y_mm,
    test_size=0.2,
    random_state=42,
    stratify=grouped["distance_mm"]
)

# -----------------------------
# XGBoost regressor
# -----------------------------
model = xgb.XGBRegressor(
    n_estimators=100000,
    max_depth=50,
    learning_rate=0.001,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict in log-space
y_pred_log = model.predict(X_test)

# Convert back to mm
y_pred_mm = np.exp(y_pred_log)
y_test_mm = np.array(y_test_mm)

# -----------------------------
# Evaluation in mm
# -----------------------------
mae = mean_absolute_error(y_test_mm, y_pred_mm)
rmse = np.sqrt(mean_squared_error(y_test_mm, y_pred_mm))

print(f"Number of grouped samples: {len(grouped)}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"MAE:  {mae:.2f} mm")
print(f"RMSE: {rmse:.2f} mm")

print("Train distances:", sorted(np.unique(y_train_mm)))
print("Test distances:", sorted(np.unique(y_test_mm)))

model.save_model("distance_model_loglog_with_std.json")

# -----------------------------
# Plot: True vs Predicted (mm)
# -----------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test_mm, y_pred_mm, alpha=0.7)
plt.plot([y_mm.min(), y_mm.max()], [y_mm.min(), y_mm.max()], "r--")
plt.xlabel("True Distance (mm)")
plt.ylabel("Predicted Distance (mm)")
plt.title("XGBoost Log-Log with Std Feature Included")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Plot: Absolute Error vs Distance (mm)
# -----------------------------
error = np.abs(y_test_mm - y_pred_mm)

model.save_model("distance_model_with_std_weight.json")

plt.figure(figsize=(7, 6))
plt.scatter(y_test_mm, error, alpha=0.7)
plt.xlabel("Distance (mm)")
plt.ylabel("Absolute Error (mm)")
plt.title("Absolute Error vs Distance")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Plot: Residual vs Distance
# -----------------------------
residual = y_pred_mm - y_test_mm

plt.figure(figsize=(7, 6))
plt.scatter(y_test_mm, residual, alpha=0.7)
plt.axhline(0, linestyle="--")
plt.xlabel("True Distance (mm)")
plt.ylabel("Residual (Predicted - True) (mm)")
plt.title("Residual vs Distance")
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